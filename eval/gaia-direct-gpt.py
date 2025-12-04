#!/usr/bin/env python3
"""
GAIA Benchmark Evaluation Script with OpenAI GPT-5.1
====================================================
Evaluates GPT-5.1 on the GAIA benchmark (gaia-benchmark/GAIA) validation set.
Uses filtered samples (web browsing only, no multimodal) - approximately 104 samples.
Computes Pass@1 metrics for Level 1, 2, 3, and overall average.

Usage:
    uv run eval/gaia-direct-gpt.py --output-dir ./gaia_results
"""

import asyncio
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI, APITimeoutError, APIConnectionError
from tqdm import tqdm

# ANSI color helpers
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_green(message: str) -> None:
    print(f"{GREEN}{message}{RESET}")


def print_yellow(message: str) -> None:
    print(f"{YELLOW}{message}{RESET}")


def print_cyan(message: str) -> None:
    print(f"{CYAN}{message}{RESET}")


# System and user prompts
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Provide the final answer inside "
    "<answer> and </answer> without extra explanation. Question: "
)

# Excluded tool keywords for filtering (same as recompute_filtered_metrics.py)
EXCLUDED_TOOL_KEYWORDS = [
    "Image recognition",
    "PowerPoint viewer",
    "Audio processing",
    "Video processing",
    "OCR",
    "Computer vision",
    "Color recognition",
    "Python IDE",
    "Video"
]


def has_excluded_tools(metadata: dict) -> bool:
    """Check if Annotator Metadata contains excluded tools."""
    if not metadata:
        return False
    
    tools_text = metadata.get("Tools", "")
    if not tools_text:
        return False
    
    tools_lower = tools_text.lower()
    for keyword in EXCLUDED_TOOL_KEYWORDS:
        if keyword.lower() in tools_lower:
            return True
    
    return False


def is_web_browsing_only(ex: dict) -> bool:
    """Check if a sample is suitable for web-browsing only evaluation."""
    # file_name check - exclude samples that require file processing
    if ex.get("file_name") not in (None, "", "NA"):
        return False
    
    # Tools check - exclude samples that require multimodal tools
    metadata = ex.get("Annotator Metadata", {})
    if has_excluded_tools(metadata):
        return False
    
    return True


def extract_answer(text: str) -> Optional[str]:
    """Extract answer even if <answer> tags are missing."""
    if not text:
        return None
    
    # Preferred: explicit answer tags
    tag_patterns = [
        r'<answer>(.*?)</answer>',
        r'<final_answer>(.*?)</final_answer>',
        r'<final>(.*?)</final>',
    ]
    for pattern in tag_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    
    # Fallback: look for lines starting with "answer:" variants
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        prefixes = ["answer:", "final answer:", "final:", "result:", "response:"]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                value = line[len(prefix):].strip()
                if value:
                    return value
    
    # Last resort: return last non-empty line
    return lines[-1] if lines else text.strip()


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()
    # Remove punctuation at the end
    answer = answer.rstrip('.,;:!?')
    return answer


def check_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth (case-insensitive, normalized)."""
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return pred_norm == gt_norm


def run_single_query(
    client: OpenAI,
    model_name: str,
    query: str,
    reasoning_effort: str = "medium",
    max_retries: int = 3,
    timeout: int = 300,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Run a single-turn inference with GPT-5.1 reasoning model.
    Returns (answer, reasoning_content) tuple.
    """
    
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX + query},
    ]
    
    for attempt in range(max_retries):
        try:
            # GPT-5.1 supports reasoning_effort parameter via extra_body
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=timeout,
                extra_body={"reasoning_effort": reasoning_effort},
            )
            
            message = response.choices[0].message
            content = message.content or ""
            
            # Extract reasoning content if available (for reasoning models like GPT-5.1)
            reasoning_content = None
            # Try different possible attribute names for reasoning content
            for attr in ['reasoning_content', 'reasoning', 'thinking']:
                if hasattr(message, attr):
                    val = getattr(message, attr)
                    if val:
                        reasoning_content = val if isinstance(val, str) else str(val)
                        break
            
            # Log the response
            print_cyan(f"\n[Reasoning Content Preview]")
            if reasoning_content:
                # Show first 500 chars of reasoning
                preview = reasoning_content[:500] + "..." if len(reasoning_content) > 500 else reasoning_content
                print(preview)
            else:
                print("(No reasoning content available)")
            
            print_cyan(f"\n[Model Response]")
            print(content)
            
            answer = extract_answer(content)
            return (answer if answer else content, reasoning_content)
            
        except (APITimeoutError, APIConnectionError) as e:
            print_yellow(f"[Warning] Request timeout/connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print_yellow(f"[Info] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print_yellow(f"[Error] All retry attempts failed.")
                return (None, None)
        except Exception as e:
            print_yellow(f"[Error] Direct inference failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return (None, None)
    
    return (None, None)


def load_checkpoint(checkpoint_file: Path) -> Dict:
    """Load checkpoint from file."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_task_ids": [], "results": []}


def save_checkpoint(checkpoint_file: Path, completed_task_ids: List[str], results: List[Dict]):
    """Save checkpoint to file."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump({
            "completed_task_ids": completed_task_ids,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)


def print_final_report(final_metrics: Dict, model_name: str) -> str:
    """Generate and print final evaluation report."""
    report_lines = []
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("📊 GAIA Benchmark Evaluation Report (GPT-5.1)")
    report_lines.append("=" * 80)
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("-" * 40)
    report_lines.append("Results by Level:")
    report_lines.append("-" * 40)
    
    for level in ["1", "2", "3"]:
        level_key = f"Level {level}"
        if level_key in final_metrics:
            m = final_metrics[level_key]
            report_lines.append(f"  Level {level}: {m['pass@1']:.2f}% ({m['correct']}/{m['total']})")
    
    report_lines.append("-" * 40)
    
    if "Average" in final_metrics:
        m = final_metrics["Average"]
        report_lines.append(f"  Overall: {m['pass@1']:.2f}% ({m['correct']}/{m['total']})")
    
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    print_green(report)
    return report


async def evaluate_gaia(
    model_name: str = "gpt-5.1",
    output_dir: str = "./gaia_results",
    reasoning_effort: str = "medium",
    checkpoint_interval: int = 5,
    resume: bool = True,
):
    """Evaluate GPT-5.1 on GAIA benchmark validation set (filtered samples only)."""
    
    print_green("=" * 80)
    print_green("GAIA Benchmark Evaluation (GPT-5.1)")
    print_green("=" * 80)
    print_green(f"Model: {model_name}")
    print_green(f"Reasoning Effort: {reasoning_effort}")
    print_green("Filter: Web-browsing only (no file/multimodal)")
    print_green("=" * 80)
    print()
    
    # Setup checkpoint file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint_gpt51.json"
    
    # Load checkpoint if resuming
    completed_task_ids = []
    all_results = []
    if resume:
        checkpoint = load_checkpoint(checkpoint_file)
        completed_task_ids = checkpoint.get("completed_task_ids", [])
        all_results = checkpoint.get("results", [])
        if completed_task_ids:
            print_green(f"Resuming from checkpoint: {len(completed_task_ids)} tasks already completed")
    
    # Load dataset
    print_green("Loading GAIA validation dataset...")
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
    print_green(f"Loaded {len(dataset)} total samples")
    
    # Filter to web-browsing only samples
    filtered_dataset = [sample for sample in dataset if is_web_browsing_only(sample)]
    print_green(f"Filtered to {len(filtered_dataset)} web-browsing only samples")
    
    # Count by level
    level_counts = defaultdict(int)
    for sample in filtered_dataset:
        level_counts[sample["Level"]] += 1
    print_green(f"  Level 1: {level_counts['1']}, Level 2: {level_counts['2']}, Level 3: {level_counts['3']}")
    print()
    
    # Filter out completed tasks
    if completed_task_ids:
        remaining_dataset = [sample for sample in filtered_dataset if sample["task_id"] not in completed_task_ids]
        print_green(f"Skipping {len(completed_task_ids)} completed tasks, {len(remaining_dataset)} remaining\n")
    else:
        remaining_dataset = filtered_dataset
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    client = OpenAI(api_key=api_key)
    
    # Track results by level
    results_by_level = defaultdict(list)
    
    # Rebuild results_by_level from checkpoint
    for result in all_results:
        results_by_level[result["level"]].append(result)
    
    # Evaluate each sample
    pbar = tqdm(
        remaining_dataset, 
        desc="Evaluating", 
        dynamic_ncols=True, 
        initial=len(completed_task_ids), 
        total=len(filtered_dataset)
    )
    
    for idx, sample in enumerate(pbar):
        task_id = sample["task_id"]
        question = sample["Question"]
        level = sample["Level"]
        ground_truth = sample["Final answer"]
        
        print_green(f"\n{'='*80}")
        print_green(f"[Question {len(completed_task_ids) + idx + 1}/{len(filtered_dataset)} | Task {task_id} | Level {level}]")
        print_green(f"Q: {question[:200]}..." if len(question) > 200 else f"Q: {question}")
        print_green("=" * 80)
        
        # Run inference
        prediction, reasoning_content = run_single_query(
            client, model_name, question, reasoning_effort
        )
        
        # Check if correct
        is_correct = check_exact_match(prediction or "", ground_truth)
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        
        print_green("\n" + "-" * 40)
        print_green(f"Status: {status}")
        print_green(f"Prediction: {prediction or '(no prediction)'}")
        print_green(f"Ground Truth: {ground_truth}")
        print_green("-" * 40)
        
        # Store result
        result = {
            "task_id": task_id,
            "question": question,
            "level": level,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "reasoning_content": reasoning_content,
        }
        
        results_by_level[level].append(result)
        all_results.append(result)
        completed_task_ids.append(task_id)
        
        # Save checkpoint periodically
        if (idx + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_file, completed_task_ids, all_results)
            print_green(f"[Checkpoint] Saved progress at {len(completed_task_ids)} completed tasks")
        
        # Calculate current metrics
        total_correct = sum(1 for r in all_results if r["correct"])
        total_samples = len(all_results)
        avg_pass_at_1 = total_correct / total_samples * 100 if total_samples > 0 else 0
        
        # Calculate per-level metrics
        level_metrics = {}
        for lvl in ["1", "2", "3"]:
            lvl_results = results_by_level.get(lvl, [])
            if lvl_results:
                lvl_correct = sum(1 for r in lvl_results if r["correct"])
                lvl_pass_at_1 = lvl_correct / len(lvl_results) * 100
                level_metrics[f"L{lvl}"] = lvl_pass_at_1
        
        # Update progress bar
        metrics_str = " | ".join([f"{k}: {v:.1f}%" for k, v in level_metrics.items()])
        pbar.set_postfix_str(f"Avg: {avg_pass_at_1:.1f}% | {metrics_str}")
    
    # Save final checkpoint
    save_checkpoint(checkpoint_file, completed_task_ids, all_results)
    print_green(f"\n[Checkpoint] Final checkpoint saved")
    
    # Calculate final metrics
    final_metrics = {}
    for level in ["1", "2", "3"]:
        level_results = results_by_level.get(level, [])
        if level_results:
            correct = sum(1 for r in level_results if r["correct"])
            total = len(level_results)
            pass_at_1 = correct / total * 100
            final_metrics[f"Level {level}"] = {
                "pass@1": pass_at_1,
                "correct": correct,
                "total": total,
            }
    
    # Overall average
    total_correct = sum(1 for r in all_results if r["correct"])
    total_samples = len(all_results)
    avg_pass_at_1 = total_correct / total_samples * 100 if total_samples > 0 else 0
    final_metrics["Average"] = {
        "pass@1": avg_pass_at_1,
        "correct": total_correct,
        "total": total_samples,
    }
    
    # Print final report
    report = print_final_report(final_metrics, model_name)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results with reasoning content
    results_file = output_dir / f"gaia_gpt51_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "reasoning_effort": reasoning_effort,
            "timestamp": timestamp,
            "filter": "web_browsing_only",
            "metrics": final_metrics,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    
    print_green(f"\nDetailed results saved to: {results_file}")
    
    # Save summary (without reasoning content for quick access)
    summary_file = output_dir / f"gaia_gpt51_summary_{timestamp}.json"
    summary_results = [
        {
            "task_id": r["task_id"],
            "level": r["level"],
            "prediction": r["prediction"],
            "ground_truth": r["ground_truth"],
            "correct": r["correct"],
        }
        for r in all_results
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "reasoning_effort": reasoning_effort,
            "metrics": final_metrics,
            "results": summary_results,
        }, f, indent=2, ensure_ascii=False)
    
    print_green(f"Summary saved to: {summary_file}")
    
    # Save report as text file
    report_file = output_dir / f"gaia_gpt51_report_{timestamp}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\n")
        f.write("Per-sample results:\n")
        f.write("-" * 80 + "\n")
        for r in all_results:
            status = "✅" if r["correct"] else "❌"
            f.write(f"{status} [{r['level']}] {r['task_id']}\n")
            f.write(f"   Q: {r['question'][:100]}...\n" if len(r['question']) > 100 else f"   Q: {r['question']}\n")
            f.write(f"   Pred: {r['prediction']}\n")
            f.write(f"   GT: {r['ground_truth']}\n")
            f.write("-" * 80 + "\n")
    
    print_green(f"Report saved to: {report_file}")
    print()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.1 on GAIA benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="Model name to evaluate (default: gpt-5.1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gaia_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level (default: medium)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N samples"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch, ignoring any existing checkpoint"
    )
    
    args = parser.parse_args()
    
    await evaluate_gaia(
        model_name=args.model,
        output_dir=args.output_dir,
        reasoning_effort=args.reasoning_effort,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    asyncio.run(main())

