#!/usr/bin/env python3
"""
GAIA Benchmark Evaluation Script
=================================
Evaluates models on the GAIA benchmark (gaia-benchmark/GAIA) validation set.
Computes Pass@1 metrics for Level 1, 2, 3, and overall average.

uv run eval/gaia-direct.py --model Qwen/Qwen3-4B-Instruct-2507 --output-dir ./gaia_results_qwen3_4b_instruct_2507
"""

import asyncio
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ANSI color helpers
GREEN = "\033[92m"
RESET = "\033[0m"


def print_green(message: str) -> None:
    print(f"{GREEN}{message}{RESET}")


DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Provide the final answer inside "
    "<answer> and </answer> without extra explanation. Question: "
)


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


async def run_single_query(
    client: OpenAI,
    model_name: str,
    query: str,
) -> Optional[str]:
    """Run a single-turn inference without tool use."""
    
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX + query},
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
        )
        message = response.choices[0].message
        print(message.content)
        answer = extract_answer(message.content)
        return answer if answer else message.content
    except Exception as e:
        print_green(f"[Error] Direct inference failed: {str(e)}")
        return None


async def evaluate_gaia(
    model_name: str = "Seungyoun/Qwen3-4B-search-r1-w-selective-plan",
    output_dir: str = "./gaia_results",
):
    """Evaluate on GAIA benchmark validation set."""
    
    print_green("=" * 80)
    print_green("GAIA Benchmark Evaluation")
    print_green("=" * 80)
    print_green(f"Model: {model_name}")
    print_green("=" * 80)
    print()
    
    # Load dataset
    print_green("Loading GAIA validation dataset...")
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
    print_green(f"Loaded {len(dataset)} samples\n")
    
    # Initialize OpenAI client (pointing to vLLM server)
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="EMPTY",
    )
    
    # Track results by level
    results_by_level = defaultdict(list)
    all_results = []
    
    # Evaluate each sample
    pbar = tqdm(dataset, desc="Evaluating", dynamic_ncols=True)
    
    for idx, sample in enumerate(pbar):
        task_id = sample["task_id"]
        question = sample["Question"]
        level = sample["Level"]
        ground_truth = sample["Final answer"]
        
        print_green(f"\n[Question {idx + 1}/{len(dataset)} | Task {task_id}] {question}")
        
        # Run inference
        prediction = await run_single_query(client, model_name, question)
        
        # Check if correct
        is_correct = check_exact_match(prediction or "", ground_truth)
        status = "✅ correct" if is_correct else "❌ incorrect"
        print_green("=====")
        print_green(f"{task_id} {status}")
        print_green(f"pred : {prediction or '(no prediction)'}")
        print_green(f"gt   : {ground_truth}")
        print_green("-======")
        
        # Store result
        result = {
            "task_id": task_id,
            "question": question,
            "level": level,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": is_correct,
        }
        
        results_by_level[level].append(result)
        all_results.append(result)
        
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
                level_metrics[f"Level {lvl}"] = lvl_pass_at_1
        
        # Update progress bar
        metrics_str = " | ".join([f"{k}: {v:.1f}%" for k, v in level_metrics.items()])
        pbar.set_postfix_str(f"Avg: {avg_pass_at_1:.1f}% | {metrics_str}")
    
    print_green("\n" + "=" * 80)
    print_green("Evaluation Complete!")
    print_green("=" * 80)
    
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
            print_green(f"Level {level} Pass@1: {pass_at_1:.2f}% ({correct}/{total})")
    
    # Overall average
    total_correct = sum(1 for r in all_results if r["correct"])
    total_samples = len(all_results)
    avg_pass_at_1 = total_correct / total_samples * 100
    final_metrics["Average"] = {
        "pass@1": avg_pass_at_1,
        "correct": total_correct,
        "total": total_samples,
    }
    print_green(f"\nAverage Pass@1: {avg_pass_at_1:.2f}% ({total_correct}/{total_samples})")
    print_green("=" * 80)
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = output_dir / f"gaia_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "timestamp": timestamp,
            "metrics": final_metrics,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    
    print_green(f"\nResults saved to: {results_file}")
    
    # Save summary only
    summary_file = output_dir / f"gaia_summary_{timestamp}.json"
    summary_results = [
        {
            "task_id": r["task_id"],
            "prediction": r["prediction"],
            "ground_truth": r["ground_truth"],
        }
        for r in all_results
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    print_green(f"Summary saved to: {summary_file}")
    print()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate on GAIA benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="Seungyoun/Qwen3-4B-search-r1-w-selective-plan",
        help="Model name to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gaia_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    await evaluate_gaia(
        model_name=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())