#!/usr/bin/env python3
"""
GAIA Benchmark Evaluation Script
=================================
Evaluates models on the GAIA benchmark (gaia-benchmark/GAIA) validation set.
Computes Pass@1 metrics for Level 1, 2, 3, and overall average.

nohup uv run eval/gaia.py --model Qwen/Qwen3-4B-Instruct-2507 --max-turns 60 --output-dir ./gaia_results_qwen3_4b_instruct_2507 --backend exa > gaia_results_qwen3_4b_instruct_2507.log 2>&1 &
"""

import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# ANSI color helpers
GREEN = "\033[92m"
BLUE = "\033[94m"
ORANGE = "\033[33m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_green(message: str) -> None:
    print(f"{GREEN}{message}{RESET}")


def print_blue(message: str) -> None:
    print(f"{BLUE}{message}{RESET}")


def print_gray(message: str) -> None:
    print(f"{GRAY}{message}{RESET}")


def print_orange(message: str) -> None:
    print(f"{ORANGE}{message}{RESET}")

# minimal-web-browser-env import
sys.path.insert(0, "/home/robin/minimal-web-browser/src")
from minimal_web_browser_env import DuckDuckGoBackend, ExaBackend, WebBrowserEnv


# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web and get a list of search results with links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    },
                    "topn": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Open a link by its ID from the current page, or open a URL directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": ["integer", "string"],
                        "description": "Link ID (integer) or URL (string) to open",
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Optional: cursor position in history to use as base page",
                    },
                    "loc": {
                        "type": "integer",
                        "description": "Optional: line number to start viewing from",
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": "Optional: number of lines to display",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Search for a pattern within the currently active page and show matching excerpts. Returns a new view with only the matching sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for in the page",
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Optional: cursor position in history to search from",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select",
            "description": "Follow a link using its ID or a substring of its label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": ["integer", "string"],
                        "description": "Link ID (integer) or substring of link label (string)",
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Optional: cursor position in history",
                    },
                    "loc": {
                        "type": "integer",
                        "description": "Optional: line number to start viewing from",
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": "Optional: number of lines to display",
                    },
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "thinking",
            "description": "A planning tool to help you think through a problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The current goal or task."},
                    "status": {"type": "string", "description": "What you know so far."},
                    "next_step": {"type": "string", "description": "The single next step to take."}
                },
                "required": ["goal", "status", "next_step"],
            },
        },
    }
]


DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. Call the `thinking` tool whenever a short planning note will help. (DO NOT REPEAT THE SAME THINKING NOTE TWICE)"
    "Fill the fields `goal`, `status`, and `next_step` to track your current focus, what you know so far, "
    "and the single next move. Use the `search` tool if you need external knowledge—the results will appear "
    "between <tool_response> and </tool_response>. Invoke tools as needed, then provide the final answer inside "
    "<answer> and </answer> without extra explanation. Question: "
)


async def execute_tool(env: WebBrowserEnv, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a browser tool and return the result as a string."""
    # Handle thinking tool separately
    if tool_name == "thinking":
        print_orange(f"[Thinking] {arguments}")
        return "NOTE_ACCEPTED_PROCEED_WITH_NEXT_STEP"
    
    try:
        print_blue(f"[Tool Call] {tool_name} {arguments}")
        state = await env.step(tool_name, **arguments)
        
        # Format the response
        result_parts = []
        
        if state.get("current_page"):
            page = state["current_page"]
            result_parts.append(f"Title: {page['title']}")
            result_parts.append(f"URL: {page['url']}")
            result_parts.append(f"\n{page['preview']}")
        
        if state.get("links") and tool_name in ["search", "open"]:
            result_parts.append("\n--- Available Links ---")
            for link in state["links"][:15]:  # Limit to first 15 links
                result_parts.append(f"  [{link['id']}] {link['label']}")
        
        response_text = "\n".join(result_parts)
        preview = response_text if len(response_text) <= 200 else response_text[:200] + "... (truncated)"
        print_gray(f"[Tool Response] {preview}")
        return response_text
    
    except Exception as e:
        error_msg = str(e).lower() if str(e) else ""
        error_type = type(e).__name__
        
        # Check for timeout errors by type first
        if isinstance(e, asyncio.TimeoutError) or any(keyword in error_msg for keyword in ['timeout', 'timed out']):
            print_gray(f"[Tool Error] Timeout for {tool_name}: [{error_type}] {str(e) or 'Request timed out'}")
            return "Request timed out. The operation took too long. Please try again."
        
        # Check for rate limit errors
        elif any(keyword in error_msg for keyword in ['rate limit', 'ratelimit', 'too many requests', '429']):
            print_gray(f"[Tool Error] Rate Limit detected for {tool_name}: [{error_type}] {str(e)}")
            return "Rate limit reached. Please wait before retrying."
        
        # Check for other network errors
        elif any(keyword in error_msg for keyword in ['connection', 'network']):
            print_gray(f"[Tool Error] Network issue for {tool_name}: [{error_type}] {str(e)}")
            return f"Network error occurred: {str(e) or error_type}"
        
        # Log other errors as errors
        else:
            print_gray(f"[Tool Error] Failed to execute {tool_name}: [{error_type}] {str(e) or '(no message)'}")
            return f"Error executing {tool_name}: {str(e) or error_type}"


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    if not text:
        return None
    
    # Try to find answer tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If no tags found, return the whole text
    return text.strip()


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
    env: WebBrowserEnv,
    model_name: str,
    query: str,
    max_turns: int = 30
) -> Optional[str]:
    """Run inference on a single query and return the answer."""
    
    # Reset environment
    await env.reset()
    
    # Initialize conversation
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX + query},
    ]
    
    # Agentic loop
    for turn in range(max_turns):
        try:
            # Call the model
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
                top_p=1.0,
            )
            
            choice = response.choices[0]
            message = choice.message
            
            # Check if model wants to call tools
            if message.tool_calls:
                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    result = await execute_tool(env, func_name, func_args)
                    
                    # Add tool response to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            
            else:
                # Model provided a final answer
                answer = extract_answer(message.content)
                return answer if answer else message.content
        
        except Exception as e:
            print_green(f"[Error] Inference failed on turn {turn}: {str(e)}")
            return None
    
    # Max turns reached
    print_green(f"[Warn] Max turns ({max_turns}) reached without getting final answer")
    return None


async def evaluate_gaia(
    model_name: str = "Seungyoun/Qwen3-4B-search-r1-w-selective-plan",
    max_turns: int = 60,
    output_dir: str = "./gaia_results",
    backend: str = "duckduckgo"
):
    """Evaluate on GAIA benchmark validation set."""
    
    print_green("=" * 80)
    print_green("GAIA Benchmark Evaluation")
    print_green("=" * 80)
    print_green(f"Model: {model_name}")
    print_green(f"Max turns per query: {max_turns}")
    print_green(f"Search backend: {backend}")
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
    
    # Initialize search backend
    if backend.lower() == "exa":
        search_backend = ExaBackend()
        print_green("Using Exa search backend")
    elif backend.lower() == "duckduckgo":
        search_backend = DuckDuckGoBackend()
        print_green("Using DuckDuckGo search backend")
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'exa' or 'duckduckgo'")
    
    # Initialize browser environment
    env = WebBrowserEnv(
        backend=search_backend,
        preview_lines=240,
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
        prediction = await run_single_query(client, env, model_name, question, max_turns)
        
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
        "--max-turns",
        type=int,
        default=60,
        help="Maximum turns per query"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./gaia_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="exa",
        choices=["duckduckgo", "exa"],
        help="Search backend to use (duckduckgo or exa)"
    )
    
    args = parser.parse_args()
    
    await evaluate_gaia(
        model_name=args.model,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        backend=args.backend,
    )


if __name__ == "__main__":
    asyncio.run(main())