#!/usr/bin/env python3
"""

vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --max-model-len 262144 --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser hermes -tp 4
vllm serve Seungyoun/Qwen3-4B-search-r1-w-selective-plan --max-model-len 262144 --enable-auto-tool-choice --tool-call-parser hermes -tp 4

Qwen3 Agentic Web Browser Demo
================================
"""

import asyncio
import json
import sys
import time
import threading
from typing import Any, Dict, List, Optional

from openai import OpenAI

# minimal-web-browser-env import
sys.path.insert(0, "/home/robin/minimal-web-browser/src")
from minimal_web_browser_env import DuckDuckGoBackend, WebBrowserEnv


# ANSI Color codes and control
class Colors:
    BLUE = "\033[94m"
    ORANGE = "\033[38;5;214m"
    GRAY = "\033[90m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[2K"
    CURSOR_UP = "\033[1A"


class ThinkingAnimation:
    """Animated thinking indicator that runs in a background thread."""
    
    def __init__(self, goal: str = ""):
        self.goal = goal
        self.running = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.frame_idx = 0
        
    def start(self):
        """Start the animation in a background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the animation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        # Clear the animation line
        print(f"\r{Colors.CLEAR_LINE}", end="", flush=True)
    
    def update_goal(self, goal: str):
        """Update the goal text."""
        self.goal = goal
    
    def _animate(self):
        """Animation loop."""
        while self.running:
            frame = self.frames[self.frame_idx % len(self.frames)]
            goal_display = self.goal[:70] + "..." if len(self.goal) > 70 else self.goal
            print(
                f"\r{Colors.CYAN}{frame} {Colors.BOLD}Next Step:{Colors.RESET} {Colors.CYAN}{goal_display}{Colors.RESET}",
                end="",
                flush=True
            )
            self.frame_idx += 1
            time.sleep(0.1)


# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web using DuckDuckGo and get a list of search results with links.",
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
            "description": "Search for a pattern within the currently active page and show matching excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for in the page",
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Optional: cursor position in history",
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
    "Answer the given question. Call the `thinking` tool whenever a short planning note will help. "
    "Fill the fields `goal`, `status`, and `next_step` to track your current focus, what you know so far, "
    "and the single next move. Use the `search` tool if you need external knowledge—the results will appear "
    "between <tool_response> and </tool_response>. Invoke tools as needed, then provide the final answer inside "
    "<answer> and </answer> without extra explanation. Question: "
)

async def execute_tool(env: WebBrowserEnv, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a browser tool and return the result as a string."""
    # Handle thinking tool separately
    if tool_name == "thinking":
        return "NOTE_ACCEPTED_PROCEED_WITH_NEXT_STEP"
    
    try:
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
        
        return "\n".join(result_parts)
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


async def run_agent(MODEL_NAME: str, query: str, max_turns: int = 10):
    """Run the agentic loop with Qwen3 model."""
    
    # Initialize OpenAI client (pointing to vLLM server)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",  # vLLM doesn't need a real API key
    )
    
    # Initialize browser environment
    env = WebBrowserEnv(
        backend=DuckDuckGoBackend(),
        preview_lines=60,
    )
    await env.reset()
    
    # Initialize conversation
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": DEFAULT_USER_CONTENT_PREFIX + query},
    ]
    
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}User Query: {query}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")
    
    # Initialize thinking animation
    animation = ThinkingAnimation(goal="Starting...")
    current_next_step = "Starting..."
    completed_steps = []  # Track completed steps
    
    # Start animation from the beginning
    animation.update_goal(current_next_step)
    animation.start()
    
    # Agentic loop
    for turn in range(max_turns):
        # Call the model
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
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
                
                # Handle thinking tool specially
                if func_name == "thinking":
                    # Mark previous step as completed
                    if current_next_step and current_next_step != "Starting...":
                        animation.stop()
                        print(f"\r{Colors.CLEAR_LINE}", end="")
                        completed_steps.append(current_next_step)
                        # Print all completed steps
                        for step in completed_steps:
                            step_display = step[:70] + "..." if len(step) > 70 else step
                            print(f"{Colors.GRAY}✓ {step_display}{Colors.RESET}")
                    
                    # Update the current next_step
                    current_next_step = func_args.get("next_step", current_next_step)
                    animation.update_goal(current_next_step)
                    animation.start()
                    
                    # Add tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "NOTE_ACCEPTED_PROCEED_WITH_NEXT_STEP",
                    })
                    continue
                
                # For non-thinking tools, keep animation running
                # Execute the tool silently
                result = await execute_tool(env, func_name, func_args)
                
                # Add tool response to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        
        else:
            # Model provided a final answer
            animation.stop()
            print(f"\r{Colors.CLEAR_LINE}", end="")  # Clear the animation line
            
            # Mark the last step as completed
            if current_next_step and current_next_step not in completed_steps:
                completed_steps.append(current_next_step)
            
            # Print all completed steps
            for step in completed_steps:
                step_display = step[:70] + "..." if len(step) > 70 else step
                print(f"{Colors.GRAY}✓ {step_display}{Colors.RESET}")
            
            print()  # Empty line before final answer
            print(f"{Colors.GREEN}{Colors.BOLD}✨ Final Answer:{Colors.RESET}")
            print(f"{Colors.GREEN}{message.content}{Colors.RESET}\n")
            
            print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
            print(f"{Colors.BOLD}Task completed in {turn + 1} turns{Colors.RESET}")
            print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")
            break
    
    else:
        animation.stop()
        print(f"\r{Colors.CLEAR_LINE}", end="")  # Clear the animation line
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}Max turns ({max_turns}) reached without completion{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")


async def main():
    """Main entry point."""
    # Example query
    query = "채널톡 PU 가 무슨뜻이야?"
    
    await run_agent("Seungyoun/Qwen3-4B-search-r1-w-selective-plan", query, max_turns=60)


if __name__ == "__main__":
    asyncio.run(main())
