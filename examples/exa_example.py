#!/usr/bin/env python3
"""
Example: Using ExaBackend for high-quality AI-powered search
=============================================================

This example demonstrates how to use ExaBackend, which provides:
- Neural search capabilities
- High-quality, relevant results
- Full text content extraction

Requirements:
- EXA_API_KEY environment variable set
- exa_py package installed: uv pip install exa_py
"""

import asyncio
import sys
sys.path.insert(0, "/home/robin/minimal-web-browser/src")

from minimal_web_browser_env import ExaBackend, WebBrowserEnv


async def main():
    print("=" * 80)
    print("Exa Backend Example")
    print("=" * 80)
    print()
    
    # Initialize Exa backend (requires EXA_API_KEY env var)
    backend = ExaBackend()
    env = WebBrowserEnv(backend=backend, preview_lines=50)
    
    # Example 1: Search for AI/ML content
    print("1. Searching for 'machine learning transformers'...")
    state = await env.step("search", query="machine learning transformers", topn=5)
    env.pretty_print(state)
    print()
    
    # Example 2: Open first result
    if state.get("links"):
        print("2. Opening first result...")
        state = await env.step("open", id=0)
        env.pretty_print(state)
        print()
    
    # Example 3: Find specific content
    print("3. Finding 'attention' in the page...")
    state = await env.step("find", pattern="attention")
    env.pretty_print(state)
    

if __name__ == "__main__":
    asyncio.run(main())

