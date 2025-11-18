"""
Interactive CLI for the minimal browser environment.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from typing import Any, Dict

from .backend import DuckDuckGoBackend, TutorialBackend
from .browser import BrowserError
from .env import ACTION_DEFINITIONS, WebBrowserEnv

INT_FIELDS = {"cursor", "loc", "num_lines", "topn", "max_lines"}
BOOL_FIELDS = {"include_links"}

PROMPT = """
Commands:
  - search query="gpt-oss"
  - open id=0
  - open id=https://seungyoun.github.io
  - select target="browser actions"
  - snapshot
  - find pattern=browser
  - help
  - quit
"""


def _coerce_kwargs(kwargs: Dict[str, str]) -> Dict[str, Any]:
    coerced: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in INT_FIELDS:
            coerced[key] = int(value)
        elif key in BOOL_FIELDS:
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "y"}:
                coerced[key] = True
            elif lowered in {"false", "0", "no", "n"}:
                coerced[key] = False
            else:
                raise BrowserError(f"Expected boolean for `{key}`, got `{value}`")
        elif key in {"id", "target"} and value.lstrip("-").isdigit():
            coerced[key] = int(value)
        else:
            coerced[key] = value
    return coerced


def _print_action_specs(env: WebBrowserEnv) -> None:
    print("Available actions:")
    for name, spec in env.action_space.items():
        print(f"- {name}: {spec['description']}")
        if spec["required"]:
            print(f"    required: {spec['required']}")
        if spec["optional"]:
            print(f"    optional: {spec['optional']}")


async def run_cli() -> None:
    backend_name = os.getenv("MINI_BROWSER_BACKEND", "tutorial").lower()
    if backend_name == "duckduckgo":
        backend = DuckDuckGoBackend()
    else:
        backend = TutorialBackend()

    view_lines = os.getenv("MINI_BROWSER_VIEW_LINES")
    try:
        preview_lines = int(view_lines) if view_lines is not None else 18
    except ValueError:
        raise BrowserError(
            f"Invalid MINI_BROWSER_VIEW_LINES value `{view_lines}` (expected integer)."
        )

    env = WebBrowserEnv(backend=backend, preview_lines=preview_lines)
    state = await env.reset()
    print(PROMPT)
    _print_action_specs(env)
    env.pretty_print(state)

    while True:
        raw = input("action> ").strip()
        if not raw:
            continue
        if raw.lower() in {"quit", "exit"}:
            break
        if raw.lower() == "help":
            _print_action_specs(env)
            continue
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            print(f"[parse error] {exc}")
            continue
        action, *arg_chunks = parts
        kwargs = {}
        try:
            for chunk in arg_chunks:
                if "=" not in chunk:
                    raise BrowserError(f"Expected key=value, got `{chunk}`")
                key, value = chunk.split("=", 1)
                kwargs[key] = value
            kwargs = _coerce_kwargs(kwargs)
            state = await env.step(action, **kwargs)
            env.pretty_print(state)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}")


def main() -> None:
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()

