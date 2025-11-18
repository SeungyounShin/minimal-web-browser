"""
Env-style wrapper around :class:`BrowserTool`.
"""

from __future__ import annotations

import shlex
from typing import Any, Dict

from .backend import TutorialBackend
from .browser import BrowserError, BrowserTool


ACTION_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "search": {
        "description": "Push a new synthetic search results page.",
        "required": ["query"],
        "optional": ["topn"],
    },
    "open": {
        "description": "Open a link id from the active page or pass a URL string.",
        "required": [],
        "optional": ["id", "cursor", "loc", "num_lines"],
    },
    "find": {
        "description": "Search for `pattern` within the active page.",
        "required": ["pattern"],
        "optional": ["cursor"],
    },
    "select": {
        "description": "Follow a link using its id or label substring.",
        "required": ["target"],
        "optional": ["cursor", "loc", "num_lines"],
    },
    "snapshot": {
        "description": "Dump the entire page as Markdown for offline grep.",
        "required": [],
        "optional": ["cursor", "max_lines", "include_links"],
    },
}


class WebBrowserEnv:
    """Minimal agent env API for the tutorial browser."""

    def __init__(
        self,
        backend: TutorialBackend | None = None,
        preview_lines: int = 18,
    ) -> None:
        self.backend = backend or TutorialBackend()
        self.preview_lines = preview_lines
        self.browser = BrowserTool(backend=self.backend, view_lines=preview_lines)

    @property
    def action_space(self) -> Dict[str, Dict[str, Any]]:
        return ACTION_DEFINITIONS

    async def reset(self) -> Dict[str, Any]:
        self.browser = BrowserTool(backend=self.backend, view_lines=self.preview_lines)
        return self.get_state()

    async def step(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        action = action.lower()
        if action not in ACTION_DEFINITIONS:
            raise BrowserError(f"Unknown action `{action}`")
        required = ACTION_DEFINITIONS[action]["required"]
        missing = [arg for arg in required if arg not in kwargs]
        if missing:
            raise BrowserError(f"Missing required args for `{action}`: {missing}")
        handler = getattr(self.browser, action)
        await handler(**kwargs)
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "cursor": self.browser.current_cursor if self.browser.history() else None,
            "history": self.browser.history(),
            "current_page": None,
            "links": self.browser.links(),
            "available_actions": self.action_space,
            "last_response": self.browser.last_render(),
        }
        if self.browser.history():
            page = self.browser.active_page()
            state["current_page"] = {
                "title": page.title,
                "url": page.url,
                "preview": self.browser.last_render()[0]
                if self.browser.last_render()
                else "",
            }
        return state

    def pretty_print(self, state: Dict[str, Any] | None = None) -> None:
        state = state or self.get_state()
        print("-" * 72)
        print("Cursor:", state["cursor"])
        print("History:")
        if state["history"]:
            for entry in state["history"]:
                print(f"  [{entry['cursor']}] {entry['url']}")
        else:
            print("  (empty)")
        print("-" * 72)
        if state["current_page"]:
            print(state["current_page"]["preview"])
        else:
            print("No active page. Run `search` to populate the stack.")
        if state["links"]:
            print("\nLinks on active page:")
            for link in state["links"]:
                label = link.get("label", link["url"])
                print(f"  id={link['id']} ({label}) -> {link['url']}")
        if state["last_response"]:
            print("\nLast tool response:")
            for chunk in state["last_response"]:
                print(chunk)
        print("-" * 72)


def parse_kwargs(chunks: list[str]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for chunk in chunks:
        if "=" not in chunk:
            raise BrowserError(f"Expected key=value, got `{chunk}`")
        key, value = chunk.split("=", 1)
        kwargs[key] = value
    return kwargs

