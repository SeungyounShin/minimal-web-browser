"""
Minimal synthetic browser environment.

Exports:
    TutorialBackend -- in-memory backend with deterministic pages
    BrowserTool -- lightweight browser action handler
    WebBrowserEnv -- Env-style wrapper around BrowserTool
"""

from .backend import DuckDuckGoBackend, TutorialBackend
from .browser import BrowserTool, BrowserError
from .env import WebBrowserEnv

__all__ = [
    "TutorialBackend",
    "DuckDuckGoBackend",
    "BrowserTool",
    "BrowserError",
    "WebBrowserEnv",
]

