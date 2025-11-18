"""
Lightweight browser actions that mimic the interface of the tutorial script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .backend import Page, TutorialBackend


class BrowserError(RuntimeError):
    """User-facing error for invalid cursor or link operations."""


@dataclass
class BrowserResponse:
    cursor: int
    title: str
    url: str
    scrollbar: str
    body: str

    def render(self) -> str:
        header = f"[{self.cursor}] {self.title} ({self.url})\n**{self.scrollbar}**\n\n"
        return header + self.body


class BrowserTool:
    """Minimal version of the gpt-oss SimpleBrowserTool."""

    def __init__(self, backend: TutorialBackend | None = None, view_lines: int = 20):
        self.backend = backend or TutorialBackend()
        self.view_lines = view_lines
        self._pages: List[Page] = []
        self._last_response: BrowserResponse | None = None

    # ------------------------------------------------------------------ state
    @property
    def current_cursor(self) -> int:
        return len(self._pages) - 1

    def history(self) -> List[dict]:
        return [{"cursor": idx, "url": page.url} for idx, page in enumerate(self._pages)]

    def active_page(self) -> Page:
        if not self._pages:
            raise BrowserError("No pages loaded yet. Run `search` first.")
        return self._pages[self.current_cursor]

    def links(self) -> List[dict]:
        if not self._pages:
            return []
        page = self.active_page()
        result = []
        for link_id, url in page.links.items():
            result.append(
                {
                    "id": link_id,
                    "url": url,
                    "label": page.link_labels.get(link_id, url),
                }
            )
        return result

    def last_render(self) -> list[str]:
        if not self._last_response:
            return []
        return [self._last_response.render()]

    # ---------------------------------------------------------------- actions
    async def search(self, query: str, topn: int = 10) -> BrowserResponse:
        page = await self.backend.search(query=query, topn=topn)
        self._pages.append(page)
        self._last_response = self._render_page(page, loc=0, num_lines=self.view_lines)
        return self._last_response

    async def open(
        self,
        id: int | str | None = None,
        cursor: int | None = None,
        loc: int | None = None,
        num_lines: int | None = None,
    ) -> BrowserResponse:
        base_page = self._get_page(cursor) if self._pages else None

        target_url: str | None = None
        line_hint: int | None = None
        if isinstance(id, str) and not id.lstrip("-").isdigit():
            target_url = id
        elif id is not None and int(id) >= 0:
            link_id = str(int(id))
            if base_page is None:
                raise BrowserError("No pages loaded yet. Provide a full URL to open.")
            if link_id not in base_page.links:
                raise BrowserError(f"Invalid link id `{link_id}` on current page.")
            target_url = base_page.links[link_id]
            line_hint = base_page.link_line_index.get(link_id)

        if target_url:
            try:
                page = await self.backend.fetch(target_url)
            except Exception as exc:  # noqa: BLE001
                raise BrowserError(f"Failed to fetch `{target_url}`: {exc}") from exc
            self._pages.append(page)
        else:
            if base_page is None:
                raise BrowserError("No pages available. Run `search` or open a URL first.")
            page = base_page

        loc_to_use = loc
        if loc_to_use is None:
            if line_hint is not None:
                loc_to_use = max(line_hint - 2, 0)
            else:
                loc_to_use = 0

        response = self._render_page(
            page,
            loc=loc_to_use,
            num_lines=num_lines if num_lines is not None else self.view_lines,
        )
        self._last_response = response
        return response

    async def find(self, pattern: str, cursor: int | None = None) -> BrowserResponse:
        if not pattern:
            raise BrowserError("`pattern` must be non-empty.")
        page = self._get_page(cursor)
        find_page = self._run_find(page, pattern)
        self._pages.append(find_page)
        response = self._render_page(find_page, loc=0, num_lines=self.view_lines)
        self._last_response = response
        return response

    async def snapshot(
        self,
        cursor: int | None = None,
        max_lines: int | None = None,
        include_links: bool = True,
    ) -> BrowserResponse:
        page = self._get_page(cursor)
        response = self._render_snapshot(page, max_lines=max_lines, include_links=include_links)
        self._last_response = response
        return response

    async def select(
        self,
        target: int | str,
        cursor: int | None = None,
        loc: int | None = None,
        num_lines: int | None = None,
    ) -> BrowserResponse:
        page = self._get_page(cursor)
        link_id = self._resolve_link_id(page, target)
        return await self.open(id=link_id, cursor=cursor, loc=loc, num_lines=num_lines)

    # ---------------------------------------------------------------- helpers
    def _get_page(self, cursor: int | None) -> Page:
        if not self._pages:
            raise BrowserError("No pages loaded yet.")
        if cursor is None or cursor < 0:
            return self._pages[self.current_cursor]
        if cursor >= len(self._pages):
            raise BrowserError(
                f"Cursor `{cursor}` is out of range. "
                f"Valid range: 0-{self.current_cursor}."
            )
        return self._pages[cursor]

    def _render_page(self, page: Page, loc: int, num_lines: int) -> BrowserResponse:
        lines = page.text.splitlines()
        total_lines = max(len(lines) - 1, 0)
        loc = max(0, min(loc, len(lines)))
        end = len(lines) if num_lines < 0 else min(len(lines), loc + num_lines)
        snippet = lines[loc:end]
        if not snippet:
            snippet = [""]
        body_lines = [
            f"L{loc + idx}: {line}" for idx, line in enumerate(snippet)
        ]
        if end < len(lines):
            body_lines.append(f"... ({len(lines) - end} more lines)")
        scrollbar = f"viewing lines [{loc} - {max(end - 1, loc)}] of {total_lines}"
        return BrowserResponse(
            cursor=self.current_cursor,
            title=page.title,
            url=page.url,
            scrollbar=scrollbar,
            body="\n".join(body_lines),
        )

    def _render_snapshot(
        self,
        page: Page,
        *,
        max_lines: int | None,
        include_links: bool,
    ) -> BrowserResponse:
        lines = page.text.splitlines()
        total_lines = len(lines)
        if max_lines is not None and max_lines >= 0:
            limit = min(max_lines, total_lines)
        else:
            limit = total_lines

        body_lines: list[str] = []
        if include_links and page.links:
            body_lines.append("Links:")
            for link_id, url in page.links.items():
                label = page.link_labels.get(link_id, url)
                body_lines.append(f"  [{link_id}] {label} -> {url}")
            body_lines.append("")  # separator

        for idx in range(limit):
            body_lines.append(f"L{idx}: {lines[idx]}")

        if limit < total_lines:
            body_lines.append(f"... ({total_lines - limit} more lines)")
        if not body_lines:
            body_lines.append("(empty page)")

        scrollbar = f"snapshot of {total_lines} lines"
        return BrowserResponse(
            cursor=self.current_cursor,
            title=page.title,
            url=page.url,
            scrollbar=scrollbar,
            body="\n".join(body_lines),
        )

    def _run_find(self, page: Page, pattern: str) -> Page:
        target = pattern.lower()
        lines = page.text.splitlines()
        chunks: List[str] = []
        for idx, line in enumerate(lines):
            if target in line.lower():
                window = lines[idx : idx + 4]
                block = "\n".join([f"L{idx + offset}: {value}" for offset, value in enumerate(window)])
                chunks.append(f"Match at L{idx}\n{block}")
        if not chunks:
            body = f"No matches for `{pattern}`"
        else:
            body = "\n\n".join(chunks)
        return Page(
            url=f"{page.url}#find:{pattern}",
            title=f"Find `{pattern}` in {page.title}",
            text=body,
            links={},
        )

    def _resolve_link_id(self, page: Page, target: int | str) -> str:
        if isinstance(target, int):
            link_id = str(target)
            if link_id in page.links:
                return link_id
        elif isinstance(target, str):
            if target.isdigit() and target in page.links:
                return target
            lowered = target.lower()
            for link_id, label in page.link_labels.items():
                if lowered in label.lower():
                    return link_id
        raise BrowserError(f"No link matching `{target}` on current page.")

