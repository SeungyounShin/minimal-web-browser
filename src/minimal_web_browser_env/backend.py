"""
In-memory backend that mimics a browsing data source.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup
from html2text import HTML2Text

logger = logging.getLogger(__name__)

try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False
    logger.warning("exa_py not installed. ExaBackend will not be available.")


@dataclass(frozen=True)
class Page:
    """Plaintext page representation surfaced to the browser tool."""

    url: str
    title: str
    text: str
    links: Dict[str, str] = field(default_factory=dict)
    link_labels: Dict[str, str] = field(default_factory=dict)
    link_line_index: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _PageDefinition:
    url: str
    title: str
    summary: str
    paragraphs: List[str]
    links: List[tuple[str, str]]


REMOTE_HEADERS = {
    "User-Agent": "minimal-web-browser-env/0.1 (+https://github.com/)",
}


class TutorialBackend:
    """
    Supplies deterministic tutorial pages so the browser can run without
    network requests or API keys.
    """

    source: str = "tutorial-data"

    def __init__(self, max_links: int = 30, request_timeout: float = 30.0) -> None:
        self._pages = self._build_pages()
        self._page_map: Dict[str, Page] = {
            page.url: self._page_to_contents(page) for page in self._pages
        }
        self.max_links = max_links
        self.request_timeout = request_timeout

    async def search(self, query: str, topn: int = 10) -> Page:
        intro = [
            f"Search results for `{query}`",
            "Use `open id=<number>` to load a snippet.",
        ]
        urls: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        snippets: List[str] = []
        for idx, page in enumerate(self._pages[:topn]):
            urls[str(idx)] = page.url
            labels[str(idx)] = page.title
            snippets.append(f"【{idx}†{page.title}】\n{page.summary}")

        body = "\n\n".join(intro + snippets)
        return Page(
            url="tutorial://search",
            title=f"Tutorial search: {query}",
            text=body,
            links=urls,
            link_labels=labels,
            link_line_index={},
        )

    async def fetch(self, url: str) -> Page:
        resolved_url = self._unwrap_redirect(url)
        if resolved_url in self._page_map:
            return self._page_map[resolved_url]
        if resolved_url.startswith(("http://", "https://")):
            return await self._fetch_remote(resolved_url)
        raise ValueError(f"No tutorial page for `{url}`")

    def _unwrap_redirect(self, url: str) -> str:
        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            query = parse_qs(parsed.query)
            uddg_values = query.get("uddg")
            if uddg_values:
                return unquote(uddg_values[0])
        return url

    def _page_to_contents(self, page: _PageDefinition) -> Page:
        link_map: Dict[str, str] = {}
        label_map: Dict[str, str] = {}
        text_blocks = list(page.paragraphs)
        if page.links:
            tokens = []
            for idx, (label, target) in enumerate(page.links):
                link_map[str(idx)] = target
                label_map[str(idx)] = label
                tokens.append(f"【{idx}†{label}】")
            text_blocks.append("Related: " + ", ".join(tokens))
        rendered = "\n\n".join(text_blocks)
        line_map = _compute_link_line_indices(rendered, link_map)
        return Page(
            url=page.url,
            title=page.title,
            text=rendered,
            links=link_map,
            link_labels=label_map,
            link_line_index=line_map,
        )

    def _build_pages(self) -> List[_PageDefinition]:
        return [
            _PageDefinition(
                url="tutorial://gpt-oss",
                title="What is gpt-oss?",
                summary="Overview of the open-source GPT reproduction project.",
                paragraphs=[
                    "gpt-oss is an educational reproduction that showcases how small "
                    "teams can experiment with reasoning and tool use.",
                    "It provides interchangeable inference backends plus simple tools "
                    "like browsing so you can plug in your own checkpoints.",
                    "Components stay decoupled, which makes it easy to embed the model "
                    "inside custom agent loops.",
                ],
                links=[
                    ("browser actions", "tutorial://browser-actions"),
                    ("citations", "tutorial://citations"),
                ],
            ),
            _PageDefinition(
                url="tutorial://browser-actions",
                title="Browser actions",
                summary="How search, open, and find interact with the page stack.",
                paragraphs=[
                    "Every call pushes a new page to the stack and updates the active cursor.",
                    "search(query=...) loads a synthetic results page.",
                    "open(id=...) follows a numbered link. Passing a raw URL string also works.",
                    "find(pattern=...) scans only the active page and yields snippets.",
                ],
                links=[("tutorial home", "tutorial://gpt-oss")],
            ),
            _PageDefinition(
                url="tutorial://citations",
                title="Citations and cursors",
                summary="Line windows and cursor indices for referencing snippets.",
                paragraphs=[
                    "Pages are rendered with line numbers so a model can cite specific spans.",
                    "The history list mirrors the cursor indices that appear in tool output.",
                ],
                links=[("browser actions", "tutorial://browser-actions")],
            ),
        ]

    async def _fetch_remote(self, url: str) -> Page:
        timeout = ClientTimeout(total=self.request_timeout)
        
        try:
            async with ClientSession(timeout=timeout, headers=REMOTE_HEADERS) as session:
                async with session.get(url) as response:
                    # Check for rate limiting or other status codes
                    if response.status == 429:
                        logger.warning(f"⚠️ Rate Limit (429) when fetching: {url}")
                    elif response.status == 503:
                        logger.warning(f"⚠️ Service Unavailable (503) when fetching: {url}")
                    elif response.status >= 400:
                        logger.warning(f"⚠️ HTTP {response.status} when fetching: {url}")
                    
                    response.raise_for_status()
                    
                    # Check content type for binary files
                    content_type = response.headers.get('Content-Type', '').lower()
                    if any(binary_type in content_type for binary_type in ['pdf', 'octet-stream', 'zip', 'binary']):
                        logger.warning(f"⚠️ Binary file detected: {url} (Content-Type: {content_type})")
                        return Page(
                            url=url,
                            title=f"Binary File: {urlparse(url).path.split('/')[-1]}",
                            text=f"⚠️ This is a binary file ({content_type}). Content cannot be displayed as text.\n\nURL: {url}\n\nPlease try searching for the HTML version or abstract page instead.",
                            links={},
                            link_labels={},
                            link_line_index={},
                        )
                    
                    # Try to decode as text
                    try:
                        html = await response.text()
                    except UnicodeDecodeError as e:
                        logger.warning(f"⚠️ Unicode decode error for {url}: {e}")
                        return Page(
                            url=url,
                            title=f"Binary Content: {urlparse(url).path.split('/')[-1]}",
                            text=f"⚠️ Unable to decode content as text. This may be a binary file.\n\nURL: {url}\n\nPlease try the HTML version instead.",
                            links={},
                            link_labels={},
                            link_line_index={},
                        )
                    
                    logger.debug(f"✓ Successfully fetched: {url}")
                    return self._html_to_page(url, html)
                    
        except Exception as e:
            error_msg = str(e).lower() if str(e) else ""
            error_type = type(e).__name__
            
            # Categorize errors by type first, then by message
            if isinstance(e, asyncio.TimeoutError) or 'timeout' in error_msg or 'timed out' in error_msg:
                logger.warning(f"⚠️ Timeout when fetching '{url}': [{error_type}] {e or 'Request timed out'}")
            elif 'rate limit' in error_msg or '429' in error_msg:
                logger.warning(f"⚠️ Rate Limit when fetching '{url}': [{error_type}] {e}")
            elif 'connection' in error_msg or 'network' in error_msg:
                logger.warning(f"⚠️ Connection error when fetching '{url}': [{error_type}] {e}")
            else:
                logger.error(f"❌ Error fetching '{url}': [{error_type}] {e or '(no message)'}")
            
            # Re-raise to let caller handle it
            raise

    def _html_to_page(self, url: str, html: str) -> Page:
        # Check if this is a PDF or binary file
        if url.lower().endswith('.pdf'):
            return Page(
                url=url,
                title=f"PDF: {urlparse(url).path.split('/')[-1]}",
                text=f"⚠️ This is a PDF file. PDF content cannot be displayed in text format.\n\nURL: {url}\n\nPlease try searching for the HTML version or abstract page instead.",
                links={},
                link_labels={},
                link_line_index={},
            )
        
        soup = BeautifulSoup(html, "html.parser")
        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else urlparse(url).netloc or url
        )

        links: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        for idx, anchor in enumerate(soup.find_all("a", href=True)):
            if idx >= self.max_links:
                break
            href = urljoin(url, anchor["href"])
            text = anchor.get_text(" ", strip=True) or href
            links[str(idx)] = href
            labels[str(idx)] = text
            anchor.replace_with(f"【{idx}†{text}】")

        converter = HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.body_width = 0
        converter.ignore_anchors = True
        converter.ignore_tables = False
        markdown = converter.handle(str(soup)).strip()

        line_map = _compute_link_line_indices(markdown, links)
        return Page(
            url=url,
            title=title,
            text=markdown or "(페이지 내용을 불러오지 못했습니다.)",
            links=links,
            link_labels=labels,
            link_line_index=line_map,
        )


def _compute_link_line_indices(text: str, link_map: Dict[str, str]) -> Dict[str, int]:
    """Approximate the line index where each link placeholder first appears."""
    if not link_map:
        return {}
    line_map: Dict[str, int] = {}
    for link_id in link_map.keys():
        token = f"【{link_id}†"
        pos = text.find(token)
        if pos == -1:
            continue
        line_idx = text.count("\n", 0, pos)
        line_map[link_id] = line_idx
    return line_map


class DuckDuckGoBackend(TutorialBackend):
    """
    Backend that proxies queries to DuckDuckGo Lite search.
    """

    source: str = "duckduckgo"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def search(self, query: str, topn: int = 120) -> Page:  # noqa: ARG002
        search_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        timeout = ClientTimeout(total=self.request_timeout)
        
        try:
            async with ClientSession(timeout=timeout, headers=REMOTE_HEADERS) as session:
                async with session.get(search_url) as response:
                    # Check for rate limiting before raising for status
                    if response.status == 429:
                        logger.warning(f"⚠️ DuckDuckGo Rate Limit (429) for query: {query}")
                    elif response.status == 503:
                        logger.warning(f"⚠️ DuckDuckGo Service Unavailable (503) for query: {query}")
                    
                    response.raise_for_status()
                    html = await response.text()
                    
                    logger.debug(f"✓ DuckDuckGo search successful for query: {query}")
                    return self._html_to_page(search_url, html)
                    
        except Exception as e:
            error_msg = str(e).lower() if str(e) else ""
            error_type = type(e).__name__
            
            # Categorize errors by type first, then by message
            if isinstance(e, asyncio.TimeoutError) or 'timeout' in error_msg or 'timed out' in error_msg:
                logger.warning(f"⚠️ DuckDuckGo Timeout for query '{query}': [{error_type}] {e or 'Request timed out'}")
            elif 'rate limit' in error_msg or '429' in error_msg:
                logger.warning(f"⚠️ DuckDuckGo Rate Limit for query '{query}': [{error_type}] {e}")
            elif 'connection' in error_msg or 'network' in error_msg:
                logger.warning(f"⚠️ DuckDuckGo Connection error for query '{query}': [{error_type}] {e}")
            else:
                logger.error(f"❌ DuckDuckGo search error for query '{query}': [{error_type}] {e or '(no message)'}")
            
            # Re-raise to let caller handle it
            raise


class ExaBackend(TutorialBackend):
    """
    Backend that uses Exa AI search API for high-quality results.
    Requires EXA_API_KEY environment variable or api_key parameter.
    """

    source: str = "exa"

    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        
        if not EXA_AVAILABLE:
            raise ImportError(
                "exa_py is not installed. Install it with: pip install exa_py"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EXA_API_KEY must be provided as parameter or environment variable"
            )
        
        self.exa = Exa(api_key=self.api_key)
        logger.info("✓ ExaBackend initialized successfully")

    async def search(self, query: str, topn: int = 10) -> Page:
        """
        Search using Exa API and return formatted results.
        """
        try:
            logger.debug(f"Exa search: {query} (topn={topn})")
            
            # Run synchronous Exa call in executor to not block async loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.exa.search_and_contents(
                    query,
                    text=True,
                    type="auto",
                    num_results=min(topn, 10),  # Exa has rate limits
                )
            )
            
            # Format results as a search page
            intro = [
                f"Exa Search results for `{query}`",
                f"Found {len(result.results)} high-quality results.",
                "Use `open id=<number>` to view full content.",
            ]
            
            urls: Dict[str, str] = {}
            labels: Dict[str, str] = {}
            snippets: List[str] = []
            
            for idx, item in enumerate(result.results):
                urls[str(idx)] = item.url
                labels[str(idx)] = item.title or item.url
                
                # Create snippet from text
                text_preview = ""
                if item.text:
                    # Get first 300 chars of text
                    text_preview = item.text[:300].strip()
                    if len(item.text) > 300:
                        text_preview += "..."
                
                snippet = f"【{idx}†{item.title or item.url}】\n{text_preview or '(No preview available)'}"
                snippets.append(snippet)
            
            body = "\n\n".join(intro + snippets)
            
            # Calculate line indices for links
            line_map = _compute_link_line_indices(body, urls)
            
            logger.debug(f"✓ Exa search successful: {len(result.results)} results")
            
            return Page(
                url=f"exa://search?q={query}",
                title=f"Exa Search: {query}",
                text=body,
                links=urls,
                link_labels=labels,
                link_line_index=line_map,
            )
            
        except Exception as e:
            error_msg = str(e).lower() if str(e) else ""
            error_type = type(e).__name__
            
            # Categorize errors
            if 'rate limit' in error_msg or '429' in error_msg:
                logger.warning(f"⚠️ Exa Rate Limit for query '{query}': [{error_type}] {e}")
            elif 'api key' in error_msg or 'authentication' in error_msg or '401' in error_msg:
                logger.error(f"❌ Exa Authentication error: [{error_type}] {e}")
            elif 'timeout' in error_msg or isinstance(e, asyncio.TimeoutError):
                logger.warning(f"⚠️ Exa Timeout for query '{query}': [{error_type}] {e or 'Request timed out'}")
            else:
                logger.error(f"❌ Exa search error for query '{query}': [{error_type}] {e or '(no message)'}")
            
            # Re-raise to let caller handle it
            raise

