from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Protocol

from src.orchestrator.types import RuntimeConfig, SearchResult


class SearchTool(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        ...


class NullSearchTool:
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        return []


class MockSearchTool:
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for idx in range(max_results):
            rank = idx + 1
            results.append(
                SearchResult(
                    title=f"Mock result {rank} for {query}",
                    url=f"https://example.test/search/{rank}?q={urllib.parse.quote(query)}",
                    snippet=(
                        f"Synthetic evidence item {rank} for '{query}'. "
                        "Use this only for integration testing."
                    ),
                    source="mock",
                )
            )
        return results


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "") or ""
        if tag == "a" and "result__a" in class_name:
            self._in_title = True
            href = attrs_dict.get("href", "") or ""
            self._current_url = _decode_duckduckgo_href(href)
            self._current_title = []
            self._current_snippet = []
        if tag == "a" and "result__snippet" in class_name:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
            title = " ".join("".join(self._current_title).split())
            snippet = " ".join("".join(self._current_snippet).split())
            if title and self._current_url:
                self.results.append(
                    SearchResult(
                        title=title,
                        url=self._current_url,
                        snippet=snippet,
                        source="duckduckgo",
                    )
                )
        if tag == "a" and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current_title.append(data)
        if self._in_snippet:
            self._current_snippet.append(data)


def _decode_duckduckgo_href(href: str) -> str:
    if not href:
        return ""
    parsed = urllib.parse.urlparse(href)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return urllib.parse.unquote(query["uddg"][0])
    return href


class DuckDuckGoSearchTool:
    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://html.duckduckgo.com/html/?{encoded}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "grok_multiagent/0.2 (+https://github.com/local/grok_multiagent)"
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
        parser = _DuckDuckGoHTMLParser()
        parser.feed(payload)
        return parser.results[:max_results]


class SerperSearchTool:
    def __init__(self, api_key: str, timeout_seconds: int = 20) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        body = json.dumps({"q": query, "num": max_results}).encode("utf-8")
        request = urllib.request.Request(
            "https://google.serper.dev/search",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        items = payload.get("organic", [])[:max_results]
        return [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("link", "")),
                snippet=str(item.get("snippet", "")),
                source="serper",
            )
            for item in items
            if item.get("title") and item.get("link")
        ]


def build_search_tool(runtime_config: RuntimeConfig) -> SearchTool:
    provider = str(runtime_config.search.get("provider", "mock")).lower()
    timeout_seconds = int(runtime_config.search.get("timeout_seconds", 20))

    if provider == "none":
        return NullSearchTool()
    if provider == "mock":
        return MockSearchTool()
    if provider == "duckduckgo":
        return DuckDuckGoSearchTool(timeout_seconds=timeout_seconds)
    if provider == "serper":
        api_key = os.environ.get("SERPER_API_KEY", "")
        if not api_key:
            raise RuntimeError("SERPER_API_KEY is required when search.provider=serper")
        return SerperSearchTool(api_key=api_key, timeout_seconds=timeout_seconds)
    raise ValueError(f"unsupported search provider: {provider}")
