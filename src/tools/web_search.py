from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Protocol

from src.orchestrator.types import RuntimeConfig, SearchRequest, SearchResult, SearchTrace


class SearchTool(Protocol):
    def search(self, request: SearchRequest) -> SearchTrace:
        ...


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
                        domain=_domain_for_url(self._current_url),
                        source_type="web",
                        provider="duckduckgo",
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


def _domain_for_url(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


class DuckDuckGoSearchTool:
    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, request: SearchRequest) -> SearchTrace:
        encoded = urllib.parse.urlencode({"q": request.query})
        url = f"https://html.duckduckgo.com/html/?{encoded}"
        http_request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "grok_multiagent/0.3 (+https://github.com/local/grok_multiagent)"
            },
            method="GET",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            payload = response.read().decode("utf-8", errors="replace")
        parser = _DuckDuckGoHTMLParser()
        parser.feed(payload)
        return SearchTrace(
            query=request.query,
            mode=request.mode,
            provider="duckduckgo",
            results=parser.results[: request.max_results],
            error=None,
        )


class SerperSearchTool:
    def __init__(self, api_key: str, timeout_seconds: int = 20) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(self, request: SearchRequest) -> SearchTrace:
        payload: dict[str, Any] = {
            "q": request.query,
            "num": request.max_results,
        }
        if request.user_location:
            payload["gl"] = request.user_location.lower()
        if request.language:
            payload["hl"] = request.language.lower()

        response = _post_json(
            "https://google.serper.dev/search",
            payload=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
            },
            timeout_seconds=self.timeout_seconds,
        )
        items = response.get("organic", [])[: request.max_results]
        results = [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("link", "")),
                snippet=str(item.get("snippet", "")),
                source="serper",
                domain=_domain_for_url(str(item.get("link", ""))),
                published_at=str(item.get("date", "")) or None,
                source_type="web",
                provider="serper",
            )
            for item in items
            if item.get("title") and item.get("link")
        ]
        return SearchTrace(
            query=request.query,
            mode=request.mode,
            provider="serper",
            results=results,
            error=None,
        )


class ExaSearchTool:
    def __init__(self, api_key: str, timeout_seconds: int = 30, config: dict[str, Any] | None = None) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.config = config or {}

    def search(self, request: SearchRequest) -> SearchTrace:
        payload: dict[str, Any] = {
            "query": request.query,
            "numResults": request.max_results,
            "type": _exa_search_type_for_mode(request.mode),
        }
        if request.include_domains:
            payload["includeDomains"] = request.include_domains
        if request.exclude_domains:
            payload["excludeDomains"] = request.exclude_domains
        if request.category:
            payload["category"] = request.category

        response = _post_json(
            "https://api.exa.ai/search",
            payload=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            },
            timeout_seconds=self.timeout_seconds,
        )

        items = response.get("results", [])[: request.max_results]
        results = [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("text", ""))[:400] if item.get("text") else "",
                source="exa",
                domain=_domain_for_url(str(item.get("url", ""))),
                published_at=str(item.get("publishedDate", "")) or None,
                source_type="web",
                provider="exa",
            )
            for item in items
            if item.get("title") and item.get("url")
        ]

        if _should_fetch_contents(request.mode, self.config) and results:
            contents = self._fetch_contents(
                urls=[result.url for result in results],
                query=request.query,
            )
            for result in results:
                content = contents.get(result.url, {})
                result.text = str(content.get("text", ""))[:4000]
                result.summary = _normalize_summary(content.get("summary"))
                result.highlights = [
                    str(item).strip() for item in content.get("highlights", []) if str(item).strip()
                ]
                if result.highlights and not result.snippet:
                    result.snippet = " ".join(result.highlights[:2])
                if result.summary and not result.snippet:
                    result.snippet = result.summary

        return SearchTrace(
            query=request.query,
            mode=request.mode,
            provider="exa",
            results=results,
            error=None,
        )

    def _fetch_contents(self, urls: list[str], query: str) -> dict[str, dict[str, Any]]:
        highlights = {
            "query": query,
            "numSentences": int(self.config.get("highlight_num_sentences", 2)),
            "highlightsPerUrl": int(self.config.get("highlights_per_url", 2)),
        }
        payload: dict[str, Any] = {
            "urls": urls,
            "text": bool(self.config.get("include_text", True)),
            "highlights": highlights,
        }
        if self.config.get("include_summary", False):
            payload["summary"] = {"query": query}

        response = _post_json(
            "https://api.exa.ai/contents",
            payload=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
            },
            timeout_seconds=self.timeout_seconds,
        )
        items = response.get("results", [])
        return {str(item.get("url", "")): item for item in items if item.get("url")}


class SearchRouter:
    def __init__(self, providers: dict[str, SearchTool], routing: dict[str, str], fallback_provider: str | None) -> None:
        self.providers = providers
        self.routing = routing
        self.fallback_provider = fallback_provider

    def search(self, request: SearchRequest) -> SearchTrace:
        provider_name = self.routing.get(request.mode) or self.routing.get("default")
        if not provider_name:
            raise RuntimeError(f"no provider configured for search mode: {request.mode}")

        tool = self.providers[provider_name]
        try:
            return tool.search(request)
        except Exception as exc:
            if not self.fallback_provider or self.fallback_provider == provider_name:
                raise
            fallback = self.providers.get(self.fallback_provider)
            if fallback is None:
                raise
            trace = fallback.search(request)
            if trace.error:
                trace.error = f"primary:{provider_name}:{exc}; fallback:{trace.error}"
            else:
                trace.error = f"primary:{provider_name}:{exc}"
            return trace


class FixedProviderSearchTool:
    def __init__(self, provider_name: str, tool: SearchTool) -> None:
        self.provider_name = provider_name
        self.tool = tool

    def search(self, request: SearchRequest) -> SearchTrace:
        trace = self.tool.search(request)
        trace.provider = self.provider_name
        return trace


def _normalize_summary(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _exa_search_type_for_mode(mode: str) -> str:
    if mode == "fast_lookup":
        return "fast"
    if mode == "deep_research":
        return "deep"
    return "auto"


def _should_fetch_contents(mode: str, config: dict[str, Any]) -> bool:
    if mode == "deep_research":
        return bool(config.get("fetch_contents_on_deep", True))
    if mode == "standard_research":
        return bool(config.get("fetch_contents_on_standard", True))
    return bool(config.get("fetch_contents_on_fast", False))


def build_search_tool(runtime_config: RuntimeConfig) -> SearchTool:
    search_config = runtime_config.search
    provider = str(search_config.get("provider", "router")).lower()

    providers: dict[str, SearchTool] = {
        "duckduckgo": DuckDuckGoSearchTool(
            timeout_seconds=int(search_config.get("duckduckgo", {}).get("timeout_seconds", 20))
        ),
    }

    serper_api_key = os.environ.get("SERPER_API_KEY", "")
    if serper_api_key:
        providers["serper"] = SerperSearchTool(
            api_key=serper_api_key,
            timeout_seconds=int(search_config.get("serper", {}).get("timeout_seconds", 20)),
        )
    exa_api_key = os.environ.get("EXA_API_KEY", "")
    if exa_api_key:
        providers["exa"] = ExaSearchTool(
            api_key=exa_api_key,
            timeout_seconds=int(search_config.get("exa", {}).get("timeout_seconds", 30)),
            config=dict(search_config.get("exa", {})),
        )

    if provider == "router":
        routing = {
            "fast_lookup": str(search_config.get("routing", {}).get("fast_lookup", "serper")),
            "standard_research": str(search_config.get("routing", {}).get("standard_research", "exa")),
            "deep_research": str(search_config.get("routing", {}).get("deep_research", "exa")),
            "default": str(search_config.get("routing", {}).get("default", "serper")),
        }
        fallback_provider = str(search_config.get("routing", {}).get("fallback", "duckduckgo"))
        missing = [name for name in set(routing.values()) | {fallback_provider} if name and name not in providers]
        if missing:
            raise RuntimeError(f"missing search provider credentials or configuration for: {sorted(missing)}")
        return SearchRouter(
            providers=providers,
            routing=routing,
            fallback_provider=fallback_provider,
        )

    if provider not in providers:
        raise ValueError(f"unsupported search provider: {provider}")
    return FixedProviderSearchTool(provider_name=provider, tool=providers[provider])
