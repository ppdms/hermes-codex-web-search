"""OpenAI Codex Hosted Web Search provider.

This mirrors the native Codex Responses request used by Oh My Pi (OMP):
``tools=[{"type": "web_search"}]`` plus the ChatGPT OAuth account header.
Search is native Codex Hosted Search. Extraction is model-mediated: Codex is
asked to read each supplied URL and its answer is returned as the document
content. Codex does not expose a separate raw page-fetch endpoint.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_TIMEOUT = 90
_DEFAULT_INSTRUCTIONS = (
    "You are a helpful assistant with web search capabilities. "
    "Search the web to answer the user's question accurately and cite your sources."
)
_AUTH_CLAIM = "https://api.openai.com/auth"


def _codex_account_id(token: str) -> str:
    """Extract chatgpt_account_id from a Codex OAuth JWT, if present."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return ""
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        auth = claims.get(_AUTH_CLAIM)
        return str(auth.get("chatgpt_account_id") or "") if isinstance(auth, dict) else ""
    except Exception:  # noqa: BLE001 - malformed credentials become an auth error
        return ""


def _codex_responses_url(base_url: str) -> str:
    """Resolve both Hermes' default /codex URL and custom Codex base URLs."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        base = "https://chatgpt.com/backend-api/codex"
    if base.endswith("/responses"):
        return base
    if base.endswith("/codex"):
        return f"{base}/responses"
    return f"{base}/codex/responses"


def _configured_model() -> str:
    """Use an explicit search model, then Hermes' active Codex model."""
    configured = os.getenv("HERMES_CODEX_WEB_SEARCH_MODEL", "").strip()
    if configured:
        return configured
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web = cfg.get("web", {}) if isinstance(cfg, dict) else {}
        codex = web.get("codex", {}) if isinstance(web, dict) else {}
        value = codex.get("model") if isinstance(codex, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip().split("/")[-1]
        model = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        value = model.get("default") if isinstance(model, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip().split("/")[-1]
    except Exception:  # noqa: BLE001 - default remains usable
        pass
    return _DEFAULT_MODEL


def _configured_timeout() -> float:
    try:
        value = float(os.getenv("HERMES_CODEX_WEB_SEARCH_TIMEOUT", str(_DEFAULT_TIMEOUT)))
        return max(5.0, min(value, 300.0))
    except (TypeError, ValueError):
        return float(_DEFAULT_TIMEOUT)


def _iter_sse_json(response: Any) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a Responses API SSE response."""
    for line in response.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            logger.debug("Ignoring malformed Codex SSE event")
            continue
        if isinstance(event, dict):
            yield event


def _add_source(sources: List[Dict[str, str]], url: str, title: str = "") -> None:
    url = str(url or "").strip()
    if not url or not url.startswith(("http://", "https://")):
        return
    if any(item["url"] == url for item in sources):
        return
    sources.append({"url": url, "title": title.strip() or url})


def _sources_from_text(text: str) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)", text):
        _add_source(sources, url.rstrip(".,!?"), title)
    for url in re.findall(r"https?://[^\s)\]>]+", text):
        _add_source(sources, url.rstrip(".,!?"))
    return sources


def _error_message(response: Any) -> str:
    try:
        body = response.text[:1000]
    except Exception:  # noqa: BLE001
        body = ""
    return f"Codex web search returned HTTP {response.status_code}: {body}".strip()


class CodexWebSearchProvider(WebSearchProvider):
    """Codex Hosted Search plus model-mediated URL extraction."""

    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex Hosted Search"

    def is_available(self) -> bool:
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(
                refresh_if_expiring=False,
                refresh_skew_seconds=0,
            )
            return bool(str(creds.get("api_key") or "").strip())
        except Exception:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except Exception:  # noqa: BLE001
            pass

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials()
            token = str(creds.get("api_key") or "").strip()
            if not token:
                return {"success": False, "error": "No Codex OAuth credentials found. Run `hermes auth add openai-codex`."}
            account_id = _codex_account_id(token)
            if not account_id:
                return {"success": False, "error": "Codex OAuth token has no chatgpt_account_id claim; re-authenticate with `hermes auth add openai-codex`."}

            from agent.auxiliary_client import _codex_cloudflare_headers

            headers = _codex_cloudflare_headers(token)
            headers.update(
                {
                    "Authorization": f"Bearer {token}",
                    "ChatGPT-Account-ID": account_id,
                    "OpenAI-Beta": "responses=experimental",
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                }
            )
            model = _configured_model()
            body = {
                "model": model,
                "stream": True,
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": query}],
                    }
                ],
                "tools": [
                    {
                        "type": "web_search",
                        "search_context_size": "high",
                    }
                ],
                "tool_choice": {"type": "web_search"},
                "instructions": _DEFAULT_INSTRUCTIONS,
            }

            import httpx

            with httpx.Client(timeout=_configured_timeout()) as client:
                with client.stream(
                    "POST",
                    _codex_responses_url(str(creds.get("base_url") or "")),
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code >= 400:
                        return {"success": False, "error": _error_message(response)}
                    answer_parts: List[str] = []
                    sources: List[Dict[str, str]] = []
                    searched = False
                    for event in _iter_sse_json(response):
                        event_type = str(event.get("type") or "")
                        if event_type.startswith("response.web_search_call"):
                            searched = True
                        if event_type == "response.output_text.delta":
                            delta = event.get("delta")
                            if isinstance(delta, str):
                                answer_parts.append(delta)
                        if event_type == "response.output_item.done":
                            item = event.get("item")
                            if not isinstance(item, dict):
                                continue
                            if item.get("type") == "web_search_call":
                                searched = True
                            if item.get("type") != "message":
                                continue
                            for part in item.get("content") or []:
                                if not isinstance(part, dict) or part.get("type") != "output_text":
                                    continue
                                text = part.get("text")
                                if isinstance(text, str):
                                    if not answer_parts:
                                        answer_parts.append(text)
                                    for annotation in part.get("annotations") or []:
                                        if isinstance(annotation, dict) and annotation.get("type") == "url_citation":
                                            _add_source(sources, annotation.get("url", ""), annotation.get("title", ""))
                        if event_type in {"error", "response.failed"}:
                            return {"success": False, "error": "Codex web search failed: " + str(event.get("error") or event.get("message") or event)}

            if not searched:
                return {"success": False, "error": "Codex returned an answer without invoking hosted web search."}
            answer = "".join(answer_parts).strip()
            if not sources:
                sources = _sources_from_text(answer)
            results = [
                {
                    "title": source["title"],
                    "url": source["url"],
                    "description": answer if index == 0 else "",
                    "position": index + 1,
                }
                for index, source in enumerate(sources[: max(1, min(int(limit), 10))])
            ]
            return {"success": True, "data": {"web": results}}
        except Exception as exc:  # noqa: BLE001 - provider errors are tool results
            logger.warning("Codex web search error: %s", exc)
            return {"success": False, "error": f"Codex web search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Read URLs through Codex Hosted Search's model-mediated web access.

        This is intentionally one request per URL so each returned document
        remains attributable to its requested source. The text is Codex's
        readable answer, not a byte-for-byte scrape of the page.
        """
        documents: List[Dict[str, Any]] = []
        for url in urls:
            url = str(url or "").strip()
            if not url:
                continue
            result = self.search(
                (
                    "Read the webpage at this exact URL and extract its main "
                    "readable content. Preserve important headings, facts, "
                    "code, and links. Do not discuss the search process; "
                    "return only the page content. URL: "
                    f"{url}"
                ),
                limit=1,
            )
            if not result.get("success"):
                documents.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": result.get("error", "Codex extraction failed"),
                    "metadata": {"sourceURL": url, "provider": "codex"},
                })
                continue

            rows = result.get("data", {}).get("web", [])
            first = rows[0] if rows else {}
            content = str(first.get("description") or "").strip()
            title = str(first.get("title") or url).strip()
            if not content:
                documents.append({
                    "url": url,
                    "title": title,
                    "content": "",
                    "raw_content": "",
                    "error": "Codex returned no readable page content",
                    "metadata": {"sourceURL": url, "provider": "codex"},
                })
                continue
            documents.append({
                "url": url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": {
                    "sourceURL": url,
                    "provider": "codex",
                    "mode": "model-mediated",
                },
            })
        return documents

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "oauth",
            "tag": "Search + model-mediated URL extraction via existing Codex OAuth; no separate API key.",
            "env_vars": [],
        }
