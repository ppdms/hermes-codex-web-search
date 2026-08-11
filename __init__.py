"""User plugin registration for the Codex Hosted Web Search provider."""

from __future__ import annotations

from .provider import CodexWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(CodexWebSearchProvider())
