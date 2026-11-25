"""Thin wrapper around the Anthropic SDK used by the orchestrating agent and the LLM judge."""
from __future__ import annotations

import os
from typing import Any

import anthropic

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_client() -> anthropic.Anthropic:
    # Credentials resolve the usual SDK way: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
    # or a profile from `ant auth login`.
    return anthropic.Anthropic(max_retries=4)


def has_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return os.path.isdir(os.path.expanduser("~/.config/anthropic"))


def create(client: anthropic.Anthropic, **kwargs: Any):
    """messages.create with server-side refusal fallbacks enabled.

    If a safety classifier declines a request, the API re-runs it on a fallback model
    inside the same call instead of returning an empty refusal.
    """
    return client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
