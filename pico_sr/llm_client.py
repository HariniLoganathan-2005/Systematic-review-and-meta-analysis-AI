"""Unified chat completion: local Ollama or cloud Groq (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import time
from typing import Any

from pico_sr.config import settings

logger = logging.getLogger(__name__)


class LLMConfigError(ValueError):
    """Missing API key or invalid provider configuration."""


class LLMTransportError(RuntimeError):
    """Network / auth / upstream failure."""

    def __init__(self, message: str, technical: str = ""):
        super().__init__(message)
        self.message = message
        self.technical = technical


def complete_chat(
    user_prompt: str,
    temperature: float = 0.1,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Send a single user message; return assistant text.

    Improvements over original:
    - Retry logic for transient errors (rate limits, timeouts)
    - Better error messages
    - Logs token usage when available
    """
    provider = (settings.llm_provider or "ollama").strip().lower()

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "groq":
                result = _complete_groq(user_prompt, temperature)
            else:
                result = _complete_ollama(user_prompt, temperature)

            # Log response length for debugging
            logger.debug(
                "LLM response: %d chars (attempt %d)", len(result), attempt
            )
            return result

        except LLMTransportError as e:
            if attempt < max_retries:
                logger.warning(
                    "LLM attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt, max_retries, e.message, retry_delay
                )
                time.sleep(retry_delay * attempt)  # exponential backoff
            else:
                logger.error(
                    "LLM failed after %d attempts: %s", max_retries, e.message
                )
                raise

        except LLMConfigError:
            # Config errors won't be fixed by retrying
            raise

    return ""  # unreachable but satisfies type checker


def _complete_ollama(user_prompt: str, temperature: float) -> str:
    """Call local Ollama instance."""
    try:
        import ollama
    except ImportError as e:
        raise LLMConfigError(
            "Ollama SDK not installed. Run: pip install ollama"
        ) from e

    try:
        client = ollama.Client(host=settings.ollama_host)
        resp: dict[str, Any] = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": user_prompt}],
            options={"temperature": temperature},
        )
        content = (resp.get("message") or {}).get("content") or ""
        if not content:
            logger.warning("Ollama returned empty content")
        return content

    except ConnectionError as e:
        raise LLMTransportError(
            f"Cannot connect to Ollama at {settings.ollama_host}. "
            "Start the Ollama app and run: ollama pull " + settings.ollama_model,
            str(e),
        ) from e
    except Exception as e:
        raise LLMTransportError(
            f"Ollama error: {str(e)[:200]}",
            str(e),
        ) from e


def _complete_groq(user_prompt: str, temperature: float) -> str:
    """Call Groq cloud API (OpenAI-compatible)."""
    if not (settings.groq_api_key or "").strip():
        raise LLMConfigError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys "
            "or set LLM_PROVIDER=ollama to use a local model."
        )

    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            OpenAI,
            OpenAIError,
            RateLimitError,
        )
    except ImportError as e:
        raise LLMConfigError(
            "OpenAI SDK required for Groq. Run: pip install openai"
        ) from e

    client = OpenAI(
        api_key=settings.groq_api_key.strip(),
        base_url=settings.groq_base_url,
    )

    try:
        r = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            # Request JSON output to reduce parsing failures
            response_format={"type": "text"},
        )

        choice = r.choices[0].message
        content = (choice.content or "").strip()

        # Log token usage if available
        if hasattr(r, "usage") and r.usage:
            logger.debug(
                "Groq tokens — prompt: %d, completion: %d, total: %d",
                r.usage.prompt_tokens,
                r.usage.completion_tokens,
                r.usage.total_tokens,
            )

        if not content:
            logger.warning("Groq returned empty content for model %s", settings.groq_model)

        return content

    except RateLimitError as e:
        raise LLMTransportError(
            "Groq rate limit hit. Your extraction will retry automatically. "
            "Consider upgrading your Groq plan or reducing batch size.",
            str(e),
        ) from e

    except APIConnectionError as e:
        raise LLMTransportError(
            "Cannot reach Groq API. Check your internet connection.",
            str(e),
        ) from e

    except APIStatusError as e:
        if e.status_code == 401:
            raise LLMTransportError(
                "Groq API key is invalid or expired. "
                "Check GROQ_API_KEY in your .env file.",
                str(e),
            ) from e
        elif e.status_code == 429:
            raise LLMTransportError(
                "Groq quota exceeded. Check your usage at console.groq.com",
                str(e),
            ) from e
        else:
            raise LLMTransportError(
                f"Groq API error {e.status_code}: {str(e)[:200]}",
                str(e),
            ) from e

    except OpenAIError as e:
        raise LLMTransportError(
            "Groq API error. Check GROQ_API_KEY, credits, and GROQ_MODEL in .env",
            str(e),
        ) from e


def health_llm_sync() -> dict[str, Any]:
    """
    Synchronous probe for /health/llm endpoint.
    Returns provider status and configuration details.
    """
    provider = (settings.llm_provider or "ollama").strip().lower()

    if provider == "groq":
        return _health_groq()
    return _health_ollama()


def _health_groq() -> dict[str, Any]:
    """Check Groq API health."""
    if not (settings.groq_api_key or "").strip():
        return {
            "provider": "groq",
            "ok": False,
            "message": "GROQ_API_KEY not set in .env file",
        }
    try:
        from openai import OpenAI

        c = OpenAI(
            api_key=settings.groq_api_key.strip(),
            base_url=settings.groq_base_url,
        )
        models = c.models.list()
        model_ids = [m.id for m in models.data] if hasattr(models, "data") else []

        return {
            "provider": "groq",
            "ok": True,
            "base_url": settings.groq_base_url,
            "model": settings.groq_model,
            "model_available": settings.groq_model in model_ids,
            "available_models": model_ids[:5],  # show first 5
        }
    except Exception as e:
        return {
            "provider": "groq",
            "ok": False,
            "message": str(e)[:200],
        }


def _health_ollama() -> dict[str, Any]:
    """Check Ollama local instance health."""
    try:
        import ollama

        client = ollama.Client(host=settings.ollama_host)
        models_resp = client.list()

        # Extract model names
        model_names = []
        if hasattr(models_resp, "models"):
            model_names = [
                m.model if hasattr(m, "model") else str(m)
                for m in models_resp.models
            ]

        model_available = any(
            settings.ollama_model in name for name in model_names
        )

        return {
            "provider": "ollama",
            "ok": True,
            "host": settings.ollama_host,
            "model": settings.ollama_model,
            "model_available": model_available,
            "available_models": model_names[:5],
        }
    except Exception as e:
        return {
            "provider": "ollama",
            "ok": False,
            "message": str(e)[:200],
            "hint": (
                "Make sure Ollama is running and you have pulled the model: "
                f"ollama pull {settings.ollama_model}"
            ),
        }