"""Provider-neutral JSON chat calls: OpenAI GPT-6 Luna, local Ollama, or the offline sandbox.

The agent never imports a provider SDK directly; ``LLM_PROVIDER`` selects one
without changing agent code. Every billed call is metered and budget-checked.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from .usage import BudgetExceeded, UsageLedger


@dataclass
class LLMResult:
    data: Any
    text: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached_in: int = 0
    reasoning: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    estimated: bool = False
    temperature: float | None = None
    seed: int | None = None
    error: str | None = None
    dropped_params: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """~4 characters per token; used only where a provider reports no usage."""
    return max(1, round(len(text) / 4))


def parse_json(text: str) -> Any:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


class BaseLLM:
    provider = "base"
    model = "none"
    stochastic = False

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class SandboxLLM(BaseLLM):
    """Deterministic rule-based stand-in. Nodes implement their own sandbox policy.

    ``noise`` is the per-call probability of an LLM-like mistake; it simulates
    natural failures for data generation and is 0 for demos.
    """
    provider = "sandbox"

    def __init__(self, noise: float | None = None, tool_noise: float | None = None):
        self.noise = settings().sandbox_noise if noise is None else noise
        # Transient tool timeouts default to a quarter of the LLM mistake rate.
        self.tool_noise = self.noise * 0.25 if tool_noise is None else tool_noise
        self.model = "rule-based-v2" if not self.noise else f"rule-based-v2-noise{self.noise:g}"
        self.stochastic = self.noise > 0 or self.tool_noise > 0


class ChatLLM(BaseLLM):
    stochastic = True
    _unsupported_lock = threading.Lock()

    def __init__(self, provider: str, model: str | None = None, ledger: UsageLedger | None = None):
        cfg = settings()
        self.provider = provider
        self.model = model or (cfg.openai_model if provider == "openai" else cfg.ollama_model)
        self.ledger = ledger or UsageLedger()
        self.unsupported: set[str] = set()
        if provider == "openai" and not cfg.openai_ready:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")

    def _client(self, temperature, seed, model):
        cfg = settings()
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            kwargs: dict[str, Any] = {"model": model, "api_key": cfg.openai_api_key, "timeout": cfg.request_timeout,
                                      "max_retries": cfg.max_retries}
            if cfg.openai_base_url:
                kwargs["base_url"] = cfg.openai_base_url
            if "reasoning_effort" not in self.unsupported and cfg.reasoning_effort:
                kwargs["reasoning_effort"] = cfg.reasoning_effort
            if "temperature" not in self.unsupported and temperature is not None:
                kwargs["temperature"] = temperature
            if "seed" not in self.unsupported and seed is not None:
                kwargs["seed"] = seed
            if "response_format" not in self.unsupported:
                kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
            return ChatOpenAI(**kwargs)
        from langchain_ollama import ChatOllama
        # num_predict caps each answer: small local models in JSON mode can otherwise loop forever.
        return ChatOllama(model=model, base_url=cfg.ollama_base_url, format="json", num_ctx=cfg.ollama_num_ctx,
                          num_predict=int(__import__("os").environ.get("OLLAMA_NUM_PREDICT", "512")),
                          temperature=temperature if temperature is not None else 0, seed=seed,
                          client_kwargs={"timeout": cfg.request_timeout})

    def _invoke(self, system: str, user: str, temperature, seed, model):
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for _ in range(5):
            try:
                return self._client(temperature, seed, model).invoke(messages)
            except Exception as exc:  # Unsupported-parameter errors are provider specific.
                text = str(exc).lower()
                dropped = next((name for name in ("temperature", "seed", "reasoning_effort", "response_format")
                                if name in text
                                and any(w in text for w in ("unsupported", "not support", "invalid", "unrecognized", "does not"))), None)
                if dropped is None:
                    raise
                # Another thread may already have learned this; either way retry without it.
                with self._unsupported_lock:
                    self.unsupported.add(dropped)
        raise RuntimeError("Unable to call the provider with any supported parameter set")

    def complete_json(self, system: str, user: str, *, seed: int | None = None, temperature: float | None = None,
                      model: str | None = None, purpose: str = "agent", run_id: str | None = None,
                      node: str | None = None) -> LLMResult:
        cfg = settings()
        model = model or self.model
        temperature = cfg.temperature if temperature is None else temperature
        self.ledger.check(self.provider)
        start = time.perf_counter()
        message = self._invoke(system, user, temperature, seed, model)
        latency = (time.perf_counter() - start) * 1000
        text = message.content if isinstance(message.content, str) else json.dumps(message.content)
        usage = getattr(message, "usage_metadata", None) or {}
        tokens_in = int(usage.get("input_tokens") or estimate_tokens(system + user))
        tokens_out = int(usage.get("output_tokens") or estimate_tokens(text))
        cached = int((usage.get("input_token_details") or {}).get("cache_read") or 0)
        reasoning = int((usage.get("output_token_details") or {}).get("reasoning") or 0)
        cost = cfg.price_usd(tokens_in, tokens_out, cached) if self.provider == "openai" else 0.0
        self.ledger.record(provider=self.provider, model=model, purpose=purpose, run_id=run_id, node=node,
                           tokens_in=tokens_in, tokens_out=tokens_out, cached_in=cached, reasoning=reasoning,
                           cost_usd=cost)
        try:
            data, error = parse_json(text), None
        except (json.JSONDecodeError, ValueError) as exc:
            data, error = None, f"Model returned invalid JSON: {exc}"
        return LLMResult(data=data, text=text, provider=self.provider, model=model, tokens_in=tokens_in,
                         tokens_out=tokens_out, cached_in=cached, reasoning=reasoning, cost_usd=cost,
                         latency_ms=latency, estimated=not usage,
                         temperature=None if "temperature" in self.unsupported else temperature,
                         seed=None if "seed" in self.unsupported else seed, error=error,
                         dropped_params=sorted(self.unsupported))


_cache: dict[str, BaseLLM] = {}


def get_llm(provider: str | None = None, *, noise: float | None = None) -> BaseLLM:
    """Resolve the configured provider. Falls back to the sandbox only when asked to."""
    provider = (provider or settings().provider).lower()
    if provider == "sandbox":
        return SandboxLLM(noise)
    if provider not in ("openai", "ollama"):
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; use sandbox, openai or ollama")
    if provider not in _cache:
        _cache[provider] = ChatLLM(provider)
    return _cache[provider]


def provider_status() -> dict:
    cfg = settings()
    ledger = UsageLedger()
    return {"provider": cfg.provider, "openai_model": cfg.openai_model, "openai_key_configured": cfg.openai_ready,
            "ollama_model": cfg.ollama_model, "reasoning_effort": cfg.reasoning_effort,
            "budget_usd": cfg.budget_usd, "spent_usd": round(ledger.spent("openai"), 6),
            "budget_start": cfg.budget_start, "spent_toward_budget_usd": round(ledger.spent("openai", cfg.budget_start), 6),
            "pricing_per_million": {"input": cfg.price_input, "cached_input": cfg.price_cached_input,
                                    "output": cfg.price_output},
            "sandbox_noise": cfg.sandbox_noise}


__all__ = ["BaseLLM", "SandboxLLM", "ChatLLM", "LLMResult", "BudgetExceeded", "get_llm", "provider_status",
           "estimate_tokens", "parse_json"]
