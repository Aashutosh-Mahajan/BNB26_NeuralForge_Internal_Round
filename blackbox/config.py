"""Environment-driven settings. Secrets are read from .env and never persisted."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional; uvicorn --env-file also works.
        return
    load_dotenv(ROOT / ".env", override=False)


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: str
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str | None
    reasoning_effort: str
    temperature: float | None
    seed: int
    ollama_base_url: str
    ollama_model: str
    ollama_num_ctx: int
    explainer_llm: str
    alert_webhook: str | None
    alert_threshold: float
    budget_usd: float
    budget_start: str | None
    price_input: float
    price_cached_input: float
    price_output: float
    sandbox_noise: float
    request_timeout: float
    max_retries: int
    usage_db: str
    model_dir: str
    otel_endpoint: str | None
    otel_console: bool
    explainer_adapter: str | None
    explainer_narrative: str

    @property
    def openai_ready(self) -> bool:
        key = self.openai_api_key or ""
        return bool(key) and not key.startswith("sk-your")

    def price_usd(self, tokens_in: int, tokens_out: int, cached_in: int = 0) -> float:
        cached_in = min(cached_in, tokens_in)
        return ((tokens_in - cached_in) * self.price_input + cached_in * self.price_cached_input
                + tokens_out * self.price_output) / 1_000_000


@lru_cache(maxsize=1)
def settings() -> Settings:
    _load_dotenv()
    temperature = os.environ.get("LLM_TEMPERATURE", "0").strip()
    return Settings(
        provider=os.environ.get("LLM_PROVIDER", "sandbox").strip().lower(),
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
        openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "low"),
        temperature=None if temperature.lower() in {"", "none", "default"} else float(temperature),
        seed=int(os.environ.get("LLM_SEED", "7")),
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        ollama_num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
        # The narrative explainer never bills OpenAI unless explicitly set to "openai".
        explainer_llm=os.environ.get("EXPLAINER_LLM", "ollama").strip().lower(),
        alert_webhook=os.environ.get("ALERT_WEBHOOK_URL") or None,
        alert_threshold=_float("ALERT_P_FAIL_THRESHOLD", 0.8),
        budget_usd=_float("LLM_BUDGET_USD", 0.5),
        budget_start=os.environ.get("LLM_BUDGET_START") or None,
        # USD per 1M tokens; .env sets the live model's prices (gpt-5.4-nano: 0.20 / 0.02 / 1.25).
        price_input=_float("LLM_PRICE_INPUT_PER_M", 0.10),
        price_cached_input=_float("LLM_PRICE_CACHED_INPUT_PER_M", 0.01),
        price_output=_float("LLM_PRICE_OUTPUT_PER_M", 0.50),
        sandbox_noise=_float("SANDBOX_NOISE", 0.0),
        request_timeout=_float("LLM_TIMEOUT_SECONDS", 60.0),
        max_retries=int(os.environ.get("LLM_MAX_RETRIES", "4")),
        usage_db=os.environ.get("BLACKBOX_USAGE_DB", str(ROOT / "data" / "llm_usage.db")),
        model_dir=os.environ.get("BLACKBOX_MODEL_DIR", str(ROOT / "data" / "models")),
        otel_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
        otel_console=_flag("BLACKBOX_OTEL_CONSOLE", False),
        explainer_adapter=os.environ.get("EXPLAINER_ADAPTER_PATH") or None,
        explainer_narrative=os.environ.get("EXPLAINER_NARRATIVE", "template").strip().lower(),
    )
