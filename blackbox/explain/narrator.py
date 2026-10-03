"""Plain-English diagnosis (M5): template, provider LLM, or a QLoRA-tuned Qwen2.5-1.5B adapter.

The narrative only rephrases evidence the models already produced; it never
changes the ranking.
"""
from __future__ import annotations

import json
import threading

from ..config import settings
from ..judge import serialize
from .evidence import narrative

SYSTEM = ("You explain AI-agent failure diagnoses to engineers in 3-4 plain sentences. Use only the evidence "
          "given. Name the guilty step, why it is guilty, what it affected, and whether replay confirmed it.")
_qlora = {"model": None, "tokenizer": None, "error": None}
_lock = threading.Lock()


def prompt_for(run: dict, diagnosis: dict) -> str:
    evidence = {k: diagnosis.get("evidence", {}).get(k) for k in ("summary", "data_flow", "factors", "model_votes",
                                                                   "nearest_success", "diverges_at", "counterfactual")}
    return (f"{serialize(run)}\n\nDiagnosis: root cause step {diagnosis['root_cause']['step']} "
            f"({diagnosis['root_cause']['node']}), blame {diagnosis['root_cause']['confidence']:.2f}, "
            f"p_fail {diagnosis.get('p_fail', 0):.2f}\nEvidence: {json.dumps(evidence, default=str)[:3000]}")


def _load_qlora():
    with _lock:
        if _qlora["model"] is not None or _qlora["error"]:
            return
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            adapter = settings().explainer_adapter
            config = PeftConfig.from_pretrained(adapter)
            tokenizer = AutoTokenizer.from_pretrained(adapter)
            base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16,
                                                        device_map="auto")
            _qlora.update(model=PeftModel.from_pretrained(base, adapter).eval(), tokenizer=tokenizer)
        except Exception as exc:
            _qlora["error"] = str(exc)


def _qlora_generate(text: str) -> str:
    import torch
    _load_qlora()
    if _qlora["error"]:
        raise RuntimeError(_qlora["error"])
    tokenizer, model = _qlora["tokenizer"], _qlora["model"]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(ids, max_new_tokens=180, do_sample=False)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


def narrate(run: dict, diagnosis: dict | None, mode: str = "auto") -> dict:
    if not diagnosis or not diagnosis.get("root_cause"):
        return {"mode": "template", "text": "No diagnosis is available for this run."}
    cfg = settings()
    if mode == "auto":
        mode = "qlora" if cfg.explainer_adapter else "llm" if cfg.explainer_narrative == "llm" else "template"
    if mode == "qlora":
        try:
            return {"mode": "qlora", "text": _qlora_generate(prompt_for(run, diagnosis)), "adapter": cfg.explainer_adapter}
        except Exception as exc:
            return {"mode": "template", "text": narrative(diagnosis, run), "fallback_reason": f"QLoRA explainer unavailable: {exc}"}
    if mode == "llm":
        from ..llm import get_llm
        try:
            llm = get_llm(cfg.provider if cfg.provider != "sandbox" else "openai")
            result = llm.complete_json(SYSTEM + " Respond with JSON only: {\"narrative\": \"...\"}",
                                       prompt_for(run, diagnosis), seed=5, purpose="explainer", run_id=run.get("run_id"))
            text = (result.data or {}).get("narrative") if isinstance(result.data, dict) else None
            if text:
                return {"mode": "llm", "text": text, "model": result.model, "cost_usd": result.cost_usd}
            raise RuntimeError(result.error or "empty narrative")
        except Exception as exc:
            return {"mode": "template", "text": narrative(diagnosis, run), "fallback_reason": f"LLM explainer unavailable: {exc}"}
    return {"mode": "template", "text": narrative(diagnosis, run)}
