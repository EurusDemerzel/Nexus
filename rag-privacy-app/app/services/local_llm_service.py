#!/usr/bin/env python3
"""local_llm_service.py — GPTQ 模型加载（直达修复版）"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

_DEVICE = os.getenv("LOCAL_LLM_DEVICE", "cuda:0")
_MODEL_PATH = os.getenv("LOCAL_LLM_PATH", "")
_model = None
_tokenizer = None

_GPTQModel = None
for _imp in ("from auto_gptq import AutoGPTQForCausalLM",):
    try: exec(_imp); _GPTQModel = locals().get("AutoGPTQForCausalLM"); break
    except ImportError: pass


def _patch_quantize_config(model_path: str):
    cfg = os.path.join(model_path, "quantize_config.json")
    if not os.path.isfile(cfg): return
    raw = Path(cfg).read_text(encoding="utf-8")
    if '"QuantizeConfig"' in raw:
        Path(cfg).write_text(raw.replace('"QuantizeConfig"', '"BaseQuantizeConfig"'), encoding="utf-8")
        print("[local_llm] Patched quantize_config.json")


def load_model(force_reload: bool = False) -> bool:
    global _model, _tokenizer
    if _model is not None and not force_reload: return True

    mp = _MODEL_PATH or str(Path(__file__).resolve().parents[2] / "models_for_server" / "Qwen2.5-1.5B-Instruct-GPTQ-Int4")
    if not os.path.isdir(mp):
        print(f"[local_llm] Not found: {mp}"); return False

    print(f"[local_llm] Loading from {mp} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    _tokenizer = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)

    if _GPTQModel is not None:
        print("[local_llm] AutoGPTQForCausalLM")
        _model = _GPTQModel.from_quantized(mp, device=_DEVICE, trust_remote_code=True)
    else:
        _patch_quantize_config(mp)
        print("[local_llm] AutoModelForCausalLM")
        _model = AutoModelForCausalLM.from_pretrained(mp, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)

    print(f"[local_llm] Loaded on {_model.device}")
    return True


def generate(prompt: str, max_tokens: int = 128) -> str:
    if _model is None and not load_model(): return "错误: 本地 LLM 未加载"
    print(f"[local_llm] prompt_len={len(prompt)}")
    try:
        messages = [{"role": "user", "content": prompt}]
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        import torch
        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            outputs = _model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=False, pad_token_id=_tokenizer.eos_token_id)
        resp = _tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"[local_llm] resp_len={len(resp)}")
        return resp
    except Exception as e:
        return f"推理失败: {e}"


def unload_model():
    global _model, _tokenizer
    if _model is not None:
        import torch; del _model; del _tokenizer; _model = _tokenizer = None; torch.cuda.empty_cache()
        print("[local_llm] Unloaded")


if __name__ == "__main__":
    if load_model():
        print("\n" + generate("法国的首都是哪里？", 50))
        unload_model()
    else:
        sys.exit(1)
