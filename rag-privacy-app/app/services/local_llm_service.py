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
    """彻底绕过 optimum GPTQ：删 quantize_config.json + 清 config.json 中的 quantization_config"""
    # ---- 1. 修 config.json：移除 quantization_config 字段 ----
    cfg_json = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_json):
        raw = Path(cfg_json).read_text(encoding="utf-8")
        cfg = json.loads(raw)
        if "quantization_config" in cfg:
            del cfg["quantization_config"]
            Path(cfg_json).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[local_llm] Removed quantization_config from config.json")

    # ---- 2. 重命名 quantize_config.json（禁用 GPTQ） ----
    qt_cfg = os.path.join(model_path, "quantize_config.json")
    if os.path.isfile(qt_cfg):
        os.rename(qt_cfg, qt_cfg + ".bak")
        print("[local_llm] Disabled quantize_config.json")


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
        print("[local_llm] AutoModelForCausalLM (FP16)")
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
