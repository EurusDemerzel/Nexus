#!/usr/bin/env python3
"""
local_llm_service.py
在 3090 GPU 上加载 GPTQ 量化模型并进行推理。

依赖:
  pip install torch transformers accelerate optimum auto-gptq

环境变量:
  LOCAL_LLM_PATH  - 模型路径（如 /home/vision/Nexus/models_for_server/Qwen1.5-7B-Chat-GPTQ）
  LOCAL_LLM_DEVICE - 设备映射（默认 "cuda:0"）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEVICE = os.getenv("LOCAL_LLM_DEVICE", "cuda:0")
_MODEL_PATH = os.getenv("LOCAL_LLM_PATH", "")

_model = None
_tokenizer = None


def load_model(force_reload: bool = False) -> bool:
    """Load the GPTQ model and tokenizer into GPU memory."""
    global _model, _tokenizer

    if _model is not None and not force_reload:
        return True

    model_path = _MODEL_PATH
    if not model_path:
        # Fallback: check project-relative path
        model_path = str(
            Path(__file__).resolve().parents[2]
            / "models_for_server"
            / "Qwen1.5-7B-Chat-GPTQ"
        )

    if not os.path.isdir(model_path):
        print(f"[local_llm] Model path not found: {model_path}")
        print("[local_llm] Set LOCAL_LLM_PATH env var or place model in models_for_server/")
        return False

    print(f"[local_llm] Loading model from {model_path} ...")
    print(f"[local_llm] Device: {_DEVICE}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[local_llm] transformers not installed. Run: pip install transformers")
        return False

    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=_DEVICE,
            trust_remote_code=True,
            # GPTQ models use the quantized config automatically
        )

        print(f"[local_llm] Model loaded successfully on {_model.device}")
        return True
    except Exception as e:
        print(f"[local_llm] Failed to load model: {type(e).__name__}: {e}")
        _model = None
        _tokenizer = None
        return False


def generate(prompt: str, max_tokens: int = 128) -> str:
    """Generate text using the local GPTQ model."""
    global _model, _tokenizer

    if _model is None:
        if not load_model():
            return "错误: 本地 LLM 模型未加载，请检查 LOCAL_LLM_PATH"

    prompt_preview = prompt.replace("\n", " ")[:200]
    print(f"[local_llm.generate] prompt_len={len(prompt)} preview={prompt_preview}...")

    try:
        messages = [{"role": "user", "content": prompt}]
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        import torch
        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1,
                do_sample=False,
                pad_token_id=_tokenizer.eos_token_id,
            )

        response = _tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        resp_preview = response.replace("\n", " ")[:200]
        print(f"[local_llm.generate] response_len={len(response)} preview={resp_preview}...")
        return response

    except Exception as e:
        print(f"[local_llm.generate][ERROR] {type(e).__name__}: {e}")
        return f"本地 LLM 推理失败: {e}"


def unload_model():
    """Free GPU memory by deleting the model."""
    global _model, _tokenizer
    if _model is not None:
        import torch
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        torch.cuda.empty_cache()
        print("[local_llm] Model unloaded, GPU memory freed.")


if __name__ == "__main__":
    # Quick smoke test
    if load_model():
        resp = generate("Hello! What is the capital of France?", max_tokens=50)
        print(f"\nSmoke test response:\n{resp}")
        unload_model()
    else:
        sys.exit(1)
