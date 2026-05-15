#!/usr/bin/env python3
"""
llm_server_gguf.py — GGUF 本地模型 API 服务器
在 3090 服务器上运行，供主应用通过 GGUF_URL 调用。

用法:
  python llm_server_gguf.py --port 8081
  python llm_server_gguf.py --port 8081 --model-path ./models/Qwen2.5-7B-Q4_K_M.gguf

测试:
  curl http://localhost:8081/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"What is the capital of France?","max_tokens":32}'

客户端配置:
  LLM_CLIENT_MODE=gguf
  GGUF_URL=http://localhost:8081/generate
"""

import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, request, jsonify

app = Flask(__name__)

# ---- 默认模型路径 ----
_DEFAULT_MODEL_DIR = ROOT_DIR / "models_for_server"
_DEFAULT_MODEL = _DEFAULT_MODEL_DIR / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

# ---- 全局 llama 实例 ----
_LLM = None


def load_gguf_model(model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
    """加载 GGUF 模型（使用 llama-cpp-python）。"""
    global _LLM
    try:
        from llama_cpp import Llama
        print(f"[llm_server_gguf] Loading model: {model_path}")
        print(f"[llm_server_gguf] n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")
        _LLM = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("[llm_server_gguf] Model loaded successfully.")
        return True
    except Exception as e:
        print(f"[llm_server_gguf][ERROR] Failed to load model: {e}")
        return False


@app.route("/generate", methods=["POST"])
def generate():
    """GGUF 模型生成接口。
    请求 JSON: {"prompt": "...", "max_tokens": 128, "temperature": 0.1, ...}
    返回 JSON: {"content": "..."}
    """
    if _LLM is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", data.get("n_predict", 128))
    temperature = data.get("temperature", 0.1)
    top_p = data.get("top_p", 0.95)
    top_k = data.get("top_k", 40)
    repeat_penalty = data.get("repeat_penalty", 1.1)
    stop = data.get("stop", [])

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        output = _LLM(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop if stop else None,
        )
        # llama-cpp-python 返回 {"choices": [{"text": "..."}]} 格式
        text = ""
        if isinstance(output, dict):
            choices = output.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
        return jsonify({"content": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if _LLM is not None else "model_not_loaded",
    })


def main():
    parser = argparse.ArgumentParser(description="Launch GGUF LLM API server")
    parser.add_argument("--port", type=int, default=8081,
                        help="Server port (default: 8081)")
    parser.add_argument("--model-path", type=str, default=str(_DEFAULT_MODEL),
                        help=f"Path to GGUF model file (default: {_DEFAULT_MODEL})")
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="Context window size (default: 4096)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU layers, -1 = all (default: -1)")
    args = parser.parse_args()

    model_path = args.model_path
    if not Path(model_path).is_file():
        print(f"[llm_server_gguf][ERROR] Model file not found: {model_path}")
        # 尝试 glob 查找目录下的第一个 .gguf 文件
        model_dir = Path(model_path).parent
        gguf_files = list(model_dir.glob("*.gguf"))
        if gguf_files:
            model_path = str(gguf_files[0])
            print(f"[llm_server_gguf] Auto-selected: {model_path}")
        else:
            sys.exit(1)

    if not load_gguf_model(model_path, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers):
        sys.exit(1)

    print(f"[llm_server_gguf] Starting on http://0.0.0.0:{args.port}")
    print(f"[llm_server_gguf] Model: {model_path}")
    print(f"[llm_server_gguf] Test: curl http://localhost:{args.port}/health")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
