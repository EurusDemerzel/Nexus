#!/usr/bin/env python3
"""
llm_server.py — 独立的 LLM API 服务器（OpenAI 兼容）
在 3090 服务器上运行，供主应用通过 LLM_URL 调用。

用法:
  python llm_server.py --port 8080

测试:
  curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
"""

import os
import sys
import argparse
from pathlib import Path

# 将项目根目录加入路径，复用 local_llm_service.py
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, request, jsonify

# 导入本地推理服务
os.environ.setdefault("LOCAL_LLM_PATH", str(ROOT_DIR / "models_for_server" / "Qwen2.5-1.5B-Instruct-GPTQ-Int4"))
from app.services.local_llm_service import load_model, generate, unload_model

app = Flask(__name__)


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI 兼容的聊天补全接口"""
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 128)
    temperature = data.get("temperature", 0.1)

    # 拼装 prompt（取最后一条 user 消息）
    prompt = ""
    for msg in messages:
        if msg.get("role") == "user":
            prompt = msg.get("content", "")

    if not prompt:
        return jsonify({"error": "No user message found"}), 400

    try:
        text = generate(prompt, max_tokens=max_tokens)
        return jsonify({
            "choices": [{"message": {"content": text}}]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def main():
    parser = argparse.ArgumentParser(description="Launch local LLM API server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to local model directory")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0 / cpu)")
    args = parser.parse_args()

    if args.model_path:
        os.environ["LOCAL_LLM_PATH"] = args.model_path
    if args.device:
        os.environ["LOCAL_LLM_DEVICE"] = args.device

    print(f"[llm_server] Loading model ...")
    if not load_model():
        sys.exit(1)

    print(f"[llm_server] Starting on http://0.0.0.0:{args.port}")
    print(f"[llm_server] Model: {os.environ.get('LOCAL_LLM_PATH')}")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
