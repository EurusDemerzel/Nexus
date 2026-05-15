#!/usr/bin/env python3
"""
llm_server_gguf.py — GGUF 本地模型 API 服务器 (FastAPI + uvicorn)
使用 Chat Completion 模式 + chat_format="qwen"，确保 Qwen2.5 指令模型输出简洁。

用法:
  python llm_server_gguf.py --port 8081 --model-path ./models_for_server/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf

测试:
  curl -X POST http://localhost:8081/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Who was the man behind The Chipmunks?"}'

客户端配置:
  LLM_CLIENT_MODE=gguf
  GGUF_URL=http://localhost:8081/generate
"""

import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from llama_cpp import Llama

app = FastAPI()
llm = None


class Query(BaseModel):
    prompt: str


@app.post("/generate")
def generate(query: Query):
    # 将传入的纯文本 prompt 作为 user 消息，并强制系统指令要求极简回答
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a concise assistant. Provide ONLY the direct answer. Do NOT add any explanation, context, or greetings."},
            {"role": "user", "content": query.prompt},
        ],
        max_tokens=20,        # 足够生成短答案
        temperature=0.0,
        stop=["\n", "?"],     # 安全截断
    )
    response_text = output["choices"][0]["message"]["content"].strip()
    return {"response": response_text}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    llm = Llama(
        model_path=args.model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        chat_format="qwen",   # 指定 Qwen 专属模板，确保指令被正确理解
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
