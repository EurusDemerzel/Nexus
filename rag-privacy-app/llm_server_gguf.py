#!/usr/bin/env python3
"""
llm_server_gguf.py — GGUF 本地模型 API 服务器 (FastAPI + uvicorn)
已验证稳定版本，在 3090 服务器上成功处理过成千上万次请求。

用法:
  python llm_server_gguf.py --port 8081 --model-path ./models_for_server/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf

测试:
  curl -X POST http://localhost:8081/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Question: What is the capital of France?\n\nAnswer (ONLY the answer, no explanation):"}'

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
    output = llm(
        query.prompt,
        max_tokens=5,          # 强制短答
        temperature=0.0,
        stop=["\n", ".", "?", "Context:", "You are"],  # 截断后续废话
        echo=False,
    )
    response_text = output["choices"][0]["text"].strip()
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
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
