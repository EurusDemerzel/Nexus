import requests
import os

# 使用 Tailscale IP（也可通过环境变量 LLM_URL 覆盖）
DEFAULT_URL = "http://100.92.149.102:8080/v1/chat/completions"
LLM_URL = os.getenv("LLM_URL", DEFAULT_URL)

def generate(prompt: str, max_tokens=500) -> str:
    """调用 Mac mini 上的 llama.cpp 服务生成回答"""
    try:
        resp = requests.post(
            LLM_URL,
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        # llama.cpp server 兼容 OpenAI API 格式
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM 调用失败: {e}"