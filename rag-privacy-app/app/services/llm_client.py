import requests
import os

# 使用 Tailscale IP（也可通过环境变量 LLM_URL 覆盖）
DEFAULT_URL = "http://100.106.140.63:8080/v1/chat/completions"
LLM_URL = os.getenv("LLM_URL", DEFAULT_URL)

def generate(prompt: str, max_tokens: int = 128) -> str:
    """调用 Mac mini 上的 llama.cpp 服务生成回答"""
    try:
        # MOD: 调试日志 - 打印 prompt 长度和前 200 字符
        safe_prompt = prompt if isinstance(prompt, str) else str(prompt)
        prompt_preview = safe_prompt.replace("\n", " ")[:200]
        print(f"[llm_client.generate] prompt_len={len(safe_prompt)} preview={prompt_preview}...")

        resp = requests.post(
            LLM_URL,
            json={
                "messages": [{"role": "user", "content": safe_prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "do_sample": False,
                "top_p": 0.95,
                "top_k": 40,
                "repetition_penalty": 1.1,
            },
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        # llama.cpp server 兼容 OpenAI API 格式
        response_text = data["choices"][0]["message"]["content"]

        # MOD: 调试日志 - 打印 response 长度和前 200 字符
        response_preview = (response_text or "").replace("\n", " ")[:200]
        print(f"[llm_client.generate] response_len={len(response_text or '')} preview={response_preview}...")
        return response_text
    except Exception as e:
        # MOD: 调试日志 - 打印异常信息
        print(f"[llm_client.generate][ERROR] {type(e).__name__}: {e}")
        return f"LLM 调用失败: {e}"