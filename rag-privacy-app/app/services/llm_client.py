import os

# ---- 模式选择 ----
# 通过 LLM_CLIENT_MODE 环境变量切换:
#   "local"  -> 本地 GPU (3090, 使用 local_llm_service.py)
#   "ark"    -> 火山引擎 Ark API (无需 GPU)
#   "gguf"   -> 本地 GGUF 模型服务 (http://.../generate)
#   "remote" -> 远程 llama.cpp 服务器 (OpenAI 兼容, 默认)
_CLIENT_MODE = os.getenv("LLM_CLIENT_MODE", "remote").strip().lower()

# ---- 远程 llama.cpp / OpenAI 兼容配置 (mode=remote) ----
_REMOTE_URL = os.getenv("LLM_URL", "http://100.106.140.63:8080/v1/chat/completions")

# ---- GGUF 服务配置 (mode=gguf) ----
_GGUF_URL = os.getenv("GGUF_URL", "http://localhost:8080/generate")


def generate(prompt: str, max_tokens: int = 128) -> str:
    """根据 LLM_CLIENT_MODE 选择对应的后端生成回答。"""
    safe_prompt = prompt if isinstance(prompt, str) else str(prompt)
    print(f"[llm_client] mode={_CLIENT_MODE} prompt_len={len(safe_prompt)}")

    if _CLIENT_MODE == "local":
        return _generate_local(safe_prompt, max_tokens)
    elif _CLIENT_MODE == "ark":
        return _generate_ark(safe_prompt, max_tokens)
    elif _CLIENT_MODE == "gguf":
        return _generate_gguf(safe_prompt, max_tokens)
    else:
        return _generate_remote(safe_prompt, max_tokens)


# =================== 本地 GPU 推理 ===================
def _generate_local(prompt: str, max_tokens: int) -> str:
    try:
        from app.services.local_llm_service import generate as local_gen
        return local_gen(prompt, max_tokens=max_tokens)
    except Exception as e:
        return f"本地 LLM 调用失败: {e}"


# =================== 火山引擎 Ark API ===================
def _generate_ark(prompt: str, max_tokens: int) -> str:
    import requests

    api_key = os.getenv("LLM_SECRET_ACCESS_KEY", "")
    model_id = os.getenv("LLM_MODEL_ID", "")
    api_url = os.getenv("LLM_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")

    if not api_key or not model_id:
        return "错误: .env 中 LLM_SECRET_ACCESS_KEY 或 LLM_MODEL_ID 未设置"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        print(f"[llm_client.ark] resp_len={len(text)}")
        return text
    except Exception as e:
        print(f"[llm_client.ark][ERROR] {e}")
        return f"LLM 调用失败(ark): {e}"


# =================== GGUF 模型服务 (http://.../generate) ===================
def _generate_gguf(prompt: str, max_tokens: int) -> str:
    """
    调用本地 GGUF 服务（如 llama.cpp 启动的 /generate 端点）。
    支持的 JSON 格式: {"prompt": "...", "max_tokens": N}
    返回格式: {"content": "..."} 或 纯文本
    """
    import requests

    print(f"[llm_client.gguf] url={_GGUF_URL}")

    try:
        resp = requests.post(
            _GGUF_URL,
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "n_predict": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # 兼容多种常见 GGUF 服务返回格式
        if isinstance(data, dict):
            text = (
                data.get("content")
                or data.get("response")
                or data.get("text")
                or data.get("choices", [{}])[0].get("text")
                or data.get("choices", [{}])[0].get("message", {}).get("content")
                or str(data)
            )
        else:
            text = str(data)

        print(f"[llm_client.gguf] resp_len={len(text)}")
        return text
    except Exception as e:
        print(f"[llm_client.gguf][ERROR] {e}")
        return f"GGUF 调用失败: {e}"


# =================== 远程 llama.cpp / OpenAI 兼容服务 ===================
def _generate_remote(prompt: str, max_tokens: int) -> str:
    import requests

    print(f"[llm_client.remote] url={_REMOTE_URL}")

    try:
        resp = requests.post(
            _REMOTE_URL,
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "do_sample": False,
                "top_p": 0.95,
                "top_k": 40,
                "repetition_penalty": 1.1,
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        print(f"[llm_client.remote] resp_len={len(text)}")
        return text
    except Exception as e:
        print(f"[llm_client.remote][ERROR] {e}")
        return f"LLM 调用失败(remote): {e}"