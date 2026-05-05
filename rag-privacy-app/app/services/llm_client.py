import os

# ---- 模式选择 ----
# 通过 LLM_CLIENT_MODE 环境变量切换:
#   "local"  -> 本地 GPU (3090, 使用 local_llm_service.py)
#   "ark"    -> 火山引擎 Ark API (无需 GPU)
#   "remote" -> 远程 llama.cpp 服务器 (默认, 当前 Mac Mini)
_CLIENT_MODE = os.getenv("LLM_CLIENT_MODE", "remote").strip().lower()

# ---- 远程 llama.cpp 配置 (mode=remote) ----
_REMOTE_URL = os.getenv(
    "LLM_URL",
    "http://100.106.140.63:8080/v1/chat/completions"
)


def generate(prompt: str, max_tokens: int = 128) -> str:
    """根据 LLM_CLIENT_MODE 选择对应的后端生成回答。"""
    safe_prompt = prompt if isinstance(prompt, str) else str(prompt)
    prompt_preview = safe_prompt.replace("\n", " ")[:200]
    print(f"[llm_client] mode={_CLIENT_MODE} prompt_len={len(safe_prompt)}")

    if _CLIENT_MODE == "local":
        return _generate_local(safe_prompt, max_tokens)
    elif _CLIENT_MODE == "ark":
        return _generate_ark(safe_prompt, max_tokens)
    else:
        return _generate_remote(safe_prompt, max_tokens, prompt_preview)


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
    api_url = os.getenv(
        "LLM_API_URL",
        "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    )

    if not api_key or not model_id:
        return "错误: .env 中 LLM_SECRET_ACCESS_KEY 或 LLM_MODEL_ID 未设置"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        resp_preview = (text or "").replace("\n", " ")[:200]
        print(f"[llm_client.ark] response_len={len(text or '')} preview={resp_preview}...")
        return text
    except Exception as e:
        print(f"[llm_client.ark][ERROR] {type(e).__name__}: {e}")
        return f"LLM 调用失败(ark): {e}"


# =================== 远程 llama.cpp 服务器 ===================
def _generate_remote(prompt: str, max_tokens: int, preview: str) -> str:
    import requests

    print(f"[llm_client.remote] url={_REMOTE_URL} preview={preview}...")

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
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        resp_preview = (text or "").replace("\n", " ")[:200]
        print(f"[llm_client.remote] response_len={len(text or '')} preview={resp_preview}...")
        return text
    except Exception as e:
        print(f"[llm_client.remote][ERROR] {type(e).__name__}: {e}")
        return f"LLM 调用失败(remote): {e}"