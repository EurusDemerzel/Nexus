def mock_generate(prompt: str) -> str:
    if "你好" in prompt:
        return "你好！我是你的隐私保护助手。"
    return f"[模拟回答] 收到问题：{prompt[:100]}..."
