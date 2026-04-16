from __future__ import annotations

import base64
import math
import os
import random


def _sample_laplace(scale: float) -> float:
    """通过逆变换采样拉普拉斯噪声。"""
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


def add_laplace_noise(text: str, epsilon: float = 0.1) -> str:
    """
    路径B：差分隐私文本混淆（简化实现）。

    说明：严格 DP 多用于数值统计/向量扰动；此处按你的要求做文本层可解释 mock：
    - 用拉普拉斯噪声控制扰动比例
    - 对文本做随机替换/插入
    """
    value = text or ""
    if not value:
        return value

    safe_eps = max(1e-6, float(epsilon))
    scale = 1.0 / safe_eps
    noise_value = abs(_sample_laplace(scale))

    # 将噪声映射到字符扰动比例，限制在 [1%, 30%]
    perturb_ratio = max(0.01, min(0.30, noise_value * 0.02))
    edit_count = max(1, int(len(value) * perturb_ratio))

    chars = list(value)
    inject_pool = "abcdefghijklmnopqrstuvwxyz0123456789隐私协同安全"

    for _ in range(edit_count):
        if not chars:
            chars.append(random.choice(inject_pool))
            continue

        idx = random.randint(0, len(chars) - 1)
        if random.random() < 0.5:
            # 替换字符
            chars[idx] = random.choice(inject_pool)
        else:
            # 插入字符
            chars.insert(idx, random.choice(inject_pool))

    return "".join(chars)


class PrivacyLayer:
    """
    专利第一层隐私保护：数据隐身与混淆。

    支持三种模式：
    - none: 不处理
    - he: 路径A，同态加密 mock（base64），保留接口可替换为真实 HE
    - dp: 路径B，拉普拉斯噪声混淆
    """

    def __init__(self, mode: str | None = None, epsilon: float | None = None):
        self.mode = (mode or os.getenv("NEXUS_PRIVACY_MODE", "none")).strip().lower()
        self.epsilon = float(epsilon if epsilon is not None else os.getenv("NEXUS_PRIVACY_EPSILON", "0.1"))

    def homomorphic_encrypt(self, plaintext: str) -> str:
        """
        路径A占位：当前为 mock。
        后续可替换为真实同态加密 SDK 接口，不改调用方。
        """
        payload = (plaintext or "").encode("utf-8")
        return base64.b64encode(payload).decode("ascii")

    def protect_text(self, text: str) -> str:
        if self.mode in {"", "none", "off"}:
            return text

        if self.mode in {"a", "he", "homomorphic", "path_a"}:
            return self.homomorphic_encrypt(text)

        if self.mode in {"b", "dp", "laplace", "path_b"}:
            return add_laplace_noise(text, epsilon=self.epsilon)

        if self.mode in {"both", "he+dp", "dp+he"}:
            encrypted = self.homomorphic_encrypt(text)
            return add_laplace_noise(encrypted, epsilon=self.epsilon)

        # 未知模式时保持可用性，默认透传
        return text
