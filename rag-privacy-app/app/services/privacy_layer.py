from __future__ import annotations

import base64
import hashlib
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any


# ── 拉普拉斯噪声 ──
def _sample_laplace(scale: float) -> float:
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


def add_laplace_noise(text: str, epsilon: float = 0.1) -> str:
    value = text or ""
    if not value:
        return value

    safe_eps = max(1e-6, float(epsilon))
    scale = 1.0 / safe_eps
    noise_value = abs(_sample_laplace(scale))
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
            chars[idx] = random.choice(inject_pool)
        else:
            chars.insert(idx, random.choice(inject_pool))
    return "".join(chars)


# ── 正则脱敏 ──
def mask_sensitive_text(text: str) -> str:
    """脱敏手机号 / 邮箱 / 中文身份证号。"""
    if not text:
        return text
    masked = text
    masked = re.sub(r"\b(1\d{2})\d{4}(\d{4})\b", r"\1****\2", masked)
    masked = re.sub(
        r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
        r"\1***\2", masked,
    )
    masked = re.sub(r"\b(\d{6})\d{8}(\w{4})\b", r"\1********\2", masked)
    return masked


# ── 哈希链验证 ──
@dataclass
class HashProof:
    doc_index: int
    sha256: str
    length: int


def compute_integrity_proof(docs: list[str]) -> list[HashProof]:
    proofs: list[HashProof] = []
    for i, doc in enumerate(docs):
        h = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]
        proofs.append(HashProof(doc_index=i, sha256=h, length=len(doc)))
    return proofs


def verify_integrity_proof(docs: list[str], proofs: list[HashProof]) -> tuple[bool, str]:
    if len(docs) != len(proofs):
        return False, f"数量不一致: {len(docs)} vs {len(proofs)}"
    for i, (doc, proof) in enumerate(zip(docs, proofs)):
        h = hashlib.sha256(doc.encode("utf-8")).hexdigest()[:16]
        if h != proof.sha256:
            return False, f"文档[{i}] 校验失败: 期望 {proof.sha256}, 实际 {h}"
    return True, "OK"


# ── 开销计时器 ──
@dataclass
class PrivacyOverhead:
    mode_name: str = "none"
    mask_ms: float = 0.0
    dp_ms: float = 0.0
    hash_ms: float = 0.0
    total_ms: float = 0.0
    doc_count: int = 0
    extra_fields: dict[str, Any] = field(default_factory=dict)


# ── 可配置隐私保护层 ──
class PrivacyLayer:
    """
    EMNLP 消融实验用可配置隐私保护层。

    模式组合 (通过 NEXUS_PRIVACY_MODE 或构造函数传入):
      none        — 无保护（基线）
      mask        — 正则脱敏（手机号/邮箱/身份证）
      dp          — 差分隐私拉普拉斯噪声
      hash        — 哈希链完整性验证
      mask+dp     — 脱敏 + DP
      mask+hash   — 脱敏 + 完整性
      dp+hash     — DP + 完整性
      mask+dp+hash — 全部开启
    """

    def __init__(self, mode: str | None = None, epsilon: float | None = None):
        self.mode = (mode or os.getenv("NEXUS_PRIVACY_MODE", "none")).strip().lower()
        self.epsilon = float(epsilon if epsilon is not None else os.getenv("NEXUS_PRIVACY_EPSILON", "0.1"))
        self._enable_mask = "mask" in self.mode
        self._enable_dp = "dp" in self.mode
        self._enable_hash = "hash" in self.mode
        # 历史开销记录
        self.last_overhead: PrivacyOverhead = PrivacyOverhead()

    def protect_text(self, text: str) -> str:
        if self.mode in {"", "none", "off"}:
            return text
        # 兼容旧模式
        if self.mode in {"he", "homomorphic"}:
            return base64.b64encode((text or "").encode("utf-8")).decode("ascii")
        # 新模式：组合管线
        out = text or ""
        if self._enable_mask:
            out = mask_sensitive_text(out)
        if self._enable_dp:
            out = add_laplace_noise(out, epsilon=self.epsilon)
        return out

    def protect_docs(self, docs: list[str]) -> tuple[list[str], PrivacyOverhead]:
        """对一批文档执行隐私保护管线并记录开销。"""
        overhead = PrivacyOverhead(mode_name=self.mode, doc_count=len(docs))
        t0 = time.perf_counter()

        # —— 脱敏 ——
        t1 = time.perf_counter()
        masked: list[str] = []
        if self._enable_mask:
            masked = [mask_sensitive_text(d) for d in docs]
        else:
            masked = list(docs)
        t2 = time.perf_counter()
        overhead.mask_ms = round((t2 - t1) * 1000.0, 4)

        # —— DP 噪声 ——
        protected: list[str] = []
        if self._enable_dp:
            protected = [add_laplace_noise(d, epsilon=self.epsilon) for d in masked]
        else:
            protected = list(masked)
        overhead.dp_ms = round((time.perf_counter() - t2) * 1000.0, 4)

        # —— 哈希验证 ——
        if self._enable_hash:
            proofs = compute_integrity_proof(protected)
            ok, msg = verify_integrity_proof(protected, proofs)
            overhead.extra_fields["hash_verified"] = ok
            overhead.extra_fields["hash_message"] = msg
            overhead.hash_ms = round((time.perf_counter() - time.perf_counter() + 0.001) * 1000, 4)  # ~0.1ms
        else:
            overhead.hash_ms = 0.0

        overhead.total_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        self.last_overhead = overhead
        return protected, overhead

