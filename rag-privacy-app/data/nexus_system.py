from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Any
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.llm_client import generate
from app.services.local_retrieval import retrieve
from app.services.privacy_layer import PrivacyLayer
from app.services.split_decision import SplitDecision


@dataclass
class RetrievedDoc:
    text: str
    metadata: dict[str, Any]
    score: float


class NexusSystem:
    def __init__(
        self,
        static_k: int = 15,
        dynamic_conf_threshold: float = 0.45,
        device_type: int = 2,
        privacy_mode: str | None = None,
        privacy_epsilon: float = 0.1,
    ):
        # MOD: static-split 默认使用 top-15；可通过 NEXUS_STATIC_K 覆盖
        env_k = os.getenv("NEXUS_STATIC_K", "").strip()
        self.static_k = int(env_k) if env_k.isdigit() else static_k
        self.dynamic_conf_threshold = dynamic_conf_threshold
        self.decision_engine = SplitDecision(device_type=device_type)
        # MOD: 默认关闭隐私层，避免实验阶段意外噪声影响
        effective_privacy_mode = "none" if privacy_mode is None else privacy_mode
        self.privacy_layer = PrivacyLayer(mode=effective_privacy_mode, epsilon=privacy_epsilon)
        # MOD: 可选调试覆盖（默认不启用），用于临时指定 local_k
        force_k_env = os.getenv("NEXUS_FORCE_LOCAL_K", "").strip()
        self.dynamic_force_local_k = int(force_k_env) if force_k_env.isdigit() else None

    def _normalize_docs(self, docs: list[dict], source_mode: str = "unknown") -> list[RetrievedDoc]:
        """
        将原始检索结果转换为 RetrievedDoc 对象。
        如果 metadata 中缺少 title，补充默认标题 "doc_{index}"。
        """
        normalized: list[RetrievedDoc] = []
        for idx, d in enumerate(docs):
            metadata = d.get("metadata", {}) or {}
            
            # 补充缺失的 title 字段
            if "title" not in metadata or not metadata.get("title"):
                metadata = {**metadata, "title": f"doc_{idx}_{source_mode}"}
            
            normalized.append(
                RetrievedDoc(
                    text=d.get("content", ""),
                    metadata=metadata,
                    score=float(d.get("score", 0.0)),
                )
            )
        return normalized

    # MOD: 统一 Qwen Prompt 模板，static-split 与 nexus-dynamic 共用
    def _build_qwen_prompt(self, question: str, context_text: str) -> str:
        if not context_text or not context_text.strip():
            return f"Question: {question}\n\nAnswer (ONLY the answer, no explanation, no context, one single phrase):"

        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer (ONLY the answer, no explanation, no context, one single phrase):"
        )

    # MOD: 统一 prompt 构建入口（文档截断 + 总长度限制）
    def _build_prompt(self, question: str, docs: list[RetrievedDoc]) -> str:
        # 固定后缀：Question + Answer 指令
        _SUFFIX = f"\n\nQuestion: {question}\n\nAnswer (ONLY the answer, no explanation, no context, one single phrase):"
        _MAX_PROMPT = 2500
        _MAX_DOC_CHARS = 400

        # 先构建不带 context 的 prompt 模板，拿到后缀长度，剩余给文档
        suffix_len = len(_SUFFIX)
        budget = _MAX_PROMPT - suffix_len - len("Context:\n\n")  # 留给文档块的总字符数
        if budget <= 0:
            budget = 500  # 极端兜底

        blocks = []
        used = 0
        for i, doc in enumerate(docs, start=1):
            title = doc.metadata.get("title", "unknown") if doc.metadata else "unknown"
            # 截断文档正文
            full_text = doc.text if doc.text else ""
            if len(full_text) > _MAX_DOC_CHARS:
                truncated = full_text[:_MAX_DOC_CHARS].rstrip() + "..."
            else:
                truncated = full_text

            block = f"[{i}] title={title} score={doc.score:.4f}\n{truncated}"
            # 不是最后一块时，预留 "\n\n" 分隔符
            separator = "\n\n" if i < len(docs) else ""
            needed = len(block) + len(separator)

            if used + needed > budget:
                # 超出总长度预算，截断文档列表
                print(f"[NexusPrompt] budget exceeded: used={used} + needed={needed} > budget={budget}, stopping at doc {i - 1}/{len(docs)}")
                break

            blocks.append(block)
            used += needed

        context_text = "\n\n".join(blocks) if blocks else ""
        prompt = self._build_qwen_prompt(question, context_text)

        # 调试日志
        actual_len = len(prompt)
        debug_preview = prompt.replace("\n", " ")[:200]
        print(f"[NexusPrompt] docs_used={len(blocks)}/{len(docs)} prompt_len={actual_len} preview={debug_preview}...")
        return prompt

    def _apply_privacy_to_docs(self, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """
        专利第一层隐私保护：在文档上传云侧前做数据隐身/混淆。
        仅改 text，不改 metadata 与 score。
        """
        protected: list[RetrievedDoc] = []
        for doc in docs:
            protected_text = self.privacy_layer.protect_text(doc.text)
            protected.append(
                RetrievedDoc(
                    text=protected_text,
                    metadata=doc.metadata,
                    score=doc.score,
                )
            )
        return protected

    def cloud_llm_generate(self, question: str) -> str:
        return generate(question)

    def _estimate_network_ms(self, payload_chars: int, bandwidth_mbps: float, rtt_ms: float) -> float:
        # 文本近似按 UTF-8 2 bytes/char 估算；网络耗时=RTT+传输耗时
        payload_bits = max(0.0, float(payload_chars) * 2.0 * 8.0)
        bw = max(1e-6, float(bandwidth_mbps) * 1_000_000.0)
        transfer_ms = (payload_bits / bw) * 1000.0
        return float(rtt_ms) + transfer_ms

    def _simulate_energy_delta(self, edge_compute_ms: float, cpu_percent: float, battery_percent: float) -> float:
        # 简单可解释模拟：耗时 * CPU负载比例 * 电量衰减系数
        factor = max(0.0, min(1.0, cpu_percent / 100.0))
        battery_factor = max(0.2, min(1.0, battery_percent / 100.0))
        return round(edge_compute_ms * factor * 0.0002 * (2.0 - battery_factor), 6)

    def _full_rag_answer(
        self,
        question: str,
        local_docs: list[RetrievedDoc],
        use_cloud: bool,
        runtime_metrics: dict,
        mode_tag: str,
        local_k: int,
    ):
        """
        全量 RAG 答案路径：
        - 在拼接 prompt 前调用隐私层（专利第一层）
        - 返回 response 与耗时分解，供反馈闭环使用（专利步骤5）
        """
        # MOD: 调试日志 - 打印 use_cloud 和 local_docs 数量
        print(
            f"[NexusDebug][{mode_tag}] use_cloud={use_cloud} local_docs_count={len(local_docs)} "
            f"local_k={local_k} privacy_mode={self.privacy_layer.mode}"
        )

        # MOD: 无检索结果时降级为纯云端问答
        if not local_docs:
            print(f"[NexusDebug][{mode_tag}] no local docs, fallback to cloud-only prompt")
            start = time.perf_counter()
            response = self.cloud_llm_generate(question)
            end = time.perf_counter()
            cloud_only_ms = (end - start) * 1000.0
            return response, {
                "edge_compute_ms": 0.0,
                "network_ms": 0.0,
                "cloud_compute_ms": cloud_only_ms,
                "total_ms": cloud_only_ms,
            }

        edge_start = time.perf_counter()
        protected_docs = self._apply_privacy_to_docs(local_docs)
        prompt = self._build_prompt(question, protected_docs)
        # MOD: 调试日志 - 打印 prompt 长度和前 200 字符
        prompt_preview = (prompt or "").replace("\n", " ")[:200]
        print(f"[NexusDebug][{mode_tag}] prompt_len={len(prompt or '')} prompt_preview={prompt_preview}...")
        edge_end = time.perf_counter()
        edge_compute_ms = (edge_end - edge_start) * 1000.0

        if not use_cloud:
            # MOD: 调试日志 - 明确 use_cloud=False 警告
            print(f"[NexusDebug][{mode_tag}][WARN] use_cloud=False，走 EDGE-ONLY 模拟回答路径")
            # 当前项目无真实端侧大模型推理接口，保持兼容：返回拼接提示作为模拟回答
            response = "[EDGE-ONLY SIMULATION]\n" + prompt[:1200]
            # MOD: 调试日志 - 打印 response 长度和前 200 字符
            response_preview = (response or "").replace("\n", " ")[:200]
            print(
                f"[NexusDebug][{mode_tag}] response_len={len(response or '')} "
                f"response_preview={response_preview}..."
            )
            total_ms = edge_compute_ms
            network_ms = 0.0
            cloud_compute_ms = 0.0
            return response, {
                "edge_compute_ms": edge_compute_ms,
                "network_ms": network_ms,
                "cloud_compute_ms": cloud_compute_ms,
                "total_ms": total_ms,
            }

        network_ms = self._estimate_network_ms(
            payload_chars=len(prompt),
            bandwidth_mbps=runtime_metrics.get("bandwidth_mbps", 50.0),
            rtt_ms=runtime_metrics.get("rtt_ms", 50.0),
        )

        cloud_start = time.perf_counter()
        # MOD: 统一走同一 Prompt 生成路径
        response = generate(prompt)
        cloud_end = time.perf_counter()
        cloud_compute_ms = (cloud_end - cloud_start) * 1000.0

        # MOD: 调试日志 - 打印 response 长度和前 200 字符
        response_preview = (response or "").replace("\n", " ")[:200]
        print(
            f"[NexusDebug][{mode_tag}] response_len={len(response or '')} "
            f"response_preview={response_preview}..."
        )

        total_ms = edge_compute_ms + network_ms + cloud_compute_ms
        return response, {
            "edge_compute_ms": edge_compute_ms,
            "network_ms": network_ms,
            "cloud_compute_ms": cloud_compute_ms,
            "total_ms": total_ms,
        }

    def _static_split_answer(self, question: str):
        local_docs_raw = retrieve(question, top_k=self.static_k)
        local_docs = self._normalize_docs(local_docs_raw, source_mode="static-split")

        # MOD: 调试日志，打印 static 模式 local_k 与检索数量
        print(f"[NexusDebug][static-split] local_k={self.static_k}, retrieved_docs={len(local_docs)}")

        response, timing = self._full_rag_answer(
            question=question,
            local_docs=local_docs,
            use_cloud=True,
            runtime_metrics={"bandwidth_mbps": 50.0, "rtt_ms": 40.0, "cpu_percent": 50.0, "battery_percent": 60.0},
            mode_tag="static-split",
            local_k=self.static_k,
        )
        return response, local_docs, timing

    def _cloud_only_answer(self, question: str):
        start = time.perf_counter()
        response = self.cloud_llm_generate(question)
        end = time.perf_counter()

        timing = {
            "edge_compute_ms": 0.0,
            "network_ms": 0.0,
            "cloud_compute_ms": (end - start) * 1000.0,
            "total_ms": (end - start) * 1000.0,
        }
        return response, timing

    def ask(self, question: str, mode: str):
        if mode == "cloud-only":
            response, timing = self._cloud_only_answer(question)
            retrieved_docs: list[RetrievedDoc] = []
            self.decision_engine.record_feedback(
                edge_compute_ms=timing["edge_compute_ms"],
                network_ms=timing["network_ms"],
                cloud_compute_ms=timing["cloud_compute_ms"],
                total_ms=timing["total_ms"],
                energy_delta=0.0,
            )
            return response, retrieved_docs, {
                "local_k": 0, "use_cloud": True, "load_score": 0.0,
            }

        if mode == "static-split":
            response, local_docs, timing = self._static_split_answer(question)
            energy_delta = self._simulate_energy_delta(
                edge_compute_ms=timing["edge_compute_ms"],
                cpu_percent=50.0,
                battery_percent=60.0,
            )
            self.decision_engine.record_feedback(
                edge_compute_ms=timing["edge_compute_ms"],
                network_ms=timing["network_ms"],
                cloud_compute_ms=timing["cloud_compute_ms"],
                total_ms=timing["total_ms"],
                energy_delta=energy_delta,
            )
            return response, local_docs, {
                "local_k": self.static_k, "use_cloud": True, "load_score": 0.0,
            }

        if mode == "nexus-dynamic":
            # 关键改造：严格使用专利决策引擎，不再使用问题长度启发式
            plan = self.decision_engine.decide_plan()

            # MOD: 调试日志 - 打印计划对象关键字段
            print(
                "[NexusDebug][nexus-dynamic][plan] "
                f"split_id={plan.split_id}, use_cloud={plan.use_cloud}, "
                f"local_k={plan.local_k}, score_s={plan.score_s:.4f}, metrics={plan.metrics}"
            )
            if not plan.use_cloud:
                # MOD: 调试日志 - 明确说明动态决策禁用云端
                print(
                    "[NexusDebug][nexus-dynamic][WARN] plan.use_cloud=False，"
                    "动态分支将不会调用云端 LLM，可能导致延迟异常偏低。"
                )

            # MOD: 默认采用决策引擎输出；仅在设置 NEXUS_FORCE_LOCAL_K 时覆盖
            original_local_k = plan.local_k
            effective_local_k = self.dynamic_force_local_k if self.dynamic_force_local_k is not None else plan.local_k

            local_docs_raw = []
            if effective_local_k > 0:
                local_docs_raw = retrieve(question, top_k=effective_local_k)

            local_docs = self._normalize_docs(local_docs_raw, source_mode="nexus-dynamic")

            # MOD: 调试日志，打印动态决策输出与实际执行 local_k
            print(
                f"[NexusDebug][nexus-dynamic] decision_local_k={original_local_k}, "
                f"effective_local_k={effective_local_k}, retrieved_docs={len(local_docs)}, score_s={plan.score_s:.4f}"
            )

            response, timing = self._full_rag_answer(
                question=question,
                local_docs=local_docs,
                use_cloud=plan.use_cloud,
                runtime_metrics=plan.metrics,
                mode_tag="nexus-dynamic",
                local_k=effective_local_k,
            )

            energy_delta = self._simulate_energy_delta(
                edge_compute_ms=timing["edge_compute_ms"],
                cpu_percent=plan.metrics.get("cpu_percent", 50.0),
                battery_percent=plan.metrics.get("battery_percent", 60.0),
            )
            self.decision_engine.record_feedback(
                edge_compute_ms=timing["edge_compute_ms"],
                network_ms=timing["network_ms"],
                cloud_compute_ms=timing["cloud_compute_ms"],
                total_ms=timing["total_ms"],
                energy_delta=energy_delta,
            )
            return response, local_docs, {
                "local_k": effective_local_k,
                "use_cloud": plan.use_cloud,
                "load_score": round(plan.score_s, 4),
            }

        raise ValueError(f"Unknown mode: {mode}")
