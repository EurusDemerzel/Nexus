from dataclasses import dataclass
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
        static_k: int = 10,
        dynamic_conf_threshold: float = 0.45,
        device_type: int = 2,
        privacy_mode: str | None = None,
        privacy_epsilon: float = 0.1,
    ):
        self.static_k = static_k
        self.dynamic_conf_threshold = dynamic_conf_threshold
        self.decision_engine = SplitDecision(device_type=device_type)
        self.privacy_layer = PrivacyLayer(mode=privacy_mode, epsilon=privacy_epsilon)

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

    def _build_prompt(self, question: str, docs: list[RetrievedDoc]) -> str:
        if not docs:
            # 无检索结果警告
            print(f"⚠️ 警告: 无检索文档，问题={question[:50]}...")
            return question

        blocks = []
        for i, doc in enumerate(docs, start=1):
            title = doc.metadata.get("title", "unknown")
            blocks.append(f"[{i}] title={title} score={doc.score:.4f}\n{doc.text}")

        context = "\n\n".join(blocks)
        prompt = (
            "Answer the question using the retrieved evidence when possible. "
            "If evidence is insufficient, provide the best possible answer and mention uncertainty.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved Evidence:\n{context}\n\n"
            "Answer:"
        )
        
        # 调试输出：打印 prompt 前 250 字符以验证上下文格式
        debug_preview = prompt.replace("\n", " ")[:250]
        print(f"✓ Prompt 预览: {debug_preview}...")
        print(f"  └─ 检索文档数: {len(docs)}, 第1个标题: {docs[0].metadata.get('title', 'unknown')}")
        
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

    def _full_rag_answer(self, question: str, local_docs: list[RetrievedDoc], use_cloud: bool, runtime_metrics: dict):
        """
        全量 RAG 答案路径：
        - 在拼接 prompt 前调用隐私层（专利第一层）
        - 返回 response 与耗时分解，供反馈闭环使用（专利步骤5）
        """
        edge_start = time.perf_counter()
        protected_docs = self._apply_privacy_to_docs(local_docs)
        prompt = self._build_prompt(question, protected_docs)
        edge_end = time.perf_counter()
        edge_compute_ms = (edge_end - edge_start) * 1000.0

        if not use_cloud:
            # 当前项目无真实端侧大模型推理接口，保持兼容：返回拼接提示作为模拟回答
            response = "[EDGE-ONLY SIMULATION]\n" + prompt[:1200]
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
        response = generate(prompt)
        cloud_end = time.perf_counter()
        cloud_compute_ms = (cloud_end - cloud_start) * 1000.0

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

        response, timing = self._full_rag_answer(
            question=question,
            local_docs=local_docs,
            use_cloud=True,
            runtime_metrics={"bandwidth_mbps": 50.0, "rtt_ms": 40.0, "cpu_percent": 50.0, "battery_percent": 60.0},
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
            return response, retrieved_docs

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
            return response, local_docs

        if mode == "nexus-dynamic":
            # 关键改造：严格使用专利决策引擎，不再使用问题长度启发式
            plan = self.decision_engine.decide_plan()

            local_docs_raw = []
            if plan.local_k > 0:
                local_docs_raw = retrieve(question, top_k=plan.local_k)

            local_docs = self._normalize_docs(local_docs_raw, source_mode="nexus-dynamic")

            response, timing = self._full_rag_answer(
                question=question,
                local_docs=local_docs,
                use_cloud=plan.use_cloud,
                runtime_metrics=plan.metrics,
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
            return response, local_docs

        raise ValueError(f"Unknown mode: {mode}")
