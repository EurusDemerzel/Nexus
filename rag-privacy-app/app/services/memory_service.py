import json
import re
from pathlib import Path


class MemoryService:
    def __init__(self, user_id: int = 1):
        self.user_id = int(user_id)
        root_dir = Path(__file__).resolve().parents[2]
        self.data_dir = root_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.data_dir / f"memory_user_{self.user_id}.json"
        self.memory = self._load_memory()

    def _load_memory(self) -> dict:
        default = {"facts": [], "history": []}
        if not self.memory_file.exists():
            return default

        try:
            with self.memory_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return default
            payload.setdefault("facts", [])
            payload.setdefault("history", [])
            if not isinstance(payload["facts"], list):
                payload["facts"] = []
            if not isinstance(payload["history"], list):
                payload["history"] = []
            return payload
        except Exception:
            return default

    def _save_memory(self):
        with self.memory_file.open("w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def add_fact(self, fact: str):
        fact = (fact or "").strip()
        if not fact:
            return
        if fact not in self.memory["facts"]:
            self.memory["facts"].append(fact)
            self._save_memory()

    def add_history(self, question: str, answer: str):
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question and not answer:
            return
        self.memory["history"].append({"question": question, "answer": answer})
        self._save_memory()

    def retrieve_relevant_memory(self, query: str, top_k: int = 3) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []

        candidates: list[str] = []
        candidates.extend(self.memory.get("facts", []))

        for item in self.memory.get("history", []):
            if not isinstance(item, dict):
                continue
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            candidates.append(f"历史问答: Q: {q} | A: {a}")

        terms = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", query)
        if not terms:
            terms = [query]

        scored = []
        for text in candidates:
            score = 0
            for term in terms:
                if term and term in text:
                    score += len(term)
            if score > 0:
                scored.append((score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [text for _, text in scored[:max(1, top_k)]]

        if selected:
            return selected
        return candidates[-max(1, top_k):]

    def get_injected_context(self, query: str) -> str:
        memory_items = self.retrieve_relevant_memory(query=query, top_k=3)
        if not memory_items:
            return ""
        sanitized = [self._sanitize_text(item) for item in memory_items]
        return "\n".join([f"- {line}" for line in sanitized])

    def _sanitize_text(self, text: str) -> str:
        # Basic masking for common sensitive patterns.
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
        text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE]", text)
        text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "[ID]", text)
        text = re.sub(r"(?<!\d)\d{12,19}(?!\d)", "[CARD]", text)
        return text
