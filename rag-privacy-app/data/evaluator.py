import re
import string
import time
from collections import Counter
from typing import Any

try:
    import psutil as _psutil  # type: ignore[import-untyped]
    _PROC = _psutil.Process()
except Exception:
    _psutil = None
    _PROC = None

# NLTK BLEU（可选依赖）
_nltk_available = True
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _bleu_smoother = SmoothingFunction().method1
except ImportError:
    _nltk_available = False
    sentence_bleu = None  # type: ignore[assignment]


_scorer = None


def get_scorer():
    """Lazy-load and cache the ROUGE scorer."""
    global _scorer
    if _scorer is None:
        try:
            from rouge_score import rouge_scorer  # type: ignore[reportMissingImports]
        except Exception as exc:
            raise SystemExit("缺少依赖 rouge-score，请先执行: pip install rouge-score") from exc
        _scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _scorer


# NEW: 标准化答案（小写、去标点、合并空格）
def normalize_answer(s: str) -> str:
    text = (s or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def as_gold_answers(gold: Any) -> list[str]:
    if isinstance(gold, str):
        value = gold.strip()
        return [value] if value else []

    if isinstance(gold, (list, tuple)):
        values: list[str] = []
        for item in gold:
            text = str(item or "").strip()
            if text:
                values.append(text)
        return values

    text = str(gold or "").strip()
    return [text] if text else []


# NEW: Exact Match
def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


# NEW: Token-level F1
def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# NEW: 判断 gold 是否为 yes/no 或纯数字
def is_yes_no_or_number(gold_answer: Any) -> bool:
    for ans in as_gold_answers(gold_answer):
        gold_norm = normalize_answer(ans)
        if gold_norm in {"yes", "no"}:
            return True
        if bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", gold_norm)):
            return True
    return False


def calculate_retrieval_precision(
    retrieved_docs: list[Any],
    gold_supporting_facts: list[list[Any]],
    gold_answer: Any = "",
    question: str = "",
) -> float:
    """
    计算检索精度。支持两种模式：
    1. 标题匹配模式（当 gold_supporting_facts 不为空）
    2. 答案匹配模式（当 gold_supporting_facts 为空）
    """
    if not retrieved_docs:
        print(f"⚠️  检索精度计算: 无检索文档 (问题={question[:40]}...)")
        return 0.0

    gold_titles = {
        fact[0]
        for fact in gold_supporting_facts
        if isinstance(fact, list) and len(fact) >= 1 and isinstance(fact[0], str)
    }

    if gold_titles:
        hit_count = 0
        for doc in retrieved_docs:
            doc_title = None
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    doc_title = metadata.get("title")
            else:
                metadata = getattr(doc, "metadata", None)
                if isinstance(metadata, dict):
                    doc_title = metadata.get("title")
                elif metadata is not None:
                    doc_title = getattr(metadata, "title", None)

            if isinstance(doc_title, str) and doc_title in gold_titles:
                hit_count += 1

        return hit_count / len(retrieved_docs)

    gold_answers = as_gold_answers(gold_answer)
    if gold_answers:
        normalized_answers = [normalize_answer(x) for x in gold_answers if normalize_answer(x)]
        hit_count = 0
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                doc_text = doc.get("text", "") or doc.get("content", "")
            else:
                doc_text = getattr(doc, "text", "") or ""
            normalized_text = normalize_answer(doc_text)
            if any(ans in normalized_text for ans in normalized_answers):
                hit_count += 1
        return hit_count / len(retrieved_docs)

    print(f"⚠️  检索精度计算: 无金标支持事实且无金标答案 (问题={question[:40]}...)")
    return 0.0


def calculate_rouge_l(response: str, gold_answer: Any) -> float:
    try:
        response_text = (response or "").strip()
        gold_candidates = as_gold_answers(gold_answer)
        if not response_text or not gold_candidates:
            return 0.0

        best = 0.0
        scorer = get_scorer()
        for gold_text in gold_candidates:
            score_dict = scorer.score(gold_text, response_text)
            best = max(best, score_dict["rougeL"].fmeasure)
        return best
    except Exception as exc:
        print(f"ROUGE 计算异常: {exc}")
        return 0.0


# ── BLEU 计算 ──
def compute_bleu(reference: str, hypothesis: str, ngram: int = 4) -> float:
    """
    基于 nltk.translate.bleu_score 计算 BLEU-1 ~ BLEU-4。
    返回 float，如果 nltk 不可用则返回 -1。
    """
    if not _nltk_available:
        return -1.0

    try:
        ref_tokens = normalize_answer(reference).split()
        hyp_tokens = normalize_answer(hypothesis).split()
        if not hyp_tokens or not ref_tokens:
            return 0.0

        n = max(1, min(4, int(ngram)))
        weights = [1.0 / n] * n
        return sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=tuple(weights),
            smoothing_function=_bleu_smoother,
        )
    except Exception as exc:
        print(f"BLEU 计算异常: {exc}")
        return 0.0


def _max_bleu(refs: list[str], hyp: str, ngram: int) -> float:
    """多参考答案下取最高 BLEU。"""
    if not _nltk_available:
        return -1.0
    return max(compute_bleu(ref, hyp, ngram=ngram) for ref in refs)


def evaluate_single_query(
    question: str,
    gold_answer: Any,
    gold_supporting_facts: list[list[Any]],
    mode: str,
    nexus_system: Any,
) -> dict:
    """
    评估单个问题。
    NEW: 增加 exact_match / token_f1；yes/no 或纯数字题优先参考 exact_match。
    """
    try:
        print(f"\n--- 问题 [{mode}]: {question[:60]}... ---")

        start_time = time.perf_counter()
        cpu_before = _PROC.cpu_times().user if _PROC else 0.0
        mem_before = _PROC.memory_info().rss if _PROC else 0

        response, retrieved_docs, debug_meta = nexus_system.ask(question, mode)

        end_time = time.perf_counter()
        cpu_after = _PROC.cpu_times().user if _PROC else 0.0
        mem_after = _PROC.memory_info().rss if _PROC else 0

        latency_ms = (end_time - start_time) * 1000.0
        cpu_time_sec = round(cpu_after - cpu_before, 4)
        mem_bytes = max(0, mem_after - mem_before)

        model_answer = (response or "").strip()
        gold_candidates = as_gold_answers(gold_answer)
        if not gold_candidates:
            gold_candidates = [""]

        print(f"← 响应获取耗时: {latency_ms:.2f} ms, 检索文档数: {len(retrieved_docs) if retrieved_docs else 0}")

        rouge_l = calculate_rouge_l(model_answer, gold_candidates)
        em = max(exact_match(model_answer, ans) for ans in gold_candidates)
        f1 = max(token_f1(model_answer, ans) for ans in gold_candidates)
        bleu1 = _max_bleu(gold_candidates, model_answer, ngram=1)
        bleu4 = _max_bleu(gold_candidates, model_answer, ngram=4)
        best_answer_for_log = max(gold_candidates, key=lambda ans: token_f1(model_answer, ans))

        # NEW: 依据题型提示优先指标
        if is_yes_no_or_number(gold_candidates):
            print(f"📌 题型=Yes/No或数字，优先指标=ExactMatch ({em:.4f})")
        else:
            print(f"📌 题型=开放问答，优先指标=TokenF1 ({f1:.4f})")

        retrieval_precision = calculate_retrieval_precision(
            retrieved_docs,
            gold_supporting_facts,
            gold_answer=gold_candidates,
            question=question,
        )

        result = {
            # MOD: 输出中增加 gold_answer / model_answer，便于后续人工评估抽样
            "question": question,
            "gold_answer": best_answer_for_log,
            "model_answer": model_answer,
            "mode": mode,
            "latency_ms": round(latency_ms, 4),
            "cpu_time_sec": cpu_time_sec if _PROC else -1,
            "mem_bytes": mem_bytes if _PROC else -1,
            "rouge_l": round(rouge_l, 6),
            # NEW: 新增指标
            "exact_match": round(em, 6),
            "token_f1": round(f1, 6),
            "bleu_1": round(bleu1, 6) if bleu1 >= 0 else -1,
            "bleu_4": round(bleu4, 6) if bleu4 >= 0 else -1,
            "retrieval_precision": round(retrieval_precision, 6),
            # ── 隐私开销 ──
            "privacy_mode": getattr(getattr(nexus_system, "privacy_layer", None), "mode", "unknown"),
            "privacy_overhead_ms": round(getattr(getattr(nexus_system, "privacy_layer", None), "last_overhead", type("x", (), {"total_ms": -1})()).total_ms, 4),
            # ── 动态决策调试字段 ──
            "local_k": debug_meta.get("local_k", -1),
            "use_cloud": int(debug_meta.get("use_cloud", False)),
            "load_score": debug_meta.get("load_score", -1.0),
        }
        print(
            f"→ 结果: ROUGE-L={result['rouge_l']}, EM={result['exact_match']}, "
            f"Token-F1={result['token_f1']}, 检索精度={result['retrieval_precision']}"
        )
        return result

    except Exception as exc:
        import traceback

        print(f"❌ 评估异常 (问题: {question[:50]}...)")
        print(f"   错误: {exc}")
        print(f"   堆栈:\n{traceback.format_exc()}")
        gold_candidates = as_gold_answers(gold_answer)
        logged_gold = gold_candidates[0] if gold_candidates else ""
        return {
            "question": question,
            "gold_answer": logged_gold,
            "model_answer": f"ERROR: {exc}",
            "mode": mode,
            "latency_ms": -1,
            "cpu_time_sec": -1,
            "mem_bytes": -1,
            "rouge_l": -1,
            "exact_match": -1,
            "token_f1": -1,
            "retrieval_precision": -1,
        }
