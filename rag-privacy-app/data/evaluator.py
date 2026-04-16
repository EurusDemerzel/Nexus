import time
from typing import Any

import psutil


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


def calculate_retrieval_precision(
    retrieved_docs: list[Any],
    gold_supporting_facts: list[list[Any]],
    gold_answer: str = "",
    question: str = "",
) -> float:
    """
    计算检索精度。支持两种模式：
    1. 标题匹配模式（当 gold_supporting_facts 不为空）
       精度 = (命中的金标题数) / len(retrieved_docs)
    2. 答案匹配模式（当 gold_supporting_facts 为空）
       精度 = (包含答案的文档数) / len(retrieved_docs)
    
    Args:
        retrieved_docs: 检索结果列表，每个元素为字典或对象，包含 metadata.title 和 text
        gold_supporting_facts: 金标支持事实，格式为 [[title, sent_id], ...]
        gold_answer: 金标回答，用于答案匹配模式
        question: 调试用的问题文本
    
    Returns:
        检索精度 (0.0 - 1.0)
    """
    # 调试输出：检查输入
    if not retrieved_docs:
        print(f"⚠️  检索精度计算: 无检索文档 (问题={question[:40]}...)")
        return 0.0

    # 提取金标题集合
    gold_titles = {
        fact[0]
        for fact in gold_supporting_facts
        if isinstance(fact, list) and len(fact) >= 1 and isinstance(fact[0], str)
    }
    
    # ============ 模式1: 标题匹配模式 ============
    if gold_titles:
        print(f"📌 检索精度计算模式: 标题匹配 (金标事实数={len(gold_titles)})")
        hit_count = 0
        for doc in retrieved_docs:
            doc_title = None
            
            # 处理字典形式的文档
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                if isinstance(metadata, dict):
                    doc_title = metadata.get("title")
            # 处理对象形式的文档（RetrievedDoc）
            else:
                metadata = getattr(doc, "metadata", None)
                if metadata is None and hasattr(doc, "__dict__"):
                    # 尝试直接属性访问
                    doc_title = getattr(doc, "title", None)
                elif isinstance(metadata, dict):
                    doc_title = metadata.get("title")
                else:
                    doc_title = getattr(metadata, "title", None)
            
            # 检查是否命中
            if isinstance(doc_title, str) and doc_title in gold_titles:
                hit_count += 1
        
        precision = hit_count / len(retrieved_docs)
        
        # 调试输出
        first_title = None
        if retrieved_docs:
            doc = retrieved_docs[0]
            if isinstance(doc, dict):
                first_title = doc.get("metadata", {}).get("title", "N/A")
            else:
                first_title = getattr(getattr(doc, "metadata", None), "title", "N/A")
        
        print(f"✓ 检索精度: {precision:.4f} (命中={hit_count}/{len(retrieved_docs)}, "
              f"第1个标题={first_title})")
        
        return precision
    
    # ============ 模式2: 答案匹配模式 ============
    elif gold_answer and gold_answer.strip():
        print(f"📌 检索精度计算模式: 答案匹配 (金标答案长度={len(gold_answer)})")
        
        # 规范化答案：转小写，移除多余空格
        normalized_answer = " ".join(gold_answer.lower().split())
        
        hit_count = 0
        for idx, doc in enumerate(retrieved_docs):
            doc_text = ""
            
            # 提取文档文本
            if isinstance(doc, dict):
                doc_text = doc.get("text", "") or doc.get("content", "")
            else:
                doc_text = getattr(doc, "text", "") or ""
            
            # 规范化文档文本
            normalized_text = " ".join(doc_text.lower().split())
            
            # 检查答案是否在文档中（不区分大小写、不考虑空格差异）
            if normalized_answer in normalized_text:
                hit_count += 1
                print(f"   └─ 文档 #{idx+1} ✓ 包含答案")
            else:
                print(f"   └─ 文档 #{idx+1} ✗ 不包含答案")
        
        precision = hit_count / len(retrieved_docs)
        
        print(f"✓ 检索精度: {precision:.4f} (命中={hit_count}/{len(retrieved_docs)}, "
              f"基于答案字符串匹配)")
        
        return precision
    
    # ============ 回退: 无法计算精度 ============
    else:
        print(f"⚠️  检索精度计算: 无金标支持事实且无金标答案 "
              f"(问题={question[:40]}..., 检索文档数={len(retrieved_docs)})")
        return 0.0


def calculate_rouge_l(response: str, gold_answer: str) -> float:
    """
    计算 ROUGE-L F-measure
    
    Args:
        response: 模型生成的回答
        gold_answer: 金标回答
    
    Returns:
        ROUGE-L F-measure (0.0 - 1.0)
    """
    try:
        # 处理空值
        response_text = (response or "").strip()
        gold_text = (gold_answer or "").strip()
        
        if not response_text or not gold_text:
            return 0.0
        
        score_dict = get_scorer().score(gold_text, response_text)
        return score_dict["rougeL"].fmeasure
    except Exception as exc:
        print(f"ROUGE 计算异常: {exc}")
        return 0.0


def evaluate_single_query(
    question: str,
    gold_answer: str,
    gold_supporting_facts: list[list[Any]],
    mode: str,
    nexus_system: Any,
) -> dict:
    """
    评估单个问题
    
    Args:
        question: 输入问题
        gold_answer: 金标回答
        gold_supporting_facts: 金标支持事实
        mode: 运行模式 (cloud-only/static-split/nexus-dynamic)
        nexus_system: NexusSystem 实例
    
    Returns:
        包含各项指标的字典，所有异常均标记为 -1
    """
    try:
        print(f"\n--- 问题 [{mode}]: {question[:60]}... ---")
        
        # 记录起始时间
        start_time = time.perf_counter()
        
        # 调用 RAG 系统
        response, retrieved_docs = nexus_system.ask(question, mode)
        
        # 记录结束时间
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        print(f"← 响应获取耗时: {latency_ms:.2f} ms, 检索文档数: {len(retrieved_docs) if retrieved_docs else 0}")
        
        # 计算 ROUGE-L
        rouge_l = calculate_rouge_l(response, gold_answer)
        
        # 计算检索精度
        retrieval_precision = calculate_retrieval_precision(
            retrieved_docs, 
            gold_supporting_facts,
            gold_answer=gold_answer,
            question=question
        )
        
        result = {
            "question": question,
            "mode": mode,
            "latency_ms": round(latency_ms, 4),
            "cpu_time_sec": -1,  # 标记为未测量（避免测量开销影响主逻辑）
            "mem_bytes": -1,     # 标记为未测量
            "rouge_l": round(rouge_l, 6),
            "retrieval_precision": round(retrieval_precision, 6),
        }
        print(f"→ 结果: ROUGE-L={result['rouge_l']}, 检索精度={result['retrieval_precision']}")
        return result
    
    except Exception as exc:
        import traceback
        print(f"❌ 评估异常 (问题: {question[:50]}...)")
        print(f"   错误: {exc}")
        print(f"   堆栈:\n{traceback.format_exc()}")
        return {
            "question": question,
            "mode": mode,
            "latency_ms": -1,
            "cpu_time_sec": -1,
            "mem_bytes": -1,
            "rouge_l": -1,
            "retrieval_precision": -1,
        }
