#!/usr/bin/env python3
"""
快速验证脚本：测试单条查询的检索和生成是否正常

使用方式:
  python data/test_single.py --question "What is machine learning?" --mode nexus-dynamic
"""

import argparse
import json
from pathlib import Path

from nexus_system import NexusSystem
from evaluator import evaluate_single_query

# 示例 HotpotQA 问题（可选）
SAMPLE_QUESTIONS = {
    "Metallica_1": {
        "question": "What is the main discipline of James Hetfield?",
        "answer": "James Hetfield is best known as the main discipline of heavy metal music.",
        "supporting_facts": [["James Hetfield", 0], ["Metallica", 1]],
    },
    "Demo_1": {
        "question": "What color is the sky?",
        "answer": "The sky is typically blue.",
        "supporting_facts": [["Sky", 0]],
    },
}


def get_sample_question(key: str) -> dict:
    """获取预定义的示例问题"""
    if key in SAMPLE_QUESTIONS:
        return SAMPLE_QUESTIONS[key]
    else:
        available = ", ".join(SAMPLE_QUESTIONS.keys())
        raise ValueError(f"示例 {key} 不存在。可用的示例: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description="单个查询快速验证脚本")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="输入问题（如果不指定，使用 --sample）",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default="Demo_1",
        help="使用预定义的示例问题（如果 --question 未指定）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cloud-only", "static-split", "nexus-dynamic"],
        default="nexus-dynamic",
        help="运行模式",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细输出",
    )
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"🔍 Nexus 单查询验证脚本")
    print(f"{'='*70}\n")

    # 确定问题来源
    if args.question:
        question = args.question
        gold_answer = "N/A (自定义问题，无金标回答)"
        gold_facts = []
        print(f"📝 自定义问题: {question}\n")
    else:
        try:
            sample = get_sample_question(args.sample)
            question = sample["question"]
            gold_answer = sample["answer"]
            gold_facts = sample["supporting_facts"]
            print(f"📝 示例问题 [{args.sample}]:")
            print(f"   问题: {question}")
            print(f"   金标答案: {gold_answer}")
            print(f"   支持事实: {gold_facts}\n")
        except ValueError as e:
            print(f"❌ 错误: {e}")
            return

    # 初始化 NexusSystem
    print(f"初始化 NexusSystem (模式={args.mode})...")
    nexus = NexusSystem()
    print(f"✓ NexusSystem 已初始化\n")

    # 运行查询
    print(f"{'='*70}")
    print(f"📡 执行查询...")
    print(f"{'='*70}\n")

    try:
        result = evaluate_single_query(
            question=question,
            gold_answer=gold_answer,
            gold_supporting_facts=gold_facts,
            mode=args.mode,
            nexus_system=nexus,
        )

        print(f"\n{'='*70}")
        print(f"📊 评估结果:")
        print(f"{'='*70}")
        for key, value in result.items():
            if key != "question":
                status = "✓" if value >= 0 else "⚠️ "
                print(f"  {status} {key:25s}: {value}")
        print(f"{'='*70}\n")

        # 添加诊断建议
        print(f"🔧 诊断建议:")
        if result["retrieval_precision"] == 0.0 and gold_facts:
            print(f"  ⚠️  检索精度为 0，可能原因:")
            print(f"    1. 检索的文档标题与金标支持事实不匹配")
            print(f"    2. 知识库未包含相关文档")
            print(f"    3. 文档元数据格式有问题")
        
        if result["rouge_l"] < 0.1 and gold_answer != "N/A (自定义问题，无金标回答)":
            print(f"  ⚠️  ROUGE-L 分数很低，可能原因:")
            print(f"    1. LLM 没有接收到充足的上下文")
            print(f"    2. LLM 生成的回答与金标答案差异很大")
            print(f"    3. 检索结果被隐私层过度掩盖")

        if result["latency_ms"] < 0:
            print(f"  ⚠️  无法测量延迟，可能发生异常")

    except Exception as exc:
        import traceback
        print(f"❌ 查询执行失败!")
        print(f"   错误: {exc}")
        print(f"   堆栈:\n{traceback.format_exc()}")
        return

    print(f"\n✅ 验证完成!")


if __name__ == "__main__":
    main()
