import argparse
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.local_retrieval import add_document


DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class LocalVectorStoreAdapter:
    def add_document(self, text: str, metadata: dict) -> bool:
        return add_document(text, metadata)


def download_and_save_hotpotqa(output_path: Path) -> Path:
    try:
        from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    except Exception as exc:
        raise SystemExit("缺少依赖 datasets，请先执行: pip install datasets") from exc

    # 与项目其余模块保持一致，默认走镜像端点（若用户已设置则不覆盖）。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

    print("正在下载 HotpotQA 数据集（开发集）...")

    download_config = DownloadConfig(
        resume_download=True,
        max_retries=20,
    )

    # Hugging Face 上常见命名是 hotpot_qa；保留回退候选，避免源差异导致报错。
    candidates = [
        ("hotpot_qa", "distractor"),
        ("hotpotqa", "distractor"),
        ("hotpot_qa", None),
    ]

    dataset = None
    last_error = None
    for name, config in candidates:
        try:
            if config:
                dataset = load_dataset(
                    name,
                    config,
                    split="validation",
                    download_config=download_config,
                )
            else:
                dataset = load_dataset(
                    name,
                    split="validation",
                    download_config=download_config,
                )
            print(f"数据集加载成功: name={name}, config={config}")
            break
        except Exception as exc:
            last_error = exc
            print(f"候选数据集加载失败: name={name}, config={config}, error={exc}")

    if dataset is None:
        raise SystemExit(
            "HotpotQA 下载失败。请确认 datasets 版本、VPN/网络连通性，"
            "并尝试: pip install -U datasets huggingface_hub。"
            "若下载较慢请不要中断，支持断点续传。"
        ) from last_error

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)

    print(f"已保存到 {output_path}，共 {len(dataset)} 条数据。")
    return output_path


def _supporting_title_set(item: dict) -> set[str]:
    facts = item.get("supporting_facts", [])
    return {fact[0] for fact in facts if isinstance(fact, list) and fact}


def build_knowledge_base_from_hotpotqa(
    json_path: Path,
    vector_store_api: LocalVectorStoreAdapter,
    limit: int | None = None,
) -> tuple[int, int]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    added = 0
    skipped = 0

    for idx, item in enumerate(data, start=1):
        context = item.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])
        support_titles = _supporting_title_set(item)

        for title_idx, title in enumerate(titles):
            if title_idx >= len(sentences_list):
                continue

            sentences = sentences_list[title_idx]
            if not isinstance(sentences, list):
                continue

            paragraph_text = " ".join([str(x).strip() for x in sentences if str(x).strip()]).strip()
            if len(paragraph_text) < 20:
                skipped += 1
                continue

            metadata = {
                "source": "hotpotqa",
                "title": title,
                "original_question": item.get("question", ""),
                "supporting_fact": title in support_titles,
                "dataset_index": idx,
            }

            ok = vector_store_api.add_document(paragraph_text, metadata)
            if ok:
                added += 1
            else:
                skipped += 1

            if added and added % 1000 == 0:
                print(f"已添加 {added} 个文档...")

    print(f"知识库构建完成，共添加 {added} 个段落，跳过 {skipped} 个段落。")
    return added, skipped


def reset_local_store(store_path: Path) -> None:
    store_path.write_text("[]", encoding="utf-8")
    print(f"已重置本地存储: {store_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 HotpotQA 并导入本地向量库")
    parser.add_argument("--json", default=str(DATA_DIR / "hotpotqa_dev.json"), help="HotpotQA JSON 输出路径")
    parser.add_argument("--limit", type=int, default=None, help="仅导入前 N 条，用于调试")
    parser.add_argument("--download-only", action="store_true", help="仅下载，不导入")
    parser.add_argument("--reset-store", action="store_true", help="导入前清空 local_notes_store.json")
    args = parser.parse_args()

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        download_and_save_hotpotqa(json_path)
    else:
        print(f"检测到本地文件，跳过下载: {json_path}")

    if args.download_only:
        return

    store_path = DATA_DIR / "local_notes_store.json"
    if args.reset_store:
        reset_local_store(store_path)

    retriever = LocalVectorStoreAdapter()
    build_knowledge_base_from_hotpotqa(json_path, retriever, limit=args.limit)


if __name__ == "__main__":
    main()
