#!/usr/bin/env python3
"""
download_bge_large.py — 从 HuggingFace 下载 BAAI/bge-large-en-v1.5 到本地
用法:
  python data/download_bge_large.py
  python data/download_bge_large.py --output ./models_for_server/bge-large-en-v1.5
"""

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT_DIR / "models_for_server" / "bge-large-en-v1.5"


def download_model(model_id: str, save_dir: Path):
    """使用 snapshot_download 下载完整模型（含 tokenizer、config、权重）。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Model: {model_id}")
    print(f"[download] Target: {save_dir}")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model.msgpack", "tf_model.h5"],
        )
    except ImportError:
        print("[download] huggingface_hub 未安装，尝试 pip install huggingface_hub ...")
        os.system(f"{_get_python()} -m pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model.msgpack", "tf_model.h5"],
        )

    print(f"[download] Done. Files saved to {save_dir}")

    # 验证关键文件
    required = ["config.json", "tokenizer.json", "model.safetensors"]
    missing = [f for f in required if not (save_dir / f).exists()]
    # bge-large-en-v1.5 可能用 pytorch_model.bin 而非 safetensors
    alt_check = (save_dir / "pytorch_model.bin").exists() or (save_dir / "model.safetensors").exists()
    if missing and "tokenizer.json" in missing:
        # sentence_transformers 也可能用 tokenizer_config.json
        pass
    if not alt_check:
        print("[download][WARN] 未找到模型权重文件 (model.safetensors / pytorch_model.bin)，请检查下载是否完整")

    # 快速验证可用性
    print("[download] Verifying model can be loaded ...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(save_dir), local_files_only=True)
        dim = model.get_sentence_embedding_dimension()
        print(f"[download] ✓ Model loaded OK, embedding dim = {dim}")
    except Exception as e:
        print(f"[download][WARN] 模型加载验证失败（可能在离线环境正常）: {e}")


def _get_python() -> str:
    import sys
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description="Download BAAI/bge-large-en-v1.5 model")
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model-id", type=str, default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    download_model(args.model_id, Path(args.output))


if __name__ == "__main__":
    main()
