#!/usr/bin/env python3
"""
download_assets.py
用途：在本地 Windows 电脑上下载 LLM 和 Embedding 模型，打包后传输到服务器。
下载目标：
  - Qwen/Qwen1.5-7B-Chat-GPTQ (4-bit 量化, ~5GB 显存)
  - sentence-transformers/all-MiniLM-L6-v2 (embedding, ~90MB)
输出目录：./models_for_server/
"""

import os
from huggingface_hub import snapshot_download

# ======================= 配置 =======================
LLM_REPO = "Qwen/Qwen1.5-7B-Chat-GPTQ"
LLM_REVISION = "gptq-4bit-128g-actorder_True"

EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = "./models_for_server"

# 国内下载加速（取消注释即可启用）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ===================================================


def download(repo_id: str, subdir: str, revision: str | None = None):
    save_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading {repo_id}")
    if revision:
        print(f"  revision: {revision}")
    print(f"  -> {save_dir}")
    print(f"{'='*60}")

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"  Done: {repo_id}\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 下载 Qwen LLM (GPTQ 量化版)
    download(LLM_REPO, "Qwen1.5-7B-Chat-GPTQ", revision=LLM_REVISION)

    # 2. 下载 Embedding 模型
    download(EMBEDDING_REPO, "all-MiniLM-L6-v2")

    total = sum(
        f.stat().st_size
        for root, dirs, files in os.walk(OUTPUT_DIR)
        for f in (os.path.join(root, f) and [None] or []) if False
    )
    # count actual files
    file_count = sum(len(files) for _, _, files in os.walk(OUTPUT_DIR))
    print(f"\nAll done! {file_count} files in {OUTPUT_DIR}")
    print("Run the following to pack and upload:")
    print(f"  tar -czf server_models.tar.gz {OUTPUT_DIR}")
    print("  scp server_models.tar.gz vision@<server_ip>:~/Nexus/")
