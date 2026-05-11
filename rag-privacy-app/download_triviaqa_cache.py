#!/usr/bin/env python3
"""
download_triviaqa_cache.py
在笔记本上下载 TriviaQA 数据集（unfiltered/validation）到 HF 缓存，
然后打印缓存目录路径，方便打包传输到离线服务器。
"""

import os
from pathlib import Path

# 使用国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset, DownloadConfig


def main():
    print("=" * 60)
    print("下载 TriviaQA (unfiltered/validation) ...")
    print("=" * 60)

    # 下载到缓存（不 stream，确保完整缓存）
    ds = load_dataset(
        "trivia_qa",
        "unfiltered",
        split="validation",
        streaming=False,
        download_config=DownloadConfig(max_retries=5),
    )

    # 获取缓存文件路径
    if ds.cache_files:
        cache_path = Path(ds.cache_files[0]["filename"]).parent
    else:
        # fallback: 从 HF 环境变量获取默认缓存根目录
        from datasets import config as datasets_config
        cache_path = (
            Path(datasets_config.HF_DATASETS_CACHE)
            / "trivia_qa"
            / "unfiltered"
            / "validation"
        )

    print(f"\n✅ 下载完成！")
    print(f"📁 缓存目录: {cache_path}")

    # 计算大小
    total_bytes = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
    print(f"📦 总大小: {total_bytes / 1024**3:.2f} GB")

    # 打印打包命令
    parent_dir = cache_path.parent
    # 找到 trivia_qa 的根缓存目录（含 unfiltered 和 rc）
    root_cache = parent_dir  # .../trivia_qa/
    # 打包应从 datasets 层级进行，因为 triva_qa 是子目录
    datasets_parent = root_cache.parent  # .../datasets/
    print(f"\n📋 打包命令（在笔记本终端执行）:")
    print(f"  cd {datasets_parent}")
    print(f"  tar -czf triviaqa_cache.tar.gz trivia_qa")
    print()
    print(f"📋 上传命令:")
    print(f"  scp triviaqa_cache.tar.gz vision@100.64.91.120:~/")
    print()


if __name__ == "__main__":
    main()
