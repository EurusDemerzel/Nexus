#!/usr/bin/env python3
"""
download_assets.py
用途：在本地 Windows 电脑上下载 LLM 和 Embedding 模型，打包后传输到服务器。

=== 重要: 模型授权 ===
Qwen/Qwen1.5-7B-Chat-GPTQ 是受限模型，下载前需要：
  1. 登录 https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ
  2. 点击 "Agree and access repository" 接受许可
  3. 在 https://huggingface.co/settings/tokens 创建 token
  4. 设置环境变量: set HF_TOKEN=hf_你的token

如果无法获取授权，脚本会自动回退到可公开访问的替代模型。
"""

import os
import sys
from huggingface_hub import snapshot_download, HfApi

# ======================= 配置 =======================
# 首选模型（需要授权）
LLM_REPO_PRIMARY = "Qwen/Qwen1.5-7B-Chat-GPTQ"
LLM_REVISION = "gptq-4bit-128g-actorder_True"

# 备用模型（公开，无需授权）— 如果首选下载失败则使用此模型
LLM_REPO_FALLBACK = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
LLM_FALLBACK_REVISION = "main"

EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = "./models_for_server"

# 国内下载加速（取消注释即可启用）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ===================================================


def check_repo_exists(repo_id: str, revision: str | None = None) -> bool:
    """检查仓库是否可访问（检测是否需要授权）。"""
    try:
        api = HfApi()
        api.repo_info(repo_id=repo_id, revision=revision)
        return True
    except Exception as e:
        status = getattr(e, "response", None)
        code = getattr(status, "status_code", None) if status else None
        if code == 401:
            print(f"  [WARN] 仓库 {repo_id} 需要授权 (401)。请设置 HF_TOKEN。")
        else:
            print(f"  [WARN] 仓库 {repo_id} 不可用: {type(e).__name__}")
        return False


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
        resume_download=True,
    )
    print(f"  Done: {repo_id}\n")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1. 下载 LLM ----------
    LLM_REPO = LLM_REPO_PRIMARY
    LLM_VER = LLM_REVISION
    LLM_DIR = "Qwen1.5-7B-Chat-GPTQ"

    if not check_repo_exists(LLM_REPO_PRIMARY, revision=LLM_REVISION):
        print("\n  >>> 首选模型需要授权或不存在。")
        print(f"  >>> 回退到公开模型: {LLM_REPO_FALLBACK}")
        LLM_REPO = LLM_REPO_FALLBACK
        LLM_VER = LLM_FALLBACK_REVISION
        LLM_DIR = "Qwen2.5-1.5B-Instruct-GPTQ-Int4"

    download(LLM_REPO, LLM_DIR, revision=LLM_VER)

    # ---------- 2. 下载 Embedding 模型 ----------
    download(EMBEDDING_REPO, "all-MiniLM-L6-v2")

    # ---------- 统计 ----------
    file_count = sum(len(files) for _, _, files in os.walk(OUTPUT_DIR))
    total_bytes = sum(
        f.stat().st_size
        for root, _, files in os.walk(OUTPUT_DIR)
        for f_name in files
        for f in [os.path.join(root, f_name)]
    )
    print(f"\n{'='*60}")
    print(f"All done! {file_count} files, {total_bytes / 1024**3:.2f} GB")
    print(f"{'='*60}")
    print()
    print("打包并上传到服务器:")
    print(f"  tar -czf server_models.tar.gz -C {OUTPUT_DIR} .")
    print("  scp server_models.tar.gz vision@<server_ip>:~/Nexus/")
    print()
    print(f"下载的模型: {LLM_REPO}")
    print(f"如需使用首选模型，请先设置 HF_TOKEN 后重新运行。")
