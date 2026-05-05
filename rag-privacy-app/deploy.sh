#!/bin/bash
# ============================================================
# Nexus RAG Privacy App - 服务器部署脚本
# 用法: bash deploy.sh [--install-deps]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[Nexus Deploy] Project dir: $SCRIPT_DIR"

# --- Python 环境检测 ---
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "[Nexus Deploy] ERROR: Python not found. Please install Python 3.9+"
    exit 1
fi

echo "[Nexus Deploy] Python: $($PYTHON --version)"

# --- 虚拟环境 ---
VENV_DIR="$SCRIPT_DIR/../.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[Nexus Deploy] Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"  # Windows fallback
fi
echo "[Nexus Deploy] VENV Python: $VENV_PYTHON"

# --- 安装依赖 ---
if [ "$1" = "--install-deps" ]; then
    echo "[Nexus Deploy] Installing dependencies..."
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r requirements.txt
fi

# --- 环境变量 ---
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "[Nexus Deploy] Loading .env file..."
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
else
    echo "[Nexus Deploy] WARNING: .env file not found, using .env.example defaults"
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        set -a
        source "$SCRIPT_DIR/.env.example"
        set +a
    fi
fi

# --- 端口检查 ---
PORT="${FLASK_PORT:-5000}"
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[Nexus Deploy] Port $PORT occupied, killing existing process..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# --- 启动服务 ---
export PYTHONPATH="$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

echo "[Nexus Deploy] Starting server on http://0.0.0.0:$PORT"
"$VENV_PYTHON" -c "
from app.app import create_app
app = create_app()
app.run(host='0.0.0.0', port=$PORT, debug=False, use_reloader=False)
"
