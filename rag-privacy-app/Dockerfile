# ============================================================
# Nexus RAG Privacy App - Docker 部署
# 构建: docker build -t nexus-rag .
# 运行: docker run -p 5000:5000 --env-file .env nexus-rag
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "-c", "from app.app import create_app; app = create_app(); app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)"]
