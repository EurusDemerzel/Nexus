# Nexus — 端云协同隐私 RAG 问答系统

## 项目简介

Nexus 是一个基于 Flask 的端云协同 RAG（Retrieval-Augmented Generation）智能问答系统，融合本地/云端混合检索、动态切分决策、多层隐私保护，支持专利和课程两大知识域的混合问答。项目面向 EMNLP 论文实验设计。

## 项目结构

```
rag-privacy-app/
├── app/                          # Flask 应用核心
│   ├── app.py                    # 应用工厂入口
│   ├── models.py                 # SQLAlchemy 数据模型
│   ├── routes.py                 # API 路由 (/ask, /search 等)
│   └── services/                 # 服务模块 (16个)
│       ├── embedding_service.py  # 文本向量化 (MiniLM)
│       ├── hybrid_retriever.py   # 混合检索 (FAISS + BM25)
│       ├── llm_client.py         # LLM 请求客户端
│       ├── llm_service.py        # LLM 生成服务
│       ├── local_retrieval.py    # 本地检索统一入口
│       ├── memory_service.py     # 用户记忆管理
│       ├── mock_cloud.py         # 云端模拟服务
│       ├── policy_service.py     # 策略检索配置
│       ├── predictor_service.py  # 性能预测
│       ├── privacy_layer.py      # 隐私加密层
│       ├── privacy_service.py    # 隐私保护服务
│       ├── retrieval_service.py  # MySQL 检索服务
│       ├── risk_service.py       # 风险评估
│       ├── secure_channel.py     # 安全通信信道
│       ├── split_decision.py     # 动态切分决策引擎
│       └── weighting_service.py  # 权重管理
├── data/                         # 数据与实验脚本
│   ├── triviaqa_dev.json         # TriviaQA 验证集 (已导出)
│   ├── hotpotqa_dev.json         # HotpotQA 数据集
│   ├── course_knowledge.sql      # 课程知识 SQL
│   ├── evaluator.py              # 评估核心 (Token-F1, ROUGE, EM)
│   ├── nexus_system.py           # 实验系统封装
│   ├── run_experiments.py        # 批量实验主入口
│   ├── eval_retrieval.py         # 检索 Recall@k 评估
│   ├── build_triviaqa_kb.py      # TriviaQA 知识库构建
│   ├── build_triviaqa_kb_advanced.py  # 高级分块知识库构建
│   ├── load_triviaqa.py          # TriviaQA 数据加载
│   ├── populate_db.py            # 专利数据库填充
│   ├── import_course_sql.py      # 课程数据库导入
│   └── sample_human_eval.py      # 人工评估抽样
├── templates/                    # 前端模板
│   └── index.html
├── tests/                        # 单元测试
│   ├── test_pipeline_services.py
│   └── test_split_decision_engine.py
├── paper/                        # 论文相关 (LaTeX, 图表)
│   ├── method.tex
│   └── figures/
├── split_cnn_demo/               # SplitCNN 原型
│   ├── cloud_server.py
│   ├── common_model.py
│   └── end_client.py
├── triviaqa_kb/                  # TriviaQA FAISS 知识库 (需构建)
├── config.py                     # 应用配置
├── requirements.txt              # Python 依赖
├── .env                          # 环境变量 (不提交git)
├── .env.example                  # 环境变量模板
├── .gitignore                    # Git 忽略规则
├── .dockerignore                 # Docker 忽略规则
├── Dockerfile                    # Docker 部署文件
├── deploy.sh                     # Linux 服务器部署脚本
├── start_nexus.bat               # Windows 启动脚本
├── start_nexus.ps1               # PowerShell 启动 (已弃用)
└── README.md                     # 本文档
```

## 环境要求

- Python 3.9+
- MySQL 5.7+ (或 MariaDB 10.x)
- 推荐 8GB+ RAM (运行嵌入模型)

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd rag-privacy-app
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入数据库和 LLM 配置
```

### 3. 安装依赖

```bash
python -m venv ../.venv
source ../.venv/bin/activate  # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. 初始化数据库

```bash
python data/populate_db.py          # 创建专利数据表
python data/import_course_sql.py    # 导入课程知识
```

### 5. 启动服务

**Windows:**
```batch
start_nexus.bat
# 或首次安装依赖: start_nexus.bat --install-deps
```

**Linux/服务器:**
```bash
bash deploy.sh
# 或首次安装依赖: bash deploy.sh --install-deps
```

**Docker:**
```bash
docker build -t nexus-rag .
docker run -p 5000:5000 --env-file .env nexus-rag
```

### 6. 访问

- 打开浏览器 `http://localhost:5000`
- 问答 API: `POST /ask` `{"question": "...", "user_id": 1}`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/ask` | 接口说明 |
| `POST` | `/ask` | 问答请求 (JSON) |
| `GET` | `/health` | 健康检查 |

**请求示例:**
```json
POST /ask
{
    "question": "What is the capital of France?",
    "user_id": 1,
    "source": "auto",
    "debug": true,
    "force_split_point": null
}
```

**响应示例:**
```json
{
    "answer": "The capital of France is Paris.",
    "source": "patents",
    "retrieved_context": [...],
    "debug": {...}
}
```

## 检索后端切换

通过环境变量 `RETRIEVAL_BACKEND` 或 `NEXUS_RETRIEVER` 切换：

| 值 | 说明 |
|------|------|
| `notes` | 本地笔记检索 (默认) |
| `faiss` | TriviaQA FAISS 向量检索 |
| `hybrid` | FAISS + BM25 混合检索 |

## 实验运行

```bash
# 构建 TriviaQA 知识库 (直连HF主站)
python data/build_triviaqa_kb_advanced.py --max-docs 10000 --chunk-size 512 --overlap 64

# 运行批量实验
python data/run_experiments.py --mode static-split --limit 50 --output results_static_50.csv

# 评估检索召回
python data/eval_retrieval.py --limit 50
```

## 测试

```bash
pytest -q
# 或单独运行
pytest tests/test_split_decision_engine.py -vv
```

## 论文相关

- `paper/method.tex` — 方法描述 LaTeX
- `paper/figures/` — 论文图表输出
- `data/plot_paper_figures.py` — 论文图表生成脚本

## 许可

项目用于学术研究目的。
