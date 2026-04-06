# NEXUS App

## 项目简介

RAG Privacy App 是一个基于 Flask 的智能问答系统，旨在通过检索增强生成（RAG）技术提供高效的问答服务。该应用从 MySQL 数据库中检索相关内容，并结合大语言模型（LLM）生成回答。项目特别关注隐私保护，计划实现多层隐私保护功能。

## 功能

- 用户通过网页输入问题。
- 系统从 MySQL 数据库中检索相关内容。
- 返回基于检索的模拟回答。
- 未来将接入真实大模型 API。
- 实现隐私保护功能，如输入脱敏、加密传输和差分隐私。

## 项目结构

```
rag-privacy-app
├── app
│   ├── __init__.py          # 初始化 Flask 应用
│   ├── app.py               # 应用主入口
│   ├── models.py            # 数据库模型定义
│   ├── routes.py            # 路由定义
│   └── services
│       ├── __init__.py      # 服务模块初始化
│       ├── llm_service.py   # LLM 服务逻辑
│       └── retrieval_service.py # 数据检索服务逻辑
├── data
│   └── populate_db.py       # 数据库填充脚本
├── templates
│   └── index.html           # 前端模板
├── config.py                # 配置设置
├── requirements.txt         # 项目依赖
└── README.md                # 项目文档
```

## 安装步骤

1. 克隆项目：

   ```
   git clone <repository-url>
   cd rag-privacy-app
   ```
2. 创建虚拟环境并激活：

   ```
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate
   ```
3. 安装依赖：

   ```
   pip install -r requirements.txt
   ```
4. 配置数据库连接信息：

   - 编辑 `config.py` 文件，设置 MySQL 数据库连接信息。
5. 填充数据库：

   ```
   python data/populate_db.py
   ```
6. 启动应用（推荐仅使用这一种方式）：

   ```
   start_nexus.bat
   ```

   首次需要安装依赖时可使用：

   ```
   start_nexus.bat --install-deps
   ```

## 使用说明

- 打开浏览器，访问 `http://localhost:5000`。
- `GET /ask` 会返回接口使用说明（不再报 405）。
- 实际问答请使用 `POST /ask`，并发送 JSON 请求体，例如：

   ```json
   {
      "question": "我的学习计划是什么？",
      "user_id": 1,
      "debug": true
   }
   ```

## 未来工作

- 完善数据库中的中文数据。
- 实现隐私保护功能。
- 撰写相关论文，分享项目成果。
