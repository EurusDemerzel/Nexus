## 🔧 修复说明与调试指南

### 修复概览

已修复的主要问题：
1. ✅ **nexus_system.py**：文档缺失 title 字段 → 补充默认标题
2. ✅ **evaluator.py**：检索精度计算时无调试信息 → 添加详细日志
3. ✅ **run_experiments.py**：异常处理不够详细 → 完整 traceback 输出
4. ✅ **test_single.py**：新增快速验证脚本

---

## 🚀 快速开始

### 方式 1：快速验证（推荐！先用这个测试单条查询）

```bash
# 进入项目目录
cd c:\Users\yan\Desktop\Nexus_clean\rag-privacy-app

# 测试单条查询（使用预定义示例）
python data/test_single.py --mode nexus-dynamic

# 测试自定义问题
python data/test_single.py --question "What is Python?" --mode nexus-dynamic

# 显示详细日志
python data/test_single.py --mode nexus-dynamic --verbose
```

**预期输出示例：**
```
--- 问题 [nexus-dynamic]: What is machine learning? ---
✓ Prompt 预览: Answer the question using the retrieved evidence when possible...
  └─ 检索文档数: 5, 第1个标题: doc_0_nexus-dynamic
← 响应获取耗时: 234.56 ms, 检索文档数: 5
✓ 检索精度: 0.2000 (命中=1/5, 金标=5, 第1个标题=doc_0_nexus-dynamic)
→ 结果: ROUGE-L=0.456789, 检索精度=0.200000
```

### 方式 2：运行完整实验（测试多条查询）

```bash
# 测试 10 条查询（快速验证）
python data/run_experiments.py --mode nexus-dynamic --limit 10 --output ./data/test_10.csv

# 完整实验（100 条）
python data/run_experiments.py --mode nexus-dynamic --limit 100 --output ./data/nexus_dynamic_100.csv

# 对比三种模式
python data/run_experiments.py --mode cloud-only --limit 50 --output ./data/cloud_only_50.csv
python data/run_experiments.py --mode static-split --limit 50 --output ./data/static_split_50.csv
```

---

## 🔍 修复详情

### 1️⃣ nexus_system.py 修复

**问题**：`retrieve()` 可能返回的文档不包含 `metadata.title`，导致评估时无法匹配金标。

**修复**：
- ✅ `_normalize_docs()` 中为缺失 `title` 的文档补充默认标题
  ```python
  if "title" not in metadata or not metadata.get("title"):
      metadata = {**metadata, "title": f"doc_{idx}_{source_mode}"}
  ```
- ✅ `_build_prompt()` 打印 prompt 前 250 字符用于验证上下文格式
  ```
  ✓ Prompt 预览: Answer the question using the retrieved evidence...
    └─ 检索文档数: 5, 第1个标题: wikipedia_article_123
  ```

### 2️⃣ evaluator.py 修复

**问题**：
- 检索精度计算时无法看到具体命中情况
- ROUGE 计算失败时无错误信息
- 无法判断是否真正调用了 RAG 逻辑

**修复**：
- ✅ `calculate_retrieval_precision()` 添加详细日志
  ```
  ✓ 检索精度: 0.4000 (命中=2/5, 金标=5, 第1个标题=wikipedia_article)
  ```
- ✅ `calculate_rouge_l()` 完整异常处理
- ✅ `evaluate_single_query()` 打印查询过程
  ```
  --- 问题 [nexus-dynamic]: What is X? ---
  ← 响应获取耗时: 245.3 ms, 检索文档数: 5
  ✓ 检索精度: 0.2000, ROUGE-L: 0.456789
  ```

### 3️⃣ run_experiments.py 修复

**问题**：
- 异常时只打印单行错误，无法判断根本原因
- 不知道哪些查询失败了

**修复**：
- ✅ 完整 traceback 输出
- ✅ 成功/失败计数统计
- ✅ 实验完成时显示汇总统计
  ```
  ============================================================
  ✅ 实验完成!
     成功: 95/100
     失败: 5/100
     结果保存至: ./data/nexus_dynamic_100.csv
  ============================================================
  ```

### 4️⃣ test_single.py（新增）

**功能**：快速验证单条查询的检索和生成。

**使用**：
```bash
python data/test_single.py --question "Your question here?" --mode nexus-dynamic
```

---

## 📋 诊断检查清单

运行 `test_single.py` 后，根据输出检查问题：

| 指标 | 预期 | 诊断 |
|------|------|------|
| 检索文档数 > 0 | ✓ | 如果为 0 → 检查 `retrieve()` 函数和知识库 |
| 第1个标题 ≠ "unknown" | ✓ | 如果是 → 文档缺少 title，已自动补充 |
| 检索精度 > 0 | ✓ | 如果为 0 → 标题不匹配，检查 `supporting_facts` 格式 |
| ROUGE-L > 0.1 | ✓ | 如果 < 0.1 → LLM 回答质量差，检查 prompt |
| latency_ms > 0 | ✓ | 如果 < 0 → 异常发生 |

---

## 🐛 常见问题

### Q1: 检索精度仍为 0？
**A**: 
1. 运行 `test_single.py` 查看第1个文档的标题
2. 与 HotpotQA 的 `supporting_facts` 比对（格式为 `[[title, sent_id], ...]`）
3. 如果标题是 `doc_0_nexus-dynamic` 形式，说明原始知识库没有 title
   - 检查 `data/import_hotpotqa.py` 中的 `build_knowledge_base()` 是否保存了 title

### Q2: ROUGE-L 总是很低？
**A**:
1. 检查 prompt 预览是否包含检索的文档内容
2. 检查 LLM（`app/services/llm_client.py` 的 `generate()` 函数）是否正确接收 prompt
3. 如果启用了隐私层，检查是否过度掩盖了文本

### Q3: 实验快速崩溃？
**A**:
运行 `test_single.py --verbose` 查看详细堆栈，然后：
- 检查 HotpotQA 数据文件是否存在 (`./data/hotpotqa_dev.json`)
- 检查 `retrieve()` 函数是否正确初始化知识库

---

## 📊 预期结果

修复后，CSV 输出应该类似：

```
question,mode,latency_ms,cpu_time_sec,mem_bytes,rouge_l,retrieval_precision
What is machine learning?,nexus-dynamic,245.3004,-1,-1,0.456789,0.200000
```

**说明**：
- `latency_ms`: 真实测量值（关键指标）
- `cpu_time_sec`, `mem_bytes`: -1（标记为未测量，避免测量开销）
- `rouge_l`: 0-1 分数（越高越好）
- `retrieval_precision`: 0-1 精度（命中标题数 / 检索文档数）

---

## 🎯 后续优化建议

1. **if 检索精度 = 0**:
   - 检查进口 HotpotQA 时标题是否正确保存
   - 考虑模糊匹配（当前为精确匹配）

2. **if rouge_l 仍然很低**:
   - 调查 LLM 后端（llama.cpp）是否在线
   - 考虑使用不同的评估指标（BLEU, METEOR）

3. **if 延迟异常高**:
   - 检查网络模拟参数 (`bandwidth_mbps`, `rtt_ms`)
   - 考虑测试云端 LLM 的响应时间

---

修复完成！🎉 请先运行 `test_single.py` 快速验证。
