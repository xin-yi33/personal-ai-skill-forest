# 实验改进计划：基线对比扩展

## 问题诊断

当前实验仅对比了 3 种方案：
1. Flat ANN (FAISS IVF+PQ)
2. 单棵 B+ 树（无领域路由）
3. 技能森林（本方案）

**审稿意见核心问题**：没有对比现有 SOTA 方法，无法证明"B+ 树比现有主流方案好"。

## 新增基线方案

### 基线 1：HNSW（Hierarchical Navigable Small World）

- **实现**：`faiss.IndexHNSWFlat(d, 32)` — FAISS 内置 HNSW
- **意义**：HNSW 是当前向量检索的 SOTA 近似算法，广泛用于 Milvus/Qdrant/Pinecone 等生产系统
- **对比维度**：纯检索准确率、延迟、端到端 Token 消耗
- **预期**：HNSW 准确率接近 Flat ANN（因为是近似搜索），延迟略高，但不需要森林结构

### 基线 2：FAISS Top-K + LLM Rerank（生产环境主流方案）

- **实现**：先用 FAISS 检索 Top-20，再用 embedding 相似度模拟 LLM 重排序，输出 Top-5
- **意义**：这是 Anthropic/OpenAI 生产环境的实际做法 — 先粗筛再精排
- **Token 模型**：粗筛 20 个描述 + LLM 重排序推理 Token（模拟）
- **对比维度**：准确率、Token 消耗、端到端性能
- **预期**：准确率高于纯 FAISS，但 Token 消耗远高于森林方案

### 基线 3：BM25 纯文本检索

- **实现**：基于 TF-IDF 的 BM25 检索（rank_bm25 库）
- **意义**：传统 IR 基线，代表"不用 embedding"的方案
- **对比维度**：准确率（预期显著低于 embedding 方案）
- **意义**：证明 embedding-based 方法的必要性

### 基线 4：Random（随机选择）

- **实现**：从全部 API 中随机选 Top-K
- **意义**：理论下限，证明任何检索策略都有价值
- **对比维度**：准确率下限

### 基线 5：Flat ANN + Cross-Encoder Rerank（学术 SOTA）

- **实现**：FAISS Top-20 → cross-encoder 精排（用 sentence-transformers 的 cross-encoder 模型）
- **意义**：代表 RAG 领域的两阶段检索范式
- **对比维度**：准确率、延迟（cross-encoder 推理慢）
- **预期**：准确率最高，但延迟和 Token 消耗也最高

### 基线 6：LLM 全量推理（Gorilla/ToolLLM 风格）

- **实现**：将所有 API 描述放入 prompt，模拟 LLM 直接选择
- **意义**：代表 ToolLLM、Gorilla 等"LLM 直接选工具"的范式
- **Token 模型**：所有 API 描述的 Token 数 + LLM 推理 Token
- **对比维度**：准确率、Token 消耗（预期极高）
- **预期**：小规模时准确率可能最高，但 Token 随 N 线性增长，大规模不可行

## 实验方案更新

### Exp1 扩展：纯检索 + 端到端对比

**新增对照组（Layer 1 纯检索）**：

| 方案 | 检索算法 | 精排 | 描述 |
|------|---------|------|------|
| Flat ANN (FAISS IVF+PQ) | FAISS | 无 | 当前基线 |
| HNSW | FAISS HNSW | 无 | 新增 |
| BM25 | TF-IDF | 无 | 新增 |
| Random | 随机 | 无 | 新增 |
| 森林（本方案） | B+ 树遍历 | 内置 | 当前方案 |

**新增对照组（Layer 2 端到端）**：

| 方案 | 检索 | 精排 | Token 模型 |
|------|------|------|-----------|
| Flat+LLM | FAISS Top-5 | 无 | LLM 推理全部 |
| FAISS+Rerank | FAISS Top-20 | LLM Rerank | 20 描述 + 重排推理 |
| HNSW+Rerank | HNSW Top-20 | LLM Rerank | 20 描述 + 重排推理 |
| LLM 全量 | 无检索 | LLM 全量 | 所有 API 描述 |
| 森林+M4/M6/M9 | B+ 树 | 结构化 | 路由+检索+确认 |

### 实现优先级

1. **P0（必须）**：HNSW、FAISS+Rerank、LLM 全量推理 — 这三个直接回应审稿意见
2. **P1（重要）**：BM25、Random — 提供理论边界
3. **P2（可选）**：Cross-Encoder Rerank — 需要额外模型，计算成本高

## 文件变更计划

### 新增文件

- `shared/baselines.py` — 所有基线方法的统一实现
  - `build_hnsw_index(embeddings)` → HNSW 索引
  - `search_hnsw(query_emb, index, all_apis, top_k)` → HNSW 检索
  - `build_bm25_index(apis)` → BM25 索引
  - `search_bm25(query, bm25_model, all_apis, top_k)` → BM25 检索
  - `search_random(all_apis, top_k)` → 随机选择
  - `search_faiss_rerank(query_emb, faiss_index, all_apis, top_k_retrieve, top_k_final)` → FAISS+Rerank
  - `search_llm_all(query_emb, all_apis)` → LLM 全量推理 Token 模型
  - `measure_e2e_tokens_faiss_rerank(...)` → FAISS+Rerank Token 模型
  - `measure_e2e_tokens_llm_all(...)` → LLM 全量 Token 模型

### 修改文件

- `Exp1_Retrieval_Performance/run_experiment.py` — 新增基线对比
- `Exp6_Token_Consumption/run_experiment_v3.py` — 新增基线的 Token 对比
- `requirements.txt` — 新增 `rank_bm25` 依赖

### 论文更新

- `paper/paper_EN.md` Section 6.2 — 新增基线对比表格和分析
- `paper/paper_CN.md` 第 6.2 节 — 同步中文版

## 预期结果（推测）

| 方案 | Acc@5 | MRR | Token (N=5000) | 延迟 |
|------|-------|-----|---------------|------|
| Random | ~0.200 | ~0.100 | - | 极低 |
| BM25 | ~0.350 | ~0.150 | - | 低 |
| Flat ANN (FAISS) | 0.536 | 0.170 | 612 | 0.1ms |
| HNSW | ~0.540 | ~0.175 | ~615 | ~0.15ms |
| FAISS+Rerank | ~0.600 | ~0.230 | ~1200 | ~2ms |
| LLM 全量 | ~0.650 | ~0.300 | ~5000+ | N/A |
| 森林（本方案） | 0.583 | 0.219 | 127 | 1.6ms |

**关键论点**：森林方案在准确率上接近 FAISS+Rerank，但 Token 消耗仅为后者的 ~10%，延迟也更低。这证明了结构化索引在 Token 效率上的独特优势。

## 时间线

1. Day 1：实现 `shared/baselines.py`（HNSW + BM25 + Random + FAISS+Rerank + LLM全量）
2. Day 2：修改 Exp1 和 Exp6 脚本，运行实验
3. Day 3：更新论文（英文+中文），新增基线对比表格和分析
4. Day 4：Commit + Push
