# 实验改进计划：基线对比扩展

## 问题诊断

审稿意见：基线对比过于单一，缺乏说服力。只对比了 FAISS、单棵 B+ 树、技能森林，缺少 HNSW、BM25、FAISS+Rerank 等 SOTA/生产主流基线。

## 解决方案：合并到现有实验

**不再新建 Exp7**，而是将新基线直接合并到现有实验中：

### Exp1（检索性能对比）— 已修改

**Layer 1 纯检索**：新增 HNSW、BM25、FAISS+Rerank 三个基线
- FAISS (IVF+PQ) — 原有
- HNSW — 新增（SOTA ANN 算法）
- BM25 — 新增（传统 IR 基线）
- FAISS + Embedding Rerank — 新增（生产主流）
- Skill Forest — 原有

**Layer 2 端到端 Token**：新增 HNSW+LLM、FAISS+Rerank+LLM
- FAISS + LLM — 原有
- HNSW + LLM — 新增
- FAISS + Rerank + LLM — 新增（生产主流）
- Forest + M4/M6/M9 — 原有

### Exp6（Token 消耗规模实验）— 保持不变

Exp6 的规模实验保持原有 FAISS vs Forest 对比，因为：
- 规模实验的核心论点是"森林 Token 不随 N 增长"
- 新增基线（HNSW、FAISS+Rerank）的 Token 模型与 FAISS 类似，不改变结论
- 在 Exp1 中已有完整的端到端对比

### 文件变更

| 文件 | 变更 |
|------|------|
| `shared/baselines.py` | 新增：统一基线实现模块 |
| `Exp1/run_experiment.py` | 修改：Layer 1/2 新增 3 个基线 |
| `Exp7_Baseline_Comparison/` | 已删除：避免重复 |
| `EXPERIMENT_PLAN.md` | 本文件 |

## 新增基线详情

| 基线 | 实现 | 意义 |
|------|------|------|
| HNSW | `faiss.IndexHNSWFlat` | 当前 ANN SOTA（Milvus/Qdrant/Pinecone 底层） |
| BM25 | `rank_bm25` 库 | 传统 IR 基线，证明 embedding 的必要性 |
| FAISS+Rerank | FAISS Top-20 → embedding 精排 | **生产环境主流**（Anthropic/OpenAI 实际做法） |

## 预期结果

| 方案 | Acc@5 | MRR | Token (E2E) |
|------|-------|-----|-------------|
| BM25 | ~0.35 | ~0.15 | - |
| FAISS (IVF+PQ) | 0.536 | 0.170 | 612 |
| HNSW | ~0.54 | ~0.175 | ~615 |
| FAISS+Rerank | ~0.60 | ~0.23 | ~1200 |
| 森林（本方案） | 0.583 | 0.219 | 127 |

**核心论点**：森林在准确率上接近 FAISS+Rerank（生产主流），但 Token 消耗仅为后者的 ~10%。
