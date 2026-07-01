# 个人 AI 技能森林

> **基于多棵 B+ 树的多级索引与自进化系统——面向智能体技能管理**

[English](README.md) | 中文

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-SCI%20Ready-orange.svg)](paper/)

---

## 概述

**个人 AI 技能森林**是一种将 B+ 树多级索引引入智能体技能管理的新颖架构。随着大语言模型（LLM）从单轮问答向长程任务执行转型，管理数百个工具成为结构性障碍——Anthropic 报告显示，仅 58 个工具即消耗约 55K tokens。本系统解决五大核心挑战：**可扩展性**、**个性化**、**可进化性**、**可解释性**与 **Token 效率**。

### 核心结果

| 指标 | 改进幅度 |
|------|---------|
| Token 节省 | **79.3%**（612 → 127 tokens） |
| MRR 提升 | **+28.8%**（vs. 平铺 ANN） |
| 链完整率 | **0.363 → 1.000**（+175.5%） |
| 自进化成功率 | **+21.6pp**（3 轮学习） |
| 跨规模 Token 节省 | **~82%**（500–5,000 API 稳定） |

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    用户查询                           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│              M2: 森林级路由                            │
│    （根向量相似度 → 选择领域树）                        │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ 文档树   │ │ 数据树   │ │ 代码树   │  ...（5 棵树）
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         ▼            ▼            ▼
    ┌──────────────────────────────────────┐
    │      B+ 树多级遍历                     │
    │  根 → 中间节点 → 叶子 → Top-K          │
    └──────────────────────┬───────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────┐
    │       核心机制（M4–M9）                │
    │  M4: 依赖回溯                         │
    │  M5: 多候选选择                        │
    │  M6: 参数合并                          │
    │  M7: 私人技能遮蔽                      │
    │  M9: 角色降维                          │
    └──────────────────────┬───────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────┐
    │     自进化系统（M10–M12）              │
    │  行动反思（4 个提取器）                │
    │  思维反思（认知循环）                  │
    └──────────────────────────────────────┘
```

---

## 项目结构

```
personal-ai-skill-forest/
├── .gitignore                             # Git 忽略规则
├── README.md                              # 英文 README
├── README_CN.md                           # 中文 README（本文件）
├── requirements.txt                       # Python 依赖
├── verify_fixes.py                        # 验证脚本
│
├── paper/                                 # 学术论文
│   ├── paper_EN.md                        # 英文论文（SCI 格式）
│   └── paper_CN.md                        # 中文论文
│
├── shared/                                # 核心实现与共享数据
│   ├── __init__.py
│   ├── bplus_tree.py                      # B+ 树多级遍历实现
│   ├── mechanisms.py                      # M4/M5/M6/M7/M9 机制实现
│   ├── data_generator.py                  # 合成数据集生成器
│   ├── enrich_data.py                     # 数据增强（依赖、对话）
│   ├── prepare_data.py                    # 数据准备流水线
│   ├── visualization_utils.py             # 绘图工具库
│   └── data/                              # ** 所有数据集的唯一数据源 **
│       ├── all_apis.json                  # 5,000 个 API（原始）
│       ├── all_apis_enriched.json         # 5,000 个 API（含完整依赖）
│       ├── test_queries.json              # 200 个测试查询
│       ├── private_skills.json            # 50 个私有技能
│       ├── param_conflict_cases.json      # 20 个参数冲突用例
│       ├── m7_test_cases.json             # 30 个 M7 测试用例
│       └── conversations_dataset.json     # 100 条对话（70 真实 + 30 噪声）
│
├── Exp1_Retrieval_Performance/            # 实验一：检索性能对比
│   ├── run_experiment.py                  # 实验脚本
│   ├── results/                           # JSON 结果
│   └── Visualization/                     # 生成的图表
│
├── Exp2_Ablation_Study/                   # 实验二：消融实验
│   ├── run_experiment_v3.py               # 实验脚本
│   ├── run_m2_experiment.py               # M2 专项实验
│   ├── results/
│   └── Visualization/
│
├── Exp3_Threshold_Sensitivity/            # 实验三：阈值 δ 敏感性分析
│   ├── run_experiment_v3.py               # 实验脚本
│   ├── results/
│   └── Visualization/
│
├── Exp4_Action_Reflection/                # 实验四：模式提取器有效性
│   ├── run_experiment_v2.py               # 实验脚本
│   ├── results/
│   └── Visualization/
│
├── Exp5_Thought_Reflection/               # 实验五：元认知策略有效性
│   ├── run_experiment_v2.py               # 实验脚本
│   ├── results/
│   └── Visualization/
│
└── Exp6_Token_Consumption/                # 实验六：Token 消耗理论与实证
    ├── run_experiment_v3.py               # 实验脚本
    ├── results/
    └── Visualization/
```

> **说明**：所有实验统一从 `shared/data/` 加载数据（唯一数据源），实验文件夹内无重复数据文件。

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/xin-yi33/personal-ai-skill-forest.git
cd personal-ai-skill-forest

# 安装依赖
pip install -r requirements.txt
```

### 依赖项

```
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pandas>=2.0.0
```

### 运行实验

```bash
# 第 1 步：准备数据
python shared/prepare_data.py
python shared/enrich_data.py

# 第 2 步：运行实验
python Exp1_Retrieval_Performance/run_experiment.py
python Exp2_Ablation_Study/run_experiment_v3.py
python Exp3_Threshold_Sensitivity/run_experiment_v3.py
python Exp4_Action_Reflection/run_experiment_v2.py
python Exp5_Thought_Reflection/run_experiment_v2.py
python Exp6_Token_Consumption/run_experiment_v3.py

# 第 3 步：验证结果
python verify_fixes.py
```

---

## 实验结果

### 实验一：检索性能对比

| 指标 | Flat ANN (FAISS) | 技能森林 | 差异 |
|------|------------------|---------|------|
| Accuracy@5 | 0.536 | **0.583** | +8.8% |
| MRR | 0.170 | **0.219** | +28.8% |
| 端到端 Token | 612 | **127** | -79.3% |
| 链完整率 | 0.363 | **1.000** | +175.5% |

### 实验二：消融实验

| 机制 | 贡献量 |
|------|--------|
| M4（依赖回溯） | +0.592 链完整率 |
| M6（参数合并） | +0.533 冲突消解率 |
| M7（私人遮蔽） | +0.467 命中率 |
| M5（ABCD 选择） | +0.340 任务完成率 |
| M9（角色降维） | -210 tokens（63% 降低） |

### 实验三：最优阈值

- **最优 δ = 0.25**（任务完成率：0.585）
- Δ 分布：均值=0.0466，标准差=0.0481

### 实验四：行动反思——模式提取器

| 提取器 | F1（清洁） | F1（30% 噪声） |
|--------|-----------|----------------|
| E1：错误模式 | 0.894 | 0.834 |
| E2：默认偏好 | 0.909 | 0.907 |
| E3：工作流 | 0.926 | 0.891 |
| E4：显式指令 | 0.795 | 0.781 |
| **宏平均** | **0.881** | **0.853** |

### 实验五：思维反思——元认知策略

| 轮次 | 步骤数 | Token | 成功率 |
|------|--------|-------|--------|
| Round 1 | 4.29 | 2,050 | 55.43% |
| Round 2 | 2.62 | 1,370 | 75.93% |
| Round 3 | 2.50 | 1,380 | **77.01%** |

**改进**：步骤 -41.6%，Token -32.7%，成功率 +21.6pp

### 实验六：Token 效率与规模

| N | Flat+LLM | 森林 | 节省 |
|---|----------|------|------|
| 500 | 512 | 88 | 82.9% |
| 1,000 | 497 | 88 | 82.4% |
| 3,000 | 486 | 89 | 81.7% |
| 5,000 | 485 | 89 | 81.7% |

---

## 论文

完整学术论文位于 `paper/` 目录：

- **英文版**：[`paper/paper_EN.md`](paper/paper_EN.md) — 完整 SCI 格式论文
- **中文版**：[`paper/paper_CN.md`](paper/paper_CN.md) — 完整中文翻译

### 论文摘要

> 本文提出"个人 AI 技能森林"架构，基于多棵并行 B+ 树，系统性解决技能管理在规模化场景下的五大核心挑战。系统由 12 个互锁机制组成，分为森林级路由层、核心操作机制组（依赖回溯、多候选选择、参数合并、私人技能遮蔽、角色降维）和双层自进化系统。在 5,000-API 数据集上的实验表明：Token 消耗降低 79.3%，自进化成功率提升 21.6 个百分点，跨规模 Token 节省稳定在约 82%。

---

## 数据集

| 领域 | API 数量 | 子类别 | 依赖关系 |
|------|---------|--------|---------|
| 文档创作 | 1,000 | 6 | ✅ 全部（最大深度 3） |
| 数据分析 | 1,000 | 6 | ✅ 全部（最大深度 3） |
| 通信协作 | 1,000 | 6 | ✅ 全部（最大深度 3） |
| 代码工程 | 1,000 | 6 | ✅ 全部（最大深度 3） |
| 设计创意 | 1,000 | 6 | ✅ 全部（最大深度 3） |

- **API 总数**：5,000
- **测试查询**：200（140 清晰 + 60 歧义）
- **对话数据**：100（70 真实 + 30 噪声）
- **Embedding 模型**：all-MiniLM-L6-v2（384 维）

---

## 重要声明

1. **合成数据集**：本实验使用模板化 API 描述配合真实 embedding 的合成数据。合成数据的领域区分度可能高于真实数据。

2. **Token 模型**：Token 消耗基于内容长度估算，非 LLM 实际 API 调用测量。模型透明可复现，但可能与真实 LLM Token 消耗存在差异。

3. **确定性机制**：M4/M5/M6/M7/M9 均实现为操作真实数据结构的确定性算法，非随机模拟。

---

## 引用

如使用本工作，请引用：

```bibtex
@article{ruan2026skillforest,
  title={Personal AI Skill Forest: A Multi-B+ Tree Based Multi-Level Indexing and Self-Evolution System for Intelligent Agent Skill Management},
  author={Ruan, Xinyi},
  year={2026}
}
```

---

## 许可证

本项目基于 MIT 许可证开源——详见 [LICENSE](LICENSE) 文件。

---

## 联系方式

- **作者**：阮心一（Xinyi Ruan）
- **GitHub**：[xin-yi33](https://github.com/xin-yi33)

---

> **声明**：本研究在 AI 工具辅助下完成。所有实验数据由合成数据集上的实际代码执行产生。关键应用场景请用户独立验证结果。
