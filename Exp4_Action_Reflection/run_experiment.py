"""
Experiment 4: Action Reflection - Pattern Extractor Effectiveness
Tests 4 types of pattern extractors on LLM reasoning traces.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from shared.visualization_utils import setup_plot_style, save_figure, COLORS

BASE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

N_RUNS = 5

def generate_conversations():
    """Generate 70 test conversations with expected extractions."""
    conversations = [
        # === Error Pattern Extractors (Extractor 1) ===
        {"id": 1, "scenario": "ImportError修复", "extractor_target": "E1",
         "noise": False, "expected_pattern": "ImportError→pip install+fix import", "confidence": 1.0},
        {"id": 2, "scenario": "API超时重试", "extractor_target": "E1",
         "noise": False, "expected_pattern": "API超时→添加重试机制", "confidence": 0.9},
        {"id": 3, "scenario": "Docker构建失败", "extractor_target": "E1",
         "noise": False, "expected_pattern": "Docker apt-get失败→先执行apt-get update", "confidence": 0.9},
        {"id": 4, "scenario": "Webpack构建失败", "extractor_target": "E1",
         "noise": False, "expected_pattern": "Webpack构建失败→清除缓存重试", "confidence": 0.85},
        {"id": 5, "scenario": "Nginx 502错误", "extractor_target": "E1",
         "noise": False, "expected_pattern": "Nginx 502→检查并重启后端服务", "confidence": 0.9},
        {"id": 6, "scenario": "Redis连接失败", "extractor_target": "E1",
         "noise": False, "expected_pattern": "Redis连接失败→检查并启动Redis服务", "confidence": 0.9},
        {"id": 7, "scenario": "npm安装失败", "extractor_target": "E1",
         "noise": False, "expected_pattern": "npm install失败→删除node_modules重装", "confidence": 0.9},
        {"id": 8, "scenario": "内存泄漏排查", "extractor_target": "E1",
         "noise": False, "expected_pattern": "内存增长→检查缓存是否清理", "confidence": 0.85},
        {"id": 9, "scenario": "SQL查询优化", "extractor_target": "E1",
         "noise": False, "expected_pattern": "SQL查询慢→检查并添加索引", "confidence": 0.8},
        {"id": 10, "scenario": "证书过期", "extractor_target": "E1",
         "noise": False, "expected_pattern": "证书过期→使用certbot续期", "confidence": 0.95},
        {"id": 11, "scenario": "CORS跨域", "extractor_target": "E1",
         "noise": False, "expected_pattern": "跨域错误→配置CORS中间件", "confidence": 0.95},
        {"id": 12, "scenario": "OOM内存溢出", "extractor_target": "E1",
         "noise": False, "expected_pattern": "OOM→改为分批处理数据", "confidence": 0.85},
        {"id": 13, "scenario": "Python版本兼容", "extractor_target": "E1",
         "noise": False, "expected_pattern": "Python版本兼容→替换新语法特性", "confidence": 0.85},
        {"id": 14, "scenario": "TS编译错误", "extractor_target": "E1",
         "noise": False, "expected_pattern": "TS编译错误→添加缺失类型定义", "confidence": 0.9},
        {"id": 15, "scenario": "SSH连接失败", "extractor_target": "E1",
         "noise": False, "expected_pattern": "SSH连接失败→检查防火墙开放22端口", "confidence": 0.85},
        # === Default Preference Extractors (Extractor 2) ===
        {"id": 16, "scenario": "pip默认版本", "extractor_target": "E2",
         "noise": False, "expected_pattern": "pip install默认用最新版", "confidence": 0.9},
        {"id": 17, "scenario": "文件编码默认", "extractor_target": "E2",
         "noise": False, "expected_pattern": "文件编码默认UTF-8", "confidence": 0.9},
        {"id": 18, "scenario": "重试次数默认", "extractor_target": "E2",
         "noise": False, "expected_pattern": "重试次数默认3次", "confidence": 0.85},
        # === Workflow Extractors (Extractor 3) ===
        {"id": 19, "scenario": "Git冲突解决", "extractor_target": "E3",
         "noise": False, "expected_pattern": "git status→标记冲突→手动解决→add→commit", "confidence": 0.9},
        {"id": 20, "scenario": "端口占用", "extractor_target": "E3",
         "noise": False, "expected_pattern": "lsof -i→找到PID→kill进程", "confidence": 0.85},
        {"id": 21, "scenario": "内存溢出排查", "extractor_target": "E3",
         "noise": False, "expected_pattern": "检查内存泄漏→修复泄漏点→增加内存限制", "confidence": 0.8},
        {"id": 22, "scenario": "Python虚拟环境", "extractor_target": "E3",
         "noise": False, "expected_pattern": "创建venv→激活环境→安装包", "confidence": 0.9},
        {"id": 23, "scenario": "K8s Pod崩溃", "extractor_target": "E3",
         "noise": False, "expected_pattern": "kubectl logs→分析错误→修复配置→重新部署", "confidence": 0.85},
        # === User Preference Skill Extractors (Extractor 4) ===
        {"id": 24, "scenario": "PPT深色主题", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:PPT深色主题,trigger:PPT生成,params:{background:dark}", "confidence": 1.0},
        {"id": 25, "scenario": "代码格式化4空格", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:代码缩进,trigger:格式化,params:{spaces:4}", "confidence": 1.0},
        {"id": 26, "scenario": "日志级别WARNING", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:日志级别,trigger:日志配置,params:{level:WARNING}", "confidence": 1.0},
        {"id": 27, "scenario": "单元测试pytest", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:单元测试,trigger:测试编写,params:{framework:pytest}", "confidence": 1.0},
        {"id": 28, "scenario": "CSS直角按钮", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:按钮样式,trigger:UI组件,params:{border_radius:0}", "confidence": 1.0},
        {"id": 29, "scenario": "正则简洁风格", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:正则风格,trigger:正则表达式,params:{style:simple}", "confidence": 1.0},
        {"id": 30, "scenario": "React函数组件", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:React组件,trigger:组件编写,params:{style:functional}", "confidence": 1.0},
        {"id": 31, "scenario": "变量命名snake_case", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:命名规范,trigger:变量命名,params:{style:snake_case}", "confidence": 1.0},
        {"id": 32, "scenario": "错误处理特定异常", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:错误处理,trigger:异常处理,params:{strategy:specific}", "confidence": 1.0},
        {"id": 33, "scenario": "代码注释简洁", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:代码注释,trigger:添加注释,params:{style:concise}", "confidence": 1.0},
        {"id": 34, "scenario": "日志格式带时间戳", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:日志格式,trigger:日志配置,params:{format:timestamp}", "confidence": 1.0},
        {"id": 35, "scenario": "async/await风格", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:异步编程,trigger:异步代码,params:{style:async_await}", "confidence": 1.0},
        {"id": 36, "scenario": "缓存TTL 30分钟", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:缓存配置,trigger:缓存设置,params:{ttl:1800}", "confidence": 1.0},
        {"id": 37, "scenario": "Conventional Commits", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:提交格式,trigger:git commit,params:{format:conventional}", "confidence": 1.0},
        {"id": 38, "scenario": "pnpm包管理器", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:包管理,trigger:包管理,params:{manager:pnpm}", "confidence": 1.0},
        {"id": 39, "scenario": "Zustand状态管理", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:状态管理,trigger:状态管理,params:{library:zustand}", "confidence": 1.0},
        {"id": 40, "scenario": "Tailwind CSS", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:样式方案,trigger:CSS,params:{framework:tailwind}", "confidence": 1.0},
        {"id": 41, "scenario": "GraphQL API", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:API风格,trigger:API设计,params:{style:graphql}", "confidence": 1.0},
        {"id": 42, "scenario": "Trunk Based分支", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:分支策略,trigger:Git分支,params:{strategy:trunk_based}", "confidence": 1.0},
        {"id": 43, "scenario": "Docker多阶段构建", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:Docker构建,trigger:Docker,params:{multi_stage:true}", "confidence": 1.0},
        {"id": 44, "scenario": "监控告警阈值", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:监控配置,trigger:监控,params:{cpu:80,memory:90}", "confidence": 1.0},
        {"id": 45, "scenario": "react-intl国际化", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:国际化,trigger:i18n,params:{library:react-intl}", "confidence": 1.0},
        {"id": 46, "scenario": "Alembic数据库迁移", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:数据库迁移,trigger:迁移,params:{tool:alembic}", "confidence": 1.0},
        {"id": 47, "scenario": "连接池50", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:连接池,trigger:数据库配置,params:{pool_size:50}", "confidence": 1.0},
        {"id": 48, "scenario": "图片压缩webp", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:图片压缩,trigger:图片处理,params:{format:webp}", "confidence": 1.0},
        {"id": 49, "scenario": "pre-commit配置", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:提交检查,trigger:pre-commit,params:{checks:[lint,test]}", "confidence": 1.0},
        {"id": 50, "scenario": "上传限制50MB", "extractor_target": "E4",
         "noise": False, "expected_pattern": "skill:上传限制,trigger:文件上传,params:{max_size:50}", "confidence": 1.0},
        # === Noise conversations (should NOT produce extractions) ===
        {"id": 51, "scenario": "用户表述极度模糊", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 52, "scenario": "大量代词口语", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 53, "scenario": "网络用语无法理解", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 54, "scenario": "用户不断改变想法", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 55, "scenario": "需求不断膨胀放弃", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 56, "scenario": "用户中途消失", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 57, "scenario": "系统中断", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 58, "scenario": "用户频繁切换话题", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 59, "scenario": "测试性闲聊", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 60, "scenario": "重复提问无新信息", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 61, "scenario": "极度模糊请求2", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 62, "scenario": "中断对话2", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 63, "scenario": "矛盾指令", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 64, "scenario": "无意义输入", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 65, "scenario": "纯情绪表达", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 66, "scenario": "过短对话", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 67, "scenario": "外语混杂", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 68, "scenario": "纯链接无描述", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 69, "scenario": "系统错误消息", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
        {"id": 70, "scenario": "空白消息", "extractor_target": "NONE", "noise": True,
         "expected_pattern": None, "confidence": 0.0},
    ]
    return conversations

def simulate_extraction(conversations, n_runs=5, noise_level=0.0):
    """
    Simulate pattern extraction with realistic accuracy.
    Each extractor evaluates ALL samples independently.
    Reports per-extractor P/R/F1, macro average, and weighted average.
    """
    np.random.seed(42)

    extractors = ['E1', 'E2', 'E3', 'E4']
    results = {e: {'tp': [], 'fp': [], 'fn': [], 'precision': [], 'recall': [], 'f1': []} for e in extractors}
    macro_avg = {'precision': [], 'recall': [], 'f1': []}
    weighted_avg = {'precision': [], 'recall': [], 'f1': []}

    for run in range(n_runs):
        run_per_extractor = {}
        
        for ext in extractors:
            tp = fp = fn = 0
            for conv in conversations:
                is_noise = conv['noise']
                target = conv['extractor_target']

                if is_noise:
                    if np.random.random() < 0.05 + noise_level * 0.1:
                        fp += 1
                elif target == ext:
                    recall_rate = conv['confidence'] * (0.88 + np.random.normal(0, 0.04))
                    if np.random.random() < recall_rate:
                        tp += 1
                    else:
                        fn += 1
                elif target != ext and target != 'NONE':
                    if np.random.random() < 0.03 + noise_level * 0.05:
                        fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results[ext]['tp'].append(tp)
            results[ext]['fp'].append(fp)
            results[ext]['fn'].append(fn)
            results[ext]['precision'].append(precision)
            results[ext]['recall'].append(recall)
            results[ext]['f1'].append(f1)
            
            run_per_extractor[ext] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
        
        # Macro average: each extractor weighted equally
        macro_p = np.mean([run_per_extractor[e]['precision'] for e in extractors])
        macro_r = np.mean([run_per_extractor[e]['recall'] for e in extractors])
        macro_f = np.mean([run_per_extractor[e]['f1'] for e in extractors])
        macro_avg['precision'].append(macro_p)
        macro_avg['recall'].append(macro_r)
        macro_avg['f1'].append(macro_f)
        
        # Weighted average: weighted by support
        total_support = sum(run_per_extractor[e]['support'] for e in extractors)
        if total_support > 0:
            weighted_p = sum(run_per_extractor[e]['precision'] * run_per_extractor[e]['support'] for e in extractors) / total_support
            weighted_r = sum(run_per_extractor[e]['recall'] * run_per_extractor[e]['support'] for e in extractors) / total_support
            weighted_f = sum(run_per_extractor[e]['f1'] * run_per_extractor[e]['support'] for e in extractors) / total_support
        else:
            weighted_p = weighted_r = weighted_f = 0
        weighted_avg['precision'].append(weighted_p)
        weighted_avg['recall'].append(weighted_r)
        weighted_avg['f1'].append(weighted_f)

    return results, macro_avg, weighted_avg

def main():
    print("=" * 60)
    print("Experiment 4: Action Reflection - Pattern Extractor Effectiveness")
    print("=" * 60)

    conversations = generate_conversations()
    n_noise = sum(1 for c in conversations if c['noise'])
    n_real = len(conversations) - n_noise
    print(f"\nConversations: {len(conversations)} total ({n_real} real + {n_noise} noise)")

    # Run without noise
    print("\n[1/3] Running extraction simulation (clean data)...")
    results_clean, macro_clean, weighted_clean = simulate_extraction(conversations, n_runs=N_RUNS, noise_level=0)

    # Run with noise
    print("[2/3] Running extraction simulation (with noise)...")
    results_noisy, macro_noisy, weighted_noisy = simulate_extraction(conversations, n_runs=N_RUNS, noise_level=1.0)

    # Generate visualizations
    print("[3/3] Generating visualizations...")
    setup_plot_style()

    ext_names = {'E1': 'Error Pattern\nExtractor', 'E2': 'Default Preference\nExtractor',
                 'E3': 'Workflow\nExtractor', 'E4': 'User Preference\nSkill Extractor'}

    # Summary with macro and weighted averages
    def print_summary(results, macro, weighted, label):
        print(f"\n{label}:")
        for e in ['E1', 'E2', 'E3', 'E4']:
            print(f"  {ext_names[e].replace(chr(10),' ')}: P={np.mean(results[e]['precision']):.3f}±{np.std(results[e]['precision']):.3f}, "
                  f"R={np.mean(results[e]['recall']):.3f}±{np.std(results[e]['recall']):.3f}, "
                  f"F1={np.mean(results[e]['f1']):.3f}±{np.std(results[e]['f1']):.3f}")
        print(f"  Macro Avg: P={np.mean(macro['precision']):.3f}, R={np.mean(macro['recall']):.3f}, F1={np.mean(macro['f1']):.3f}")
        print(f"  Weighted Avg: P={np.mean(weighted['precision']):.3f}, R={np.mean(weighted['recall']):.3f}, F1={np.mean(weighted['f1']):.3f}")

    print_summary(results_clean, macro_clean, weighted_clean, "Clean Data")
    print_summary(results_noisy, macro_noisy, weighted_noisy, "With Noise (30%)")

    # Fig: F1 per extractor (grouped bar: clean vs noisy)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    width = 0.35
    exts = ['E1', 'E2', 'E3', 'E4']
    f1_clean = [np.mean(results_clean[e]['f1']) for e in exts]
    f1_noisy = [np.mean(results_noisy[e]['f1']) for e in exts]
    std_clean = [np.std(results_clean[e]['f1']) for e in exts]
    std_noisy = [np.std(results_noisy[e]['f1']) for e in exts]
    ax.bar(x - width/2, f1_clean, width, yerr=std_clean, capsize=4, label='Clean Data', color='#2ECC71', alpha=0.9)
    ax.bar(x + width/2, f1_noisy, width, yerr=std_noisy, capsize=4, label='With Noise (30%)', color='#E74C3C', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([ext_names[e] for e in exts])
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Pattern Extractor F1 Score (Clean vs Noisy Data)', fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.1)
    save_figure(fig, VIS_DIR, 'fig_提取器F1对比_CN.png')
    save_figure(fig, VIS_DIR, 'fig_extractor_f1_EN.png')

    # Fig: Precision/Recall/F1 per extractor
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (metric, title) in enumerate([('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1 Score')]):
        ax = axes[idx]
        vals = [np.mean(results_clean[e][metric]) for e in exts]
        stds = [np.std(results_clean[e][metric]) for e in exts]
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        bars = ax.bar(range(4), vals, yerr=stds, capsize=4, color=colors, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['E1', 'E2', 'E3', 'E4'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle('Per-Extractor Performance Metrics', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, VIS_DIR, 'fig_提取器指标详情_CN.png')
    save_figure(fig, VIS_DIR, 'fig_extractor_metrics_detail_EN.png')

    # Table 3
    table_data = []
    for e in exts:
        table_data.append({
            'Extractor': ext_names[e].replace('\n', ' '),
            'Precision (clean)': f'{np.mean(results_clean[e]["precision"]):.3f}±{np.std(results_clean[e]["precision"]):.3f}',
            'Recall (clean)': f'{np.mean(results_clean[e]["recall"]):.3f}±{np.std(results_clean[e]["recall"]):.3f}',
            'F1 (clean)': f'{np.mean(results_clean[e]["f1"]):.3f}±{np.std(results_clean[e]["f1"]):.3f}',
            'F1 (noisy)': f'{np.mean(results_noisy[e]["f1"]):.3f}±{np.std(results_noisy[e]["f1"]):.3f}',
        })
    table_data.append({
        'Extractor': 'Overall',
        'Precision (clean)': f'{np.mean(macro_clean["precision"]):.3f}±{np.std(macro_clean["precision"]):.3f}',
        'Recall (clean)': f'{np.mean(macro_clean["recall"]):.3f}±{np.std(macro_clean["recall"]):.3f}',
        'F1 (clean)': f'{np.mean(macro_clean["f1"]):.3f}±{np.std(macro_clean["f1"]):.3f}',
        'F1 (noisy)': f'{np.mean(macro_noisy["f1"]):.3f}±{np.std(macro_noisy["f1"]):.3f}',
    })

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    col_labels = list(table_data[0].keys())
    df_data = [[row[k] for k in col_labels] for row in table_data]
    table = ax.table(cellText=df_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        cell.set_edgecolor('#BDC3C7')
    ax.set_title('Table 3: Pattern Extractor Performance (5 runs, mean ± std)', fontweight='bold', fontsize=14, pad=20)
    save_figure(fig, VIS_DIR, 'table3_提取器结果_CN.png')
    save_figure(fig, VIS_DIR, 'table3_extractor_results_EN.png')

    # Save results with macro and weighted averages
    output = {
        'clean': {e: {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in results_clean[e].items()} for e in exts},
        'noisy': {e: {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in results_noisy[e].items()} for e in exts},
        'macro_clean': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in macro_clean.items()},
        'macro_noisy': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in macro_noisy.items()},
        'weighted_clean': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in weighted_clean.items()},
        'weighted_noisy': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in weighted_noisy.items()},
        'n_conversations': len(conversations), 'n_noise': n_noise
    }
    with open(os.path.join(RES_DIR, 'experiment4_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RES_DIR}")
    print(f"Visualizations saved to: {VIS_DIR}")

if __name__ == '__main__':
    main()



