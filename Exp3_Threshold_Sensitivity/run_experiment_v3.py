"""
Experiment 3 v3: Threshold δ Sensitivity Analysis (Comprehensive Redesign)

Design:
- Test specific δ values: 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
- Record: routing accuracy, task completion rate, intent ambiguity rate
- Explain best δ selection based on Δ distribution (mean, std)
- Full table output with all comparison results
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from shared.data_generator import load_dataset

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), 'shared', 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']


def main():
    print("=" * 60)
    print("Experiment 3 v3: Threshold δ Sensitivity (Comprehensive)")
    print("=" * 60)

    # === Step 1: Load data and compute Δ distribution ===
    all_apis, test_queries = load_dataset(DATA_DIR)

    domain_apis = {d: [] for d in DOMAINS}
    for api in all_apis:
        if api['domain'] in domain_apis:
            domain_apis[api['domain']].append(api)
    domain_vectors = {}
    for d, apis in domain_apis.items():
        embs = np.array([a['embedding'] for a in apis])
        domain_vectors[d] = np.mean(embs, axis=0)

    # Compute per-query: Δ (gap between top-1 and top-2), routing correctness, ambiguity
    deltas = []
    correct_top1 = []
    is_ambiguous = []
    top2_domains = []  # For intent ambiguity analysis

    for q in test_queries:
        q_emb = np.array(q['query_embedding'])
        scores = {}
        for d, vec in domain_vectors.items():
            scores[d] = cosine_similarity(q_emb.reshape(1, -1), vec.reshape(1, -1))[0][0]
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        delta = sorted_items[0][1] - sorted_items[1][1]
        deltas.append(delta)

        best_domain = sorted_items[0][0]
        correct_top1.append(1 if best_domain == q['correct_domain'] else 0)
        is_ambiguous.append(q.get('cross_domain_ambiguous', False))
        top2_domains.append((sorted_items[0][0], sorted_items[1][0]))

    deltas = np.array(deltas)
    correct_top1 = np.array(correct_top1)
    is_ambiguous = np.array(is_ambiguous)
    routing_acc = float(np.mean(correct_top1))
    ambiguity_rate = float(np.mean(is_ambiguous))

    # === Step 2: Δ distribution statistics ===
    delta_mean = float(np.mean(deltas))
    delta_std = float(np.std(deltas))
    delta_median = float(np.median(deltas))
    delta_p25 = float(np.percentile(deltas, 25))
    delta_p75 = float(np.percentile(deltas, 75))
    delta_min = float(np.min(deltas))
    delta_max = float(np.max(deltas))

    print(f"\n{'='*60}")
    print(f"Δ DISTRIBUTION STATISTICS")
    print(f"{'='*60}")
    print(f"  Mean Δ = {delta_mean:.4f}")
    print(f"  Std Δ  = {delta_std:.4f}")
    print(f"  Median = {delta_median:.4f}")
    print(f"  P25    = {delta_p25:.4f}")
    print(f"  P75    = {delta_p75:.4f}")
    print(f"  Min    = {delta_min:.4f}")
    print(f"  Max    = {delta_max:.4f}")
    print(f"  Routing Accuracy (Top-1) = {routing_acc:.1%}")
    print(f"  Intent Ambiguity Rate    = {ambiguity_rate:.1%}")

    # === Step 3: Test specific δ values ===
    test_deltas = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    results_table = []

    for th in test_deltas:
        triggered = deltas < th
        n_trig = int(np.sum(triggered))
        trigger_rate = n_trig / len(deltas)

        # Among triggered queries: how many would be misrouted?
        misroute_triggered = int(np.sum(triggered & (correct_top1 == 0)))
        misroute_rate = misroute_triggered / n_trig if n_trig > 0 else 0.0

        # Among triggered queries: how many are ambiguous?
        ambiguous_triggered = int(np.sum(triggered & is_ambiguous))
        ambiguity_in_triggered = ambiguous_triggered / n_trig if n_trig > 0 else 0.0

        # Task completion model:
        # Without ABCD: system picks top-1 → success = routing_acc
        # With ABCD: user picks from candidates → success = 1 - misroute_rate
        abcd_success = 1 - misroute_rate
        no_abcd_success = routing_acc
        task_comp = trigger_rate * abcd_success + (1 - trigger_rate) * no_abcd_success

        # Token overhead estimate: triggered queries cost extra tokens for ABCD presentation
        # Each ABCD candidate: ~30 tokens for domain description
        token_overhead = trigger_rate * 4 * 30  # 4 candidates × 30 tokens

        results_table.append({
            'delta': th,
            'trigger_rate': trigger_rate,
            'n_triggered': n_trig,
            'misroute_rate': misroute_rate,
            'ambiguity_in_triggered': ambiguity_in_triggered,
            'abcd_success': abcd_success,
            'task_completion': task_comp,
            'token_overhead': token_overhead,
        })

    # Find optimal δ
    best_idx = max(range(len(results_table)), key=lambda i: results_table[i]['task_completion'])
    optimal = results_table[best_idx]['delta']

    # === Step 4: Print comprehensive table ===
    print(f"\n{'='*120}")
    print(f"THRESHOLD SENSITIVITY ANALYSIS TABLE")
    print(f"{'='*120}")
    print(f"{'δ':<8} {'触发率':<10} {'触发数':<8} {'误路由率':<10} {'歧义占比':<10} {'ABCD成功率':<12} {'任务完成率':<12} {'Token开销':<10} {'评价'}")
    print(f"{'-'*120}")

    for r in results_table:
        # Evaluation label
        if r['delta'] == optimal:
            label = "★ 最优"
        elif r['delta'] < delta_mean - delta_std:
            label = "过小(高误路由)"
        elif r['delta'] > delta_mean + 2 * delta_std:
            label = "过大(高触发率)"
        elif r['delta'] < delta_mean:
            label = "偏保守"
        else:
            label = "偏激进"

        print(f"{r['delta']:<8.2f} {r['trigger_rate']:<10.1%} {r['n_triggered']:<8d} "
              f"{r['misroute_rate']:<10.1%} {r['ambiguity_in_triggered']:<10.1%} "
              f"{r['abcd_success']:<12.1%} {r['task_completion']:<12.3f} "
              f"{r['token_overhead']:<10.0f} {label}")

    # === Step 5: Best δ analysis ===
    print(f"\n{'='*60}")
    print(f"BEST δ SELECTION ANALYSIS")
    print(f"{'='*60}")
    print(f"  Δ distribution: mean={delta_mean:.4f}, std={delta_std:.4f}")
    print(f"  Optimal δ = {optimal:.2f}")
    print(f"  At optimal δ:")
    best_r = results_table[best_idx]
    print(f"    Trigger rate     = {best_r['trigger_rate']:.1%}")
    print(f"    Misroute rate    = {best_r['misroute_rate']:.1%}")
    print(f"    Task completion  = {best_r['task_completion']:.3f}")
    print(f"    Token overhead   = {best_r['token_overhead']:.0f}")

    print(f"\n  Selection rationale:")
    print(f"    - Δ mean = {delta_mean:.4f}: most queries have small gap (ambiguous intent)")
    print(f"    - Δ std  = {delta_std:.4f}: high variance indicates mixed clear/ambiguous queries")
    print(f"    - δ = {optimal:.2f} triggers ABCD for {best_r['trigger_rate']:.0%} of queries")
    print(f"    - This captures queries where top-1/top-2 similarity gap < δ (intent unclear)")
    print(f"    - For these queries, ABCD lets user pick correct domain (success={best_r['abcd_success']:.0%})")
    print(f"    - For clear queries (gap ≥ δ), system uses top-1 (success={routing_acc:.0%})")

    # === Step 6: Generate visualizations ===
    print(f"\nGenerating visualizations...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Chart 1: Dual-axis (trigger rate + task completion) CN
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ths = [r['delta'] for r in results_table]
    trigs = [r['trigger_rate'] for r in results_table]
    comps = [r['task_completion'] for r in results_table]

    bars = ax1.bar(ths, trigs, width=0.008, alpha=0.6, color='#3498DB', label='多候选触发率')
    ax1.set_xlabel('阈值 δ', fontsize=12)
    ax1.set_ylabel('多候选触发率', color='#3498DB', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#3498DB')

    ax2 = ax1.twinx()
    ax2.plot(ths, comps, 'o-', color='#E74C3C', linewidth=2, markersize=5, label='任务完成率')
    ax2.set_ylabel('任务完成率', color='#E74C3C', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#E74C3C')

    ax1.axvline(x=optimal, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=delta_mean, color='orange', linestyle=':', alpha=0.7, linewidth=2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2 + [f'最优δ={optimal:.2f}', f'平均Δ={delta_mean:.4f}'],
               loc='best', framealpha=0.9)
    ax1.set_title(f'阈值δ敏感性分析\n最优δ={optimal:.2f}, 路由准确率={routing_acc:.1%}, Δ均值={delta_mean:.4f}±{delta_std:.4f}',
                  fontweight='bold', fontsize=13, pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    fig.savefig(os.path.join(VIS_DIR, 'fig_阈值敏感性_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Chart 1 EN
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(ths, trigs, width=0.008, alpha=0.6, color='#3498DB', label='Multi-candidate Trigger Rate')
    ax1.set_xlabel('Threshold δ', fontsize=12)
    ax1.set_ylabel('Trigger Rate', color='#3498DB', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#3498DB')
    ax2 = ax1.twinx()
    ax2.plot(ths, comps, 'o-', color='#E74C3C', linewidth=2, markersize=5, label='Task Completion Rate')
    ax2.set_ylabel('Task Completion Rate', color='#E74C3C', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    ax1.axvline(x=optimal, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=delta_mean, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2 + [f'Optimal δ={optimal:.2f}', f'Mean Δ={delta_mean:.4f}'],
               loc='best', framealpha=0.9)
    ax1.set_title(f'Threshold δ Sensitivity Analysis\nOptimal δ={optimal:.2f}, Routing Acc={routing_acc:.1%}, Δ mean={delta_mean:.4f}±{delta_std:.4f}',
                  fontweight='bold', fontsize=13, pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    fig.savefig(os.path.join(VIS_DIR, 'fig_threshold_sensitivity_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Chart 2: Δ distribution histogram CN
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deltas, bins=40, alpha=0.7, color='#3498DB', edgecolor='white')
    ax.axvline(x=optimal, color='red', linestyle='--', linewidth=2, label=f'最优δ={optimal:.2f}')
    ax.axvline(x=delta_mean, color='orange', linestyle='-', linewidth=2, label=f'平均Δ={delta_mean:.4f}')
    ax.axvline(x=delta_mean + delta_std, color='orange', linestyle=':', alpha=0.5, label=f'Δ+σ={delta_mean + delta_std:.4f}')
    ax.axvline(x=delta_mean - delta_std, color='orange', linestyle=':', alpha=0.5, label=f'Δ-σ={delta_mean - delta_std:.4f}')
    ax.set_xlabel('Δ值 (Top-1与Top-2相似度差值)', fontsize=12)
    ax.set_ylabel('查询数量', fontsize=12)
    ax.set_title(f'Δ值分布直方图\n平均Δ={delta_mean:.4f}, 标准差={delta_std:.4f}, 中位数={delta_median:.4f}',
                 fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_Delta分布_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Chart 2 EN
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deltas, bins=40, alpha=0.7, color='#3498DB', edgecolor='white')
    ax.axvline(x=optimal, color='red', linestyle='--', linewidth=2, label=f'Optimal δ={optimal:.2f}')
    ax.axvline(x=delta_mean, color='orange', linestyle='-', linewidth=2, label=f'Mean Δ={delta_mean:.4f}')
    ax.axvline(x=delta_mean + delta_std, color='orange', linestyle=':', alpha=0.5, label=f'Δ+σ={delta_mean + delta_std:.4f}')
    ax.axvline(x=delta_mean - delta_std, color='orange', linestyle=':', alpha=0.5, label=f'Δ-σ={delta_mean - delta_std:.4f}')
    ax.set_xlabel('Δ (Top-1 vs Top-2 Similarity Difference)', fontsize=12)
    ax.set_ylabel('Number of Queries', fontsize=12)
    ax.set_title(f'Delta Distribution Histogram\nMean={delta_mean:.4f}, Std={delta_std:.4f}, Median={delta_median:.4f}',
                 fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_delta_distribution_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Chart 3: Token overhead CN
    fig, ax = plt.subplots(figsize=(10, 6))
    token_overheads = [r['token_overhead'] for r in results_table]
    ax.plot(ths, token_overheads, 's-', color='#F39C12', linewidth=2, markersize=5)
    ax.axvline(x=optimal, color='red', linestyle='--', alpha=0.7, label=f'最优δ={optimal:.2f}')
    ax.set_xlabel('阈值 δ', fontsize=12)
    ax.set_ylabel('平均Token开销', fontsize=12)
    ax.set_title('Token消耗随阈值δ变化\n(触发ABCD选择时Token更高)', fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_Token随阈值变化_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Chart 3 EN
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ths, token_overheads, 's-', color='#F39C12', linewidth=2, markersize=5)
    ax.axvline(x=optimal, color='red', linestyle='--', alpha=0.7, label=f'Optimal δ={optimal:.2f}')
    ax.set_xlabel('Threshold δ', fontsize=12)
    ax.set_ylabel('Average Token Overhead', fontsize=12)
    ax.set_title('Token Consumption vs Threshold δ\n(ABCD selection costs more tokens)', fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_token_vs_threshold_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Chart 4: Combined analysis (new) - Task completion vs trigger rate scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_scatter = ['#E74C3C' if r['delta'] == optimal else '#3498DB' for r in results_table]
    sizes = [100 if r['delta'] == optimal else 50 for r in results_table]
    ax.scatter([r['trigger_rate'] for r in results_table],
               [r['task_completion'] for r in results_table],
               c=colors_scatter, s=sizes, alpha=0.8, edgecolors='white', linewidth=1.5)
    for r in results_table:
        ax.annotate(f'δ={r["delta"]:.2f}',
                    (r['trigger_rate'], r['task_completion']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Multi-candidate Trigger Rate', fontsize=12)
    ax.set_ylabel('Task Completion Rate', fontsize=12)
    ax.set_title(f'Trigger Rate vs Task Completion\nOptimal δ={optimal:.2f} (highlighted in red)',
                 fontweight='bold', fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_trigger_vs_completion_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Save results
    output = {
        'optimal_threshold': float(optimal),
        'routing_accuracy': routing_acc,
        'intent_ambiguity_rate': ambiguity_rate,
        'delta_stats': {
            'mean': delta_mean, 'std': delta_std, 'median': delta_median,
            'p25': delta_p25, 'p75': delta_p75, 'min': delta_min, 'max': delta_max
        },
        'threshold_table': results_table,
        'best_analysis': {
            'trigger_rate': best_r['trigger_rate'],
            'misroute_rate': best_r['misroute_rate'],
            'task_completion': best_r['task_completion'],
            'token_overhead': best_r['token_overhead'],
        }
    }
    with open(os.path.join(RES_DIR, 'experiment3_v3_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RES_DIR}")
    print(f"Visualizations saved to: {VIS_DIR}")


if __name__ == '__main__':
    main()
