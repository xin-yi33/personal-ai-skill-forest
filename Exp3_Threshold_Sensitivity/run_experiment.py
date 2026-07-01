"""
Experiment 3 v2: Threshold δ Sensitivity with bilingual charts and delta/token vis.
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
DATA_DIR = os.path.join(BASE, 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']

def main():
    print("=" * 60)
    print("Experiment 3 v2: Threshold δ Sensitivity (Bilingual)")
    print("=" * 60)

    all_apis, test_queries = load_dataset(DATA_DIR)

    domain_apis = {d: [] for d in DOMAINS}
    for api in all_apis:
        if api['domain'] in domain_apis:
            domain_apis[api['domain']].append(api)
    domain_vectors = {}
    for d, apis in domain_apis.items():
        embs = np.array([a['embedding'] for a in apis])
        domain_vectors[d] = np.mean(embs, axis=0)

    deltas = []
    correct_top1 = []
    for q in test_queries:
        q_emb = np.array(q['query_embedding'])
        scores = {}
        for d, vec in domain_vectors.items():
            scores[d] = cosine_similarity(q_emb.reshape(1,-1), vec.reshape(1,-1))[0][0]
        sorted_d = sorted(scores.values(), reverse=True)
        delta = sorted_d[0] - sorted_d[1]
        deltas.append(delta)
        best = max(scores, key=scores.get)
        correct_top1.append(1 if best == q['correct_domain'] else 0)

    deltas = np.array(deltas)
    correct_top1 = np.array(correct_top1)
    routing_acc = float(np.mean(correct_top1))

    thresholds = np.arange(0.05, 0.36, 0.01)
    # Task completion rates measured from actual routing accuracy
    # When ABCD triggers: user picks from candidates → success depends on correct domain being in top-4
    # When no ABCD: system picks top-1 → success depends on routing accuracy
    results = {'threshold': [], 'trigger_rate': [], 'misroute_rate': [], 'task_completion': []}

    for th in thresholds:
        triggered = deltas < th
        n_trig = np.sum(triggered)
        trigger_rate = n_trig / len(deltas)
        misroute = np.sum(triggered & (correct_top1 == 0)) / n_trig if n_trig > 0 else 0
        # ABCD: when triggered, user can pick correct domain from top-4 candidates
        # Without M5: just use top-1 (routing accuracy)
        # Task completion = trigger_rate * ABCD_success + (1-trigger_rate) * no_ABCD_success
        abcd_success = 1 - misroute  # ABCD helps when misroute would have happened
        no_abcd_success = routing_acc  # Without ABCD, use top-1 routing accuracy
        task_comp = trigger_rate * abcd_success + (1 - trigger_rate) * no_abcd_success
        results['threshold'].append(float(th))
        results['trigger_rate'].append(float(trigger_rate))
        results['misroute_rate'].append(float(misroute))
        results['task_completion'].append(float(task_comp))

    best_idx = np.argmax(results['task_completion'])
    optimal = results['threshold'][best_idx]
    routing_acc = float(np.mean(correct_top1))

    print(f"\nOptimal δ = {optimal:.2f}")
    print(f"Routing accuracy (Top-1): {routing_acc:.1%}")
    print(f"Delta distribution: mean={np.mean(deltas):.4f}, std={np.std(deltas):.4f}")
    print(f"Trigger rate at optimal: {results['trigger_rate'][best_idx]:.1%}")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # === Chart 1: 双轴图 (中文) ===
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(results['threshold'], results['trigger_rate'], width=0.008, alpha=0.6, color='#3498DB', label='多候选触发率')
    ax1.set_xlabel('阈值 δ', fontsize=12)
    ax1.set_ylabel('多候选触发率', color='#3498DB', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#3498DB')
    ax2 = ax1.twinx()
    ax2.plot(results['threshold'], results['task_completion'], 'o-', color='#E74C3C', linewidth=2, markersize=3, label='任务完成率')
    ax2.set_ylabel('任务完成率', color='#E74C3C', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    ax1.axvline(x=optimal, color='green', linestyle='--', alpha=0.7)
    ax1.set_title(f'阈值δ敏感性分析\n最优δ={optimal:.2f}, Top-1路由准确率={routing_acc:.1%}', fontweight='bold', fontsize=13, pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    fig.savefig(os.path.join(VIS_DIR, 'fig_阈值敏感性_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # === Chart 1 EN ===
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(results['threshold'], results['trigger_rate'], width=0.008, alpha=0.6, color='#3498DB', label='Multi-candidate Trigger Rate')
    ax1.set_xlabel('Threshold δ', fontsize=12)
    ax1.set_ylabel('Trigger Rate', color='#3498DB', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#3498DB')
    ax2 = ax1.twinx()
    ax2.plot(results['threshold'], results['task_completion'], 'o-', color='#E74C3C', linewidth=2, markersize=3, label='Task Completion Rate')
    ax2.set_ylabel('Task Completion Rate', color='#E74C3C', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    ax1.axvline(x=optimal, color='green', linestyle='--', alpha=0.7)
    ax1.set_title(f'Threshold δ Sensitivity Analysis\nOptimal δ={optimal:.2f}, Top-1 Routing Acc={routing_acc:.1%}', fontweight='bold', fontsize=13, pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    fig.savefig(os.path.join(VIS_DIR, 'fig_threshold_sensitivity_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # === Chart 2: Δ分布直方图 (中文) ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deltas, bins=30, alpha=0.7, color='#3498DB', edgecolor='white')
    ax.axvline(x=optimal, color='red', linestyle='--', linewidth=2, label=f'最优δ={optimal:.2f}')
    ax.axvline(x=np.mean(deltas), color='orange', linestyle='-', linewidth=2, label=f'平均Δ={np.mean(deltas):.4f}')
    ax.set_xlabel('Δ值 (Top-1与Top-2相似度差值)', fontsize=12)
    ax.set_ylabel('查询数量', fontsize=12)
    ax.set_title(f'Δ值分布直方图\n平均Δ={np.mean(deltas):.4f}, 标准差={np.std(deltas):.4f}', fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_Delta分布_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # === Chart 2 EN ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deltas, bins=30, alpha=0.7, color='#3498DB', edgecolor='white')
    ax.axvline(x=optimal, color='red', linestyle='--', linewidth=2, label=f'Optimal δ={optimal:.2f}')
    ax.axvline(x=np.mean(deltas), color='orange', linestyle='-', linewidth=2, label=f'Mean Δ={np.mean(deltas):.4f}')
    ax.set_xlabel('Δ (Top-1 vs Top-2 Similarity Difference)', fontsize=12)
    ax.set_ylabel('Number of Queries', fontsize=12)
    ax.set_title(f'Delta Distribution Histogram\nMean={np.mean(deltas):.4f}, Std={np.std(deltas):.4f}', fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_delta_distribution_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # === Chart 3: Token消耗随阈值变化 (中文) ===
    # Token model: triggered queries use ABCD (more tokens), non-triggered use direct Top-1
    token_with_abcd = 630  # Forest full token
    token_without_abcd = 417  # Forest base token
    token_curve = [r * token_with_abcd + (1-r) * token_without_abcd for r in results['trigger_rate']]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['threshold'], token_curve, 's-', color='#F39C12', linewidth=2, markersize=4)
    ax.axvline(x=optimal, color='red', linestyle='--', alpha=0.7, label=f'最优δ={optimal:.2f}')
    ax.set_xlabel('阈值 δ', fontsize=12)
    ax.set_ylabel('平均Token消耗', fontsize=12)
    ax.set_title('Token消耗随阈值δ变化\n(触发ABCD选择时Token更高)', fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_Token随阈值变化_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # === Chart 3 EN ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['threshold'], token_curve, 's-', color='#F39C12', linewidth=2, markersize=4)
    ax.axvline(x=optimal, color='red', linestyle='--', alpha=0.7, label=f'Optimal δ={optimal:.2f}')
    ax.set_xlabel('Threshold δ', fontsize=12)
    ax.set_ylabel('Average Token Consumption', fontsize=12)
    ax.set_title('Token Consumption vs Threshold δ\n(ABCD selection costs more tokens)', fontweight='bold', fontsize=13, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(VIS_DIR, 'fig_token_vs_threshold_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    output = {
        'optimal_threshold': float(optimal),
        'routing_accuracy': routing_acc,
        'delta_stats': {'mean': float(np.mean(deltas)), 'std': float(np.std(deltas)), 'min': float(np.min(deltas)), 'max': float(np.max(deltas))},
    }
    with open(os.path.join(RES_DIR, 'experiment3_v2_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {RES_DIR}")
    print(f"Visualizations: {VIS_DIR}")

if __name__ == '__main__':
    main()
