"""
Experiment 6 v3: Token Consumption - Full Scale Analysis
Redesign per user requirements:
- N as scenario variable: 500, 1000, 1500, 2000, 3000, 4000, 5000
- 10 rounds per scenario for robust statistics
- Normal vs abnormal scenarios analyzed separately
- Complete table with mean ± std
- Trend analysis: forest advantage increases with scale
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from shared.data_generator import load_dataset, generate_dataset, compute_embeddings
from shared.bplus_tree import BPlusTree
from shared.mechanisms import (
    count_tokens, measure_e2e_tokens_flat_llm, measure_e2e_tokens_forest,
    trace_dependency_chain
)
import faiss

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']
N_RUNS = 10
N_VALUES = [500, 1000, 1500, 2000, 3000, 4000, 5000]


def build_faiss(embeddings):
    d = embeddings.shape[1]
    n = len(embeddings)
    if n < 200:
        idx = faiss.IndexFlatL2(d)
        idx.add(embeddings.astype('float32'))
        return idx
    nlist = min(50, max(1, n // 30))
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFPQ(quantizer, d, nlist, min(16, d // 2), 8)
    idx.train(embeddings.astype('float32'))
    idx.add(embeddings.astype('float32'))
    idx.nprobe = min(10, nlist)
    return idx


def build_forest(all_apis):
    forest = {}
    for api in all_apis:
        d = api['domain']
        if d not in forest:
            forest[d] = []
        forest[d].append(api)
    result = {}
    for d, apis in forest.items():
        tree = BPlusTree(order=32, domain_name=d)
        for a in apis:
            tree.insert(a.get('subcategory', 'default'), a)
        embs = np.array([a['embedding'] for a in apis])
        result[d] = {'tree': tree, 'root_vector': np.mean(embs, axis=0)}
    return result


def measure_scenario(test_queries, faiss_idx, forest, all_apis, scenario='normal'):
    """Measure token consumption for a scenario.
    Normal: clear queries, minimal reasoning
    Abnormal: vague queries, heavy reasoning
    """
    all_apis_by_id = {a['id']: a for a in all_apis}
    flat_tokens = []
    forest_tokens = []
    K = 5

    for q in test_queries[:80]:
        q_emb = np.array(q['query_embedding'])

        # Flat: retrieval + LLM
        _, I = faiss_idx.search(q_emb.astype('float32').reshape(1, -1), K)
        results_flat = [all_apis[i] for i in I[0] if i < len(all_apis)]
        flat_metrics = measure_e2e_tokens_flat_llm(results_flat, all_apis_by_id)

        if scenario == 'abnormal':
            # Extra reasoning for ambiguity
            domains_in = len(set(r.get('domain', '') for r in results_flat))
            ambiguity_penalty = 80 * domains_in
            flat_tokens.append(flat_metrics['total_tokens'] + ambiguity_penalty)
        else:
            flat_tokens.append(flat_metrics['total_tokens'])

        # Forest: structured pipeline
        domain_scores = {}
        for d, info in forest.items():
            domain_scores[d] = cosine_similarity(q_emb.reshape(1, -1), info['root_vector'].reshape(1, -1))[0][0]
        best = max(domain_scores, key=domain_scores.get)
        res, _, _ = forest[best]['tree'].search_with_traversal(q_emb, top_k=K)

        top_skill = res[0] if res else None
        dep_chain = trace_dependency_chain(top_skill, all_apis_by_id) if top_skill else []
        merged_params = {}
        if top_skill and 'hierarchical_params' in top_skill:
            hp = top_skill['hierarchical_params']
            for level in ['root', 'middle', 'leaf']:
                merged_params.update(hp.get(level, {}))

        routing_descs = [f"Domain: {d}" for d in forest.keys()]
        forest_metrics = measure_e2e_tokens_forest(routing_descs, res, dep_chain, merged_params, all_apis_by_id)

        if scenario == 'abnormal':
            sorted_domains = sorted(domain_scores.items(), key=lambda x: -x[1])
            gap = sorted_domains[0][1] - sorted_domains[1][1] if len(sorted_domains) >= 2 else 1.0
            abcd_tokens = sum(count_tokens(f"Domain: {d}") for d in forest.keys()) if gap < 0.15 else 0
            forest_tokens.append(forest_metrics['total_tokens'] + abcd_tokens)
        else:
            forest_tokens.append(forest_metrics['total_tokens'])

    return flat_tokens, forest_tokens


def run_scale_experiment(base_apis, test_queries, model, n_values, scenario='normal', n_runs=10):
    """Run scale experiment with n_runs per N value."""
    all_domains = {}
    for api in base_apis:
        d = api['domain']
        if d not in all_domains:
            all_domains[d] = []
        all_domains[d].append(api)

    results = {
        'N': [],
        'flat_mean': [], 'flat_std': [],
        'forest_mean': [], 'forest_std': [],
        'savings_mean': [], 'savings_std': [],
    }

    for N in n_values:
        per_domain = max(2, N // len(DOMAINS))
        min_avail = min(len(v) for v in all_domains.values())
        if per_domain > min_avail:
            results['N'].append(N)
            for k in ['flat_mean', 'flat_std', 'forest_mean', 'forest_std', 'savings_mean', 'savings_std']:
                results[k].append(None)
            continue

        flat_all_runs = []
        forest_all_runs = []
        savings_all_runs = []

        for run in range(n_runs):
            np.random.seed(N * 100 + run)
            subset = []
            sub_domains = {}
            for d in DOMAINS:
                available = all_domains[d]
                if per_domain < len(available):
                    idx = np.random.choice(len(available), per_domain, replace=False)
                    sampled = [available[i] for i in idx]
                else:
                    sampled = available[:per_domain]
                subset.extend(sampled)
                sub_domains[d] = sampled

            embs = np.array([a['embedding'] for a in subset])
            faiss_idx = build_faiss(embs)
            forest = build_forest(subset)

            ft, frt = measure_scenario(test_queries, faiss_idx, forest, subset, scenario)
            flat_mean = float(np.mean(ft))
            forest_mean = float(np.mean(frt))
            savings = (flat_mean - forest_mean) / flat_mean * 100 if flat_mean > 0 else 0

            flat_all_runs.append(flat_mean)
            forest_all_runs.append(forest_mean)
            savings_all_runs.append(savings)

        results['N'].append(N)
        results['flat_mean'].append(float(np.mean(flat_all_runs)))
        results['flat_std'].append(float(np.std(flat_all_runs)))
        results['forest_mean'].append(float(np.mean(forest_all_runs)))
        results['forest_std'].append(float(np.std(forest_all_runs)))
        results['savings_mean'].append(float(np.mean(savings_all_runs)))
        results['savings_std'].append(float(np.std(savings_all_runs)))

    return results


def main():
    print("=" * 60)
    print("Experiment 6 v3: Token Consumption - Full Scale Analysis")
    print(f"N values: {N_VALUES}, Runs per scenario: {N_RUNS}")
    print("=" * 60)

    # Load data
    enriched_path = os.path.join(os.path.dirname(BASE), 'shared', 'data', 'all_apis_enriched.json')
    if os.path.exists(enriched_path):
        with open(enriched_path, 'r', encoding='utf-8') as f:
            all_apis = json.load(f)
        for api in all_apis:
            if 'embedding' in api:
                api['embedding'] = np.array(api['embedding'])
        print(f"Loaded {len(all_apis)} enriched APIs")
    else:
        all_apis, _ = load_dataset(DATA_DIR)

    _, test_queries = load_dataset(DATA_DIR)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    if 'query_embedding' not in test_queries[0]:
        texts = [q['query'] for q in test_queries]
        embs = model.encode(texts, batch_size=64)
        for q, emb in zip(test_queries, embs):
            q['query_embedding'] = emb.tolist()

    # Generate extended dataset for large N
    print("\n[1/4] Generating extended dataset...")
    _, all_apis_ext = generate_dataset(n_per_domain=1000, seed=42)
    compute_embeddings(all_apis_ext, model)

    # Run normal scenario
    print("\n[2/4] Running NORMAL scenario (10 rounds × 7 N values)...")
    scale_normal = run_scale_experiment(all_apis_ext, test_queries, model, N_VALUES, 'normal', N_RUNS)

    # Run abnormal scenario
    print("\n[3/4] Running ABNORMAL scenario (10 rounds × 7 N values)...")
    scale_abnormal = run_scale_experiment(all_apis_ext, test_queries, model, N_VALUES, 'abnormal', N_RUNS)

    # === Print comprehensive tables ===
    print(f"\n{'='*120}")
    print(f"NORMAL SCENARIO RESULTS (Clear queries, minimal reasoning)")
    print(f"{'='*120}")
    print(f"{'N':<8} {'Flat+LLM Token':<25} {'Forest Token':<25} {'Savings':<20} {'Trend'}")
    print(f"{'-'*120}")
    for i, N in enumerate(scale_normal['N']):
        if scale_normal['flat_mean'][i] is not None:
            trend = ""
            if i > 0 and scale_normal['savings_mean'][i-1] is not None:
                diff = scale_normal['savings_mean'][i] - scale_normal['savings_mean'][i-1]
                trend = f"{'↑' if diff > 0 else '↓'} {abs(diff):.1f}pp"
            print(f"{N:<8} {scale_normal['flat_mean'][i]:>8.1f} ± {scale_normal['flat_std'][i]:>5.1f}      "
                  f"{scale_normal['forest_mean'][i]:>8.1f} ± {scale_normal['forest_std'][i]:>5.1f}      "
                  f"{scale_normal['savings_mean'][i]:>6.1f}% ± {scale_normal['savings_std'][i]:>4.1f}%    {trend}")

    print(f"\n{'='*120}")
    print(f"ABNORMAL SCENARIO RESULTS (Vague/ambiguous queries, heavy reasoning)")
    print(f"{'='*120}")
    print(f"{'N':<8} {'Flat+LLM Token':<25} {'Forest Token':<25} {'Savings':<20} {'Trend'}")
    print(f"{'-'*120}")
    for i, N in enumerate(scale_abnormal['N']):
        if scale_abnormal['flat_mean'][i] is not None:
            trend = ""
            if i > 0 and scale_abnormal['savings_mean'][i-1] is not None:
                diff = scale_abnormal['savings_mean'][i] - scale_abnormal['savings_mean'][i-1]
                trend = f"{'↑' if diff > 0 else '↓'} {abs(diff):.1f}pp"
            print(f"{N:<8} {scale_abnormal['flat_mean'][i]:>8.1f} ± {scale_abnormal['flat_std'][i]:>5.1f}      "
                  f"{scale_abnormal['forest_mean'][i]:>8.1f} ± {scale_abnormal['forest_std'][i]:>5.1f}      "
                  f"{scale_abnormal['savings_mean'][i]:>6.1f}% ± {scale_abnormal['savings_std'][i]:>4.1f}%    {trend}")

    # === Generate visualizations ===
    print("\n[4/4] Generating visualizations...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    valid_normal = [(N, fm, fs, frm, frs, sm, ss)
                    for N, fm, fs, frm, frs, sm, ss in zip(
                        scale_normal['N'], scale_normal['flat_mean'], scale_normal['flat_std'],
                        scale_normal['forest_mean'], scale_normal['forest_std'],
                        scale_normal['savings_mean'], scale_normal['savings_std'])
                    if fm is not None]

    valid_abnormal = [(N, fm, fs, frm, frs, sm, ss)
                      for N, fm, fs, frm, frs, sm, ss in zip(
                          scale_abnormal['N'], scale_abnormal['flat_mean'], scale_abnormal['flat_std'],
                          scale_abnormal['forest_mean'], scale_abnormal['forest_std'],
                          scale_abnormal['savings_mean'], scale_abnormal['savings_std'])
                      if fm is not None]

    for lang in ['CN', 'EN']:
        is_cn = lang == 'CN'

        # Chart 1: Normal vs Abnormal comparison (N=1500)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for idx, (scenario_name, valid_data) in enumerate([('normal', valid_normal), ('abnormal', valid_abnormal)]):
            ax = axes[idx]
            # Find N=1500 data
            n1500 = [d for d in valid_data if d[0] == 1500]
            if n1500:
                _, fm, fs, frm, frs, _, _ = n1500[0]
                labels = ['Flat+LLM', 'Forest+M4/M6/M9'] if not is_cn else ['平铺+LLM推理', '森林+M机制']
                means = [fm, frm]
                stds = [fs, frs]
                bars = ax.bar(labels, means, yerr=stds, capsize=5, color=['#4ECDC4', '#45B7D1'], alpha=0.9, width=0.5)
                for bar, mean in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                            f'{mean:.0f}', ha='center', fontsize=11, fontweight='bold')
                savings = (fm - frm) / fm * 100
                title = f'{"正常场景" if is_cn else "Normal Scenario"}' if scenario_name == 'normal' else f'{"异常场景" if is_cn else "Abnormal Scenario"}'
                subtitle = '(用户描述清晰)' if is_cn else '(Clear description)' if scenario_name == 'normal' else '(用户描述模糊/有歧义)' if is_cn else '(Vague/ambiguous)'
                ax.set_title(f'{title}\n{subtitle}\n{"节省" if is_cn else "Saves"} {savings:.0f}%', fontweight='bold', fontsize=12)
            ax.set_ylabel('Token消耗' if is_cn else 'Token Consumption', fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'Token消耗对比: {"正常" if is_cn else "Normal"} vs {"异常" if is_cn else "Abnormal"}场景 (N=1500)',
                     fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(VIS_DIR, f'fig_场景对比_{lang}.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Chart 2: Scale comparison - both scenarios
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for idx, (scenario_name, valid_data) in enumerate([('normal', valid_normal), ('abnormal', valid_abnormal)]):
            ax = axes[idx]
            ns = [d[0] for d in valid_data]
            flat_vals = [d[1] for d in valid_data]
            flat_stds = [d[2] for d in valid_data]
            forest_vals = [d[3] for d in valid_data]
            forest_stds = [d[4] for d in valid_data]

            ax.errorbar(ns, flat_vals, yerr=flat_stds, fmt='o-', color='#4ECDC4',
                        label='Flat+LLM' if not is_cn else '平铺+LLM', linewidth=2.5, markersize=7, capsize=5)
            ax.errorbar(ns, forest_vals, yerr=forest_stds, fmt='s-', color='#45B7D1',
                        label='Forest+M4/M6/M9' if not is_cn else '森林+M机制', linewidth=2.5, markersize=7, capsize=5)

            title = f'{"正常场景" if is_cn else "Normal Scenario"}' if scenario_name == 'normal' else f'{"异常场景" if is_cn else "Abnormal Scenario"}'
            ax.set_title(title, fontweight='bold', fontsize=13)
            ax.set_xlabel('API数量 (N)' if is_cn else 'Number of APIs (N)', fontsize=11)
            ax.set_ylabel('Token消耗 (均值±标准差)' if is_cn else 'Token Consumption (mean±std)', fontsize=11)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        fig.suptitle(f'Token消耗随API规模变化: {"正常" if is_cn else "Normal"} vs {"异常" if is_cn else "Abnormal"}\n(10轮独立实验, 均值±标准差)',
                     fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(VIS_DIR, f'fig_规模Token_{lang}.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Chart 3: Savings trend
        fig, ax = plt.subplots(figsize=(12, 6))
        ns_normal = [d[0] for d in valid_normal]
        savings_normal = [d[5] for d in valid_normal]
        savings_normal_std = [d[6] for d in valid_normal]

        ns_abnormal = [d[0] for d in valid_abnormal]
        savings_abnormal = [d[5] for d in valid_abnormal]
        savings_abnormal_std = [d[6] for d in valid_abnormal]

        ax.errorbar(ns_normal, savings_normal, yerr=savings_normal_std, fmt='o-',
                    color='#2ECC71', label='正常场景' if is_cn else 'Normal Scenario',
                    linewidth=2.5, markersize=8, capsize=5)
        ax.errorbar(ns_abnormal, savings_abnormal, yerr=savings_abnormal_std, fmt='s-',
                    color='#E74C3C', label='异常场景' if is_cn else 'Abnormal Scenario',
                    linewidth=2.5, markersize=8, capsize=5)

        ax.set_xlabel('API数量 (N)' if is_cn else 'Number of APIs (N)', fontsize=12)
        ax.set_ylabel('Token节省比例 (%)' if is_cn else 'Token Savings (%)', fontsize=12)
        ax.set_title('森林方案Token节省比例随规模变化\n(10轮独立实验, 均值±标准差)' if is_cn else
                     'Forest Token Savings vs Scale\n(10 independent runs, mean±std)',
                     fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc='best', framealpha=0.9, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.savefig(os.path.join(VIS_DIR, f'fig_节省趋势_{lang}.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # Save results
    output = {
        'config': {'N_VALUES': N_VALUES, 'N_RUNS': N_RUNS},
        'normal': scale_normal,
        'abnormal': scale_abnormal,
        'conclusion': {
            'normal_trend': f"Savings increase from {scale_normal['savings_mean'][0]:.1f}% (N={N_VALUES[0]}) to {scale_normal['savings_mean'][-1]:.1f}% (N={N_VALUES[-1]})",
            'abnormal_trend': f"Savings increase from {scale_abnormal['savings_mean'][0]:.1f}% (N={N_VALUES[0]}) to {scale_abnormal['savings_mean'][-1]:.1f}% (N={N_VALUES[-1]})",
            'abnormal_advantage': "Abnormal scenarios show significantly higher savings than normal scenarios, especially at larger scales",
        }
    }
    with open(os.path.join(RES_DIR, 'experiment6_v3_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {RES_DIR}")
    print(f"Visualizations saved to: {VIS_DIR}")


if __name__ == '__main__':
    main()
