"""
Experiment 2 v3: Ablation Study with REAL mechanism implementations.

FIXES from v2:
- M4: Uses actual dependency tracing on enriched data (not simulation)
- M5: Uses actual ABCD selection based on similarity gap (not simulation)
- M6: Uses actual parameter merging on conflict cases (not simulation)
- M7: Uses actual private/public skill selection (not simulation)
- M9: Measures actual token consumption from content (not hardcoded)
- M2: Unchanged - already uses real retrieval comparison

All mechanisms now operate on real data structures and produce measurable outcomes.
"""
import sys, os, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from shared.data_generator import load_dataset
from shared.bplus_tree import BPlusTree
from shared.mechanisms import (
    trace_dependency_chain, measure_chain_completeness_with_m4,
    measure_chain_completeness_without_m4,
    select_with_m5, select_without_m5, evaluate_m5_accuracy,
    merge_params_with_m6, merge_params_without_m6, evaluate_m6_resolution,
    select_skill_with_m7, select_skill_without_m7, evaluate_m7_hit_rate,
    count_tokens, measure_tokens_with_m9, measure_tokens_without_m9,
    measure_e2e_tokens_flat_llm, measure_e2e_tokens_forest
)
from shared.visualization_utils import COLORS, ABLATION_LABELS
import faiss

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), 'shared', 'data')
SHARED_DATA_DIR = os.path.join(os.path.dirname(BASE), 'shared', 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

N_RUNS = 5
DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']

ABLATION_LABELS_CN = {
    'full_system': '完整系统',
    'no_M2': '去掉M2(路由层)',
    'no_M4': '去掉M4(依赖回溯)',
    'no_M5': '去掉M5(ABCD选择)',
    'no_M6': '去掉M6(参数合并)',
    'no_M7': '去掉M7(私人遮蔽)',
    'no_M9': '去掉M9(角色降维)',
}


def load_enriched_data():
    """Load enriched API data with dependencies and hierarchical params."""
    with open(os.path.join(SHARED_DATA_DIR, 'all_apis_enriched.json'), 'r', encoding='utf-8') as f:
        all_apis = json.load(f)
    # Convert embedding lists to numpy arrays
    for api in all_apis:
        if 'embedding' in api:
            api['embedding'] = np.array(api['embedding'])
    return all_apis


def load_aux_data():
    """Load auxiliary data files (private skills, param cases, M7 cases)."""
    aux = {}
    for fname in ['private_skills.json', 'param_conflict_cases.json', 'm7_test_cases.json']:
        fpath = os.path.join(SHARED_DATA_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                aux[fname.replace('.json', '')] = json.load(f)
    return aux


def build_forest(all_apis, domains_dict):
    forest = {}
    for domain, apis in domains_dict.items():
        tree = BPlusTree(order=32, domain_name=domain)
        for api in apis:
            tree.insert(api.get('subcategory', 'default'), api)
        embs = np.array([a['embedding'] for a in apis])
        root_vec = np.mean(embs, axis=0)
        forest[domain] = {'tree': tree, 'root_vector': root_vec}
    return forest


def build_single_tree(all_apis):
    tree = BPlusTree(order=32, domain_name='single')
    for api in all_apis:
        tree.insert(api.get('subcategory', 'default'), api)
    return tree


def build_faiss_index(all_apis):
    embeddings = np.array([a['embedding'] for a in all_apis])
    d = embeddings.shape[1]
    n = len(embeddings)
    if n < 200:
        idx = faiss.IndexFlatL2(d)
        idx.add(embeddings.astype('float32'))
    else:
        nlist = min(50, max(1, n // 30))
        quantizer = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFPQ(quantizer, d, nlist, min(16, d // 2), 8)
        idx.train(embeddings.astype('float32'))
        idx.add(embeddings.astype('float32'))
        idx.nprobe = min(10, nlist)
    return idx


def search_flat(query_emb, faiss_idx, all_apis, top_k=10):
    start = time.perf_counter()
    D, I = faiss_idx.search(query_emb.astype('float32').reshape(1, -1), top_k)
    latency = (time.perf_counter() - start) * 1000
    results = [all_apis[i] for i in I[0] if 0 <= i < len(all_apis)]
    return results, latency


def search_forest(query_emb, forest, top_k=10):
    start = time.perf_counter()
    domain_scores = {}
    for domain, info in forest.items():
        sim = cosine_similarity(query_emb.reshape(1,-1), info['root_vector'].reshape(1,-1))[0][0]
        domain_scores[domain] = sim
    best = max(domain_scores, key=domain_scores.get)
    results, _, _ = forest[best]['tree'].search_with_traversal(query_emb, top_k=top_k)
    latency = (time.perf_counter() - start) * 1000
    return results, latency, best


def search_single_tree(query_emb, tree, top_k=10):
    start = time.perf_counter()
    results, _, _ = tree.search_with_traversal(query_emb, top_k=top_k)
    latency = (time.perf_counter() - start) * 1000
    return results, latency


# ============================================================
# M2 Ablation: Forest vs Single Tree vs Flat ANN (REAL)
# ============================================================
def test_M2_ablation(test_queries, all_apis, domains_dict, forest, single_tree, faiss_idx, n_runs=5):
    """M2: Real retrieval comparison - unchanged from v2 (already real)."""
    print("Running M2 ablation: Forest vs Single Tree vs Flat ANN...")
    results = {
        'forest': {'acc@1': [], 'acc@10': [], 'purity@10': [], 'latency': [], 'tokens': [], 'routing_acc': []},
        'single_tree': {'acc@1': [], 'acc@10': [], 'purity@10': [], 'latency': [], 'tokens': []},
        'flat': {'acc@1': [], 'acc@10': [], 'purity@10': [], 'latency': [], 'tokens': []}
    }

    for run in range(n_runs):
        run_results = {
            'forest': {'acc@1': [], 'acc@10': [], 'purity@10': [], 'latency': [], 'tokens': [], 'routing_acc': []},
            'single_tree': {'acc@1': [], 'acc@10': [], 'purity@10': [], 'latency': [], 'tokens': []},
            'flat': {'acc@1': [], 'acc@10': [], 'purity@10': [], 'latency': [], 'tokens': []}
        }

        for q in test_queries:
            q_emb = np.array(q['query_embedding'])
            cdomain = q['correct_domain']

            # Forest
            res_f, lat_f, routed = search_forest(q_emb, forest, top_k=10)
            forest_domains = [r.get('domain', '') for r in res_f]
            run_results['forest']['acc@1'].append(1 if res_f and res_f[0].get('domain') == cdomain else 0)
            run_results['forest']['acc@10'].append(1 if cdomain in forest_domains else 0)
            purity = sum(1 for d in forest_domains if d == cdomain) / len(forest_domains) if forest_domains else 0
            run_results['forest']['purity@10'].append(purity)
            run_results['forest']['latency'].append(lat_f)
            run_results['forest']['tokens'].append(sum(count_tokens(r.get('description', '')) for r in res_f[:5]))
            run_results['forest']['routing_acc'].append(1 if routed == cdomain else 0)

            # Single Tree
            res_s, lat_s = search_single_tree(q_emb, single_tree, top_k=10)
            single_domains = [r.get('domain', '') for r in res_s]
            run_results['single_tree']['acc@1'].append(1 if res_s and res_s[0].get('domain') == cdomain else 0)
            run_results['single_tree']['acc@10'].append(1 if cdomain in single_domains else 0)
            purity_s = sum(1 for d in single_domains if d == cdomain) / len(single_domains) if single_domains else 0
            run_results['single_tree']['purity@10'].append(purity_s)
            run_results['single_tree']['latency'].append(lat_s)
            run_results['single_tree']['tokens'].append(sum(count_tokens(r.get('description', '')) for r in res_s[:5]))

            # Flat ANN
            res_flat, lat_flat = search_flat(q_emb, faiss_idx, all_apis, top_k=10)
            flat_domains = [r.get('domain', '') for r in res_flat]
            run_results['flat']['acc@1'].append(1 if res_flat and res_flat[0].get('domain') == cdomain else 0)
            run_results['flat']['acc@10'].append(1 if cdomain in flat_domains else 0)
            purity_flat = sum(1 for d in flat_domains if d == cdomain) / len(flat_domains) if flat_domains else 0
            run_results['flat']['purity@10'].append(purity_flat)
            run_results['flat']['latency'].append(lat_flat)
            run_results['flat']['tokens'].append(sum(count_tokens(r.get('description', '')) for r in res_flat[:5]))

        for method in ['forest', 'single_tree', 'flat']:
            for metric in ['acc@1', 'acc@10', 'purity@10', 'latency', 'tokens']:
                results[method][metric].append(np.mean(run_results[method][metric]))
            if method == 'forest':
                results[method]['routing_acc'].append(np.mean(run_results[method]['routing_acc']))

    return results


# ============================================================
# M4 Ablation: REAL dependency tracing (not simulation)
# ============================================================
def test_M4_ablation(test_queries, forest, all_apis_by_id, n_runs=5):
    """
    M4: Real dependency chain tracing.
    With M4: System traces full dependency chain using `requires` field.
    Without M4: Only leaf skill returned, LLM must guess dependencies.
    """
    print("Running M4 ablation: Real dependency tracing...")
    results = {'with_M4': [], 'without_M4': []}

    for run in range(n_runs):
        m4_completeness = []
        no_m4_completeness = []

        for q in test_queries:
            q_emb = np.array(q['query_embedding'])

            # Route to domain and get top-1 skill
            domain_scores = {}
            for domain, info in forest.items():
                sim = cosine_similarity(q_emb.reshape(1, -1), info['root_vector'].reshape(1, -1))[0][0]
                domain_scores[domain] = sim
            best_domain = max(domain_scores, key=domain_scores.get)
            res, _, _ = forest[best_domain]['tree'].search_with_traversal(q_emb, top_k=1)

            if not res:
                continue

            top_skill = res[0]

            # With M4: trace full dependency chain
            chain = trace_dependency_chain(top_skill, all_apis_by_id)
            all_reqs = set()
            _collect_all_requires(top_skill, all_apis_by_id, all_reqs, set())

            if all_reqs:
                chain_ids = set(s['id'] for s in chain)
                completeness = len(all_reqs & chain_ids) / len(all_reqs)
            else:
                completeness = 1.0  # No deps = trivially complete
            m4_completeness.append(completeness)

            # Without M4: only leaf skill, LLM guesses deps
            no_m4_completeness.append(measure_chain_completeness_without_m4(top_skill, all_apis_by_id))

        results['with_M4'].append(float(np.mean(m4_completeness)) if m4_completeness else 0)
        results['without_M4'].append(float(np.mean(no_m4_completeness)) if no_m4_completeness else 0)

    return results


def _collect_all_requires(skill, all_apis_by_id, collected, visited):
    """Recursively collect all transitive requirements."""
    skill_id = skill.get('id', '')
    if skill_id in visited:
        return
    visited.add(skill_id)
    for req_id in skill.get('requires', []):
        if req_id not in collected:
            collected.add(req_id)
            if req_id in all_apis_by_id:
                _collect_all_requires(all_apis_by_id[req_id], all_apis_by_id, collected, visited)


# ============================================================
# M5 Ablation: REAL ABCD selection (not simulation)
# ============================================================
def test_M5_ablation(test_queries, forest, delta=0.15, n_runs=5):
    """
    M5: Real ABCD selection based on similarity gap.
    With M5: When gap < delta, present candidates (user picks correctly).
    Without M5: Always pick Top-1 (may be wrong for ambiguous).
    """
    print("Running M5 ablation: Real ABCD selection...")
    return evaluate_m5_accuracy(test_queries, forest, delta=delta, top_k=5, n_runs=n_runs)


# ============================================================
# M6 Ablation: REAL parameter merging (not simulation)
# ============================================================
def test_M6_ablation(param_cases, n_runs=5):
    """
    M6: Real parameter merging on conflict cases.
    With M6: Merge by priority (user > leaf > middle > root) -> all conflicts resolved.
    Without M6: Random selection from levels -> some conflicts unresolved.
    """
    print("Running M6 ablation: Real parameter merging...")
    return evaluate_m6_resolution(param_cases, n_runs=n_runs)


# ============================================================
# M7 Ablation: REAL private skill masking (not simulation)
# ============================================================
def test_M7_ablation(m7_cases, n_runs=5):
    """
    M7: Real private skill masking.
    With M7: Private skills take priority by path.
    Without M7: Private and public compete by similarity.
    """
    print("Running M7 ablation: Real private skill masking...")
    return evaluate_m7_hit_rate(m7_cases, n_runs=n_runs, top_k=5)


# ============================================================
# M9 Ablation: REAL token measurement (not hardcoded)
# ============================================================
def test_M9_ablation(test_queries, forest, all_apis_by_id, n_runs=5):
    """
    M9: Real token measurement.
    With M9: Structured pipeline (routing + retrieval + deps + params + confirm).
    Without M9: Raw candidates + LLM reasoning.
    """
    print("Running M9 ablation: Real token measurement...")
    results = {'with_M9': [], 'without_M9': []}

    domain_names = list(forest.keys())

    for run in range(n_runs):
        m9_tokens = []
        no_m9_tokens = []

        for q in test_queries[:80]:
            q_emb = np.array(q['query_embedding'])

            # Route to domain
            domain_scores = {}
            for domain, info in forest.items():
                sim = cosine_similarity(q_emb.reshape(1, -1), info['root_vector'].reshape(1, -1))[0][0]
                domain_scores[domain] = sim
            best_domain = max(domain_scores, key=domain_scores.get)

            # Get retrieval results
            results_forest, _, _ = forest[best_domain]['tree'].search_with_traversal(q_emb, top_k=5)

            if not results_forest:
                continue

            top_skill = results_forest[0]

            # With M9: structured pipeline
            routing_descs = [f"Domain: {d}" for d in domain_names]
            dep_chain = trace_dependency_chain(top_skill, all_apis_by_id)
            merged_params = top_skill.get('hierarchical_params', {}).get('leaf', {})
            # Merge all levels
            hp = top_skill.get('hierarchical_params', {})
            merged = {}
            for level in ['root', 'middle', 'leaf']:
                merged.update(hp.get(level, {}))

            m9_tok = measure_tokens_with_m9(routing_descs, results_forest, dep_chain, merged)
            m9_tokens.append(m9_tok)

            # Without M9: raw candidates + LLM reasoning
            no_m9_tok = measure_tokens_without_m9(results_forest, results_forest)
            no_m9_tokens.append(no_m9_tok)

        results['with_M9'].append(float(np.mean(m9_tokens)) if m9_tokens else 0)
        results['without_M9'].append(float(np.mean(no_m9_tokens)) if no_m9_tokens else 0)

    return results


# ============================================================
# Visualization
# ============================================================
def generate_visualizations(m2, m4, m5, m6, m7, m9):
    print("\nGenerating visualizations...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # --- Fig 1: M2 Comparison (Forest vs Single Tree vs Flat ANN) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Accuracy comparison
    ax = axes[0]
    methods = ['Forest\n(5 trees)', 'Single B+ Tree', 'Flat ANN\n(FAISS)']
    accs = [np.mean(m2['forest']['acc@1']), np.mean(m2['single_tree']['acc@1']), np.mean(m2['flat']['acc@1'])]
    acc_stds = [np.std(m2['forest']['acc@1']), np.std(m2['single_tree']['acc@1']), np.std(m2['flat']['acc@1'])]
    colors_m2 = [COLORS['forest'], COLORS['single_tree'], COLORS['flat']]
    bars = ax.bar(methods, accs, yerr=acc_stds, capsize=5, color=colors_m2, alpha=0.9, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
    routing_acc = np.mean(m2['forest']['routing_acc'])
    ax.text(0, accs[0] - 0.05, f'Routing: {routing_acc:.1%}', ha='center', fontsize=9, color='white', fontweight='bold')
    ax.set_ylabel('Accuracy@1', fontsize=12)
    ax.set_title('M2: Retrieval Accuracy Comparison', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)

    # Right: Latency comparison
    ax = axes[1]
    lats = [np.mean(m2['forest']['latency']), np.mean(m2['single_tree']['latency']), np.mean(m2['flat']['latency'])]
    lat_stds = [np.std(m2['forest']['latency']), np.std(m2['single_tree']['latency']), np.std(m2['flat']['latency'])]
    bars = ax.bar(methods, lats, yerr=lat_stds, capsize=5, color=colors_m2, alpha=0.9, width=0.5)
    for bar, lat in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{lat:.2f}ms', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('M2: Retrieval Latency Comparison', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    fig.savefig(os.path.join(VIS_DIR, 'fig_M2_routing_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(VIS_DIR, 'fig_M2_路由对比_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # --- Fig 2: Ablation Overview (grouped bar) ---
    fig, ax = plt.subplots(figsize=(14, 7))
    configs = ['full_system', 'no_M4', 'no_M5', 'no_M6', 'no_M7', 'no_M9']
    metrics_cn = ['执行链完整率', '冲突消解率', '任务完成率', '私人技能命中率']
    metrics_en = ['Chain\nCompleteness', 'Conflict\nResolution', 'Task\nCompletion', 'Private Skill\nHit Rate']

    # Build metric data from real results
    metric_data = {
        '执行链完整率': {
            'full_system': np.mean(m4['with_M4']), 'no_M4': np.mean(m4['without_M4']),
            'no_M5': np.mean(m4['with_M4']), 'no_M6': np.mean(m4['with_M4']),
            'no_M7': np.mean(m4['with_M4']), 'no_M9': np.mean(m4['with_M4'])
        },
        '冲突消解率': {
            'full_system': np.mean(m6['with_M6']), 'no_M4': np.mean(m6['with_M6']),
            'no_M5': np.mean(m6['with_M6']), 'no_M6': np.mean(m6['without_M6']),
            'no_M7': np.mean(m6['with_M6']), 'no_M9': np.mean(m6['with_M6'])
        },
        '任务完成率': {
            'full_system': np.mean(m5['with_M5']), 'no_M4': np.mean(m5['with_M5']),
            'no_M5': np.mean(m5['without_M5']), 'no_M6': np.mean(m5['with_M5']),
            'no_M7': np.mean(m5['with_M5']), 'no_M9': np.mean(m5['with_M5'])
        },
        '私人技能命中率': {
            'full_system': np.mean(m7['with_M7']), 'no_M4': np.mean(m7['with_M7']),
            'no_M5': np.mean(m7['with_M7']), 'no_M6': np.mean(m7['with_M7']),
            'no_M7': np.mean(m7['without_M7']), 'no_M9': np.mean(m7['with_M7'])
        },
    }

    x = np.arange(len(metrics_en))
    width = 0.12
    for i, c in enumerate(configs):
        vals = [metric_data[m][c] for m in metrics_cn]
        color = COLORS.get(c, '#95A5A6')
        label = ABLATION_LABELS_CN.get(c, c)
        ax.bar(x + i * width, vals, width, color=color, label=label, alpha=0.9)

    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(metrics_en, fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study: Individual Mechanism Contributions (Real Measurements)',
                 fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.15)
    fig.savefig(os.path.join(VIS_DIR, 'fig_ablation_overview_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(VIS_DIR, 'fig_消融总览_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # --- Fig 3: M9 Token Comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Full System\n(with M9)', 'Without M9\n(Role Reduction)']
    tok_means = [np.mean(m9['with_M9']), np.mean(m9['without_M9'])]
    tok_stds = [np.std(m9['with_M9']), np.std(m9['without_M9'])]
    bars = ax.bar(labels, tok_means, yerr=tok_stds, capsize=5,
                  color=[COLORS['full_system'], COLORS['no_M9']], alpha=0.9, width=0.5)
    for bar, mean in zip(bars, tok_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                f'{mean:.0f}', ha='center', fontsize=12, fontweight='bold')
    increase = (tok_means[1] - tok_means[0]) / tok_means[0] * 100 if tok_means[0] > 0 else 0
    ax.text(0.5, 0.95, f'Without M9: Token +{increase:.0f}%', transform=ax.transAxes,
            ha='center', fontsize=14, fontweight='bold', color='red')
    ax.set_ylabel('Token Consumption', fontsize=12)
    ax.set_title('M9 Ablation: Token Efficiency of Role Reduction (Measured)',
                 fontweight='bold', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.savefig(os.path.join(VIS_DIR, 'fig_M9_token_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(VIS_DIR, 'fig_M9_Token对比_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # --- Fig 4: Per-mechanism impact (delta from full system) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    mechanisms = ['M4\n(Dependency)', 'M5\n(ABCD)', 'M6\n(Param Merge)', 'M7\n(Private)', 'M9\n(Role Red.)']
    deltas = [
        np.mean(m4['with_M4']) - np.mean(m4['without_M4']),  # Chain completeness
        np.mean(m5['with_M5']) - np.mean(m5['without_M5']),  # Task completion
        np.mean(m6['with_M6']) - np.mean(m6['without_M6']),  # Conflict resolution
        np.mean(m7['with_M7']) - np.mean(m7['without_M7']),  # Private hit rate
        0,  # M9 is token (different scale, show separately)
    ]
    # Normalize M9 delta as percentage (negative = good for M9)
    m9_delta_pct = -(np.mean(m9['without_M9']) - np.mean(m9['with_M9'])) / np.mean(m9['without_M9'])
    deltas[4] = m9_delta_pct  # Positive = improvement

    colors_bar = [COLORS['no_M4'], COLORS['no_M5'], COLORS['no_M6'], COLORS['no_M7'], COLORS['no_M9']]
    bars = ax.bar(mechanisms, deltas, color=colors_bar, alpha=0.9, width=0.5)
    for bar, d in zip(bars, deltas):
        label = f'+{d:.3f}' if d > 0 else f'{d:.3f}'
        if d > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    label, ha='center', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 0.03,
                    label, ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Improvement (full - ablated)', fontsize=12)
    ax.set_title('Per-Mechanism Impact: Improvement Lost When Ablated', fontweight='bold', fontsize=14, pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=0.8)
    fig.savefig(os.path.join(VIS_DIR, 'fig_ablation_impact_EN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(VIS_DIR, 'fig_消融影响_CN.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved visualization files to {VIS_DIR}")

    return {
        'M2': {
            'forest_acc': float(np.mean(m2['forest']['acc@1'])),
            'single_tree_acc': float(np.mean(m2['single_tree']['acc@1'])),
            'flat_acc': float(np.mean(m2['flat']['acc@1'])),
            'forest_routing_acc': float(np.mean(m2['forest']['routing_acc'])),
        },
        'M4': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in m4.items()},
        'M5': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in m5.items()},
        'M6': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in m6.items()},
        'M7': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in m7.items()},
        'M9': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in m9.items()},
        'M2_full': {
            method: {metric: {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
                      for metric, vals in m2[method].items()}
            for method in m2
        }
    }


def main():
    print("=" * 60)
    print("Experiment 2 v3: Ablation Study (REAL Mechanism Implementations)")
    print("=" * 60)

    print("\n[1/8] Loading enriched data...")
    all_apis = load_enriched_data()
    _, test_queries = load_dataset(DATA_DIR)
    aux = load_aux_data()

    # Add query embeddings if missing
    print("[2/8] Computing query embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    if 'query_embedding' not in test_queries[0]:
        texts = [q['query'] for q in test_queries]
        embs = model.encode(texts, batch_size=64)
        for q, emb in zip(test_queries, embs):
            q['query_embedding'] = emb.tolist()

    domains_dict = {d: [] for d in DOMAINS}
    for api in all_apis:
        if api['domain'] in domains_dict:
            domains_dict[api['domain']].append(api)

    all_apis_by_id = {a['id']: a for a in all_apis}

    print("[3/8] Building indices...")
    forest = build_forest(all_apis, domains_dict)
    single_tree = build_single_tree(all_apis)
    faiss_idx = build_faiss_index(all_apis)

    print("[4/8] Testing M2 (routing) - Forest vs Single Tree vs Flat ANN...")
    m2 = test_M2_ablation(test_queries, all_apis, domains_dict, forest, single_tree, faiss_idx, n_runs=N_RUNS)
    print(f"  Forest Acc@1: {np.mean(m2['forest']['acc@1']):.3f}")
    print(f"  Single Tree Acc@1: {np.mean(m2['single_tree']['acc@1']):.3f}")
    print(f"  Flat ANN Acc@1: {np.mean(m2['flat']['acc@1']):.3f}")

    print("\n[5/8] Testing M4 (dependency tracing) - REAL...")
    m4 = test_M4_ablation(test_queries, forest, all_apis_by_id, n_runs=N_RUNS)
    print(f"  With M4: {np.mean(m4['with_M4']):.3f}")
    print(f"  Without M4: {np.mean(m4['without_M4']):.3f}")

    print("\n[6/8] Testing M5 (ABCD selection) - REAL...")
    m5 = test_M5_ablation(test_queries, forest, delta=0.15, n_runs=N_RUNS)
    print(f"  With M5: {np.mean(m5['with_M5']):.3f}")
    print(f"  Without M5: {np.mean(m5['without_M5']):.3f}")

    print("\n[7a/8] Testing M6 (parameter merging) - REAL...")
    param_cases = aux.get('param_conflict_cases', [])
    m6 = test_M6_ablation(param_cases, n_runs=N_RUNS)
    print(f"  With M6: {np.mean(m6['with_M6']):.3f}")
    print(f"  Without M6: {np.mean(m6['without_M6']):.3f}")

    print("\n[7b/8] Testing M7 (private masking) - REAL...")
    m7_cases = aux.get('m7_test_cases', [])
    # Convert embedding lists to arrays
    for case in m7_cases:
        case['query_embedding'] = np.array(case['query_embedding'])
        for s in case.get('private_skills', []) + case.get('public_skills', []):
            if 'embedding' in s:
                s['embedding'] = np.array(s['embedding'])
    m7 = test_M7_ablation(m7_cases, n_runs=N_RUNS)
    print(f"  With M7: {np.mean(m7['with_M7']):.3f}")
    print(f"  Without M7: {np.mean(m7['without_M7']):.3f}")

    print("\n[7c/8] Testing M9 (token measurement) - REAL...")
    m9 = test_M9_ablation(test_queries, forest, all_apis_by_id, n_runs=N_RUNS)
    print(f"  With M9: {np.mean(m9['with_M9']):.0f} tokens")
    print(f"  Without M9: {np.mean(m9['without_M9']):.0f} tokens")

    print("\n[8/8] Generating visualizations...")
    summary = generate_visualizations(m2, m4, m5, m6, m7, m9)

    with open(os.path.join(RES_DIR, 'experiment2_v3_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY (Real Measurements)")
    print("=" * 60)
    print(f"\n{'Mechanism':<25} {'With':<12} {'Without':<12} {'Delta':<12}")
    print("-" * 60)
    print(f"{'M4 (Dependency)':<25} {np.mean(m4['with_M4']):<12.3f} {np.mean(m4['without_M4']):<12.3f} {np.mean(m4['with_M4'])-np.mean(m4['without_M4']):<+12.3f}")
    print(f"{'M5 (ABCD)':<25} {np.mean(m5['with_M5']):<12.3f} {np.mean(m5['without_M5']):<12.3f} {np.mean(m5['with_M5'])-np.mean(m5['without_M5']):<+12.3f}")
    print(f"{'M6 (Param Merge)':<25} {np.mean(m6['with_M6']):<12.3f} {np.mean(m6['without_M6']):<12.3f} {np.mean(m6['with_M6'])-np.mean(m6['without_M6']):<+12.3f}")
    print(f"{'M7 (Private)':<25} {np.mean(m7['with_M7']):<12.3f} {np.mean(m7['without_M7']):<12.3f} {np.mean(m7['with_M7'])-np.mean(m7['without_M7']):<+12.3f}")
    print(f"{'M9 (Token)':<25} {np.mean(m9['with_M9']):<12.0f} {np.mean(m9['without_M9']):<12.0f} {np.mean(m9['with_M9'])-np.mean(m9['without_M9']):<+12.0f}")

    print(f"\nResults: {RES_DIR}")
    print(f"Visualizations: {VIS_DIR}")


if __name__ == '__main__':
    main()
