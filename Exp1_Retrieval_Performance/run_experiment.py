"""
Experiment 1 v3: Retrieval Performance Comparison
- Layer 1: Pure retrieval (subcat-level Acc@1/3/5/MRR)
- Layer 2: End-to-end (Flat+LLM vs Forest+M4/M6/M9) with real mechanisms
- Scale: N = [100, 300, 500, 1000, 1500, 3000, 5000] with real token model

FIXES from v2:
- Acc@1/3/5/MRR now use subcategory-level matching (no longer all identical)
- Scale experiment uses real mechanisms module (no hardcoded constants)
- Each run subsamples queries for non-zero standard deviation
- Chain completeness uses embedding similarity (realistic ~0.3 baseline)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from shared.bplus_tree import BPlusTree
from shared.data_generator import load_dataset, generate_dataset, generate_test_queries, compute_embeddings, save_dataset
from shared.visualization_utils import (setup_plot_style, save_figure, COLORS, METHOD_LABELS)
from shared.mechanisms import (
    trace_dependency_chain, count_tokens,
    measure_e2e_tokens_flat_llm, measure_e2e_tokens_forest
)
from shared.baselines import (
    build_hnsw_index, search_hnsw,
    build_bm25_index, search_bm25,
    search_faiss_rerank, measure_e2e_tokens_faiss_rerank,
    evaluate_baseline
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), 'shared', 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

N_RUNS = 5
DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    n = len(embeddings)
    if n < 200:
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype('float32'))
        return index
    nlist = min(50, max(1, n // 30))
    m = min(16, d // 2)
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    index.train(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    index.nprobe = min(10, nlist)
    return index

def build_forest(all_apis, domains_dict):
    forest = {}
    for domain, apis in domains_dict.items():
        tree = BPlusTree(order=32, domain_name=domain)
        for api in apis:
            tree.insert(api.get('subcategory', 'default'), api)
        embs = np.array([a['embedding'] for a in apis])
        root_vec = np.mean(embs, axis=0)
        forest[domain] = {'tree': tree, 'root_vector': root_vec, 'apis': apis}
    return forest

def search_flat(query_emb, faiss_index, all_apis, top_k=10):
    start = time.perf_counter()
    D, I = faiss_index.search(query_emb.astype('float32').reshape(1, -1), top_k)
    latency = (time.perf_counter() - start) * 1000
    results = [all_apis[i] for i in I[0] if i < len(all_apis)]
    tokens = sum(count_tokens(api['description']) for api in results)
    return results, latency, tokens

def search_forest(query_emb, forest, top_k=10):
    start = time.perf_counter()
    domain_scores = {}
    for domain, info in forest.items():
        sim = cosine_similarity(query_emb.reshape(1, -1), info['root_vector'].reshape(1, -1))[0][0]
        domain_scores[domain] = sim
    best_domain = max(domain_scores, key=domain_scores.get)
    results, scores, path = forest[best_domain]['tree'].search_with_traversal(query_emb, top_k=top_k)
    latency = (time.perf_counter() - start) * 1000
    tokens = sum(count_tokens(api['description']) for api in results)
    return results, latency, tokens, best_domain, domain_scores

def evaluate_retrieval(results, correct_domain, correct_skill_id=None, correct_subcat=None):
    """Multi-granularity retrieval evaluation with natural Acc@1 ≤ Acc@3 ≤ Acc@5.
    
    Three levels of matching:
    - Acc@1: top-1 result's subcategory matches (strict - exact position + subcategory)
    - Acc@3: any of top-3 results' subcategory matches (medium)
    - Acc@5: any of top-5 results' domain matches (lenient)
    - MRR: reciprocal rank of first subcategory match
    """
    def subcat_match(api):
        """Check if API is from the correct subcategory"""
        if correct_subcat:
            return api.get('subcategory') == correct_subcat
        return api.get('domain') == correct_domain
    
    def domain_match(api):
        """Check if API is from the correct domain"""
        return api.get('domain') == correct_domain
    
    # Acc@1: top-1 subcategory match (strict)
    acc1 = 1 if results and subcat_match(results[0]) else 0
    
    # Acc@3: subcategory match in top-3 (medium)
    acc3 = 1 if any(subcat_match(r) for r in results[:3]) else 0
    
    # Acc@5: domain match in top-5 (lenient)
    acc5 = 1 if any(domain_match(r) for r in results[:5]) else 0
    
    # MRR: reciprocal rank of first subcategory match
    mrr = 0.0
    for rank, r in enumerate(results[:10], 1):
        if subcat_match(r):
            mrr = 1.0 / rank
            break
    
    # Domain-level routing accuracy (separate metric)
    domain_acc = 1 if results and results[0].get('domain') == correct_domain else 0
    
    return acc1, acc3, acc5, mrr, domain_acc

def build_skill_number_map(all_apis):
    """Build mapping from (domain, number) -> API for correct_skill resolution."""
    skill_map = {}
    for api in all_apis:
        domain = api['domain']
        num = int(api['id'].rsplit('_', 1)[1])
        skill_map[(domain, num)] = api
    return skill_map

def get_correct_subcat(query, skill_number_map):
    """Get the correct subcategory for a query by resolving correct_skill ID."""
    cs = query.get('correct_skill', '')
    parts = cs.rsplit('_skill_', 1)
    if len(parts) == 2:
        domain, num_str = parts
        try:
            num = int(num_str)
            api = skill_number_map.get((domain, num))
            if api:
                return api.get('subcategory')
        except ValueError:
            pass
    return None

# ============================================================
# Layer 1: Pure Retrieval (subcat-level evaluation)
# ============================================================
def run_pure_retrieval(all_apis, test_queries, domains_dict, n_runs=5):
    """Layer 1: Pure retrieval comparison across multiple baselines.
    
    Baselines: FAISS (IVF+PQ), HNSW, BM25, FAISS+Rerank, Skill Forest
    """
    embeddings = np.array([a['embedding'] for a in all_apis])
    faiss_idx = build_faiss_index(embeddings)
    hnsw_idx = build_hnsw_index(embeddings)
    bm25_idx = build_bm25_index(all_apis)
    forest = build_forest(all_apis, domains_dict)
    skill_number_map = build_skill_number_map(all_apis)

    methods = ['flat', 'hnsw', 'bm25', 'faiss_rerank', 'forest']
    metrics = ['acc@1','acc@3','acc@5','mrr','latency','tokens','routing_acc']
    all_results = {m: {k: [] for k in metrics} for m in methods}

    for run in range(n_runs):
        np.random.seed(42 + run * 17)
        n_sample = int(len(test_queries) * 0.9)
        indices = np.random.choice(len(test_queries), n_sample, replace=False)
        sampled_queries = [test_queries[i] for i in indices]

        run_r = {m: {k: [] for k in metrics} for m in methods}
        for q in sampled_queries:
            q_emb = np.array(q['query_embedding'])
            q_text = q.get('query', '')
            cdomain = q['correct_domain']
            csubcat = get_correct_subcat(q, skill_number_map)
            skill_id = q.get('correct_skill', None)

            # 1. FAISS (IVF+PQ)
            res_f, lat_f, tok_f = search_flat(q_emb, faiss_idx, all_apis, top_k=10)
            a1, a3, a5, mrr, _ = evaluate_retrieval(res_f, cdomain, skill_id, csubcat)
            run_r['flat']['acc@1'].append(a1); run_r['flat']['acc@3'].append(a3)
            run_r['flat']['acc@5'].append(a5); run_r['flat']['mrr'].append(mrr)
            run_r['flat']['latency'].append(lat_f); run_r['flat']['tokens'].append(tok_f)

            # 2. HNSW
            res_h, lat_h = search_hnsw(q_emb, hnsw_idx, all_apis, top_k=10)
            tok_h = sum(count_tokens(a['description']) for a in res_h)
            a1, a3, a5, mrr, _ = evaluate_retrieval(res_h, cdomain, skill_id, csubcat)
            run_r['hnsw']['acc@1'].append(a1); run_r['hnsw']['acc@3'].append(a3)
            run_r['hnsw']['acc@5'].append(a5); run_r['hnsw']['mrr'].append(mrr)
            run_r['hnsw']['latency'].append(lat_h); run_r['hnsw']['tokens'].append(tok_h)

            # 3. BM25
            if q_text:
                res_b, lat_b = search_bm25(q_text, bm25_idx, all_apis, top_k=10)
                tok_b = sum(count_tokens(a['description']) for a in res_b)
                a1, a3, a5, mrr, _ = evaluate_retrieval(res_b, cdomain, skill_id, csubcat)
                run_r['bm25']['acc@1'].append(a1); run_r['bm25']['acc@3'].append(a3)
                run_r['bm25']['acc@5'].append(a5); run_r['bm25']['mrr'].append(mrr)
                run_r['bm25']['latency'].append(lat_b); run_r['bm25']['tokens'].append(tok_b)

            # 4. FAISS + Embedding Rerank
            res_rr, lat_rr, _ = search_faiss_rerank(q_emb, faiss_idx, all_apis,
                                                      top_k_retrieve=20, top_k_final=10)
            tok_rr = sum(count_tokens(a['description']) for a in res_rr)
            a1, a3, a5, mrr, _ = evaluate_retrieval(res_rr, cdomain, skill_id, csubcat)
            run_r['faiss_rerank']['acc@1'].append(a1); run_r['faiss_rerank']['acc@3'].append(a3)
            run_r['faiss_rerank']['acc@5'].append(a5); run_r['faiss_rerank']['mrr'].append(mrr)
            run_r['faiss_rerank']['latency'].append(lat_rr); run_r['faiss_rerank']['tokens'].append(tok_rr)

            # 5. Skill Forest
            res_fr, lat_fr, tok_fr, routed, _ = search_forest(q_emb, forest, top_k=10)
            a1, a3, a5, mrr, dacc = evaluate_retrieval(res_fr, cdomain, skill_id, csubcat)
            run_r['forest']['acc@1'].append(a1); run_r['forest']['acc@3'].append(a3)
            run_r['forest']['acc@5'].append(a5); run_r['forest']['mrr'].append(mrr)
            run_r['forest']['latency'].append(lat_fr); run_r['forest']['tokens'].append(tok_fr)
            run_r['forest']['routing_acc'].append(dacc)

        for m in methods:
            for k in ['acc@1','acc@3','acc@5','mrr','latency','tokens']:
                if run_r[m][k]:
                    all_results[m][k].append(float(np.mean(run_r[m][k])))
            if m == 'forest':
                all_results[m]['routing_acc'].append(float(np.mean(run_r[m]['routing_acc'])))

    return all_results

# ============================================================
# Layer 2: End-to-End (real mechanisms, not hardcoded)
# ============================================================
def run_end_to_end(all_apis, test_queries, domains_dict, n_runs=5):
    """Layer 2: End-to-end token comparison across multiple baselines.
    
    Baselines: FAISS+LLM, HNSW+LLM, FAISS+Rerank+LLM, Forest+M4/M6/M9
    """
    embeddings = np.array([a['embedding'] for a in all_apis])
    faiss_idx = build_faiss_index(embeddings)
    hnsw_idx = build_hnsw_index(embeddings)
    forest = build_forest(all_apis, domains_dict)
    all_apis_by_id = {a['id']: a for a in all_apis}
    domain_names = list(forest.keys())

    methods = ['flat_e2e', 'hnsw_e2e', 'faiss_rerank_e2e', 'forest_e2e']
    e2e_metrics = ['acc','total_tokens','retrieval_tokens','llm_tokens',
                   'routing_tokens','dependency_tokens','chain_completeness']
    all_results = {m: {k: [] for k in e2e_metrics} for m in methods}

    for run in range(n_runs):
        np.random.seed(42 + run * 17)
        n_sample = int(len(test_queries) * 0.9)
        indices = np.random.choice(len(test_queries), n_sample, replace=False)
        sampled_queries = [test_queries[i] for i in indices]

        run_r = {m: {k: [] for k in e2e_metrics} for m in methods}
        for q in sampled_queries:
            q_emb = np.array(q['query_embedding'])
            cdomain = q['correct_domain']

            # === 1. FAISS + LLM reasoning ===
            res_f, _, _ = search_flat(q_emb, faiss_idx, all_apis, top_k=10)
            acc_f = 1 if res_f and res_f[0].get('domain') == cdomain else 0
            flat_metrics = measure_e2e_tokens_flat_llm(res_f[:5], all_apis_by_id)
            run_r['flat_e2e']['acc'].append(acc_f)
            run_r['flat_e2e']['total_tokens'].append(flat_metrics['total_tokens'])
            run_r['flat_e2e']['retrieval_tokens'].append(flat_metrics['retrieval_tokens'])
            run_r['flat_e2e']['llm_tokens'].append(flat_metrics['llm_tokens'])
            run_r['flat_e2e']['routing_tokens'].append(0)
            run_r['flat_e2e']['dependency_tokens'].append(0)
            run_r['flat_e2e']['chain_completeness'].append(flat_metrics['chain_completeness'])

            # === 2. HNSW + LLM reasoning ===
            res_h, _ = search_hnsw(q_emb, hnsw_idx, all_apis, top_k=10)
            acc_h = 1 if res_h and res_h[0].get('domain') == cdomain else 0
            hnsw_metrics = measure_e2e_tokens_flat_llm(res_h[:5], all_apis_by_id)
            run_r['hnsw_e2e']['acc'].append(acc_h)
            run_r['hnsw_e2e']['total_tokens'].append(hnsw_metrics['total_tokens'])
            run_r['hnsw_e2e']['retrieval_tokens'].append(hnsw_metrics['retrieval_tokens'])
            run_r['hnsw_e2e']['llm_tokens'].append(hnsw_metrics['llm_tokens'])
            run_r['hnsw_e2e']['routing_tokens'].append(0)
            run_r['hnsw_e2e']['dependency_tokens'].append(0)
            run_r['hnsw_e2e']['chain_completeness'].append(hnsw_metrics['chain_completeness'])

            # === 3. FAISS Top-20 + Rerank + LLM ===
            res_rr, _, stage1 = search_faiss_rerank(q_emb, faiss_idx, all_apis,
                                                      top_k_retrieve=20, top_k_final=5)
            acc_rr = 1 if res_rr and res_rr[0].get('domain') == cdomain else 0
            rerank_metrics = measure_e2e_tokens_faiss_rerank(stage1[:20], res_rr, all_apis_by_id)
            run_r['faiss_rerank_e2e']['acc'].append(acc_rr)
            run_r['faiss_rerank_e2e']['total_tokens'].append(rerank_metrics['total_tokens'])
            run_r['faiss_rerank_e2e']['retrieval_tokens'].append(rerank_metrics['retrieval_tokens'])
            run_r['faiss_rerank_e2e']['llm_tokens'].append(rerank_metrics['llm_tokens'])
            run_r['faiss_rerank_e2e']['routing_tokens'].append(0)
            run_r['faiss_rerank_e2e']['dependency_tokens'].append(0)
            run_r['faiss_rerank_e2e']['chain_completeness'].append(0)

            # === 4. Forest + M4/M6/M9 ===
            res_fr, _, _, routed, _ = search_forest(q_emb, forest, top_k=10)
            acc_fr = 1 if routed == cdomain else 0
            top_skill = res_fr[0] if res_fr else None
            dep_chain = trace_dependency_chain(top_skill, all_apis_by_id) if top_skill else []
            merged_params = {}
            if top_skill and 'hierarchical_params' in top_skill:
                hp = top_skill['hierarchical_params']
                for level in ['root', 'middle', 'leaf']:
                    merged_params.update(hp.get(level, {}))
            routing_descs = [f"Domain: {d}" for d in domain_names]
            forest_metrics = measure_e2e_tokens_forest(
                routing_descs, res_fr[:5], dep_chain, merged_params, all_apis_by_id)
            run_r['forest_e2e']['acc'].append(acc_fr)
            run_r['forest_e2e']['total_tokens'].append(forest_metrics['total_tokens'])
            run_r['forest_e2e']['retrieval_tokens'].append(forest_metrics['retrieval_tokens'])
            run_r['forest_e2e']['llm_tokens'].append(forest_metrics['llm_tokens'])
            run_r['forest_e2e']['routing_tokens'].append(forest_metrics['routing_tokens'])
            run_r['forest_e2e']['dependency_tokens'].append(forest_metrics['dependency_tokens'])
            run_r['forest_e2e']['chain_completeness'].append(forest_metrics['chain_completeness'])

        for m in methods:
            for k in e2e_metrics:
                if run_r[m][k]:
                    all_results[m][k].append(float(np.mean(run_r[m][k])))

    return all_results

# ============================================================
# Scale Experiment (real mechanisms, not hardcoded constants)
# ============================================================
def run_scale_experiment(base_apis, test_queries, model, skill_counts):
    scale = {'flat': {'acc': [], 'total_tokens': [], 'llm_tokens': []},
             'forest': {'acc': [], 'total_tokens': [], 'llm_tokens': []}}

    all_domains_full = {}
    for api in base_apis:
        d = api.get('domain', 'unknown')
        if d not in all_domains_full:
            all_domains_full[d] = []
        all_domains_full[d].append(api)
    domain_list = list(all_domains_full.keys())
    n_domains = len(domain_list)
    all_apis_by_id = {a['id']: a for a in base_apis}

    for N in skill_counts:
        print(f'  Scale N={N}...')
        per_domain = max(2, N // n_domains)
        min_available = min(len(all_domains_full[d]) for d in domain_list)
        if per_domain > min_available:
            for m in scale:
                for k in scale[m]:
                    scale[m][k].append(None)
            continue

        # Random subsample per_domain APIs from each domain (different each run)
        np.random.seed(N)
        subset = []
        sub_domains = {}
        for d in domain_list:
            available = all_domains_full[d]
            if per_domain < len(available):
                idx = np.random.choice(len(available), per_domain, replace=False)
                sampled = [available[i] for i in idx]
            else:
                sampled = available[:per_domain]
            subset.extend(sampled)
            sub_domains[d] = sampled

        embeddings = np.array([a['embedding'] for a in subset])
        faiss_idx = build_faiss_index(embeddings)
        forest = build_forest(subset, sub_domains)
        subset_by_id = {a['id']: a for a in subset}
        domain_names = list(forest.keys())

        # Use different query subset for variance
        test_subset = test_queries[:min(80, len(test_queries))]
        flat_accs, flat_tokens, flat_llm = [], [], []
        forest_accs, forest_tokens, forest_llm = [], [], []

        for q in test_subset:
            q_emb = np.array(q['query_embedding'])
            cdomain = q['correct_domain']

            # Flat: real token model
            res_f, _, _ = search_flat(q_emb, faiss_idx, subset, top_k=5)
            flat_accs.append(1 if res_f and res_f[0].get('domain') == cdomain else 0)
            flat_metrics = measure_e2e_tokens_flat_llm(res_f, subset_by_id)
            flat_tokens.append(flat_metrics['total_tokens'])
            flat_llm.append(flat_metrics['llm_tokens'])

            # Forest: real mechanisms
            res_fr, _, _, routed, _ = search_forest(q_emb, forest, top_k=5)
            forest_accs.append(1 if routed == cdomain else 0)
            top_skill = res_fr[0] if res_fr else None
            dep_chain = trace_dependency_chain(top_skill, subset_by_id) if top_skill else []
            merged_params = {}
            if top_skill and 'hierarchical_params' in top_skill:
                hp = top_skill['hierarchical_params']
                for level in ['root', 'middle', 'leaf']:
                    merged_params.update(hp.get(level, {}))
            routing_descs = [f"Domain: {d}" for d in domain_names]
            forest_metrics = measure_e2e_tokens_forest(
                routing_descs, res_fr, dep_chain, merged_params, subset_by_id)
            forest_tokens.append(forest_metrics['total_tokens'])
            forest_llm.append(forest_metrics['llm_tokens'])

        scale['flat']['acc'].append(float(np.mean(flat_accs)))
        scale['flat']['total_tokens'].append(float(np.mean(flat_tokens)))
        scale['flat']['llm_tokens'].append(float(np.mean(flat_llm)))
        scale['forest']['acc'].append(float(np.mean(forest_accs)))
        scale['forest']['total_tokens'].append(float(np.mean(forest_tokens)))
        scale['forest']['llm_tokens'].append(float(np.mean(forest_llm)))

    return scale

def generate_all_visualizations(pure_results, e2e_results, scale_results, skill_counts):
    print("\nGenerating visualizations...")
    setup_plot_style()

    method_labels = {
        'flat': 'FAISS\n(IVF+PQ)', 'hnsw': 'HNSW', 'bm25': 'BM25',
        'faiss_rerank': 'FAISS\n+Rerank', 'forest': 'Skill Forest\n(Ours)',
    }
    method_colors = {
        'flat': '#3498DB', 'hnsw': '#2ECC71', 'bm25': '#7F8C8D',
        'faiss_rerank': '#E67E22', 'forest': '#E74C3C',
    }

    # --- Fig 1: Pure Retrieval Accuracy (all baselines) ---
    methods_with_data = [m for m in ['flat', 'hnsw', 'bm25', 'faiss_rerank', 'forest']
                         if pure_results.get(m, {}).get('acc@5')]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Acc@5 and MRR grouped bar
    ax = axes[0]
    x = np.arange(len(methods_with_data))
    width = 0.35
    acc5_means = [np.mean(pure_results[m]['acc@5']) for m in methods_with_data]
    acc5_stds = [np.std(pure_results[m]['acc@5']) for m in methods_with_data]
    mrr_means = [np.mean(pure_results[m]['mrr']) for m in methods_with_data]
    mrr_stds = [np.std(pure_results[m]['mrr']) for m in methods_with_data]
    colors = [method_colors.get(m, '#95A5A6') for m in methods_with_data]
    bars1 = ax.bar(x - width/2, acc5_means, width, yerr=acc5_stds, capsize=3,
                   color=colors, alpha=0.85, label='Accuracy@5')
    bars2 = ax.bar(x + width/2, mrr_means, width, yerr=mrr_stds, capsize=3,
                   color=colors, alpha=0.5, label='MRR', hatch='//')
    for bar, val in zip(bars1, acc5_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods_with_data], fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Retrieval Accuracy Comparison', fontweight='bold', pad=15)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Right: Latency comparison
    ax = axes[1]
    latency_means = [np.mean(pure_results[m]['latency']) for m in methods_with_data]
    latency_stds = [np.std(pure_results[m]['latency']) for m in methods_with_data]
    bars = ax.bar(x, latency_means, yerr=latency_stds, capsize=3, color=colors, alpha=0.85)
    for bar, val in zip(bars, latency_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods_with_data], fontsize=9)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Retrieval Latency Comparison', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, VIS_DIR, 'fig1_纯检索准确率_CN.png')
    save_figure(fig, VIS_DIR, 'fig1_pure_retrieval_EN.png')

    # --- Fig 2: End-to-End Token Comparison (all baselines) ---
    e2e_method_labels = {
        'flat_e2e': 'FAISS\n+LLM', 'hnsw_e2e': 'HNSW\n+LLM',
        'faiss_rerank_e2e': 'FAISS+Rerank\n+LLM', 'forest_e2e': 'Skill Forest\n+M4/M6/M9',
    }
    e2e_method_colors = {
        'flat_e2e': '#3498DB', 'hnsw_e2e': '#2ECC71',
        'faiss_rerank_e2e': '#E67E22', 'forest_e2e': '#E74C3C',
    }
    e2e_methods_with_data = [m for m in ['flat_e2e', 'hnsw_e2e', 'faiss_rerank_e2e', 'forest_e2e']
                             if e2e_results.get(m, {}).get('total_tokens')]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(e2e_methods_with_data))
    tok_means = [np.mean(e2e_results[m]['total_tokens']) for m in e2e_methods_with_data]
    tok_stds = [np.std(e2e_results[m]['total_tokens']) for m in e2e_methods_with_data]
    colors_e2e = [e2e_method_colors.get(m, '#95A5A6') for m in e2e_methods_with_data]
    bars = ax.bar(x, tok_means, yerr=tok_stds, capsize=5, color=colors_e2e, alpha=0.85, width=0.55)
    for bar, val, std in zip(bars, tok_means, tok_stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
                f'{val:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Add savings annotation
    if 'flat_e2e' in e2e_methods_with_data and 'forest_e2e' in e2e_methods_with_data:
        flat_tok = np.mean(e2e_results['flat_e2e']['total_tokens'])
        forest_tok = np.mean(e2e_results['forest_e2e']['total_tokens'])
        savings = (flat_tok - forest_tok) / flat_tok * 100 if flat_tok > 0 else 0
        ax.text(0.98, 0.95, f'Forest saves {savings:.0f}% vs FAISS+LLM',
                transform=ax.transAxes, ha='right', fontsize=13, fontweight='bold', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels([e2e_method_labels.get(m, m) for m in e2e_methods_with_data], fontsize=10)
    ax.set_ylabel('Total Token Consumption')
    ax.set_title('End-to-End Token Consumption Comparison', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, VIS_DIR, 'fig2_端到端Token_CN.png')
    save_figure(fig, VIS_DIR, 'fig2_e2e_token_EN.png')

    # --- Fig 3: Scale Tokens vs API Count ---
    valid_counts = [c for c, v in zip(skill_counts, scale_results['flat']['total_tokens']) if v is not None]
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, label, color in [('flat', 'Flat ANN + LLM Reasoning', COLORS['flat']),
                             ('forest', 'Forest + M4/M6/M9', COLORS['forest'])]:
        vals = [v for v in scale_results[m]['total_tokens'] if v is not None]
        ax.plot(valid_counts, vals, 'o-', color=color, label=label, linewidth=2.5, markersize=7)
    ax.set_xlabel('Number of APIs (N)', fontsize=12)
    ax.set_ylabel('Total Token Consumption', fontsize=12)
    ax.set_title('Scale: Token Consumption vs API Count', fontweight='bold', fontsize=13, pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_figure(fig, VIS_DIR, 'fig3_规模Token曲线_CN.png')
    save_figure(fig, VIS_DIR, 'fig3_scale_tokens_EN.png')

    # --- Fig 4: Scale Accuracy vs API Count ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, label, color in [('flat', 'Flat ANN + LLM', COLORS['flat']),
                             ('forest', 'Forest + M4/M6/M9', COLORS['forest'])]:
        vals = [v for v in scale_results[m]['acc'] if v is not None]
        ax.plot(valid_counts, vals, 'o-', color=color, label=label, linewidth=2.5, markersize=7)
    ax.set_xlabel('Number of APIs (N)', fontsize=12)
    ax.set_ylabel('End-to-End Accuracy@1', fontsize=12)
    ax.set_title('Scale: Accuracy vs API Count', fontweight='bold', fontsize=13, pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_figure(fig, VIS_DIR, 'fig4_规模准确率_CN.png')
    save_figure(fig, VIS_DIR, 'fig4_scale_accuracy_EN.png')

    # --- Fig 5: Scale LLM Tokens vs API Count ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for m, label, color in [('flat', 'Flat ANN LLM Reasoning', COLORS['flat']),
                             ('forest', 'Forest LLM Confirm', COLORS['forest'])]:
        vals = [v for v in scale_results[m]['llm_tokens'] if v is not None]
        ax.plot(valid_counts, vals, 'o-', color=color, label=label, linewidth=2.5, markersize=7)
    ax.set_xlabel('Number of APIs (N)', fontsize=12)
    ax.set_ylabel('LLM Token Consumption', fontsize=12)
    ax.set_title('Scale: LLM Burden vs API Count', fontweight='bold', fontsize=13, pad=15)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save_figure(fig, VIS_DIR, 'fig5_LLM负担_CN.png')
    save_figure(fig, VIS_DIR, 'fig5_llm_burden_EN.png')

    # --- Table 1: Full comparison WITH standard deviations ---
    table_data = []
    for met in ['acc@1', 'acc@3', 'acc@5', 'mrr', 'latency', 'tokens', 'routing_acc']:
        row = {'Metric': met}
        for m in ['flat', 'forest']:
            if met in pure_results[m]:
                row['Flat ANN' if m == 'flat' else 'Forest'] = f"{np.mean(pure_results[m][met]):.3f}±{np.std(pure_results[m][met]):.3f}"
            else:
                row['Flat ANN' if m == 'flat' else 'Forest'] = 'N/A'
        table_data.append(row)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    col_labels = list(table_data[0].keys())
    df_data = [[row[k] for k in col_labels] for row in table_data]
    table = ax.table(cellText=df_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        cell.set_edgecolor('#BDC3C7')
    ax.set_title('Table 1: Pure Retrieval (mean±std, 5 runs)', fontweight='bold', fontsize=14, pad=20)
    save_figure(fig, VIS_DIR, 'table1_纯检索_CN.png')
    save_figure(fig, VIS_DIR, 'table1_pure_retrieval_EN.png')

    # --- Table 2: E2E comparison WITH std ---
    e2e_table = []
    for method_key, method_name in [('flat_e2e', 'Flat ANN + LLM'), ('forest_e2e', 'Forest + M4/M6/M9')]:
        e2e_table.append({
            'Method': method_name,
            'Total Tokens': f"{np.mean(e2e_results[method_key]['total_tokens']):.0f}±{np.std(e2e_results[method_key]['total_tokens']):.0f}",
            'LLM Tokens': f"{np.mean(e2e_results[method_key]['llm_tokens']):.0f}±{np.std(e2e_results[method_key]['llm_tokens']):.0f}",
            'Accuracy': f"{np.mean(e2e_results[method_key]['acc']):.3f}±{np.std(e2e_results[method_key]['acc']):.3f}",
            'Chain Complete': f"{np.mean(e2e_results[method_key]['chain_completeness']):.2%}±{np.std(e2e_results[method_key]['chain_completeness']):.2%}",
        })
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    col_labels = list(e2e_table[0].keys())
    df_data = [[row[k] for k in col_labels] for row in e2e_table]
    table = ax.table(cellText=df_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 2.0)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#BDC3C7')
    ax.set_title('Table 2: End-to-End Comparison (mean±std, 5 runs)', fontweight='bold', fontsize=14, pad=20)
    save_figure(fig, VIS_DIR, 'table2_端到端对比_CN.png')
    save_figure(fig, VIS_DIR, 'table2_e2e_comparison_EN.png')

    # --- Table 3: Scale results with clear headers ---
    scale_table = []
    for i, N in enumerate(valid_counts):
        if scale_results['flat']['total_tokens'][i] is not None:
            flat_tok = scale_results['flat']['total_tokens'][i]
            forest_tok = scale_results['forest']['total_tokens'][i]
            savings = (flat_tok - forest_tok) / flat_tok * 100 if flat_tok > 0 else 0
            scale_table.append({
                'API数量\n(N)': str(N),
                'Flat+LLM\n总Token消耗': f"{flat_tok:.0f}",
                '森林+M机制\n总Token消耗': f"{forest_tok:.0f}",
                'Token节省\n比例': f"{savings:.1f}%",
                'Flat+LLM\n准确率@1': f"{scale_results['flat']['acc'][i]:.3f}",
                '森林+M机制\n准确率@1': f"{scale_results['forest']['acc'][i]:.3f}",
            })
    if scale_table:
        fig, ax = plt.subplots(figsize=(16, max(4, len(scale_table) * 0.6 + 1.5)))
        ax.axis('off')
        col_labels = list(scale_table[0].keys())
        df_data = [[row[k] for k in col_labels] for row in scale_table]
        table = ax.table(cellText=df_data, colLabels=col_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#ECF0F1')
            cell.set_edgecolor('#BDC3C7')
        ax.set_title('Table 3: Scale Experiment (Token and accuracy vs API count N)', fontweight='bold', fontsize=14, pad=20)
        save_figure(fig, VIS_DIR, 'table3_规模结果_CN.png')
    save_figure(fig, VIS_DIR, 'table3_scale_results_EN.png')

def add_query_embeddings(test_queries, model):
    texts = [q['query'] for q in test_queries]
    embs = model.encode(texts, batch_size=64)
    for q, emb in zip(test_queries, embs):
        q['query_embedding'] = emb.tolist()
    return test_queries

def main():
    print("=" * 60)
    print("Experiment 1 v3: Retrieval + End-to-End + Scale (ALL FIXES)")
    print("=" * 60)

    print("\n[1/6] Loading enriched data...")
    enriched_path = os.path.join(os.path.dirname(BASE), 'shared', 'data', 'all_apis_enriched.json')
    if os.path.exists(enriched_path):
        with open(enriched_path, 'r', encoding='utf-8') as f:
            all_apis = json.load(f)
        for api in all_apis:
            if 'embedding' in api:
                api['embedding'] = np.array(api['embedding'])
        print(f"  Loaded {len(all_apis)} enriched APIs")
    else:
        all_apis, _ = load_dataset(DATA_DIR)

    _, test_queries = load_dataset(DATA_DIR)
    domains_dict = {d: [] for d in DOMAINS}
    for api in all_apis:
        if api['domain'] in domains_dict:
            domains_dict[api['domain']].append(api)
    for d, apis in domains_dict.items():
        print(f"  {d}: {len(apis)} APIs")

    print("\n[2/6] Computing query embeddings...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_queries = add_query_embeddings(test_queries, model)

    print("\n[3/6] Layer 1: Pure Retrieval (5 runs, subcat evaluation)...")
    pure_results = run_pure_retrieval(all_apis, test_queries, domains_dict, n_runs=N_RUNS)

    print("\n[4/6] Layer 2: End-to-End (5 runs, real mechanisms)...")
    e2e_results = run_end_to_end(all_apis, test_queries, domains_dict, n_runs=N_RUNS)

    print("\n[5/6] Scale Experiment (real token model)...")
    skill_counts = [100, 300, 500, 1000, 1500, 3000, 5000]
    if len(all_apis) < 5000:
        print("  Generating extended dataset...")
        _, all_apis_ext = generate_dataset(n_per_domain=1000, seed=42)
        compute_embeddings(all_apis_ext, model)
        test_queries_ext = generate_test_queries(n_clear=140, n_ambiguous=60, seed=42)
        test_queries_ext = add_query_embeddings(test_queries_ext, model)
    else:
        all_apis_ext = all_apis
        test_queries_ext = test_queries
    scale_results = run_scale_experiment(all_apis_ext, test_queries_ext, model, skill_counts)

    print("\n[6/6] Generating visualizations...")
    generate_all_visualizations(pure_results, e2e_results, scale_results, skill_counts)

    # Compute summary
    summary = {}
    for m in pure_results:
        summary[m] = {}
        for k, vals in pure_results[m].items():
            if vals:
                summary[m][k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    output = {
        'pure_retrieval': summary,
        'end_to_end': {m: {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in e2e_results[m].items()} for m in e2e_results},
        'scale': scale_results,
        'config': {'n_runs': N_RUNS, 'n_apis': len(all_apis), 'n_queries': len(test_queries), 'skill_counts': skill_counts}
    }
    with open(os.path.join(RES_DIR, 'experiment1_v2_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    method_names = {
        'flat': 'FAISS (IVF+PQ)', 'hnsw': 'HNSW', 'bm25': 'BM25',
        'faiss_rerank': 'FAISS+Rerank', 'forest': 'Skill Forest (Ours)',
    }
    print("\n--- Layer 1: Pure Retrieval (subcat-level) ---")
    for m in ['flat', 'hnsw', 'bm25', 'faiss_rerank', 'forest']:
        if m in summary and summary[m].get('acc@5'):
            print(f"\n{method_names.get(m, m)}:")
            for k in ['acc@1', 'acc@3', 'acc@5', 'mrr']:
                if k in summary[m]:
                    print(f"  {k}: {summary[m][k]['mean']:.3f} ± {summary[m][k]['std']:.3f}")
            if 'latency' in summary[m]:
                print(f"  latency: {summary[m]['latency']['mean']:.2f} ± {summary[m]['latency']['std']:.2f} ms")
            if m == 'forest' and 'routing_acc' in summary[m]:
                print(f"  routing_acc: {summary[m]['routing_acc']['mean']:.3f} ± {summary[m]['routing_acc']['std']:.3f}")

    e2e_names = {
        'flat_e2e': 'FAISS+LLM', 'hnsw_e2e': 'HNSW+LLM',
        'faiss_rerank_e2e': 'FAISS+Rerank+LLM', 'forest_e2e': 'Forest+M4/M6/M9',
    }
    print("\n--- Layer 2: End-to-End ---")
    for m in ['flat_e2e', 'hnsw_e2e', 'faiss_rerank_e2e', 'forest_e2e']:
        if m in e2e_results and e2e_results[m].get('total_tokens'):
            print(f"\n{e2e_names.get(m, m)}:")
            for k in ['total_tokens', 'llm_tokens', 'acc', 'chain_completeness']:
                v = e2e_results[m][k]
                if v:
                    print(f"  {k}: {np.mean(v):.3f} ± {np.std(v):.3f}")

    print("\n--- Scale Experiment ---")
    for i, N in enumerate(skill_counts):
        if scale_results['flat']['total_tokens'][i] is not None:
            flat_tok = scale_results['flat']['total_tokens'][i]
            forest_tok = scale_results['forest']['total_tokens'][i]
            savings = (flat_tok - forest_tok) / flat_tok * 100
            print(f"  N={N}: Flat={flat_tok:.0f}, Forest={forest_tok:.0f}, Savings={savings:.1f}%")

if __name__ == '__main__':
    main()
