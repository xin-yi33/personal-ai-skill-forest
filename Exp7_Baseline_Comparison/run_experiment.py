"""
Experiment 7: Comprehensive Baseline Comparison
Addresses reviewer feedback: "baseline comparison too narrow"

Compares 7 methods:
1. Random (theoretical lower bound)
2. BM25 (traditional IR baseline)
3. Flat ANN / FAISS IVF+PQ (current baseline)
4. HNSW (SOTA ANN algorithm)
5. FAISS + Embedding Rerank (production mainstream)
6. FAISS + Cross-Encoder Rerank (academic SOTA)
7. Skill Forest + M4/M6/M9 (our approach)
8. LLM Full-Context (Gorilla/ToolLLM style - token analysis only)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from shared.bplus_tree import BPlusTree
from shared.data_generator import load_dataset, generate_dataset, compute_embeddings, generate_test_queries
from shared.visualization_utils import setup_plot_style, save_figure, COLORS
from shared.mechanisms import count_tokens, trace_dependency_chain, measure_e2e_tokens_flat_llm
from shared.baselines import (
    build_hnsw_index, search_hnsw,
    build_bm25_index, search_bm25,
    search_random,
    search_faiss_rerank,
    measure_e2e_tokens_faiss_rerank,
    measure_e2e_tokens_llm_full,
    evaluate_baseline,
    CrossEncoderReranker, search_cross_encoder_rerank
)
import faiss

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), 'shared', 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

N_RUNS = 5
DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']

# ============================================================
# Index builders
# ============================================================
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

def search_forest(query_emb, forest, top_k=10):
    start = time.perf_counter()
    domain_scores = {}
    for domain, info in forest.items():
        sim = cosine_similarity(query_emb.reshape(1, -1), info['root_vector'].reshape(1, -1))[0][0]
        domain_scores[domain] = sim
    best_domain = max(domain_scores, key=domain_scores.get)
    results, scores, path = forest[best_domain]['tree'].search_with_traversal(query_emb, top_k=top_k)
    latency = (time.perf_counter() - start) * 1000
    return results, latency, best_domain

def search_faiss_flat(query_emb, faiss_index, all_apis, top_k=10):
    start = time.perf_counter()
    D, I = faiss_index.search(query_emb.astype('float32').reshape(1, -1), top_k)
    latency = (time.perf_counter() - start) * 1000
    results = [all_apis[i] for i in I[0] if 0 <= i < len(all_apis)]
    return results, latency

# ============================================================
# Layer 1: Pure Retrieval Comparison
# ============================================================
def run_pure_retrieval_comparison(all_apis, test_queries, domains_dict, 
                                   faiss_idx, hnsw_idx, bm25_idx, forest,
                                   skill_number_map, n_runs=5):
    """Compare all baselines on pure retrieval accuracy."""
    
    methods = ['random', 'bm25', 'flat_faiss', 'hnsw', 'faiss_rerank', 'cross_encoder', 'forest']
    metrics = ['acc@1', 'acc@3', 'acc@5', 'mrr', 'latency']
    all_results = {m: {k: [] for k in metrics} for m in methods}
    
    # Cross-encoder (optional, may fail if model not available)
    try:
        reranker = CrossEncoderReranker()
        has_cross_encoder = reranker.available
    except:
        has_cross_encoder = False
    
    for run in range(n_runs):
        np.random.seed(42 + run * 17)
        n_sample = int(len(test_queries) * 0.9)
        indices = np.random.choice(len(test_queries), n_sample, replace=False)
        sampled_queries = [test_queries[i] for i in indices]
        
        run_r = {m: {k: [] for k in metrics} for m in methods}
        
        for qi, q in enumerate(sampled_queries):
            q_emb = np.array(q['query_embedding'])
            q_text = q.get('query', '')
            cdomain = q['correct_domain']
            csubcat = None
            cs = q.get('correct_skill', '')
            parts = cs.rsplit('_skill_', 1)
            if len(parts) == 2:
                try:
                    api = skill_number_map.get((parts[0], int(parts[1])))
                    if api: csubcat = api.get('subcategory')
                except: pass
            
            # 1. Random
            res, lat = search_random(all_apis, top_k=10, seed=42+run+qi)
            a1, a3, a5, mrr = evaluate_baseline(res, cdomain, csubcat)
            run_r['random']['acc@1'].append(a1); run_r['random']['acc@3'].append(a3)
            run_r['random']['acc@5'].append(a5); run_r['random']['mrr'].append(mrr)
            run_r['random']['latency'].append(lat)
            
            # 2. BM25
            if q_text:
                res, lat = search_bm25(q_text, bm25_idx, all_apis, top_k=10)
                a1, a3, a5, mrr = evaluate_baseline(res, cdomain, csubcat)
                run_r['bm25']['acc@1'].append(a1); run_r['bm25']['acc@3'].append(a3)
                run_r['bm25']['acc@5'].append(a5); run_r['bm25']['mrr'].append(mrr)
                run_r['bm25']['latency'].append(lat)
            
            # 3. Flat FAISS
            res, lat = search_faiss_flat(q_emb, faiss_idx, all_apis, top_k=10)
            a1, a3, a5, mrr = evaluate_baseline(res, cdomain, csubcat)
            run_r['flat_faiss']['acc@1'].append(a1); run_r['flat_faiss']['acc@3'].append(a3)
            run_r['flat_faiss']['acc@5'].append(a5); run_r['flat_faiss']['mrr'].append(mrr)
            run_r['flat_faiss']['latency'].append(lat)
            
            # 4. HNSW
            res, lat = search_hnsw(q_emb, hnsw_idx, all_apis, top_k=10)
            a1, a3, a5, mrr = evaluate_baseline(res, cdomain, csubcat)
            run_r['hnsw']['acc@1'].append(a1); run_r['hnsw']['acc@3'].append(a3)
            run_r['hnsw']['acc@5'].append(a5); run_r['hnsw']['mrr'].append(mrr)
            run_r['hnsw']['latency'].append(lat)
            
            # 5. FAISS + Embedding Rerank
            res, lat, _ = search_faiss_rerank(q_emb, faiss_idx, all_apis,
                                               top_k_retrieve=20, top_k_final=10)
            a1, a3, a5, mrr = evaluate_baseline(res, cdomain, csubcat)
            run_r['faiss_rerank']['acc@1'].append(a1); run_r['faiss_rerank']['acc@3'].append(a3)
            run_r['faiss_rerank']['acc@5'].append(a5); run_r['faiss_rerank']['mrr'].append(mrr)
            run_r['faiss_rerank']['latency'].append(lat)
            
            # 6. Cross-Encoder Rerank
            if has_cross_encoder and q_text:
                res, lat = search_cross_encoder_rerank(q_text, q_emb, faiss_idx, all_apis,
                                                        reranker, top_k_retrieve=20, top_k_final=10)
                a1, a3, a5, mrr = evaluate_baseline(res, cdomain, csubcat)
                run_r['cross_encoder']['acc@1'].append(a1); run_r['cross_encoder']['acc@3'].append(a3)
                run_r['cross_encoder']['acc@5'].append(a5); run_r['cross_encoder']['mrr'].append(mrr)
                run_r['cross_encoder']['latency'].append(lat)
            
            # 7. Forest
            res, lat, routed = search_forest(q_emb, forest, top_k=10)
            a1, a3, a5, mrr = evaluate_baseline(res, cdomain, csubcat)
            run_r['forest']['acc@1'].append(a1); run_r['forest']['acc@3'].append(a3)
            run_r['forest']['acc@5'].append(a5); run_r['forest']['mrr'].append(mrr)
            run_r['forest']['latency'].append(lat)
        
        for m in methods:
            for k in metrics:
                if run_r[m][k]:
                    all_results[m][k].append(float(np.mean(run_r[m][k])))
    
    return all_results


# ============================================================
# Layer 2: End-to-End Token Comparison
# ============================================================
def run_e2e_token_comparison(all_apis, test_queries, domains_dict,
                              faiss_idx, hnsw_idx, forest, skill_number_map,
                              n_runs=5):
    """Compare end-to-end token consumption across methods."""
    
    methods = ['flat_faiss', 'hnsw', 'faiss_rerank', 'forest', 'llm_full']
    e2e_metrics = ['acc', 'total_tokens', 'retrieval_tokens', 'llm_tokens']
    all_results = {m: {k: [] for k in e2e_metrics} for m in methods}
    
    all_apis_by_id = {a['id']: a for a in all_apis}
    domain_names = list(forest.keys())
    
    for run in range(n_runs):
        np.random.seed(42 + run * 17)
        n_sample = int(len(test_queries) * 0.9)
        indices = np.random.choice(len(test_queries), n_sample, replace=False)
        sampled_queries = [test_queries[i] for i in indices]
        
        run_r = {m: {k: [] for k in e2e_metrics} for m in methods}
        
        for q in sampled_queries:
            q_emb = np.array(q['query_embedding'])
            q_text = q.get('query', '')
            cdomain = q['correct_domain']
            
            # 1. Flat FAISS + LLM
            res_f, _ = search_faiss_flat(q_emb, faiss_idx, all_apis, top_k=5)
            acc_f = 1 if res_f and res_f[0].get('domain') == cdomain else 0
            flat_m = measure_e2e_tokens_flat_llm(res_f, all_apis_by_id)
            run_r['flat_faiss']['acc'].append(acc_f)
            run_r['flat_faiss']['total_tokens'].append(flat_m['total_tokens'])
            run_r['flat_faiss']['retrieval_tokens'].append(flat_m['retrieval_tokens'])
            run_r['flat_faiss']['llm_tokens'].append(flat_m['llm_tokens'])
            
            # 2. HNSW + LLM
            res_h, _ = search_hnsw(q_emb, hnsw_idx, all_apis, top_k=5)
            acc_h = 1 if res_h and res_h[0].get('domain') == cdomain else 0
            hnsw_m = measure_e2e_tokens_flat_llm(res_h, all_apis_by_id)
            run_r['hnsw']['acc'].append(acc_h)
            run_r['hnsw']['total_tokens'].append(hnsw_m['total_tokens'])
            run_r['hnsw']['retrieval_tokens'].append(hnsw_m['retrieval_tokens'])
            run_r['hnsw']['llm_tokens'].append(hnsw_m['llm_tokens'])
            
            # 3. FAISS + Rerank
            res_rr, _, stage1 = search_faiss_rerank(q_emb, faiss_idx, all_apis,
                                                     top_k_retrieve=20, top_k_final=5)
            acc_rr = 1 if res_rr and res_rr[0].get('domain') == cdomain else 0
            rerank_m = measure_e2e_tokens_faiss_rerank(stage1, res_rr, all_apis_by_id)
            run_r['faiss_rerank']['acc'].append(acc_rr)
            run_r['faiss_rerank']['total_tokens'].append(rerank_m['total_tokens'])
            run_r['faiss_rerank']['retrieval_tokens'].append(rerank_m['retrieval_tokens'])
            run_r['faiss_rerank']['llm_tokens'].append(rerank_m['llm_tokens'])
            
            # 4. Forest + M4/M6/M9
            res_fr, _, routed = search_forest(q_emb, forest, top_k=5)
            acc_fr = 1 if routed == cdomain else 0
            top_skill = res_fr[0] if res_fr else None
            dep_chain = trace_dependency_chain(top_skill, all_apis_by_id) if top_skill else []
            merged_params = {}
            if top_skill and 'hierarchical_params' in top_skill:
                hp = top_skill['hierarchical_params']
                for level in ['root', 'middle', 'leaf']:
                    merged_params.update(hp.get(level, {}))
            routing_descs = [f"Domain: {d}" for d in domain_names]
            from shared.mechanisms import measure_e2e_tokens_forest
            forest_m = measure_e2e_tokens_forest(routing_descs, res_fr, dep_chain, merged_params, all_apis_by_id)
            run_r['forest']['acc'].append(acc_fr)
            run_r['forest']['total_tokens'].append(forest_m['total_tokens'])
            run_r['forest']['retrieval_tokens'].append(forest_m['retrieval_tokens'])
            run_r['forest']['llm_tokens'].append(forest_m['llm_tokens'])
            
            # 5. LLM Full-Context (token analysis only)
            llm_full_m = measure_e2e_tokens_llm_full(all_apis, top_skill, all_apis_by_id)
            run_r['llm_full']['acc'].append(acc_f)  # Same accuracy as flat (conceptual)
            run_r['llm_full']['total_tokens'].append(llm_full_m['total_tokens'])
            run_r['llm_full']['retrieval_tokens'].append(llm_full_m['retrieval_tokens'])
            run_r['llm_full']['llm_tokens'].append(llm_full_m['llm_tokens'])
        
        for m in methods:
            for k in e2e_metrics:
                if run_r[m][k]:
                    all_results[m][k].append(float(np.mean(run_r[m][k])))
    
    return all_results


# ============================================================
# Visualization
# ============================================================
def generate_visualizations(retrieval_results, e2e_results):
    print("\nGenerating visualizations...")
    setup_plot_style()
    
    # Method display config
    method_labels = {
        'random': 'Random',
        'bm25': 'BM25',
        'flat_faiss': 'FAISS (IVF+PQ)',
        'hnsw': 'HNSW',
        'faiss_rerank': 'FAISS+Rerank',
        'cross_encoder': 'Cross-Encoder',
        'forest': 'Skill Forest (Ours)',
        'llm_full': 'LLM Full-Context',
    }
    method_colors = {
        'random': '#95A5A6',
        'bm25': '#7F8C8D',
        'flat_faiss': '#3498DB',
        'hnsw': '#2ECC71',
        'faiss_rerank': '#E67E22',
        'cross_encoder': '#9B59B6',
        'forest': '#E74C3C',
        'llm_full': '#1ABC9C',
    }
    
    # --- Fig 1: Retrieval Accuracy Comparison (grouped bar chart) ---
    methods_with_data = [m for m in ['random', 'bm25', 'flat_faiss', 'hnsw', 
                                      'faiss_rerank', 'cross_encoder', 'forest']
                         if retrieval_results.get(m, {}).get('acc@5')]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Acc@5 and MRR
    ax = axes[0]
    x = np.arange(len(methods_with_data))
    width = 0.35
    acc5_means = [np.mean(retrieval_results[m]['acc@5']) for m in methods_with_data]
    acc5_stds = [np.std(retrieval_results[m]['acc@5']) for m in methods_with_data]
    mrr_means = [np.mean(retrieval_results[m]['mrr']) for m in methods_with_data]
    mrr_stds = [np.std(retrieval_results[m]['mrr']) for m in methods_with_data]
    colors = [method_colors[m] for m in methods_with_data]
    
    bars1 = ax.bar(x - width/2, acc5_means, width, yerr=acc5_stds, capsize=3,
                   color=colors, alpha=0.85, label='Accuracy@5')
    bars2 = ax.bar(x + width/2, mrr_means, width, yerr=mrr_stds, capsize=3,
                   color=colors, alpha=0.5, label='MRR', hatch='//')
    
    for bar, val in zip(bars1, acc5_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods_with_data], 
                        rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Retrieval Accuracy Comparison', fontweight='bold', pad=15)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Right: Latency comparison
    ax = axes[1]
    latency_means = [np.mean(retrieval_results[m]['latency']) for m in methods_with_data]
    latency_stds = [np.std(retrieval_results[m]['latency']) for m in methods_with_data]
    bars = ax.bar(x, latency_means, yerr=latency_stds, capsize=3, color=colors, alpha=0.85)
    for bar, val in zip(bars, latency_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.2f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods_with_data],
                        rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Retrieval Latency Comparison', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, VIS_DIR, 'fig7_retrieval_comparison_EN.png')
    save_figure(fig, VIS_DIR, 'fig7_检索对比_EN.png')
    
    # --- Fig 2: End-to-End Token Comparison ---
    e2e_methods = ['flat_faiss', 'hnsw', 'faiss_rerank', 'forest', 'llm_full']
    e2e_with_data = [m for m in e2e_methods if e2e_results.get(m, {}).get('total_tokens')]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Total tokens
    ax = axes[0]
    x = np.arange(len(e2e_with_data))
    tok_means = [np.mean(e2e_results[m]['total_tokens']) for m in e2e_with_data]
    tok_stds = [np.std(e2e_results[m]['total_tokens']) for m in e2e_with_data]
    colors_e2e = [method_colors.get(m, '#95A5A6') for m in e2e_with_data]
    bars = ax.bar(x, tok_means, yerr=tok_stds, capsize=4, color=colors_e2e, alpha=0.85, width=0.6)
    for bar, val in zip(bars, tok_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in e2e_with_data],
                        rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Total Tokens')
    ax.set_title('End-to-End Token Consumption', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    # Right: Accuracy vs Token scatter
    ax = axes[1]
    for m in e2e_with_data:
        if e2e_results[m].get('acc') and e2e_results[m].get('total_tokens'):
            acc_val = np.mean(e2e_results[m]['acc'])
            tok_val = np.mean(e2e_results[m]['total_tokens'])
            ax.scatter(tok_val, acc_val, s=200, c=method_colors.get(m, '#95A5A6'),
                      marker='o', zorder=5, edgecolors='black', linewidth=1.5)
            ax.annotate(method_labels.get(m, m), (tok_val, acc_val),
                       textcoords="offset points", xytext=(10, 5), fontsize=9)
    ax.set_xlabel('Total Tokens')
    ax.set_ylabel('End-to-End Accuracy')
    ax.set_title('Accuracy vs Token Trade-off', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, VIS_DIR, 'fig8_e2e_token_comparison_EN.png')
    save_figure(fig, VIS_DIR, 'fig8_端到端Token对比_EN.png')
    
    # --- Fig 3: Summary table ---
    fig, ax = plt.subplots(figsize=(18, max(4, len(e2e_with_data) * 0.7 + 1.5)))
    ax.axis('off')
    
    col_labels = ['Method', 'Acc@5', 'MRR', 'Total Tokens', 'Latency (ms)']
    table_data = []
    for m in e2e_with_data:
        acc5 = np.mean(retrieval_results[m]['acc@5']) if retrieval_results.get(m, {}).get('acc@5') else 0
        mrr = np.mean(retrieval_results[m]['mrr']) if retrieval_results.get(m, {}).get('mrr') else 0
        tok = np.mean(e2e_results[m]['total_tokens']) if e2e_results.get(m, {}).get('total_tokens') else 0
        lat = np.mean(retrieval_results[m]['latency']) if retrieval_results.get(m, {}).get('latency') else 0
        table_data.append([
            method_labels.get(m, m),
            f'{acc5:.3f}',
            f'{mrr:.3f}',
            f'{tok:.0f}',
            f'{lat:.2f}'
        ])
    
    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white', fontweight='bold')
        elif i % 2 == 0:
            cell.set_facecolor('#ECF0F1')
        cell.set_edgecolor('#BDC3C7')
    ax.set_title('Table: Comprehensive Baseline Comparison Summary', fontweight='bold', fontsize=14, pad=20)
    save_figure(fig, VIS_DIR, 'table5_baseline_summary_EN.png')
    save_figure(fig, VIS_DIR, 'table5_基线对比总览_EN.png')


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Experiment 7: Comprehensive Baseline Comparison")
    print("=" * 60)
    
    # Load data
    print("\n[1/8] Loading data...")
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
    
    # Ensure query embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for q in test_queries:
        if 'query_embedding' not in q:
            q['query_embedding'] = model.encode(q['query']).tolist()
    
    domains_dict = {d: [] for d in DOMAINS}
    for api in all_apis:
        if api['domain'] in domains_dict:
            domains_dict[api['domain']].append(api)
    
    skill_number_map = {}
    for api in all_apis:
        parts = api['id'].rsplit('_', 1)
        if len(parts) == 2:
            try:
                skill_number_map[(parts[0], int(parts[1]))] = api
            except: pass
    
    # Build all indices
    print("\n[2/8] Building indices...")
    embeddings = np.array([a['embedding'] for a in all_apis])
    
    print("  Building FAISS IVF+PQ...")
    faiss_idx = build_faiss_index(embeddings)
    
    print("  Building HNSW...")
    hnsw_idx = build_hnsw_index(embeddings)
    
    print("  Building BM25...")
    bm25_idx = build_bm25_index(all_apis)
    
    print("  Building Forest...")
    forest = build_forest(all_apis, domains_dict)
    
    # Run pure retrieval comparison
    print("\n[3/8] Layer 1: Pure Retrieval Comparison (5 runs)...")
    retrieval_results = run_pure_retrieval_comparison(
        all_apis, test_queries, domains_dict,
        faiss_idx, hnsw_idx, bm25_idx, forest,
        skill_number_map, n_runs=N_RUNS
    )
    
    # Run E2E token comparison
    print("\n[4/8] Layer 2: End-to-End Token Comparison (5 runs)...")
    e2e_results = run_e2e_token_comparison(
        all_apis, test_queries, domains_dict,
        faiss_idx, hnsw_idx, forest, skill_number_map,
        n_runs=N_RUNS
    )
    
    # Generate visualizations
    print("\n[5/8] Generating visualizations...")
    generate_visualizations(retrieval_results, e2e_results)
    
    # Save results
    print("\n[6/8] Saving results...")
    summary = {
        'retrieval': {},
        'e2e': {},
    }
    for m in retrieval_results:
        summary['retrieval'][m] = {}
        for k, vals in retrieval_results[m].items():
            if vals:
                summary['retrieval'][m][k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    for m in e2e_results:
        summary['e2e'][m] = {}
        for k, vals in e2e_results[m].items():
            if vals:
                summary['e2e'][m][k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    
    with open(os.path.join(RES_DIR, 'experiment7_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    method_labels = {
        'random': 'Random', 'bm25': 'BM25', 'flat_faiss': 'FAISS (IVF+PQ)',
        'hnsw': 'HNSW', 'faiss_rerank': 'FAISS+Rerank',
        'cross_encoder': 'Cross-Encoder', 'forest': 'Skill Forest (Ours)',
        'llm_full': 'LLM Full-Context',
    }
    
    print("\n--- Retrieval Accuracy ---")
    for m in ['random', 'bm25', 'flat_faiss', 'hnsw', 'faiss_rerank', 'cross_encoder', 'forest']:
        if m in retrieval_results and retrieval_results[m].get('acc@5'):
            acc5 = np.mean(retrieval_results[m]['acc@5'])
            mrr = np.mean(retrieval_results[m]['mrr'])
            lat = np.mean(retrieval_results[m]['latency'])
            print(f"  {method_labels.get(m, m):20s}  Acc@5={acc5:.3f}  MRR={mrr:.3f}  Latency={lat:.2f}ms")
    
    print("\n--- End-to-End Tokens ---")
    for m in ['flat_faiss', 'hnsw', 'faiss_rerank', 'forest', 'llm_full']:
        if m in e2e_results and e2e_results[m].get('total_tokens'):
            tok = np.mean(e2e_results[m]['total_tokens'])
            acc = np.mean(e2e_results[m]['acc'])
            print(f"  {method_labels.get(m, m):20s}  Tokens={tok:.0f}  Acc={acc:.3f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
