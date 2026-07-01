"""
M2 Routing Layer Experiment: Forest vs Single B+ Tree vs Flat ANN
Pure retrieval comparison - no M4/M6/M7/M9 involved.
Compares: Acc@1, Acc@10, Domain Purity@10, Latency, Token, Routing Accuracy
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from shared.bplus_tree import BPlusTree
from shared.data_generator import load_dataset, generate_dataset, compute_embeddings

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']
N_RUNS = 5

# ============================================================
# Build indexes
# ============================================================
def build_faiss(embeddings):
    d = embeddings.shape[1]
    n = len(embeddings)
    if n < 200:
        idx = faiss.IndexFlatL2(d)
        idx.add(embeddings.astype('float32'))
        return idx
    nlist = min(50, max(1, n // 30))
    m = min(16, d // 2)
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    idx.train(embeddings.astype('float32'))
    idx.add(embeddings.astype('float32'))
    idx.nprobe = min(10, nlist)
    return idx

def build_forest(all_apis, domains_dict):
    forest = {}
    for domain, apis in domains_dict.items():
        tree = BPlusTree(order=32, domain_name=domain)
        for api in apis:
            tree.insert(api.get('subcategory', 'default'), api)
        embs = np.array([a['embedding'] for a in apis])
        root_vec = np.mean(embs, axis=0)
        forest[domain] = {'tree': tree, 'root_vector': root_vec, 'n_apis': len(apis)}
    return forest

def build_single_tree(all_apis):
    tree = BPlusTree(order=32, domain_name='single')
    for api in all_apis:
        tree.insert(api.get('subcategory', 'default'), api)
    return tree

# ============================================================
# Search functions
# ============================================================
def search_flat(query_emb, faiss_idx, all_apis, top_k=10):
    start = time.perf_counter()
    D, I = faiss_idx.search(query_emb.astype('float32').reshape(1, -1), top_k)
    latency = (time.perf_counter() - start) * 1000
    results = [all_apis[i] for i in I[0] if 0 <= i < len(all_apis)]
    return results, latency

def search_forest(query_emb, forest, top_k=10):
    start = time.perf_counter()
    # Step 1: route to best domain
    domain_scores = {}
    for domain, info in forest.items():
        sim = cosine_similarity(query_emb.reshape(1, -1), info['root_vector'].reshape(1, -1))[0][0]
        domain_scores[domain] = sim
    best_domain = max(domain_scores, key=domain_scores.get)
    # Step 2: search within domain tree
    results, scores, path = forest[best_domain]['tree'].search_with_traversal(query_emb, top_k=top_k)
    latency = (time.perf_counter() - start) * 1000
    return results, latency, best_domain, domain_scores

def search_single_tree(query_emb, tree, top_k=10):
    start = time.perf_counter()
    results, scores, path = tree.search_with_traversal(query_emb, top_k=top_k)
    latency = (time.perf_counter() - start) * 1000
    return results, latency

# ============================================================
# Metrics
# ============================================================
def compute_metrics(results, correct_domain, top_k_values=[1, 3, 5, 10]):
    metrics = {}
    domains_in_results = [r.get('domain', '') for r in results]
    
    for k in top_k_values:
        top_k_domains = domains_in_results[:k]
        # Accuracy@k: is there at least one correct domain result in top-k?
        metrics[f'acc@{k}'] = 1 if correct_domain in top_k_domains else 0
        # Domain Purity@k: fraction of top-k results from correct domain
        if top_k_domains:
            metrics[f'purity@{k}'] = sum(1 for d in top_k_domains if d == correct_domain) / len(top_k_domains)
        else:
            metrics[f'purity@{k}'] = 0.0
    
    # MRR
    for rank, d in enumerate(domains_in_results, 1):
        if d == correct_domain:
            metrics['mrr'] = 1.0 / rank
            break
    else:
        metrics['mrr'] = 0.0
    
    # Token count (Top-10 descriptions)
    metrics['tokens_top10'] = sum(len(r.get('description', '').split()) * 1.3 for r in results[:10])
    
    return metrics

# ============================================================
# Run experiment
# ============================================================
def run_m2_experiment(all_apis, test_queries, domains_dict, n_runs=5, top_k=10):
    faiss_idx = build_faiss(np.array([a['embedding'] for a in all_apis]))
    forest = build_forest(all_apis, domains_dict)
    single_tree = build_single_tree(all_apis)

    methods = ['forest', 'single_tree', 'flat']
    all_results = {m: {k: [] for k in [
        'acc@1', 'acc@3', 'acc@5', 'acc@10',
        'purity@1', 'purity@3', 'purity@5', 'purity@10',
        'mrr', 'latency', 'tokens_top10', 'routing_acc'
    ]} for m in methods}

    for run in range(n_runs):
        run_r = {m: {k: [] for k in all_results[m].keys()} for m in methods}

        for q in test_queries:
            q_emb = np.array(q['query_embedding'])
            cdomain = q['correct_domain']

            # Forest
            res_f, lat_f, routed, _ = search_forest(q_emb, forest, top_k=top_k)
            m_f = compute_metrics(res_f, cdomain)
            m_f['latency'] = lat_f
            m_f['routing_acc'] = 1 if routed == cdomain else 0
            for k, v in m_f.items():
                run_r['forest'][k].append(v)

            # Single B+ Tree
            res_s, lat_s = search_single_tree(q_emb, single_tree, top_k=top_k)
            m_s = compute_metrics(res_s, cdomain)
            m_s['latency'] = lat_s
            m_s['routing_acc'] = 0  # N/A
            for k, v in m_s.items():
                run_r['single_tree'][k].append(v)

            # Flat ANN
            res_a, lat_a = search_flat(q_emb, faiss_idx, all_apis, top_k=top_k)
            m_a = compute_metrics(res_a, cdomain)
            m_a['latency'] = lat_a
            m_a['routing_acc'] = 0  # N/A
            for k, v in m_a.items():
                run_r['flat'][k].append(v)

        for m in methods:
            for k in all_results[m]:
                all_results[m][k].append(np.mean(run_r[m][k]))

    return all_results

# ============================================================
# Scale experiment
# ============================================================
def run_scale(all_apis_ext, test_queries, model, skill_counts):
    all_domains_ext = {}
    for api in all_apis_ext:
        d = api['domain']
        if d not in all_domains_ext:
            all_domains_ext[d] = []
        all_domains_ext[d].append(api)

    scale = {m: {'acc@1': [], 'acc@10': [], 'purity@10': [], 'latency': []}
             for m in ['forest', 'single_tree', 'flat']}

    for N in skill_counts:
        per_domain = max(2, N // len(DOMAINS))
        min_avail = min(len(v) for v in all_domains_ext.values())
        if per_domain > min_avail:
            for m in scale:
                for k in scale[m]:
                    scale[m][k].append(None)
            continue

        subset = []
        sub_domains = {}
        for d in DOMAINS:
            sampled = all_domains_ext[d][:per_domain]
            subset.extend(sampled)
            sub_domains[d] = sampled

        faiss_idx = build_faiss(np.array([a['embedding'] for a in subset]))
        forest = build_forest(subset, sub_domains)
        single_tree = build_single_tree(subset)

        for m_name, search_fn in [
            ('forest', lambda q: search_forest(q, forest)),
            ('single_tree', lambda q: search_single_tree(q, single_tree)),
            ('flat', lambda q: search_flat(q, faiss_idx, subset)),
        ]:
            acc1s, acc10s, pur10s, lats = [], [], [], []
            for q in test_queries[:50]:
                q_emb = np.array(q['query_embedding'])
                cdomain = q['correct_domain']
                if m_name == 'forest':
                    res, lat, _, _ = search_fn(q_emb)
                else:
                    res, lat = search_fn(q_emb)
                met = compute_metrics(res, cdomain)
                acc1s.append(met['acc@1']); acc10s.append(met['acc@10'])
                pur10s.append(met['purity@10']); lats.append(lat)
            scale[m_name]['acc@1'].append(float(np.mean(acc1s)))
            scale[m_name]['acc@10'].append(float(np.mean(acc10s)))
            scale[m_name]['purity@10'].append(float(np.mean(pur10s)))
            scale[m_name]['latency'].append(float(np.mean(lats)))

    return scale

# ============================================================
# Visualization (bilingual)
# ============================================================
def generate_visualizations(results, scale, skill_counts):
    print("\nGenerating bilingual visualizations...")

    for lang in ['CN', 'EN']:
        is_cn = lang == 'CN'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        method_colors = {'forest': '#45B7D1', 'single_tree': '#FF6B6B', 'flat': '#4ECDC4'}
        method_labels = {
            'forest': 'Forest (有M2路由)' if is_cn else 'Forest (with M2)',
            'single_tree': '单棵B+树 (无M2)' if is_cn else 'Single B+ Tree (no M2)',
            'flat': 'Flat ANN/FAISS' if is_cn else 'Flat ANN/FAISS'
        }

        # Chart 1: Accuracy comparison (Acc@1, Acc@3, Acc@5, Acc@10)
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = ['acc@1', 'acc@3', 'acc@5', 'acc@10']
        labels = ['Acc@1', 'Acc@3', 'Acc@5', 'Acc@10']
        x = np.arange(len(metrics))
        width = 0.25
        for i, m in enumerate(['forest', 'single_tree', 'flat']):
            means = [np.mean(results[m][k]) for k in metrics]
            stds = [np.std(results[m][k]) for k in metrics]
            bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                          color=method_colors[m], label=method_labels[m], alpha=0.9)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylabel('准确率' if is_cn else 'Accuracy', fontsize=12)
        ax.set_title('M2路由层对比: 检索准确率' if is_cn else 'M2 Routing: Retrieval Accuracy Comparison',
                     fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.savefig(os.path.join(VIS_DIR, f'fig_M2_准确率对比_{lang}.png' if is_cn else f'fig_M2_accuracy_{lang}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Chart 2: Domain Purity comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = ['purity@1', 'purity@3', 'purity@5', 'purity@10']
        labels = ['Purity@1', 'Purity@3', 'Purity@5', 'Purity@10']
        x = np.arange(len(metrics))
        for i, m in enumerate(['forest', 'single_tree', 'flat']):
            means = [np.mean(results[m][k]) for k in metrics]
            stds = [np.std(results[m][k]) for k in metrics]
            bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
                          color=method_colors[m], label=method_labels[m], alpha=0.9)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{mean:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylabel('域纯度' if is_cn else 'Domain Purity', fontsize=12)
        ax.set_title('M2路由层对比: Top-K结果域纯度' if is_cn else 'M2 Routing: Domain Purity@K',
                     fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.savefig(os.path.join(VIS_DIR, f'fig_M2_域纯度_{lang}.png' if is_cn else f'fig_M2_purity_{lang}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Chart 3: Latency + Token comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Latency
        ax = axes[0]
        for m in ['forest', 'single_tree', 'flat']:
            lat = np.mean(results[m]['latency'])
            ax.bar(method_labels[m], lat, color=method_colors[m], alpha=0.9, width=0.5)
        ax.set_ylabel('延迟 (ms)' if is_cn else 'Latency (ms)', fontsize=12)
        ax.set_title('检索延迟对比' if is_cn else 'Retrieval Latency', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        # Token
        ax = axes[1]
        for m in ['forest', 'single_tree', 'flat']:
            tok = np.mean(results[m]['tokens_top10'])
            ax.bar(method_labels[m], tok, color=method_colors[m], alpha=0.9, width=0.5)
        ax.set_ylabel('Token消耗 (Top-10)' if is_cn else 'Token Consumption (Top-10)', fontsize=12)
        ax.set_title('Token消耗对比' if is_cn else 'Token Consumption', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(VIS_DIR, f'fig_M2_延迟Token_{lang}.png' if is_cn else f'fig_M2_latency_token_{lang}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Chart 4: Scale - Accuracy vs N
        valid = [c for c, v in zip(skill_counts, scale['forest']['acc@1']) if v is not None]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for idx, (metric, ylabel, title) in enumerate([
            ('acc@1', 'Acc@1', 'Accuracy@1 vs API Count'),
            ('acc@10', 'Acc@10', 'Accuracy@10 vs API Count')
        ]):
            ax = axes[idx]
            for m in ['forest', 'single_tree', 'flat']:
                vals = [v for v in scale[m][metric] if v is not None]
                ax.plot(valid, vals, 'o-', color=method_colors[m], label=method_labels[m],
                        linewidth=2.5, markersize=6)
            ax.set_xlabel('API数量' if is_cn else 'Number of APIs', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            title_cn = {'acc@1': 'Acc@1 vs API数量', 'acc@10': 'Acc@10 vs API数量'}
            ax.set_title(title_cn.get(metric, title) if is_cn else title, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle('M2路由层: 规模扩展实验' if is_cn else 'M2 Routing: Scale Experiment',
                     fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(VIS_DIR, f'fig_M2_规模准确率_{lang}.png' if is_cn else f'fig_M2_scale_accuracy_{lang}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Chart 5: Scale - Purity@10 vs N
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in ['forest', 'single_tree', 'flat']:
            vals = [v for v in scale[m]['purity@10'] if v is not None]
            ax.plot(valid, vals, 'o-', color=method_colors[m], label=method_labels[m],
                    linewidth=2.5, markersize=6)
        ax.set_xlabel('API数量' if is_cn else 'Number of APIs', fontsize=12)
        ax.set_ylabel('Domain Purity@10', fontsize=12)
        ax.set_title('M2路由层: 域纯度随规模变化' if is_cn else 'M2 Routing: Domain Purity@10 vs Scale',
                     fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.savefig(os.path.join(VIS_DIR, f'fig_M2_规模域纯度_{lang}.png' if is_cn else f'fig_M2_scale_purity_{lang}.png'),
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # Table
    table_data = []
    for m in ['forest', 'single_tree', 'flat']:
        row = {'Method': m}
        for k in ['acc@1', 'acc@10', 'purity@10', 'latency', 'tokens_top10', 'mrr']:
            row[k] = f"{np.mean(results[m][k]):.3f}±{np.std(results[m][k]):.3f}"
        if m == 'forest':
            row['routing_acc'] = f"{np.mean(results[m]['routing_acc']):.3f}±{np.std(results[m]['routing_acc']):.3f}"
        else:
            row['routing_acc'] = 'N/A'
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(16, 3))
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
    ax.set_title('Table: M2 Routing Layer Comparison (mean ± std)', fontweight='bold', fontsize=14, pad=20)
    fig.savefig(os.path.join(VIS_DIR, 'table_M2_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Saved bilingual visualizations to {VIS_DIR}")

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("M2 Routing Layer Experiment")
    print("Forest (5 trees + routing) vs Single B+ Tree vs Flat ANN")
    print("=" * 60)

    print("\n[1/5] Loading data...")
    all_apis, test_queries = load_dataset(DATA_DIR)
    domains_dict = {d: [] for d in DOMAINS}
    for api in all_apis:
        if api['domain'] in domains_dict:
            domains_dict[api['domain']].append(api)
    for d, apis in domains_dict.items():
        print(f"  {d}: {len(apis)} APIs")

    print("\n[2/5] Loading model & computing query embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    if 'query_embedding' not in test_queries[0]:
        texts = [q['query'] for q in test_queries]
        embs = model.encode(texts, batch_size=64)
        for q, emb in zip(test_queries, embs):
            q['query_embedding'] = emb.tolist()

    print("\n[3/5] Running M2 comparison (5 runs, Top-10)...")
    results = run_m2_experiment(all_apis, test_queries, domains_dict, n_runs=N_RUNS, top_k=10)

    # Print summary
    print("\n" + "-" * 60)
    for m in ['forest', 'single_tree', 'flat']:
        print(f"\n{m}:")
        for k in ['acc@1', 'acc@10', 'purity@10', 'latency', 'tokens_top10', 'mrr']:
            print(f"  {k}: {np.mean(results[m][k]):.3f} ± {np.std(results[m][k]):.3f}")
        if m == 'forest':
            print(f"  routing_acc: {np.mean(results[m]['routing_acc']):.3f}")

    print("\n[4/5] Scale experiment...")
    skill_counts = [100, 300, 500, 1000, 1500, 3000, 5000]
    _, all_apis_ext = generate_dataset(n_per_domain=1000, seed=42)
    compute_embeddings(all_apis_ext, model)
    scale = run_scale(all_apis_ext, test_queries, model, skill_counts)

    print("\n[5/5] Generating visualizations...")
    generate_visualizations(results, scale, skill_counts)

    # Save results
    output = {}
    for m in results:
        output[m] = {}
        for k, v in results[m].items():
            output[m][k] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
    output['scale'] = scale
    output['config'] = {'n_runs': N_RUNS, 'n_apis': len(all_apis), 'n_queries': len(test_queries)}

    with open(os.path.join(RES_DIR, 'm2_experiment_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {RES_DIR}")
    print(f"Visualizations: {VIS_DIR}")

if __name__ == '__main__':
    main()
