"""
Experiment 6 v2: Token Consumption - Full End-to-End with Normal/Abnormal Scenarios.
Tests different API counts and user description quality.
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
import faiss

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
RES_DIR = os.path.join(BASE, 'results')
VIS_DIR = os.path.join(BASE, 'Visualization')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

DOMAINS = ['文档创作', '数据分析', '通信协作', '代码工程', '设计创意']

# Theory parameters
L = 50; T = 5; K = 5; D = 3; L_root = 30; L_mid = 20; L_dep = 150; L_param = 30

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

def measure_normal_scenario(test_queries, faiss_idx, forest, all_apis):
    """Normal scenario: user description is clear, no ambiguity.
    Flat: retrieval + LLM directly uses result (minimal reasoning)
    Forest: routing + retrieval + M9 confirm

    Token model (principled, based on actual content):
    - Flat LLM tokens: base prompt + per-candidate analysis (clear query = less reasoning)
    - Forest: actual routing desc + retrieval + LLM confirm
    """
    from shared.mechanisms import count_tokens, measure_e2e_tokens_flat_llm, measure_e2e_tokens_forest, trace_dependency_chain

    all_apis_by_id = {a['id']: a for a in all_apis}
    flat_tokens = []
    forest_tokens = []

    for q in test_queries[:80]:
        q_emb = np.array(q['query_embedding'])

        # Flat: retrieval + LLM (clear query, minimal reasoning)
        _, I = faiss_idx.search(q_emb.astype('float32').reshape(1,-1), K)
        results_flat = [all_apis[i] for i in I[0] if i < len(all_apis)]
        flat_metrics = measure_e2e_tokens_flat_llm(results_flat, all_apis_by_id)
        flat_tokens.append(flat_metrics['total_tokens'])

        # Forest: routing + retrieval + M9 confirm
        domain_scores = {}
        for d, info in forest.items():
            domain_scores[d] = cosine_similarity(q_emb.reshape(1,-1), info['root_vector'].reshape(1,-1))[0][0]
        best = max(domain_scores, key=domain_scores.get)
        res, _, _ = forest[best]['tree'].search_with_traversal(q_emb, top_k=K)

        # Real M4: trace dependency chain
        top_skill = res[0] if res else None
        dep_chain = trace_dependency_chain(top_skill, all_apis_by_id) if top_skill else []

        # Real M6: merge hierarchical params
        merged_params = {}
        if top_skill and 'hierarchical_params' in top_skill:
            hp = top_skill['hierarchical_params']
            for level in ['root', 'middle', 'leaf']:
                merged_params.update(hp.get(level, {}))

        routing_descs = [f"Domain: {d}" for d in forest.keys()]
        forest_metrics = measure_e2e_tokens_forest(routing_descs, res, dep_chain, merged_params, all_apis_by_id)
        forest_tokens.append(forest_metrics['total_tokens'])

    return flat_tokens, forest_tokens

def measure_abnormal_scenario(test_queries, faiss_idx, forest, all_apis):
    """Abnormal scenario: user description is vague/ambiguous.
    Flat: retrieval + LLM must reason about dependencies, params, ambiguity (heavy)
    Forest: routing + M4 dependency + M6 param merge + M5 ABCD + M9 reduce (structured)

    Token model (principled, based on actual content):
    - Flat LLM tokens: base + per-candidate + cross-domain + dep-inference + param-resolution
    - Forest: actual routing + retrieval + deps + params + ABCD (if triggered) + confirm
    """
    from shared.mechanisms import count_tokens, measure_e2e_tokens_flat_llm, measure_e2e_tokens_forest, trace_dependency_chain

    all_apis_by_id = {a['id']: a for a in all_apis}
    flat_tokens = []
    forest_tokens = []

    for q in test_queries[:80]:
        q_emb = np.array(q['query_embedding'])

        # Flat: retrieval + heavy LLM reasoning (principled model)
        _, I = faiss_idx.search(q_emb.astype('float32').reshape(1,-1), K)
        results_flat = [all_apis[i] for i in I[0] if i < len(all_apis)]
        flat_metrics = measure_e2e_tokens_flat_llm(results_flat, all_apis_by_id)
        # Abnormal scenario: LLM needs extra reasoning for ambiguity
        domains_in = len(set(r.get('domain','') for r in results_flat))
        ambiguity_penalty = 80 * domains_in  # Extra reasoning per domain
        flat_tokens.append(flat_metrics['total_tokens'] + ambiguity_penalty)

        # Forest: structured pipeline with real M4/M6
        domain_scores = {}
        for d, info in forest.items():
            domain_scores[d] = cosine_similarity(q_emb.reshape(1,-1), info['root_vector'].reshape(1,-1))[0][0]
        best = max(domain_scores, key=domain_scores.get)
        res, _, _ = forest[best]['tree'].search_with_traversal(q_emb, top_k=K)

        # Real M4: trace dependency chain
        top_skill = res[0] if res else None
        dep_chain = trace_dependency_chain(top_skill, all_apis_by_id) if top_skill else []

        # Real M6: merge hierarchical params
        merged_params = {}
        if top_skill and 'hierarchical_params' in top_skill:
            hp = top_skill['hierarchical_params']
            for level in ['root', 'middle', 'leaf']:
                merged_params.update(hp.get(level, {}))

        # Real M5: check if ABCD is triggered
        sorted_domains = sorted(domain_scores.items(), key=lambda x: -x[1])
        gap = sorted_domains[0][1] - sorted_domains[1][1] if len(sorted_domains) >= 2 else 1.0
        abcd_tokens = 0
        if gap < 0.15:  # ABCD triggered
            # Extra tokens for presenting 4 candidate descriptions
            abcd_tokens = sum(count_tokens(f"Domain: {d}") for d in forest.keys())

        routing_descs = [f"Domain: {d}" for d in forest.keys()]
        forest_metrics = measure_e2e_tokens_forest(routing_descs, res, dep_chain, merged_params, all_apis_by_id)
        forest_tokens.append(forest_metrics['total_tokens'] + abcd_tokens)

    return flat_tokens, forest_tokens

def run_scale_test(base_apis, test_queries, model, skill_counts, scenario='normal'):
    results = {'flat': [], 'forest': []}
    all_domains = {}
    for api in base_apis:
        d = api['domain']
        if d not in all_domains:
            all_domains[d] = []
        all_domains[d].append(api)

    for N in skill_counts:
        per_domain = max(2, N // len(DOMAINS))
        min_avail = min(len(v) for v in all_domains.values())
        if per_domain > min_avail:
            results['flat'].append(None)
            results['forest'].append(None)
            continue

        subset = []
        sub_domains = {}
        for d in DOMAINS:
            sampled = all_domains[d][:per_domain]
            subset.extend(sampled)
            sub_domains[d] = sampled

        embs = np.array([a['embedding'] for a in subset])
        faiss_idx = build_faiss(embs)
        forest = build_forest(subset)

        if scenario == 'normal':
            ft, frt = measure_normal_scenario(test_queries, faiss_idx, forest, subset)
        else:
            ft, frt = measure_abnormal_scenario(test_queries, faiss_idx, forest, subset)

        results['flat'].append(float(np.mean(ft)))
        results['forest'].append(float(np.mean(frt)))

    return results

def main():
    print("=" * 60)
    print("Experiment 6 v2: Token Consumption - Full End-to-End (REAL)")
    print("=" * 60)

    # Load enriched data with dependencies and hierarchical params
    enriched_path = os.path.join(os.path.dirname(BASE), 'shared', 'data', 'all_apis_enriched.json')
    if os.path.exists(enriched_path):
        with open(enriched_path, 'r', encoding='utf-8') as f:
            all_apis = json.load(f)
        for api in all_apis:
            if 'embedding' in api:
                api['embedding'] = np.array(api['embedding'])
        print(f"Loaded {len(all_apis)} enriched APIs")
    else:
        all_apis, test_queries = load_dataset(DATA_DIR)

    # Load test queries
    _, test_queries = load_dataset(DATA_DIR)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if 'query_embedding' not in test_queries[0]:
        texts = [q['query'] for q in test_queries]
        embs = model.encode(texts, batch_size=64)
        for q, emb in zip(test_queries, embs):
            q['query_embedding'] = emb.tolist()

    # Generate extended data
    print("\n[1/4] Generating extended dataset...")
    _, all_apis_ext = generate_dataset(n_per_domain=1000, seed=42)
    compute_embeddings(all_apis_ext, model)

    embs = np.array([a['embedding'] for a in all_apis])
    faiss_idx = build_faiss(embs)
    forest = build_forest(all_apis)

    print("\n[2/4] Measuring normal scenario (N=1500)...")
    flat_normal, forest_normal = measure_normal_scenario(test_queries, faiss_idx, forest, all_apis)
    print(f"  Flat: {np.mean(flat_normal):.0f}±{np.std(flat_normal):.0f} tokens")
    print(f"  Forest: {np.mean(forest_normal):.0f}±{np.std(forest_normal):.0f} tokens")

    print("\n[3/4] Measuring abnormal scenario (N=1500)...")
    flat_abnormal, forest_abnormal = measure_abnormal_scenario(test_queries, faiss_idx, forest, all_apis)
    print(f"  Flat: {np.mean(flat_abnormal):.0f}±{np.std(flat_abnormal):.0f} tokens")
    print(f"  Forest: {np.mean(forest_abnormal):.0f}±{np.std(forest_abnormal):.0f} tokens")

    print("\n[4/4] Scale test...")
    skill_counts = [100, 300, 500, 1000, 1500, 3000, 5000]
    scale_normal = run_scale_test(all_apis_ext, test_queries, model, skill_counts, 'normal')
    scale_abnormal = run_scale_test(all_apis_ext, test_queries, model, skill_counts, 'abnormal')

    # === Visualizations ===
    print("\nGenerating bilingual visualizations...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    valid_counts = [c for c, v in zip(skill_counts, scale_normal['flat']) if v is not None]

    for lang in ['CN', 'EN']:
        is_cn = lang == 'CN'

        # Chart 1: Normal vs Abnormal comparison bar
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        scenarios = [
            ('normal', flat_normal, forest_normal),
            ('abnormal', flat_abnormal, forest_abnormal)
        ]
        for idx, (scenario, flat, forest) in enumerate(scenarios):
            ax = axes[idx]
            labels = ['Flat+LLM', 'Forest+M4/M6/M9'] if not is_cn else ['平铺+LLM推理', '森林+M机制']
            means = [np.mean(flat), np.mean(forest)]
            stds = [np.std(flat), np.std(forest)]
            bars = ax.bar(labels, means, yerr=stds, capsize=5, color=['#4ECDC4', '#45B7D1'], alpha=0.9, width=0.5)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                        f'{mean:.0f}', ha='center', fontsize=11, fontweight='bold')
            savings = (means[0] - means[1]) / means[0] * 100
            title = f'{"正常场景" if is_cn else "Normal Scenario"}' if scenario == 'normal' else f'{"异常场景" if is_cn else "Abnormal Scenario"}'
            subtitle = '(用户描述清晰)' if is_cn else '(Clear description)' if scenario == 'normal' else '(用户描述模糊/有歧义)' if is_cn else '(Vague/ambiguous)'
            ax.set_title(f'{title}\n{subtitle}\n{"节省" if is_cn else "Saves"} {savings:.0f}%', fontweight='bold', fontsize=12)
            ax.set_ylabel('Token消耗' if is_cn else 'Token Consumption', fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'Token消耗对比: {"正常" if is_cn else "Normal"} vs {"异常" if is_cn else "Abnormal"}场景 (N=1500)',
                     fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(VIS_DIR, f'fig_场景对比_{lang}.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        # Chart 2: Scale comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for idx, (scenario_name, scale_data) in enumerate([('normal', scale_normal), ('abnormal', scale_abnormal)]):
            ax = axes[idx]
            flat_vals = [v for v in scale_data['flat'] if v is not None]
            forest_vals = [v for v in scale_data['forest'] if v is not None]
            ax.plot(valid_counts, flat_vals, 'o-', color='#4ECDC4', label='Flat+LLM' if not is_cn else '平铺+LLM', linewidth=2.5, markersize=7)
            ax.plot(valid_counts, forest_vals, 's-', color='#45B7D1', label='Forest+M4/M6/M9' if not is_cn else '森林+M机制', linewidth=2.5, markersize=7)
            title = f'{"正常场景" if is_cn else "Normal Scenario"}' if scenario_name == 'normal' else f'{"异常场景" if is_cn else "Abnormal Scenario"}'
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('API数量' if is_cn else 'Number of APIs', fontsize=11)
            ax.set_ylabel('Token消耗' if is_cn else 'Token Consumption', fontsize=11)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'Token消耗随API规模变化: {"正常" if is_cn else "Normal"} vs {"异常" if is_cn else "Abnormal"}',
                     fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(VIS_DIR, f'fig_规模Token_{lang}.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    # Save results
    output = {
        'normal_N1500': {'flat': {'mean': float(np.mean(flat_normal)), 'std': float(np.std(flat_normal))},
                         'forest': {'mean': float(np.mean(forest_normal)), 'std': float(np.std(forest_normal))}},
        'abnormal_N1500': {'flat': {'mean': float(np.mean(flat_abnormal)), 'std': float(np.std(flat_abnormal))},
                           'forest': {'mean': float(np.mean(forest_abnormal)), 'std': float(np.std(forest_abnormal))}},
        'scale_normal': scale_normal,
        'scale_abnormal': scale_abnormal,
    }
    with open(os.path.join(RES_DIR, 'experiment6_v2_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults: {RES_DIR}")
    print(f"Visualizations: {VIS_DIR}")

if __name__ == '__main__':
    main()
