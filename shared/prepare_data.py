"""
Prepare shared dataset for all experiments.
Generates 1500 APIs across 5 domains and 200 test queries.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
from shared.data_generator import generate_dataset, generate_test_queries, compute_embeddings, save_dataset
from sentence_transformers import SentenceTransformer

def main():
    print("=" * 60)
    print("Preparing Shared Dataset for Skill Forest Experiments")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "shared", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Generate synthetic dataset
    print("\n[1/4] Generating 1500 APIs across 5 domains...")
    start = time.time()
    all_domains, all_apis = generate_dataset(n_per_domain=1000, seed=42)
    for domain, apis in all_domains.items():
        print(f"  {domain}: {len(apis)} APIs")
    print(f"  Total: {len(all_apis)} APIs ({time.time()-start:.1f}s)")

    # Step 2: Generate test queries
    print("\n[2/4] Generating 200 test queries (140 clear + 60 ambiguous)...")
    test_queries = generate_test_queries(n_clear=140, n_ambiguous=60, seed=42)
    clear = sum(1 for q in test_queries if not q['cross_domain_ambiguous'])
    ambig = sum(1 for q in test_queries if q['cross_domain_ambiguous'])
    print(f"  Clear: {clear}, Ambiguous: {ambig}")

    # Step 3: Load embedding model and compute embeddings
    print("\n[3/4] Loading embedding model (all-MiniLM-L6-v2)...")
    start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Model loaded ({time.time()-start:.1f}s)")

    print("\n[4/4] Computing embeddings for 1500 APIs...")
    start = time.time()
    embeddings = compute_embeddings(all_apis, model)
    print(f"  Embeddings computed: shape={embeddings.shape} ({time.time()-start:.1f}s)")


    # Step 3b: Compute query embeddings
    print("\n[3b] Computing query embeddings...")
    query_texts = [q['query'] for q in test_queries]
    query_embs = model.encode(query_texts, show_progress_bar=False, batch_size=64)
    for q, emb in zip(test_queries, query_embs):
        q['query_embedding'] = emb.tolist()
    print(f"  Query embeddings: {len(test_queries)} queries")

    # Save
    print("\nSaving dataset...")
    save_dataset(all_domains, all_apis, test_queries, data_dir)

    # Also copy to each experiment's data folder
    for exp_dir in ['Exp1_Retrieval_Performance', 'Exp2_Ablation_Study',
                    'Exp3_Threshold_Sensitivity', 'Exp4_Action_Reflection',
                    'Exp5_Thought_Reflection', 'Exp6_Token_Consumption']:
        exp_data_dir = os.path.join(base_dir, exp_dir, 'data')
        os.makedirs(exp_data_dir, exist_ok=True)
        save_dataset(all_domains, all_apis, test_queries, exp_data_dir)

    print("\nDataset preparation complete!")
    print(f"  Location: {data_dir}")
    print(f"  APIs: {len(all_apis)}")
    print(f"  Queries: {len(test_queries)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

if __name__ == '__main__':
    main()

