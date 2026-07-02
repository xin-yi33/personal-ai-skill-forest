"""
Baseline methods for comparison experiments.
Implements: HNSW, BM25, Random, FAISS+Rerank, LLM Full-Context, Cross-Encoder Rerank.
"""
import numpy as np
import faiss
import time
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# HNSW Baseline
# ============================================================
class HNSWIndex:
    """FAISS HNSW index for approximate nearest neighbor search."""
    
    def __init__(self, d: int = 384, M: int = 32, efConstruction: int = 64):
        self.d = d
        self.index = faiss.IndexHNSWFlat(d, M)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = 64  # search-time parameter
    
    def build(self, embeddings: np.ndarray):
        """Build HNSW index from embeddings."""
        embeddings = embeddings.astype('float32')
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        query = query.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query)
        D, I = self.index.search(query, k)
        return D, I


def build_hnsw_index(embeddings: np.ndarray) -> HNSWIndex:
    """Build HNSW index from embeddings."""
    d = embeddings.shape[1]
    hnsw = HNSWIndex(d=d)
    hnsw.build(embeddings)
    return hnsw


def search_hnsw(query_emb: np.ndarray, hnsw: HNSWIndex, 
                all_apis: List[Dict], top_k: int = 10):
    """Search using HNSW index, return results with latency."""
    start = time.perf_counter()
    D, I = hnsw.search(query_emb, k=top_k)
    latency = (time.perf_counter() - start) * 1000
    results = [all_apis[i] for i in I[0] if 0 <= i < len(all_apis)]
    return results, latency


# ============================================================
# BM25 Baseline
# ============================================================
class BM25Index:
    """BM25 index using rank_bm25 library."""
    
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = []
    
    def build(self, apis: List[Dict]):
        """Build BM25 index from API descriptions."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank_bm25")
        
        # Tokenize: simple character-level for Chinese + word-level for English
        self.tokenized_corpus = []
        for api in apis:
            desc = api.get('description', '') + ' ' + api.get('name', '') + ' ' + api.get('domain', '')
            tokens = self._tokenize(desc)
            self.tokenized_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for mixed Chinese/English text."""
        import re
        # Split Chinese characters individually, keep English words
        tokens = []
        # Split by non-alphanumeric/non-CJK
        parts = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
        tokens.extend(parts)
        return tokens
    
    def search(self, query: str, k: int = 10) -> List[int]:
        """Search for top-k matching indices."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices.tolist()


def build_bm25_index(apis: List[Dict]) -> BM25Index:
    """Build BM25 index from APIs."""
    bm25 = BM25Index()
    bm25.build(apis)
    return bm25


def search_bm25(query: str, bm25: BM25Index, all_apis: List[Dict], top_k: int = 10):
    """Search using BM25, return results with latency."""
    start = time.perf_counter()
    indices = bm25.search(query, k=top_k)
    latency = (time.perf_counter() - start) * 1000
    results = [all_apis[i] for i in indices if 0 <= i < len(all_apis)]
    return results, latency


# ============================================================
# Random Baseline
# ============================================================
def search_random(all_apis: List[Dict], top_k: int = 10, seed: int = 42):
    """Random selection baseline (theoretical lower bound)."""
    start = time.perf_counter()
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(all_apis), min(top_k, len(all_apis)), replace=False)
    latency = (time.perf_counter() - start) * 1000
    results = [all_apis[i] for i in indices]
    return results, latency


# ============================================================
# FAISS Top-K + Embedding Rerank (production mainstream)
# ============================================================
def search_faiss_rerank(query_emb: np.ndarray, faiss_index, all_apis: List[Dict],
                        top_k_retrieve: int = 20, top_k_final: int = 5,
                        reranker_emb_dim: int = 384):
    """
    Two-stage retrieval: FAISS coarse retrieval → embedding similarity rerank.
    Simulates the "FAISS + LLM rerank" production pattern.
    """
    start = time.perf_counter()
    
    # Stage 1: Coarse retrieval (top-20)
    D, I = faiss_index.search(query_emb.astype('float32').reshape(1, -1), top_k_retrieve)
    candidates = [all_apis[i] for i in I[0] if 0 <= i < len(all_apis)]
    
    # Stage 2: Rerank by cosine similarity (simulating LLM rerank with embedding)
    if candidates:
        candidate_embs = np.array([a['embedding'] for a in candidates])
        sims = cosine_similarity(query_emb.reshape(1, -1), candidate_embs)[0]
        ranked_indices = np.argsort(sims)[::-1][:top_k_final]
        results = [candidates[i] for i in ranked_indices]
    else:
        results = []
    
    latency = (time.perf_counter() - start) * 1000
    return results, latency, candidates  # candidates = stage-1 results for token counting


# ============================================================
# LLM Full-Context (Gorilla/ToolLLM style)
# ============================================================
def measure_e2e_tokens_llm_full(all_apis: List[Dict], selected_api: Dict,
                                 all_apis_by_id: Dict[str, Dict]) -> Dict:
    """
    LLM Full-Context: put ALL API descriptions in prompt, LLM selects directly.
    Represents Gorilla/ToolLLM approach.
    
    Token model:
    - All API descriptions in prompt
    - LLM reasoning to select one
    - Dependency inference for selected API
    """
    from shared.mechanisms import count_tokens, _compute_flat_chain_completeness
    
    # All descriptions must be in the prompt
    all_desc_tokens = sum(count_tokens(api.get('description', '')) for api in all_apis)
    
    # Formatting overhead: each API needs "ID: description\n"
    formatting_overhead = len(all_apis) * 5  # ~5 tokens per API for formatting
    
    # LLM reasoning: must analyze all N APIs and select one
    n = len(all_apis)
    reasoning_tokens = 100 + n * 10  # ~10 tokens per API analysis (lighter than per-candidate)
    
    # Dependency inference for selected API
    dep_inference = 0
    if selected_api:
        reqs = selected_api.get('requires', [])
        dep_inference = 30 + len(reqs) * 20
    
    total = all_desc_tokens + formatting_overhead + reasoning_tokens + dep_inference
    
    return {
        'total_tokens': total,
        'retrieval_tokens': all_desc_tokens + formatting_overhead,
        'llm_tokens': reasoning_tokens + dep_inference,
        'prompt_tokens': all_desc_tokens + formatting_overhead,
        'reasoning_tokens': reasoning_tokens,
    }


def measure_e2e_tokens_faiss_rerank(stage1_results: List[Dict], 
                                     final_results: List[Dict],
                                     all_apis_by_id: Dict[str, Dict]) -> Dict:
    """
    FAISS + Rerank end-to-end token model.
    
    Token model:
    - Stage 1: FAISS top-20 descriptions in context
    - Stage 2: LLM reranking reasoning for 20 candidates
    - Dependency inference for top-1
    """
    from shared.mechanisms import count_tokens, _compute_flat_chain_completeness
    
    # Stage 1: all 20 candidate descriptions
    stage1_tokens = sum(count_tokens(api.get('description', '')) for api in stage1_results)
    
    # Stage 2: LLM reranking (reads 20 descriptions, ranks them)
    n_candidates = len(stage1_results)
    rerank_reasoning = 50 + n_candidates * 30  # ~30 tokens per candidate for reranking
    
    # Dependency inference for top-1
    top_skill = final_results[0] if final_results else None
    dep_inference = 0
    if top_skill:
        reqs = top_skill.get('requires', [])
        dep_inference = 30 + len(reqs) * 20
    
    # Cross-domain penalty
    domains = set(r.get('domain', '') for r in final_results)
    cross_domain = 100 if len(domains) > 1 else 0
    
    # Parameter resolution
    param_res = 50 if len(final_results) > 1 else 0
    
    total = stage1_tokens + rerank_reasoning + dep_inference + cross_domain + param_res
    
    return {
        'total_tokens': total,
        'retrieval_tokens': stage1_tokens,
        'llm_tokens': rerank_reasoning + dep_inference + cross_domain + param_res,
        'stage1_tokens': stage1_tokens,
        'rerank_tokens': rerank_reasoning,
    }


# ============================================================
# Cross-Encoder Rerank (academic SOTA)
# ============================================================
class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
        except Exception:
            self.available = False
            self.model = None
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank candidates using cross-encoder scores."""
        if not self.available or not candidates:
            return candidates[:top_k]
        
        pairs = [(query, api.get('description', '')) for api in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in ranked[:top_k]]


def search_cross_encoder_rerank(query: str, query_emb: np.ndarray, 
                                 faiss_index, all_apis: List[Dict],
                                 reranker: CrossEncoderReranker,
                                 top_k_retrieve: int = 20, top_k_final: int = 5):
    """
    Two-stage: FAISS coarse retrieval → Cross-encoder fine ranking.
    Academic SOTA approach.
    """
    start = time.perf_counter()
    
    # Stage 1: FAISS coarse retrieval
    D, I = faiss_index.search(query_emb.astype('float32').reshape(1, -1), top_k_retrieve)
    candidates = [all_apis[i] for i in I[0] if 0 <= i < len(all_apis)]
    
    # Stage 2: Cross-encoder rerank
    results = reranker.rerank(query, candidates, top_k=top_k_final)
    
    latency = (time.perf_counter() - start) * 1000
    return results, latency


# ============================================================
# Unified evaluation helper
# ============================================================
def evaluate_baseline(results: List[Dict], correct_domain: str, 
                      correct_subcat: str = None) -> Tuple[int, int, int, float]:
    """
    Unified retrieval evaluation for all baselines.
    Returns: (acc1, acc3, acc5, mrr)
    """
    def subcat_match(api):
        if correct_subcat:
            return api.get('subcategory') == correct_subcat
        return api.get('domain') == correct_domain
    
    def domain_match(api):
        return api.get('domain') == correct_domain
    
    acc1 = 1 if results and subcat_match(results[0]) else 0
    acc3 = 1 if any(subcat_match(r) for r in results[:3]) else 0
    acc5 = 1 if any(domain_match(r) for r in results[:5]) else 0
    
    mrr = 0.0
    for rank, r in enumerate(results[:10], 1):
        if subcat_match(r):
            mrr = 1.0 / rank
            break
    
    return acc1, acc3, acc5, mrr
