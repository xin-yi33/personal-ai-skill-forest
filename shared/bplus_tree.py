"""
B+ Tree Implementation with Multi-level Traversal
Supports representative vectors at intermediate nodes for routing decisions.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
import copy


class BPlusTreeNode:
    """B+ Tree node supporting multi-level traversal with representative vectors."""

    def __init__(self, is_leaf: bool = False):
        self.is_leaf = is_leaf
        self.keys: List[str] = []
        self.children: List['BPlusTreeNode'] = []
        self.apis: List[Dict] = []
        self.parent: Optional['BPlusTreeNode'] = None
        self.representative_vector: Optional[np.ndarray] = None
        self.metadata: Dict = {}

    def compute_representative_vector(self, model=None):
        """Compute representative vector as mean of child vectors or leaf embeddings."""
        if self.is_leaf:
            if self.apis and model is not None:
                descs = [api.get('description', '') for api in self.apis]
                if descs:
                    embeddings = model.encode(descs)
                    self.representative_vector = np.mean(embeddings, axis=0)
            return

        for child in self.children:
            child.compute_representative_vector(model)

        vectors = [c.representative_vector for c in self.children if c.representative_vector is not None]
        if vectors:
            self.representative_vector = np.mean(vectors, axis=0)


class BPlusTree:
    """B+ Tree with order parameter, supporting multi-level traversal search."""

    def __init__(self, order: int = 32, domain_name: str = ""):
        self.order = order
        self.domain_name = domain_name
        self.root = BPlusTreeNode(is_leaf=True)
        self.all_apis: List[Dict] = []
        self._depth = 0

    def insert(self, key: str, api: Dict):
        """Insert an API into the B+ tree at the leaf determined by key."""
        self.all_apis.append(api)
        leaf = self._find_leaf(key)
        leaf.apis.append(api)
        if key not in leaf.keys:
            leaf.keys.append(key)
        if len(leaf.apis) > self.order:
            self._split_leaf(leaf)

    def _find_leaf(self, key: str) -> BPlusTreeNode:
        node = self.root
        while not node.is_leaf:
            idx = self._select_child(key, node)
            node = node.children[idx]
        return node

    def _select_child(self, key: str, node: BPlusTreeNode) -> int:
        for i, k in enumerate(node.keys):
            if key < k:
                return i
        return len(node.keys)

    def _split_leaf(self, leaf: BPlusTreeNode):
        mid = len(leaf.apis) // 2
        new_leaf = BPlusTreeNode(is_leaf=True)
        new_leaf.apis = leaf.apis[mid:]
        new_leaf.keys = leaf.keys[mid:] if len(leaf.keys) > mid else []
        leaf.apis = leaf.apis[:mid]
        leaf.keys = leaf.keys[:mid] if len(leaf.keys) > mid else []

        if leaf.parent is None:
            new_root = BPlusTreeNode(is_leaf=False)
            new_root.children = [leaf, new_leaf]
            new_root.keys = [new_leaf.keys[0]] if new_leaf.keys else []
            leaf.parent = new_root
            new_leaf.parent = new_root
            self.root = new_root
        else:
            parent = leaf.parent
            idx = parent.children.index(leaf)
            parent.children.insert(idx + 1, new_leaf)
            if new_leaf.keys:
                parent.keys.insert(idx, new_leaf.keys[0])
            new_leaf.parent = parent
            if len(parent.children) > self.order:
                self._split_internal(parent)

    def _split_internal(self, node: BPlusTreeNode):
        mid = len(node.children) // 2
        new_node = BPlusTreeNode(is_leaf=False)
        new_node.children = node.children[mid:]
        new_node.keys = node.keys[mid:]
        node.children = node.children[:mid]
        node.keys = node.keys[:mid - 1] if mid > 0 else []

        for child in new_node.children:
            child.parent = new_node

        if node.parent is None:
            new_root = BPlusTreeNode(is_leaf=False)
            new_root.children = [node, new_node]
            split_key = new_node.keys[0] if new_node.keys else ""
            new_root.keys = [split_key]
            node.parent = new_root
            new_node.parent = new_root
            self.root = new_root
        else:
            parent = node.parent
            idx = parent.children.index(node)
            parent.children.insert(idx + 1, new_node)
            if new_node.keys:
                parent.keys.insert(idx, new_node.keys[0])
            new_node.parent = parent
            if len(parent.children) > self.order:
                self._split_internal(parent)

    def search_with_traversal(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[Dict], List[float], List[Dict]]:
        """
        Multi-level traversal search:
        1. Root -> select branch by similarity to child representative vectors
        2. Drill down level by level
        3. At leaf, do embedding + top-k fine-ranking
        """
        node = self.root
        traversal_path = []

        while not node.is_leaf:
            similarities = []
            for child in node.children:
                if child.representative_vector is not None:
                    sim = cosine_similarity(
                        query_vector.reshape(1, -1),
                        child.representative_vector.reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)
                else:
                    similarities.append(-1.0)

            best_idx = int(np.argmax(similarities))
            traversal_path.append({
                'level': len(traversal_path),
                'selected_child': best_idx,
                'similarity': float(similarities[best_idx]),
                'alternatives': [float(s) for s in similarities]
            })
            node = node.children[best_idx]

        if not node.apis:
            return [], [], traversal_path

        leaf_embeddings = np.array([
            api.get('embedding', np.zeros(384)) for api in node.apis
        ])
        scores = cosine_similarity(query_vector.reshape(1, -1), leaf_embeddings)[0]
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        results = [node.apis[i] for i in top_k_indices]
        result_scores = [float(scores[i]) for i in top_k_indices]

        return results, result_scores, traversal_path

    def get_depth(self) -> int:
        depth = 0
        node = self.root
        while not node.is_leaf:
            depth += 1
            node = node.children[0]
        return depth

    def get_leaf_count(self) -> int:
        count = 0
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
        queue = [self.root]
        while queue:
            n = queue.pop(0)
            if n.is_leaf:
                count += 1
            else:
                queue.extend(n.children)
        return count
