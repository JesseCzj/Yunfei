"""
DSAG Embedding Index

Provides semantic search over leaf nodes using embeddings.
Supports "semantic snap" when confidence is low.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .schema import DSAGGraph, DSAGNode, TaxonomyTree


# ============== Embedding Provider ==============

def _get_embeddings_model():
    """Get embeddings model based on environment configuration."""
    provider = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_EBD_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_EBD_API_KEY or OPENAI_API_KEY is not set")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        base_url = os.getenv("OPENAI_EBD_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if base_url:
            return OpenAIEmbeddings(api_key=api_key, model=model, base_url=base_url)
        return OpenAIEmbeddings(api_key=api_key, model=model)
    
    if provider == "deepseek":
        # DeepSeek doesn't have embeddings, fallback to OpenAI-compatible embeddings
        from langchain_openai import OpenAIEmbeddings
        api_key = os.getenv("OPENAI_EBD_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_EBD_API_KEY or OPENAI_API_KEY is required for embeddings when using DeepSeek")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        base_url = os.getenv("OPENAI_EBD_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if base_url:
            return OpenAIEmbeddings(api_key=api_key, model=model, base_url=base_url)
        return OpenAIEmbeddings(api_key=api_key, model=model)
    
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# ============== Search Result ==============

@dataclass
class SearchResult:
    """Result of an embedding search."""
    node_id: str
    node: DSAGNode
    score: float
    snapped: bool = False  # True if confidence was low but we "snapped" to nearest
    snap_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.node.label,
            "description": self.node.description,
            "score": round(self.score, 4),
            "snapped": self.snapped,
            "snap_reason": self.snap_reason,
        }


# ============== Embedding Index ==============

class EmbeddingIndex:
    """
    Index for semantic search over DSAG leaf nodes.
    """
    
    # Threshold for confident match
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    # Threshold for semantic snap (below this we snap with warning)
    SNAP_THRESHOLD = 0.45
    
    def __init__(self, graph: DSAGGraph):
        self.graph = graph
        self.embeddings_model = None
        
        # Embeddings storage: node_id -> embedding vector
        self.expert_leaf_embeddings: Dict[str, List[float]] = {}
        self.researcher_leaf_embeddings: Dict[str, List[float]] = {}
        
        # Text storage for each node (label + description)
        self.expert_leaf_texts: Dict[str, str] = {}
        self.researcher_leaf_texts: Dict[str, str] = {}
    
    def _get_embeddings_model(self):
        """Lazy load embeddings model."""
        if self.embeddings_model is None:
            self.embeddings_model = _get_embeddings_model()
        return self.embeddings_model
    
    def _node_to_text(self, node: DSAGNode) -> str:
        """Convert node to searchable text."""
        parts = [node.label]
        if node.description:
            parts.append(node.description)
        if node.aliases:
            parts.extend(node.aliases)
        return " | ".join(parts)
    
    def build_index(self) -> None:
        """
        Build embeddings for all leaf nodes.
        This should be called once after graph creation.
        """
        print("[EmbeddingIndex] Building embeddings index...")
        model = self._get_embeddings_model()
        
        # Collect expert leaves
        expert_leaves = self.graph.expert_tree.get_leaves()
        expert_texts = []
        expert_ids = []
        for leaf in expert_leaves:
            text = self._node_to_text(leaf)
            self.expert_leaf_texts[leaf.id] = text
            expert_texts.append(text)
            expert_ids.append(leaf.id)
        
        # Collect researcher leaves
        researcher_leaves = self.graph.researcher_tree.get_leaves()
        researcher_texts = []
        researcher_ids = []
        for leaf in researcher_leaves:
            text = self._node_to_text(leaf)
            self.researcher_leaf_texts[leaf.id] = text
            researcher_texts.append(text)
            researcher_ids.append(leaf.id)
        
        # Batch embed all texts
        all_texts = expert_texts + researcher_texts
        if not all_texts:
            print("[EmbeddingIndex] No leaves to embed")
            return
        
        print(f"[EmbeddingIndex] Embedding {len(expert_texts)} expert + {len(researcher_texts)} researcher leaves...")
        all_embeddings = model.embed_documents(all_texts)
        
        # Store embeddings
        for i, node_id in enumerate(expert_ids):
            self.expert_leaf_embeddings[node_id] = all_embeddings[i]
        
        offset = len(expert_ids)
        for i, node_id in enumerate(researcher_ids):
            self.researcher_leaf_embeddings[node_id] = all_embeddings[offset + i]
        
        print(f"[EmbeddingIndex] Index built: {len(self.expert_leaf_embeddings)} expert, {len(self.researcher_leaf_embeddings)} researcher embeddings")
    
    def _search_embeddings(
        self,
        query_embedding: List[float],
        embeddings_dict: Dict[str, List[float]],
        tree: TaxonomyTree,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Search for nearest nodes in an embeddings dictionary.
        """
        if not embeddings_dict:
            return []
        
        # Compute similarities
        similarities: List[Tuple[str, float]] = []
        for node_id, embedding in embeddings_dict.items():
            score = _cosine_similarity(query_embedding, embedding)
            similarities.append((node_id, score))
        
        # Sort by score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for node_id, score in similarities[:top_k]:
            node = tree.get_node(node_id)
            if node is None:
                continue
            
            # Determine if this is a confident match or semantic snap
            snapped = score < self.HIGH_CONFIDENCE_THRESHOLD
            snap_reason = ""
            if snapped and score >= self.SNAP_THRESHOLD:
                snap_reason = f"Low confidence match (score={score:.2f}), snapped to nearest concept"
            elif snapped and score < self.SNAP_THRESHOLD:
                snap_reason = f"Very low confidence (score={score:.2f}), may not be accurate"
            
            results.append(SearchResult(
                node_id=node_id,
                node=node,
                score=score,
                snapped=snapped,
                snap_reason=snap_reason,
            ))
        
        return results
    
    def search_expert_leaves(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Search expert leaves for the nearest match to a query.
        Typically used to locate expert's current statement.
        """
        if not self.expert_leaf_embeddings:
            return []
        
        model = self._get_embeddings_model()
        query_embedding = model.embed_query(query)
        
        return self._search_embeddings(
            query_embedding,
            self.expert_leaf_embeddings,
            self.graph.expert_tree,
            top_k,
        )
    
    def search_researcher_leaves(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Search researcher leaves for the nearest match to a query.
        Typically used to locate researcher's question intent.
        """
        if not self.researcher_leaf_embeddings:
            return []
        
        model = self._get_embeddings_model()
        query_embedding = model.embed_query(query)
        
        return self._search_embeddings(
            query_embedding,
            self.researcher_leaf_embeddings,
            self.graph.researcher_tree,
            top_k,
        )
    
    def get_embeddings_data(self) -> Dict[str, Dict[str, List[float]]]:
        """Get all embeddings for serialization."""
        return {
            "expert": self.expert_leaf_embeddings,
            "researcher": self.researcher_leaf_embeddings,
        }
    
    def load_embeddings_data(self, data: Dict[str, Dict[str, List[float]]]) -> None:
        """Load embeddings from serialized data."""
        self.expert_leaf_embeddings = data.get("expert", {})
        self.researcher_leaf_embeddings = data.get("researcher", {})
        
        # Rebuild text mappings
        for leaf in self.graph.expert_tree.get_leaves():
            self.expert_leaf_texts[leaf.id] = self._node_to_text(leaf)
        for leaf in self.graph.researcher_tree.get_leaves():
            self.researcher_leaf_texts[leaf.id] = self._node_to_text(leaf)


# ============== Convenience Function ==============

def build_embedding_index(graph: DSAGGraph) -> EmbeddingIndex:
    """
    Build an embedding index for a DSAG graph.
    
    Args:
        graph: The DSAG graph to index
    
    Returns:
        EmbeddingIndex with all leaf embeddings computed
    """
    index = EmbeddingIndex(graph)
    index.build_index()
    return index
