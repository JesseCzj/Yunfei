"""
DSAG Schema Definitions

Defines the JSON-serializable data structures for:
- Taxonomy tree nodes (TExp and TRes)
- Gap links (edges connecting expert and researcher leaves)
- Type-specific assistance payloads
- Complete DSAG graph structure
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal
from enum import Enum


class Tower(str, Enum):
    """Which tower a node belongs to."""
    EXPERT = "expert"
    RESEARCHER = "researcher"
    SHARED = "shared"  # For root node


class Layer(str, Enum):
    """Layer depth in the taxonomy tree."""
    ROOT = "root"      # L0 - Shared topic
    L1 = "L1"          # Perspective layer
    L2 = "L2"          # Category layer
    LEAF = "leaf"      # L3 - Specific pain points / research goals


class RelationType(str, Enum):
    """Type of mismatch/gap between linked leaves (mismap categories)."""
    LEXICAL_GAP = "LexicalGap"         # Terminology or wording mismatch
    CONCEPTUAL_GAP = "ConceptualGap"   # Mental model / concept misunderstanding
    TACIT_GAP = "TacitGap"             # Intuition/tacit knowledge not articulated
    SCOPE_GAP = "ScopeGap"             # Focus/expectation boundary mismatch
    PROCESS_GAP = "ProcessGap"         # Workflow/sequence misalignment


@dataclass
class DSAGNode:
    """
    A node in the taxonomy tree.
    
    Attributes:
        id: Unique identifier (e.g., "exp_L1_01", "res_leaf_03")
        tower: Which tower this node belongs to
        layer: Depth layer in the tree
        label: Short label for the node
        description: Optional longer description
        parent_id: ID of parent node (None for root)
        children_ids: List of child node IDs
        attributes: Tacit Knowledge Facets (implicit expert knowledge anchors),
                   spanning decision heuristics, contextual cues, data characteristics,
                   and value trade-offs (e.g., "Visual clutter level",
                   "Noise level > 80dB", "Class imbalance", "Predictability over Accuracy")
        aliases: Alternative terms that map to this node (for term indexing)
    """
    id: str
    tower: str  # Tower enum value
    layer: str  # Layer enum value
    label: str
    description: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DSAGNode:
        return cls(**data)
    
    def is_leaf(self) -> bool:
        return self.layer == Layer.LEAF.value or len(self.children_ids) == 0


@dataclass
class TaxonomyTree:
    """
    A taxonomy tree (either TExp or TRes).
    
    Attributes:
        tower: Which tower this tree represents
        root_id: ID of the root node
        nodes: Dictionary mapping node ID to DSAGNode
    """
    tower: str  # Tower enum value
    root_id: str
    nodes: Dict[str, DSAGNode] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tower": self.tower,
            "root_id": self.root_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaxonomyTree:
        nodes = {k: DSAGNode.from_dict(v) for k, v in data.get("nodes", {}).items()}
        return cls(
            tower=data["tower"],
            root_id=data["root_id"],
            nodes=nodes,
        )
    
    def get_node(self, node_id: str) -> Optional[DSAGNode]:
        return self.nodes.get(node_id)
    
    def get_leaves(self) -> List[DSAGNode]:
        """Return all leaf nodes."""
        return [n for n in self.nodes.values() if n.is_leaf()]
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """
        Get the path from a node to the root (inclusive).
        Returns list of node IDs: [node_id, parent_id, ..., root_id]
        """
        path = []
        current_id = node_id
        while current_id:
            path.append(current_id)
            node = self.nodes.get(current_id)
            if node is None:
                break
            current_id = node.parent_id
        return path
    
    def get_aligned_path(self, node_id: str) -> List[str]:
        """
        Get normalized aligned path for LCA computation.
        Returns [root_label, L1_label, L2_label, leaf_label] or shorter if node is higher.
        """
        path_ids = self.get_path_to_root(node_id)
        path_ids.reverse()  # Now: [root, ..., node]
        
        aligned = []
        for nid in path_ids:
            node = self.nodes.get(nid)
            if node:
                aligned.append(node.label)
        return aligned
    
    def get_siblings(self, node_id: str) -> List[DSAGNode]:
        """Get sibling nodes (same parent, excluding self)."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return []
        parent = self.nodes.get(node.parent_id)
        if not parent:
            return []
        return [
            self.nodes[cid] for cid in parent.children_ids
            if cid != node_id and cid in self.nodes
        ]


@dataclass
class GapLink:
    """
    An edge in the Gap Knowledge Graph connecting expert and researcher leaves.

    Attributes:
        expert_leaf_id: ID of the expert tree leaf node
        researcher_leaf_id: ID of the researcher tree leaf node
        aligned_path_expert: Normalized path [root_label, L1, L2, leaf] for expert
        aligned_path_researcher: Normalized path for researcher
        lca_layer: Layer where paths diverge (root, L1, or L2)
        conflict: Description of the divergence
        relation_type: Type of mismatch/gap between the two leaves
        assistance_payload: Type-specific assistance data (structure depends on relation_type)
        weight: Priority/frequency weight for this link
    """
    expert_leaf_id: str
    researcher_leaf_id: str
    aligned_path_expert: List[str] = field(default_factory=list)
    aligned_path_researcher: List[str] = field(default_factory=list)
    lca_layer: str = "root"  # Layer enum value
    expert_lca_id: str = ""  # Aligned LCA node in expert tree
    researcher_lca_id: str = ""  # Aligned LCA node in researcher tree
    conflict: Dict[str, str] = field(default_factory=dict)  # {expert_branch, researcher_branch}
    relation_type: str = RelationType.CONCEPTUAL_GAP.value
    assistance_payload: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GapLink:
        return cls(
            expert_leaf_id=data["expert_leaf_id"],
            researcher_leaf_id=data["researcher_leaf_id"],
            aligned_path_expert=data.get("aligned_path_expert", []),
            aligned_path_researcher=data.get("aligned_path_researcher", []),
            lca_layer=data.get("lca_layer", "root"),
            expert_lca_id=data.get("expert_lca_id", ""),
            researcher_lca_id=data.get("researcher_lca_id", ""),
            conflict=data.get("conflict", {}),
            relation_type=data.get("relation_type", RelationType.CONCEPTUAL_GAP.value),
            assistance_payload=data.get("assistance_payload", {}),
            weight=data.get("weight", 1.0),
        )


# ============== Alignment Data Structures ==============

@dataclass
class NodeAlignment:
    """
    Alignment info between an expert node and a researcher node.
    Used by Agent C (Alignment Judge) to indicate semantic similarity.
    
    Attributes:
        expert_node_id: ID of the node in expert tree
        researcher_node_id: ID of the node in researcher tree
        is_aligned: True = semantically consistent, False = inconsistent
        reason: Why they are aligned/misaligned (semantic explanation)
        semantic_similarity: 0-1 score of semantic similarity
        relation_type: Gap type for misaligned pairs (LexicalGap/ConceptualGap/TacitGap/ScopeGap/ProcessGap).
                       Set by Agent C; None for aligned pairs or when Agent C omits it.
    """
    expert_node_id: str
    researcher_node_id: str
    is_aligned: bool = False
    reason: str = ""
    semantic_similarity: float = 0.0
    relation_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NodeAlignment:
        return cls(
            expert_node_id=data.get("expert_node_id", ""),
            researcher_node_id=data.get("researcher_node_id", ""),
            is_aligned=data.get("is_aligned", False),
            reason=data.get("reason", ""),
            semantic_similarity=data.get("semantic_similarity", 0.0),
            relation_type=data.get("relation_type", None),
        )


@dataclass
class TreeAlignments:
    """
    Full alignment info between two trees (Expert and Researcher).
    Output of Agent C (Alignment Judge).
    
    Attributes:
        leaf_alignments: Alignments between leaf nodes (pain points vs goals)
        concept_alignments: Alignments between concept nodes (L1/L2 levels)
    """
    leaf_alignments: List[NodeAlignment] = field(default_factory=list)
    concept_alignments: List[NodeAlignment] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaf_alignments": [a.to_dict() for a in self.leaf_alignments],
            "concept_alignments": [a.to_dict() for a in self.concept_alignments],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeAlignments:
        return cls(
            leaf_alignments=[NodeAlignment.from_dict(a) for a in data.get("leaf_alignments", [])],
            concept_alignments=[NodeAlignment.from_dict(a) for a in data.get("concept_alignments", [])],
        )
    
    def get_aligned_concepts(self) -> Dict[str, str]:
        """
        Build a map of aligned concept node IDs: expert_node_id -> researcher_node_id.
        Used by tree traversal algorithm to find LCA.
        """
        return {
            a.expert_node_id: a.researcher_node_id
            for a in self.concept_alignments
            if a.is_aligned
        }
    
    def get_misaligned_leaf_pairs(self) -> List[NodeAlignment]:
        """Get all misaligned leaf pairs for GapLink construction."""
        return [a for a in self.leaf_alignments if not a.is_aligned]


@dataclass
class DSAGGraph:
    """
    Complete DSAG structure containing both trees and gap links.
    
    Attributes:
        topic: The interview topic (shared root label)
        researcher_bg: Researcher background description
        expert_bg: Expert background description
        expert_tree: TExp taxonomy tree
        researcher_tree: TRes taxonomy tree
        links: List of gap links between leaves
        metadata: Additional metadata (creation time, model used, etc.)
    """
    topic: str
    researcher_bg: str
    expert_bg: str
    expert_tree: TaxonomyTree
    researcher_tree: TaxonomyTree
    links: List[GapLink] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "researcher_bg": self.researcher_bg,
            "expert_bg": self.expert_bg,
            "expert_tree": self.expert_tree.to_dict(),
            "researcher_tree": self.researcher_tree.to_dict(),
            "links": [link.to_dict() for link in self.links],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DSAGGraph:
        return cls(
            topic=data["topic"],
            researcher_bg=data["researcher_bg"],
            expert_bg=data["expert_bg"],
            expert_tree=TaxonomyTree.from_dict(data["expert_tree"]),
            researcher_tree=TaxonomyTree.from_dict(data["researcher_tree"]),
            links=[GapLink.from_dict(l) for l in data.get("links", [])],
            metadata=data.get("metadata", {}),
        )
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> DSAGGraph:
        return cls.from_dict(json.loads(json_str))
    
    def get_link(self, expert_leaf_id: str, researcher_leaf_id: str) -> Optional[GapLink]:
        """Find a link by its endpoint IDs."""
        for link in self.links:
            if link.expert_leaf_id == expert_leaf_id and link.researcher_leaf_id == researcher_leaf_id:
                return link
        return None
    
    def get_links_by_expert_leaf(self, expert_leaf_id: str) -> List[GapLink]:
        """Get all links connected to a specific expert leaf."""
        return [l for l in self.links if l.expert_leaf_id == expert_leaf_id]
    
    def get_links_by_researcher_leaf(self, researcher_leaf_id: str) -> List[GapLink]:
        """Get all links connected to a specific researcher leaf."""
        return [l for l in self.links if l.researcher_leaf_id == researcher_leaf_id]
    
    @staticmethod
    def compute_cache_key(
        topic: str,
        researcher_bg: str,
        expert_bg: str,
        questionnaire: str = "",
    ) -> str:
        """Compute a stable cache key for the DSAG inputs."""
        content = f"{topic}|||{researcher_bg}|||{expert_bg}|||{questionnaire}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class DSAGState:
    """
    Runtime state for a DSAG session.

    Attributes:
        graph: The DSAG graph
        cache_key: Cache key for this graph
        expert_leaf_embeddings: Dict mapping leaf_id -> embedding vector
        researcher_leaf_embeddings: Dict mapping leaf_id -> embedding vector
        status: Current status (building, ready, error)
        error: Error message if status is error
        interview_timeline: Accumulated timeline entries for Process Gap tracking
    """
    graph: Optional[DSAGGraph] = None
    cache_key: str = ""
    expert_leaf_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    researcher_leaf_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    status: Literal["building", "ready", "error"] = "building"
    error: str = ""
    interview_timeline: List[Dict[str, Any]] = field(default_factory=list)

    def is_ready(self) -> bool:
        return self.status == "ready" and self.graph is not None


# ============== Validation Helpers ==============

def validate_node(node: DSAGNode) -> List[str]:
    """Validate a single node. Returns list of error messages."""
    errors = []
    if not node.id:
        errors.append("Node ID is required")
    if not node.label:
        errors.append(f"Node {node.id}: label is required")
    if node.tower not in [t.value for t in Tower]:
        errors.append(f"Node {node.id}: invalid tower '{node.tower}'")
    if node.layer not in [l.value for l in Layer]:
        errors.append(f"Node {node.id}: invalid layer '{node.layer}'")
    return errors


def validate_tree(tree: TaxonomyTree) -> List[str]:
    """Validate a taxonomy tree. Returns list of error messages."""
    errors = []
    if not tree.root_id:
        errors.append("Tree root_id is required")
    if tree.root_id not in tree.nodes:
        errors.append(f"Root node '{tree.root_id}' not found in nodes")
    
    for node_id, node in tree.nodes.items():
        if node_id != node.id:
            errors.append(f"Node ID mismatch: key '{node_id}' vs node.id '{node.id}'")
        errors.extend(validate_node(node))
        
        # Check parent exists
        if node.parent_id and node.parent_id not in tree.nodes:
            errors.append(f"Node {node.id}: parent '{node.parent_id}' not found")
        
        # Check children exist
        for cid in node.children_ids:
            if cid not in tree.nodes:
                errors.append(f"Node {node.id}: child '{cid}' not found")
    
    return errors


def validate_link(link: GapLink, graph: DSAGGraph) -> List[str]:
    """Validate a gap link. Returns list of error messages."""
    errors = []
    if not link.expert_leaf_id:
        errors.append("Link expert_leaf_id is required")
    if not link.researcher_leaf_id:
        errors.append("Link researcher_leaf_id is required")
    
    if link.expert_leaf_id not in graph.expert_tree.nodes:
        errors.append(f"Link: expert leaf '{link.expert_leaf_id}' not found in expert tree")
    if link.researcher_leaf_id not in graph.researcher_tree.nodes:
        errors.append(f"Link: researcher leaf '{link.researcher_leaf_id}' not found in researcher tree")
    
    if link.relation_type not in [r.value for r in RelationType]:
        errors.append(f"Link: invalid relation_type '{link.relation_type}'")
    
    return errors


def validate_graph(graph: DSAGGraph) -> List[str]:
    """Validate a complete DSAG graph. Returns list of error messages."""
    errors = []
    
    if not graph.topic:
        errors.append("Graph topic is required")
    
    errors.extend(validate_tree(graph.expert_tree))
    errors.extend(validate_tree(graph.researcher_tree))
    
    for link in graph.links:
        errors.extend(validate_link(link, graph))
    
    return errors


# ============== LCA Computation ==============

def compute_lca_layer(path_expert: List[str], path_researcher: List[str]) -> str:
    """
    Compute the LCA layer from two aligned paths.
    Returns the layer where divergence occurs.
    
    Aligned paths are: [root_label, L1_label, L2_label, leaf_label]
    LCA is the longest common prefix.
    """
    if not path_expert or not path_researcher:
        return Layer.ROOT.value
    
    # Find longest common prefix
    common_length = 0
    for i in range(min(len(path_expert), len(path_researcher))):
        if path_expert[i] == path_researcher[i]:
            common_length = i + 1
        else:
            break
    
    # Map common_length to layer
    # 0: diverge immediately -> root
    # 1: share root only -> root (diverge at L1)
    # 2: share root + L1 -> L1 (diverge at L2)
    # 3: share root + L1 + L2 -> L2 (diverge at leaf)
    # 4+: share everything -> leaf (same node, unusual)
    
    if common_length <= 1:
        return Layer.ROOT.value
    elif common_length == 2:
        return Layer.L1.value
    elif common_length >= 3:
        return Layer.L2.value
    
    return Layer.ROOT.value
