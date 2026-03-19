# DSAG - Dual-Tower Semantic Alignment Graph
# Backend module for interdisciplinary interview navigation

from .schema import (
    DSAGNode,
    TaxonomyTree,
    GapLink,
    DSAGGraph,
    DSAGState,
    Tower,
    Layer,
    RelationType,
    NodeAlignment,
    TreeAlignments,
    SubBullet,
    MainBullet,
    TranscriptSummary,
    validate_graph,
    compute_lca_layer,
)

from .factory import (
    GraphFactory,
    create_dsag_graph,
)

from .runtime import (
    RuntimeEngine,
    RuntimeAnalysis,
    Assistance,
    LocatedPosition,
    DivergenceInfo,
    SearchResult,
    analyze_turn,
)

from .transcript_summary import (
    parse_questionnaire,
    classify_and_update,
)

__all__ = [
    # Schema
    "DSAGNode",
    "TaxonomyTree",
    "GapLink",
    "DSAGGraph",
    "DSAGState",
    "Tower",
    "Layer",
    "RelationType",
    "NodeAlignment",
    "TreeAlignments",
    "SubBullet",
    "MainBullet",
    "TranscriptSummary",
    "validate_graph",
    "compute_lca_layer",
    # Factory
    "GraphFactory",
    "create_dsag_graph",
    # Runtime
    "RuntimeEngine",
    "SearchResult",
    "RuntimeAnalysis",
    "Assistance",
    "LocatedPosition",
    "DivergenceInfo",
    "analyze_turn",
    # Transcript Summary
    "parse_questionnaire",
    "classify_and_update",
]
