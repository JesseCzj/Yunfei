"""
DSAG Graph Factory

Implements the MAS (Multi-Agent System) for generating DSAG graphs.
Three agents work together:
- Agent A (ExpertPersona): Generates TExp tree
- Agent B (ResearcherPersona): Generates TRes tree
- Agent C (AlignmentJudge): Judges semantic alignments between trees

Flow:
1. Parallel: Agent A + Agent B generate trees
2. Sequential: Agent C judges alignments between trees
3. Math Algorithm: Build GapLinks from alignments using tree traversal
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .schema import (
    DSAGNode,
    TaxonomyTree,
    GapLink,
    DSAGGraph,
    Tower,
    Layer,
    RelationType,
    NodeAlignment,
    TreeAlignments,
    compute_lca_layer,
    validate_graph,
)

from .embedding_index import build_embedding_index


# ============== LLM Builder ==============

def _build_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Build LLM instance for graph generation."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        # Use a capable model for graph generation
        model = os.getenv("OPENAI_MODEL_GRAPH", os.getenv("OPENAI_MODEL", "qwen3-max"))
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=temperature)
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with fallback regex extraction."""
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Try to find JSON block in markdown
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except Exception:
            pass
    
    # Try to find any JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    
    return {}


# ============== Agent Prompts ==============

EXPERT_PERSONA_PROMPT = """You are an experienced domain expert with the following background:

{expert_bg}

You are participating in an interview about: {topic}

Please identify 8-15 key pain points, concerns, or challenges you face in this domain.
Organize them into a hierarchical taxonomy with the following structure:

- Layer 1 (Perspective): High-level viewpoints or concerns (e.g., "Clinical Outcomes", "Workflow Disruption")
- Layer 2 (Category): Categories under each perspective (e.g., "Alert Fatigue", "Time Pressure")  
- Layer 3 (Leaf): Specific, concrete pain points (e.g., "Too many false alarms", "Can't trust the predictions")

Return ONLY valid JSON with this schema:
{{
  "perspectives": [
    {{
      "label": "Perspective name",
      "description": "Brief description",
      "categories": [
        {{
          "label": "Category name",
          "description": "Brief description",
          "pain_points": [
            {{
              "label": "Specific pain point",
              "description": "Detailed description",
              "is_intuition": true/false,
              "attributes": ["attr1", "attr2"] // Only if is_intuition=true: tacit knowledge facets
            }}
          ]
        }}
      ]
    }}
  ]
}}

Scale and coverage requirements:
- 3-5 perspectives (L1)
- 2-4 categories under each perspective (L2)
- 3-5 leaves per category (L3)

Focus on pain points that:
1. Are specific to your domain expertise
2. Might be misunderstood by researchers from other fields
3. Include both concrete issues AND intuition-based concerns (mark is_intuition=true for vague feelings)

For intuition-based pain points (things you "just know" but can't easily quantify), add 2-4 "attributes" that represent Tacit Knowledge Facets. These facets can come from:
A. Decision Heuristics (implicit rules used to judge)
B. Contextual Cues (environmental constraints or conditions)
C. Data Characteristics (signals about data quality or structure)
D. Value Trade-offs (implicit preferences or priorities)
Examples: "Visual clutter level", "Noise level > 80dB", "Class imbalance", "Predictability over Accuracy".
"""

RESEARCHER_PERSONA_PROMPT = """You are an HCI/Visualization researcher with the following background:

{researcher_bg}

You are preparing to interview domain experts about: {topic}

Here is your INTERVIEW QUESTIONNAIRE / GUIDE:
\"\"\"\n{questionnaire}\n\"\"\"

Please identify 8-15 key research goals, design hypotheses, or information needs you want to explore.
Construct the Researcher Tree (TRes) based PRIMARILY on the questionnaire above:
1. Extract the core intent behind each question and turn it into a Leaf Node (Research Goal).
2. If the questionnaire is sparse, infer logical follow-up goals to fill the tree.
3. Keep the phrasing faithful to the actual interview guide when possible.
Organize them into a hierarchical taxonomy with the following structure:

- Layer 1 (Perspective): High-level research angles (e.g., "Model Interpretability", "User Experience")
- Layer 2 (Category): Research categories (e.g., "Feature Importance", "Interaction Design")
- Layer 3 (Leaf): Specific research questions or design goals (e.g., "Visualize feature contributions", "Reduce cognitive load")

Return ONLY valid JSON with this schema:
{{
  "perspectives": [
    {{
      "label": "Research perspective",
      "description": "Brief description",
      "categories": [
        {{
          "label": "Research category",
          "description": "Brief description",
          "goals": [
            {{
              "label": "Specific goal or question",
              "description": "Detailed description",
              "hci_terms": ["term1", "term2"] // HCI/UX terms that experts might not understand
            }}
          ]
        }}
      ]
    }}
  ]
}}

Scale and coverage requirements:
- 3-5 perspectives (L1)
- 2-4 categories under each perspective (L2)
- 3-5 leaves per category (L3)

Focus on goals that:
1. Require understanding from domain experts
2. Use HCI/UX concepts that might need explanation
3. Could be mapped to expert pain points
"""

# ============== Alignment Judge Prompt (Agent C - New Design) ==============
# Agent C no longer generates GapLinks directly.
# Instead, it judges semantic alignments between nodes.
# GapLinks are then constructed by tree traversal algorithm.

ALIGNMENT_JUDGE_PROMPT = """You are analyzing two taxonomy trees to identify semantic alignments between an Expert's mental model and a Researcher's mental model.

**Interview Topic:** {topic}

**Expert Tree (TExp):**
{expert_tree_json}

**Researcher Tree (TRes):**
{researcher_tree_json}

Your task is to identify which nodes are semantically ALIGNED (same concept, similar meaning) and which are MISALIGNED (different concepts, potential communication gap).

Instructions:
1. For each LEAF node in the Expert tree, determine if there's a semantically similar leaf in the Researcher tree
2. For each CONCEPT node (L1/L2 level) in the Expert tree, determine if there's a similar concept in the Researcher tree
3. Mark pairs as "aligned" (true) if they represent the same or very similar concept
4. Mark pairs as "misaligned" (false) if they represent different perspectives or concepts
5. Provide a reason explaining WHY they are aligned or misaligned
6. For every MISALIGNED pair (is_aligned=false), you MUST set "relation_type" to exactly one of the 5 gap categories below. Do NOT omit this field for misaligned pairs.

IMPORTANT:
- Root nodes are always considered aligned (they share the same topic)
- A misaligned pair indicates a potential communication gap that needs bridging
- The "reason" field should explain the semantic difference or similarity
- Focus on meaningful pairs - but cover MOST leaves
- For each expert leaf, provide at least 1 paired researcher leaf (aligned or misaligned)
- Aim for a healthy number of misaligned leaf pairs (e.g., 30-50% of leaf pairs you output)
- It is acceptable to pair one expert leaf with multiple researcher leaves if that creates meaningful gaps

Gap type definitions for "relation_type" (misaligned pairs only):
- "LexicalGap": The two sides use different terms, jargon, or labels for the same or related concept.
- "ConceptualGap": The two sides hold different mental models, analogies, or interpretations of the concept.
- "TacitGap": The expert relies on intuition, implicit knowledge, or experience that the researcher cannot easily observe or quantify.
- "ScopeGap": The two sides differ in purpose or expectations — the expert focuses on practical utility while the researcher focuses on research value, leading to inconsistent goals about "what to do."
- "ProcessGap": Factual errors disrupt the discussion, or the expert lacks standardized procedures or falls into narrow narratives (tunnel vision), causing workflow/sequence misalignment.

Classification hint: Expert leaf nodes that contain a non-empty "attributes" list indicate tacit, intuition-based knowledge (Tacit Knowledge Facets). When such a leaf is part of a misaligned pair, strongly prefer "TacitGap" as the relation_type.

Return ONLY valid JSON with this schema:
{{
  "leaf_alignments": [
    {{
      "expert_node_id": "exact ID from expert tree (e.g., exp_leaf_00_01_00)",
      "researcher_node_id": "exact ID from researcher tree (e.g., res_leaf_00_00_01)",
      "is_aligned": true/false,
      "reason": "Explanation of why they are aligned/misaligned",
      "semantic_similarity": 0.0-1.0,
      "relation_type": "REQUIRED when is_aligned=false. One of: LexicalGap | ConceptualGap | TacitGap | ScopeGap | ProcessGap. Omit only when is_aligned=true."
    }}
  ],
  "concept_alignments": [
    {{
      "expert_node_id": "L1 or L2 node ID from expert tree",
      "researcher_node_id": "L1 or L2 node ID from researcher tree",
      "is_aligned": true/false,
      "reason": "Explanation of conceptual alignment/misalignment",
      "semantic_similarity": 0.0-1.0
    }}
  ]
}}

Guidelines for alignment judgment:
- is_aligned=true: Concepts are semantically equivalent or highly similar
- is_aligned=false: Concepts differ in perspective, scope, or meaning
- semantic_similarity: 0.0 (completely different) to 1.0 (identical concept)
- Always include the root-to-root alignment as is_aligned=true in concept_alignments
"""

# Legacy prompt kept for reference (no longer used)
GRAPH_ARCHITECT_PROMPT_LEGACY = """[DEPRECATED - See ALIGNMENT_JUDGE_PROMPT]"""


# ============== Tree Building Helpers ==============

def _build_expert_tree(llm_response: Dict[str, Any], topic: str) -> TaxonomyTree:
    """Convert LLM response to TaxonomyTree for expert."""
    nodes: Dict[str, DSAGNode] = {}
    
    # Create root node
    root_id = "shared_root"
    root_node = DSAGNode(
        id=root_id,
        tower=Tower.SHARED.value,
        layer=Layer.ROOT.value,
        label=topic,
        description=f"Interview topic: {topic}",
        parent_id=None,
        children_ids=[],
    )
    nodes[root_id] = root_node
    
    perspectives = llm_response.get("perspectives", [])
    
    for p_idx, perspective in enumerate(perspectives):
        # Create L1 node
        l1_id = f"exp_L1_{p_idx:02d}"
        l1_node = DSAGNode(
            id=l1_id,
            tower=Tower.EXPERT.value,
            layer=Layer.L1.value,
            label=perspective.get("label", f"Perspective {p_idx + 1}"),
            description=perspective.get("description", ""),
            parent_id=root_id,
            children_ids=[],
        )
        nodes[l1_id] = l1_node
        root_node.children_ids.append(l1_id)
        
        categories = perspective.get("categories", [])
        for c_idx, category in enumerate(categories):
            # Create L2 node
            l2_id = f"exp_L2_{p_idx:02d}_{c_idx:02d}"
            l2_node = DSAGNode(
                id=l2_id,
                tower=Tower.EXPERT.value,
                layer=Layer.L2.value,
                label=category.get("label", f"Category {c_idx + 1}"),
                description=category.get("description", ""),
                parent_id=l1_id,
                children_ids=[],
            )
            nodes[l2_id] = l2_node
            l1_node.children_ids.append(l2_id)
            
            pain_points = category.get("pain_points", [])
            for pp_idx, pain_point in enumerate(pain_points):
                # Create leaf node
                leaf_id = f"exp_leaf_{p_idx:02d}_{c_idx:02d}_{pp_idx:02d}"
                leaf_node = DSAGNode(
                    id=leaf_id,
                    tower=Tower.EXPERT.value,
                    layer=Layer.LEAF.value,
                    label=pain_point.get("label", f"Pain point {pp_idx + 1}"),
                    description=pain_point.get("description", ""),
                    parent_id=l2_id,
                    children_ids=[],
                    attributes=pain_point.get("attributes", []) if pain_point.get("is_intuition") else [],
                    aliases=[],
                )
                nodes[leaf_id] = leaf_node
                l2_node.children_ids.append(leaf_id)
    
    return TaxonomyTree(
        tower=Tower.EXPERT.value,
        root_id=root_id,
        nodes=nodes,
    )


def _build_researcher_tree(llm_response: Dict[str, Any], topic: str) -> TaxonomyTree:
    """Convert LLM response to TaxonomyTree for researcher."""
    nodes: Dict[str, DSAGNode] = {}
    
    # Create root node (shared with expert tree conceptually)
    root_id = "shared_root"
    root_node = DSAGNode(
        id=root_id,
        tower=Tower.SHARED.value,
        layer=Layer.ROOT.value,
        label=topic,
        description=f"Interview topic: {topic}",
        parent_id=None,
        children_ids=[],
    )
    nodes[root_id] = root_node
    
    perspectives = llm_response.get("perspectives", [])
    
    for p_idx, perspective in enumerate(perspectives):
        # Create L1 node
        l1_id = f"res_L1_{p_idx:02d}"
        l1_node = DSAGNode(
            id=l1_id,
            tower=Tower.RESEARCHER.value,
            layer=Layer.L1.value,
            label=perspective.get("label", f"Perspective {p_idx + 1}"),
            description=perspective.get("description", ""),
            parent_id=root_id,
            children_ids=[],
        )
        nodes[l1_id] = l1_node
        root_node.children_ids.append(l1_id)
        
        categories = perspective.get("categories", [])
        for c_idx, category in enumerate(categories):
            # Create L2 node
            l2_id = f"res_L2_{p_idx:02d}_{c_idx:02d}"
            l2_node = DSAGNode(
                id=l2_id,
                tower=Tower.RESEARCHER.value,
                layer=Layer.L2.value,
                label=category.get("label", f"Category {c_idx + 1}"),
                description=category.get("description", ""),
                parent_id=l1_id,
                children_ids=[],
            )
            nodes[l2_id] = l2_node
            l1_node.children_ids.append(l2_id)
            
            goals = category.get("goals", [])
            for g_idx, goal in enumerate(goals):
                # Create leaf node
                leaf_id = f"res_leaf_{p_idx:02d}_{c_idx:02d}_{g_idx:02d}"
                leaf_node = DSAGNode(
                    id=leaf_id,
                    tower=Tower.RESEARCHER.value,
                    layer=Layer.LEAF.value,
                    label=goal.get("label", f"Goal {g_idx + 1}"),
                    description=goal.get("description", ""),
                    parent_id=l2_id,
                    children_ids=[],
                    attributes=[],
                    aliases=goal.get("hci_terms", []),
                )
                nodes[leaf_id] = leaf_node
                l2_node.children_ids.append(leaf_id)
    
    return TaxonomyTree(
        tower=Tower.RESEARCHER.value,
        root_id=root_id,
        nodes=nodes,
    )


def _build_links_legacy(
    llm_response: Dict[str, Any],
    expert_tree: TaxonomyTree,
    researcher_tree: TaxonomyTree,
) -> List[GapLink]:
    """[LEGACY] Convert LLM response to list of GapLinks - replaced by math-based approach."""
    links = []
    
    raw_links = llm_response.get("links", [])
    
    for raw_link in raw_links:
        expert_leaf_id = raw_link.get("expert_leaf_id", "")
        researcher_leaf_id = raw_link.get("researcher_leaf_id", "")
        
        # Skip invalid links
        if expert_leaf_id not in expert_tree.nodes:
            print(f"[GraphFactory] Warning: expert leaf '{expert_leaf_id}' not found, skipping link")
            continue
        if researcher_leaf_id not in researcher_tree.nodes:
            print(f"[GraphFactory] Warning: researcher leaf '{researcher_leaf_id}' not found, skipping link")
            continue
        
        # Compute aligned paths
        aligned_path_expert = expert_tree.get_aligned_path(expert_leaf_id)
        aligned_path_researcher = researcher_tree.get_aligned_path(researcher_leaf_id)

        # Compute LCA layer
        lca_layer = compute_lca_layer(aligned_path_expert, aligned_path_researcher)

        # Parse conflict
        conflict = raw_link.get("conflict", {})

        # Parse relation type
        relation_type = raw_link.get("relation_type", RelationType.CONCEPTUAL_GAP.value)
        if relation_type not in [r.value for r in RelationType]:
            relation_type = RelationType.CONCEPTUAL_GAP.value

        link = GapLink(
            expert_leaf_id=expert_leaf_id,
            researcher_leaf_id=researcher_leaf_id,
            aligned_path_expert=aligned_path_expert,
            aligned_path_researcher=aligned_path_researcher,
            lca_layer=lca_layer,
            conflict=conflict,
            relation_type=relation_type,
            assistance_payload=raw_link.get("assistance_payload", {}),
            weight=float(raw_link.get("weight", 5)),
        )
        links.append(link)
    
    return links


# ============== Math-based Link Construction (New Design) ==============

def _parse_alignments(llm_response: Dict[str, Any]) -> TreeAlignments:
    """Parse LLM alignment response into TreeAlignments structure."""
    leaf_alignments = []
    concept_alignments = []
    
    valid_relation_types = {r.value for r in RelationType}
    for raw in llm_response.get("leaf_alignments", []):
        rt = raw.get("relation_type", None)
        if rt not in valid_relation_types:
            rt = None
        leaf_alignments.append(NodeAlignment(
            expert_node_id=raw.get("expert_node_id", ""),
            researcher_node_id=raw.get("researcher_node_id", ""),
            is_aligned=raw.get("is_aligned", False),
            reason=raw.get("reason", ""),
            semantic_similarity=float(raw.get("semantic_similarity", 0.0)),
            relation_type=rt,
        ))
    
    for raw in llm_response.get("concept_alignments", []):
        concept_alignments.append(NodeAlignment(
            expert_node_id=raw.get("expert_node_id", ""),
            researcher_node_id=raw.get("researcher_node_id", ""),
            is_aligned=raw.get("is_aligned", False),
            reason=raw.get("reason", ""),
            semantic_similarity=float(raw.get("semantic_similarity", 0.0)),
        ))
    
    return TreeAlignments(
        leaf_alignments=leaf_alignments,
        concept_alignments=concept_alignments,
    )


def _depth_to_layer(depth: int) -> str:
    """
    Convert traversal depth to Layer enum value.
    
    Depth is the number of steps from leaf to LCA:
    - 0: LCA is at leaf level (unusual, same node)
    - 1: LCA is at L2 (parent of leaf)
    - 2: LCA is at L1 (grandparent)
    - 3+: LCA is at root
    """
    if depth <= 0:
        return Layer.LEAF.value
    elif depth == 1:
        return Layer.L2.value
    elif depth == 2:
        return Layer.L1.value
    else:
        return Layer.ROOT.value


def _infer_relation_type(reason: str) -> str:
    """
    Infer RelationType from the misalignment reason text.

    Uses a **counting mechanism** instead of first-match-wins:
    - Multi-word phrases score 2 (more discriminative).
    - Single words score 1.
    - The category with the highest total score wins.
    - On tie → default to ConceptualGap.

    This is a *fallback* classifier; Agent C's explicit ``relation_type``
    is always preferred when available.
    """
    reason_lower = reason.lower()

    # ---- Two-tier keyword lists (Agent-C analytical style) ----
    # Tier 1: single words (1pt) — broad recall, tolerate mild overlap
    # Tier 2: multi-word phrases (2pt) — high precision, discriminative
    #
    # Design principle: keywords mirror Agent C's formal analytical register.
    # Conversational phrases ("wrap my head around", "you'd know it when you
    # see it", "the way we do it", etc.) are removed — Agent C never writes them.

    lexical_keywords = [
        # -- Tier 1 (1pt) --
        "term", "terminology", "vocabulary", "definition", "language",
        "jargon", "acronym", "abbreviation", "synonym", "nomenclature",
        "shorthand", "notation", "label", "translate",
        # -- Tier 2 (2pt) --
        "technical term", "equivalent term", "refers to the same",
        "same thing", "known as", "refer to", "call it", "stands for",
        "different terminology", "different label",
    ]

    conceptual_keywords = [
        # -- Tier 1 (1pt) --
        "concept", "conceptual", "model", "analogy", "mechanism", "principle",
        "theory", "abstract", "representation", "misunderstanding", "interpretation",
        "framework", "assumption", "visualize",
        # -- Tier 2 (2pt) --
        "mental model", "mental image", "structural difference",
        "different understanding", "different interpretation",
        "conceptual mismatch", "how it works", "big picture",
        "think of it as", "not the same idea", "different framework",
    ]

    tacit_keywords = [
        # -- Tier 1 (1pt) --
        "intuition", "intuitive", "tacit", "implicit", "unspoken",
        "subjective", "experience", "instinct", "criteria",
        "quantify", "measure", "approximate", "threshold",
        "vague", "fuzzy", "depends", "varies",
        # -- Tier 2 (2pt) --
        "contextual cue", "domain-specific", "experience-based",
        "implicit knowledge", "tacit knowledge", "implicit judgment",
        "difficult to articulate", "hard to explain", "hard to pin down",
        "non-quantifiable", "rule of thumb", "good enough",
        "can't explain",
    ]

    scope_keywords = [
        # -- Tier 1 (1pt) --
        "scope", "aim", "objective", "expectation", "constraint", "boundary",
        "limitation", "resource", "priority", "tradeoff",
        "practical", "useful", "academic", "theoretical",
        "misaligned", "divergent",
        "focus", "goal", "value",        # weak votes; disambiguated by Tier 2
        # -- Tier 2 (2pt) --
        "different focus", "divergent focus", "misaligned focus",
        "research goal", "different goal",
        "practical value", "research value",
        "different scope", "different priority",
        "different level", "different levels of analysis",
        "too detailed", "too high-level", "not relevant", "out of scope",
        "bigger picture", "real-world", "day-to-day",
    ]

    process_keywords = [
        # -- Tier 1 (1pt) --
        "process", "workflow", "step", "sequence", "procedure", "stage", "phase",
        "method", "methodology", "routine", "pipeline", "order", "skip",
        "exception",
        # -- Tier 2 (2pt) --
        "different workflow", "different ordering", "different procedure",
        "procedural difference", "procedural mismatch",
        "operational mismatch", "methodological difference",
        "missing a step", "edge case", "corner case",
        "in reality",
    ]

    # ---- scoring: multi-word phrases = 2, single words = 1 ----
    category_lists = [
        (RelationType.LEXICAL_GAP.value, lexical_keywords),
        (RelationType.CONCEPTUAL_GAP.value, conceptual_keywords),
        (RelationType.TACIT_GAP.value, tacit_keywords),
        (RelationType.SCOPE_GAP.value, scope_keywords),
        (RelationType.PROCESS_GAP.value, process_keywords),
    ]

    scores: dict[str, int] = {}
    for relation_type, keywords in category_lists:
        score = 0
        for kw in keywords:
            if kw in reason_lower:
                score += 2 if " " in kw else 1
        scores[relation_type] = score

    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_type] == 0:
        return RelationType.CONCEPTUAL_GAP.value
    return best_type


def _find_path_between(leaf_id: str, ancestor_id: str, tree: TaxonomyTree) -> List[str]:
    """
    Find the path from a leaf to an ancestor (exclusive of ancestor).
    Returns [leaf_id, parent_id, ..., child_of_ancestor] or empty if not found.
    """
    path = []
    current_id = leaf_id
    
    while current_id and current_id != ancestor_id:
        path.append(current_id)
        node = tree.get_node(current_id)
        if node is None:
            break
        current_id = node.parent_id
    
    return path


def _get_ancestor_by_layer(tree: TaxonomyTree, node_id: str, layer_value: str) -> Optional[str]:
    """Find the closest ancestor (including self) at the given layer."""
    path = tree.get_path_to_root(node_id)
    for nid in path:
        node = tree.get_node(nid)
        if node and node.layer == layer_value:
            return nid
    return None


def _compute_weight_from_similarity(similarity: float) -> float:
    """
    Convert semantic similarity to link weight.
    Lower similarity = higher weight (more important to bridge).
    """
    # Invert: low similarity means high importance for bridging
    # Scale from 1-10
    return max(1.0, min(10.0, (1.0 - similarity) * 10))


def build_links_from_alignments(
    expert_tree: TaxonomyTree,
    researcher_tree: TaxonomyTree,
    alignments: TreeAlignments,
) -> List[GapLink]:
    """
    Build GapLinks from alignment judgments using tree traversal algorithm.
    
    Algorithm:
    1. Build concept alignment map (for LCA lookup)
    2. For each misaligned leaf pair:
       a. Traverse up from expert leaf to find first aligned ancestor (LCA)
       b. Record divergence path (nodes between LCA and leaf)
       c. Determine relation type from misalignment reason
       d. Create GapLink with computed LCA and divergence info
    
    Args:
        expert_tree: The expert taxonomy tree
        researcher_tree: The researcher taxonomy tree
        alignments: Alignment judgments from Agent C
    
    Returns:
        List of GapLinks constructed mathematically
    """
    print("[GraphFactory] Building links from alignments using tree traversal...")
    links = []
    
    # Step 1: Build concept alignment map (expert_node_id -> researcher_node_id)
    # Only include aligned concept pairs and normalize to same-layer mappings.
    raw_concept_map = alignments.get_aligned_concepts()
    concept_map: Dict[str, str] = {}
    for exp_id, res_id in raw_concept_map.items():
        exp_node = expert_tree.get_node(exp_id)
        res_node = researcher_tree.get_node(res_id)
        if not exp_node or not res_node:
            continue
        if exp_node.layer == res_node.layer:
            concept_map[exp_id] = res_id
            continue
        # Try to find same-layer ancestor on researcher side
        res_path = researcher_tree.get_path_to_root(res_id)
        for res_ancestor_id in res_path:
            res_ancestor = researcher_tree.get_node(res_ancestor_id)
            if res_ancestor and res_ancestor.layer == exp_node.layer:
                concept_map[exp_id] = res_ancestor_id
                break
    reverse_concept_map = {res_id: exp_id for exp_id, res_id in concept_map.items()}
    
    # Always ensure root is aligned
    concept_map["shared_root"] = "shared_root"
    
    print(f"[GraphFactory] Found {len(concept_map)} aligned concept pairs")
    
    # Step 2: Process each misaligned leaf pair
    misaligned_leaves = alignments.get_misaligned_leaf_pairs()
    print(f"[GraphFactory] Processing {len(misaligned_leaves)} misaligned leaf pairs")
    
    for leaf_align in misaligned_leaves:
        exp_leaf_id = leaf_align.expert_node_id
        res_leaf_id = leaf_align.researcher_node_id
        
        # Validate leaf IDs exist
        if exp_leaf_id not in expert_tree.nodes:
            print(f"[GraphFactory] Warning: expert leaf '{exp_leaf_id}' not found, skipping")
            continue
        if res_leaf_id not in researcher_tree.nodes:
            print(f"[GraphFactory] Warning: researcher leaf '{res_leaf_id}' not found, skipping")
            continue
        
        # Step 3: Traverse up from expert leaf to find first aligned ancestor
        exp_path = expert_tree.get_path_to_root(exp_leaf_id)  # [leaf, parent, ..., root]
        
        lca_exp_node_id = None
        lca_res_node_id = None
        divergence_depth = 0
        
        for i, exp_ancestor_id in enumerate(exp_path):
            if exp_ancestor_id in concept_map:
                lca_exp_node_id = exp_ancestor_id
                lca_res_node_id = concept_map[exp_ancestor_id]
                divergence_depth = i  # How many levels below LCA
                break
        
        # Fallback: if no aligned ancestor found, use root
        if lca_exp_node_id is None:
            lca_exp_node_id = "shared_root"
            lca_res_node_id = "shared_root"
            divergence_depth = len(exp_path) - 1

        # Validate that mapped researcher LCA is actually an ancestor of the researcher leaf
        # If not, walk up the researcher path to find the first aligned ancestor (reverse map),
        # otherwise fall back to shared_root.
        res_path = researcher_tree.get_path_to_root(res_leaf_id)
        if lca_res_node_id not in res_path:
            for res_ancestor_id in res_path:
                if res_ancestor_id in reverse_concept_map:
                    lca_res_node_id = res_ancestor_id
                    lca_exp_node_id = reverse_concept_map[res_ancestor_id]
                    break
            else:
                lca_res_node_id = "shared_root"
                lca_exp_node_id = "shared_root"

        # Recompute divergence depth based on final LCA (expert side)
        if lca_exp_node_id in exp_path:
            divergence_depth = exp_path.index(lca_exp_node_id)
        else:
            divergence_depth = len(exp_path) - 1
        
        # Step 4: Record divergence paths
        divergence_path_exp = _find_path_between(exp_leaf_id, lca_exp_node_id, expert_tree)
        divergence_path_res = _find_path_between(res_leaf_id, lca_res_node_id, researcher_tree)
        
        # Get labels for conflict description
        exp_divergence_node = expert_tree.get_node(divergence_path_exp[0]) if divergence_path_exp else None
        res_divergence_node = researcher_tree.get_node(divergence_path_res[0]) if divergence_path_res else None
        
        conflict = {
            "expert_branch": exp_divergence_node.label if exp_divergence_node else "",
            "researcher_branch": res_divergence_node.label if res_divergence_node else "",
        }
        
        # Step 5: Determine relation type — prefer Agent C's judgment, fall back to keyword matching
        relation_type = leaf_align.relation_type or _infer_relation_type(leaf_align.reason)
        
        # Step 6: Compute aligned paths for additional info
        aligned_path_expert = expert_tree.get_aligned_path(exp_leaf_id)
        aligned_path_researcher = researcher_tree.get_aligned_path(res_leaf_id)
        
        # Step 7: Determine LCA layer from aligned ancestor (expert side)
        lca_layer = _depth_to_layer(divergence_depth)
        exp_lca_node = expert_tree.get_node(lca_exp_node_id) if lca_exp_node_id else None
        if exp_lca_node:
            lca_layer = exp_lca_node.layer

        # Step 8: Create GapLink
        link = GapLink(
            expert_leaf_id=exp_leaf_id,
            researcher_leaf_id=res_leaf_id,
            aligned_path_expert=aligned_path_expert,
            aligned_path_researcher=aligned_path_researcher,
            lca_layer=lca_layer,
            expert_lca_id=lca_exp_node_id or "",
            researcher_lca_id=lca_res_node_id or "",
            conflict=conflict,
            relation_type=relation_type,
            assistance_payload={},  # To be filled by type-specific prompt
            weight=_compute_weight_from_similarity(leaf_align.semantic_similarity),
        )
        links.append(link)
    
    print(f"[GraphFactory] Built {len(links)} GapLinks from alignments")
    return links


# ============== Bridge Templates Generation ==============

# ============== Type-Specific Assistance Prompts ==============

LEXICAL_GAP_PROMPT = """You are generating a term-mapping bridge for an interview between a domain expert and a researcher.

**Expert Concept:**
- Label: {expert_label}
- Description: {expert_description}

**Researcher Concept:**
- Label: {researcher_label}
- Description: {researcher_description}

**Misalignment Reason:** {misalignment_reason}

The expert and researcher use different terminology for the same or related concept.
Perform a strict 1-to-1 semantic substitution. Map the expert's term to a native vocabulary word the researcher uses.

Return ONLY valid JSON:
{{
  "term_mapping": {{
    "expert_term": "the exact term the expert uses",
    "researcher_term": "the equivalent term in the researcher's vocabulary",
    "explanation": "1 sentence explaining how these terms relate"
  }}
}}
"""

CONCEPTUAL_GAP_PROMPT = """You are generating conceptual bridges for an interview between a domain expert and a researcher.

**Expert Concept:**
- Path: {expert_path}
- Label: {expert_label}
- Description: {expert_description}

**Researcher Concept:**
- Path: {researcher_path}
- Label: {researcher_label}
- Description: {researcher_description}

**Expert's Related Concepts (siblings under same category):**
{expert_siblings}

**Divergence Point:** {lca_layer}
**Conflict:** Expert focuses on "{expert_branch}", Researcher focuses on "{researcher_branch}"
**Misalignment Reason:** {misalignment_reason}

Their mental models differ. Generate TWO parallel explanation strategies (no required order):

Strategy 1 — Analogy Construction:
- Consider the expert's related concepts listed above as potential analogy sources — they are concepts the expert already articulated at the same abstraction level. You may also use a different concept if it provides a stronger structural mapping.
- Do NOT merely swap nouns. Map the relational structure: Inputs, Logic, Outputs.
- Explain how the researcher's concept behaves similarly to the expert's familiar concept.

Strategy 2 — Fine-grained Scenario:
- A highly specific, low-level concrete example.
- Explicitly specify inputs, outputs, and edge cases to prevent ambiguity.
- Avoid overarching metaphors that require secondary interpretation.

Return ONLY valid JSON:
{{
  "analogy": {{
    "source_concept": "A concept the expert already knows that is structurally similar",
    "structural_mapping": {{
      "inputs": "What goes in (mapped to expert's frame)",
      "logic": "How it transforms (mapped to expert's frame)",
      "outputs": "What comes out (mapped to expert's frame)"
    }},
    "explanation": "How the researcher's concept behaves similarly to the source concept"
  }},
  "scenario": {{
    "inputs": "Concrete input example the expert would recognize",
    "outputs": "What the researcher's approach would produce",
    "edge_cases": "A tricky case where the two mental models diverge"
  }}
}}
"""

TACIT_GAP_PROMPT = """You are generating probes to surface tacit knowledge from a domain expert during an interview.

**Expert Concept:**
- Label: {expert_label}
- Description: {expert_description}
- Tacit Knowledge Facets (attributes): {attributes}

**Researcher Concept:**
- Label: {researcher_label}
- Description: {researcher_description}

**Misalignment Reason:** {misalignment_reason}

The expert holds intuitive, experience-based knowledge that is hard to articulate.
The researcher needs explicit, quantifiable design parameters.

Follow a strict 3-step pipeline:

Step 1: Extract the relevant attributes (use the facets provided above).

Step 2: For each attribute, generate a MULTIPLE-CHOICE probe question to guide the expert to dismantle factors through quantifiable indicators.
  RULE: NEVER ask open-ended "Why" questions. Instead, formulate as:
  "Is your concern related to [Attribute A] or [Attribute B]?" or
  "Which matters more: A, B, or C?"

Step 3: Generate 2-3 hypothetical scenarios for optional checkout.
  RULE: Each scenario alters only ONE variable at a time to test the boundaries of their intuition.

Return ONLY valid JSON:
{{
  "attributes": ["attr1", "attr2"],
  "probes": [
    {{
      "attribute": "the attribute being probed",
      "question": "Multiple-choice question (never open-ended Why)",
      "choices": ["Option A", "Option B"]
    }}
  ],
  "hypothetical_scenarios": [
    "If [variable A] changed from X to Y, would your judgment about '{expert_label}' change?",
    "If [variable B] changed from P to Q, would your judgment about '{expert_label}' change?"
  ]
}}
"""

SCOPE_GAP_PROMPT = """You are generating a validate-and-pivot bridge for an interview where the expert and researcher have different scope/focus.

**Expert Concept:**
- Path: {expert_path}
- Label: {expert_label}
- Description: {expert_description}

**Researcher Concept:**
- Path: {researcher_path}
- Label: {researcher_label}
- Description: {researcher_description}

**Divergence Point:** {lca_layer}
**Conflict:** Expert focuses on "{expert_branch}", Researcher focuses on "{researcher_branch}"
**Misalignment Reason:** {misalignment_reason}

The expert focuses on practical functions while the researcher focuses on research value. Their expectations of "what to do" differ.

Follow a strict 2-step sequence:

Step 1 — Validate: Acknowledge the expert's practical focus genuinely.

Step 2 — Pivot: Inject the research goal by:
  - Highlighting a potential limitation in the expert's current approach
  - Presenting the research goal that addresses this limitation
  - Generating a compelling reason (value proposition)
  - Deploying a Coarse-grained Scenario (high-level macro narrative)

THREE RULES to enforce:
1. **Injection Strategy**: Use a "Validate & Pivot" template: "Your focus on [Practical Utility] is highly applicable, but we might encounter [Potential Limitation]. Solving [Research Goal] helps us overcome this because..."
2. **Persuasion Rule**: Do NOT merely state the research is a "necessary step." Actively generate compelling reasons showing how the research safeguards or enhances the expert's interests.
3. **Scenario Rule**: Deploy the Coarse-grained Scenario specifically as a persuasive tool to visualize the macro-level benefit. Keep it at the "story" level — what happens on a typical day AFTER adoption. NO technical details, NO budget arguments.

Return ONLY valid JSON:
{{
  "validate_focus": "1-2 sentences acknowledging the expert's practical focus",
  "pivot": {{
    "limitation": "A real limitation of the expert's current approach",
    "research_goal": "The researcher's goal that addresses this",
    "compelling_reason": "Why the expert should care (framed as what they GAIN)",
    "coarse_scenario": "2-3 sentence narrative of a typical day after adoption (no technical details, no budget)"
  }}
}}
"""

PROCESS_GAP_PROMPT = """You are analyzing a process/workflow mismatch between an expert and a researcher.

**Expert Concept:**
- Path: {expert_path}
- Label: {expert_label}
- Description: {expert_description}

**Researcher Concept:**
- Path: {researcher_path}
- Label: {researcher_label}
- Description: {researcher_description}

**Misalignment Reason:** {misalignment_reason}

The expert and researcher think about this workflow in a different order or with different steps.
Factual errors, lack of standardized procedures, or narrow expert narratives disrupt the discussion.

Your task: Reconstruct the expected workflow from the expert's perspective, so the interviewer can track coverage, detect skipped steps, and notice when the expert falls into tunnel vision.

Instructions:
1. Identify the key process steps the expert likely follows, IN ORDER, based on the expert concept and its context.
2. For each step, write a short description of what happens and WHY the expert considers it important.
3. Flag 1-2 "tunnel vision risks" — specific steps where the expert is likely to over-elaborate or get stuck in a narrow narrative, neglecting other steps.

Return ONLY valid JSON:
{{
  "expected_steps": [
    {{
      "label": "Short step name",
      "description": "What happens in this step and why it matters to the expert",
      "order": 1
    }}
  ],
  "tunnel_vision_risks": [
    "Step X — reason why the expert tends to over-focus here"
  ]
}}
"""


def generate_assistance_payload_for_link(
    link: GapLink,
    expert_tree: TaxonomyTree,
    researcher_tree: TaxonomyTree,
    misalignment_reason: str = "",
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    Generate type-specific assistance payload for a single GapLink.
    Dispatches to the appropriate prompt based on link.relation_type.
    """
    if llm is None:
        llm = _build_llm(temperature=0.7)

    exp_node = expert_tree.get_node(link.expert_leaf_id)
    res_node = researcher_tree.get_node(link.researcher_leaf_id)

    if not exp_node or not res_node:
        print(f"[GraphFactory] Warning: Could not find nodes for payload generation")
        return {}

    prompt_map = {
        RelationType.LEXICAL_GAP.value: LEXICAL_GAP_PROMPT,
        RelationType.CONCEPTUAL_GAP.value: CONCEPTUAL_GAP_PROMPT,
        RelationType.TACIT_GAP.value: TACIT_GAP_PROMPT,
        RelationType.SCOPE_GAP.value: SCOPE_GAP_PROMPT,
        RelationType.PROCESS_GAP.value: PROCESS_GAP_PROMPT,
    }

    prompt_template = prompt_map.get(link.relation_type, CONCEPTUAL_GAP_PROMPT)

    # Collect expert siblings for ConceptualGap analogy candidates
    expert_siblings_text = "None available"
    if link.relation_type == RelationType.CONCEPTUAL_GAP.value:
        siblings = expert_tree.get_siblings(link.expert_leaf_id)
        if siblings:
            sibling_lines = []
            for sib in siblings[:5]:  # cap at 5 to limit token cost
                sibling_lines.append(f"- \"{sib.label}\": {sib.description}")
            expert_siblings_text = "\n".join(sibling_lines)

    variables = {
        "expert_path": " > ".join(link.aligned_path_expert),
        "expert_label": exp_node.label,
        "expert_description": exp_node.description,
        "researcher_path": " > ".join(link.aligned_path_researcher),
        "researcher_label": res_node.label,
        "researcher_description": res_node.description,
        "relation_type": link.relation_type,
        "lca_layer": link.lca_layer,
        "expert_branch": link.conflict.get("expert_branch", ""),
        "researcher_branch": link.conflict.get("researcher_branch", ""),
        "misalignment_reason": misalignment_reason,
        "attributes": ", ".join(exp_node.attributes) if exp_node.attributes else "none",
        "expert_siblings": expert_siblings_text,
    }

    prompt = ChatPromptTemplate.from_messages([("user", prompt_template)])
    chain = prompt | llm

    try:
        response = chain.invoke(variables)
        content = response.content if hasattr(response, 'content') else str(response)
        data = _parse_json(content)
        return data if data else {}
    except Exception as e:
        print(f"[GraphFactory] Error generating payload for {link.relation_type}: {e}")
        return {}


def generate_all_assistance_payloads(
    links: List[GapLink],
    expert_tree: TaxonomyTree,
    researcher_tree: TaxonomyTree,
    alignments: TreeAlignments,
) -> List[GapLink]:
    """Generate type-specific assistance payloads for all GapLinks."""
    print(f"[GraphFactory] Generating assistance payloads for {len(links)} links...")
    llm = _build_llm(temperature=0.7)

    # Build lookup for misalignment reasons
    reason_lookup = {}
    for align in alignments.leaf_alignments:
        key = (align.expert_node_id, align.researcher_node_id)
        reason_lookup[key] = align.reason

    for link in links:
        key = (link.expert_leaf_id, link.researcher_leaf_id)
        reason = reason_lookup.get(key, "")

        payload = generate_assistance_payload_for_link(
            link=link,
            expert_tree=expert_tree,
            researcher_tree=researcher_tree,
            misalignment_reason=reason,
            llm=llm,
        )
        link.assistance_payload = payload

    print("[GraphFactory] Assistance payload generation complete")
    return links


# ============== Main Factory Class ==============

class GraphFactory:
    """
    Factory for generating DSAG graphs using MAS (Multi-Agent System).
    """
    
    def __init__(self):
        self.llm = None
    
    def _get_llm(self) -> ChatOpenAI:
        if self.llm is None:
            self.llm = _build_llm(temperature=0.3)
        return self.llm
    
    def generate_expert_tree(self, topic: str, expert_bg: str) -> TaxonomyTree:
        """Agent A: Generate expert taxonomy tree."""
        llm = self._get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("user", EXPERT_PERSONA_PROMPT)
        ])
        chain = prompt | llm
        
        response = chain.invoke({
            "topic": topic,
            "expert_bg": expert_bg,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        if not parsed.get("perspectives"):
            raise ValueError("Failed to generate expert tree: no perspectives returned")
        
        return _build_expert_tree(parsed, topic)
    
    def generate_researcher_tree(
        self,
        topic: str,
        researcher_bg: str,
        questionnaire: str = "",
    ) -> TaxonomyTree:
        """Agent B: Generate researcher taxonomy tree."""
        llm = self._get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("user", RESEARCHER_PERSONA_PROMPT)
        ])
        chain = prompt | llm
        
        response = chain.invoke({
            "topic": topic,
            "researcher_bg": researcher_bg,
            "questionnaire": questionnaire or "",
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        if not parsed.get("perspectives"):
            raise ValueError("Failed to generate researcher tree: no perspectives returned")
        
        return _build_researcher_tree(parsed, topic)
    
    def generate_alignments(
        self,
        topic: str,
        expert_tree: TaxonomyTree,
        researcher_tree: TaxonomyTree,
    ) -> TreeAlignments:
        """
        Agent C (new): Judge semantic alignments between trees.
        This replaces direct link generation.
        """
        print("[GraphFactory] Agent C: Judging alignments between trees...")
        llm = self._get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("user", ALIGNMENT_JUDGE_PROMPT)
        ])
        chain = prompt | llm
        
        # Prepare tree summaries for the prompt
        expert_tree_json = json.dumps(
            self._tree_to_summary(expert_tree),
            indent=2,
            ensure_ascii=False
        )
        researcher_tree_json = json.dumps(
            self._tree_to_summary(researcher_tree),
            indent=2,
            ensure_ascii=False
        )
        
        response = chain.invoke({
            "topic": topic,
            "expert_tree_json": expert_tree_json,
            "researcher_tree_json": researcher_tree_json,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        alignments = _parse_alignments(parsed)
        print(f"[GraphFactory] Agent C found {len(alignments.leaf_alignments)} leaf alignments, "
              f"{len(alignments.concept_alignments)} concept alignments")
        
        return alignments
    
    def generate_links(
        self,
        topic: str,
        expert_tree: TaxonomyTree,
        researcher_tree: TaxonomyTree,
        generate_templates: bool = True,
    ) -> Tuple[List[GapLink], TreeAlignments]:
        """
        Generate gap links using the new math-based approach.
        
        New Flow:
        1. Agent C judges alignments (LLM)
        2. Math algorithm builds links from alignments (tree traversal)
        3. Generate bridge templates (LLM, optional)
        
        Args:
            topic: Interview topic
            expert_tree: Expert taxonomy tree
            researcher_tree: Researcher taxonomy tree
            generate_templates: Whether to generate bridge templates (default True)
        
        Returns:
            Tuple of (List of GapLinks, TreeAlignments)
        """
        # Step 1: Agent C judges alignments
        alignments = self.generate_alignments(topic, expert_tree, researcher_tree)

        # Fallback pairing when coverage is low (relevance + mismatch)
        total_expert_leaves = len(expert_tree.get_leaves())
        total_researcher_leaves = len(researcher_tree.get_leaves())
        if total_expert_leaves > 0:
            misaligned_pairs = [
                a for a in alignments.leaf_alignments
                if not a.is_aligned
            ]
            misaligned_expert = {a.expert_node_id for a in misaligned_pairs}
            gap_coverage = len(misaligned_expert) / total_expert_leaves

            min_gap_coverage = float(os.getenv("DSAG_GAPLINK_MIN_COVERAGE", "0.8"))
            top_k = int(os.getenv("DSAG_FORCE_PAIRING_TOP_K", "3"))
            per_leaf = int(os.getenv("DSAG_FORCE_PAIRING_PER_LEAF", "1"))
            sim_min = float(os.getenv("DSAG_FORCE_PAIRING_SIM_MIN", "0.25"))
            sim_max = float(os.getenv("DSAG_FORCE_PAIRING_SIM_MAX", "0.75"))
            alpha = float(os.getenv("DSAG_FORCE_PAIRING_ALPHA", "0.6"))
            beta = float(os.getenv("DSAG_FORCE_PAIRING_BETA", "0.4"))

            if gap_coverage < min_gap_coverage and top_k > 0:
                print(f"[GraphFactory] Gap coverage low ({gap_coverage:.2f}), applying fallback pairing...")

                # Build temporary graph for embeddings
                temp_graph = DSAGGraph(
                    topic=topic,
                    researcher_bg="",
                    expert_bg="",
                    expert_tree=expert_tree,
                    researcher_tree=researcher_tree,
                    links=[],
                )
                index = build_embedding_index(temp_graph)

                existing_pairs = {
                    (a.expert_node_id, a.researcher_node_id)
                    for a in alignments.leaf_alignments
                }
                concept_map = alignments.get_aligned_concepts()

                added_pairs = 0
                for exp_leaf in expert_tree.get_leaves():
                    if exp_leaf.id in misaligned_expert:
                        continue

                    query_text = index.expert_leaf_texts.get(exp_leaf.id) or index._node_to_text(exp_leaf)
                    candidates = index.search_researcher_leaves(query_text, top_k=top_k)
                    if not candidates:
                        continue

                    filtered = []
                    for cand in candidates:
                        sim = cand.score
                        if sim < sim_min or sim > sim_max:
                            continue

                        aligned_path_expert = expert_tree.get_aligned_path(exp_leaf.id)
                        aligned_path_researcher = researcher_tree.get_aligned_path(cand.node_id)
                        lca_layer = compute_lca_layer(aligned_path_expert, aligned_path_researcher)

                        exp_l1 = _get_ancestor_by_layer(expert_tree, exp_leaf.id, Layer.L1.value)
                        res_l1 = _get_ancestor_by_layer(researcher_tree, cand.node_id, Layer.L1.value)
                        exp_l2 = _get_ancestor_by_layer(expert_tree, exp_leaf.id, Layer.L2.value)
                        res_l2 = _get_ancestor_by_layer(researcher_tree, cand.node_id, Layer.L2.value)
                        aligned_l1 = exp_l1 and concept_map.get(exp_l1) == res_l1
                        path_mismatch = bool(aligned_l1 and exp_l2 and res_l2 and exp_l2 != res_l2)

                        # Structural divergence filter: keep high-level mismatch or path mismatch
                        if lca_layer not in (Layer.ROOT.value, Layer.L1.value) and not path_mismatch:
                            continue

                        divergence_level = 2 if lca_layer == Layer.ROOT.value else 1 if lca_layer == Layer.L1.value else 0
                        gap_score = alpha * (1 - sim) + beta * divergence_level
                        if path_mismatch:
                            gap_score += 0.2

                        filtered.append((gap_score, sim, lca_layer, path_mismatch, cand))

                    # If no candidates passed filters, relax to top-K by similarity window only
                    if not filtered:
                        for cand in candidates:
                            sim = cand.score
                            if sim < sim_min or sim > sim_max:
                                continue
                            aligned_path_expert = expert_tree.get_aligned_path(exp_leaf.id)
                            aligned_path_researcher = researcher_tree.get_aligned_path(cand.node_id)
                            lca_layer = compute_lca_layer(aligned_path_expert, aligned_path_researcher)
                            divergence_level = 2 if lca_layer == Layer.ROOT.value else 1 if lca_layer == Layer.L1.value else 0
                            gap_score = alpha * (1 - sim) + beta * divergence_level
                            filtered.append((gap_score, sim, lca_layer, False, cand))

                    if not filtered:
                        continue

                    filtered.sort(key=lambda x: x[0], reverse=True)
                    for gap_score, sim, lca_layer, path_mismatch, cand in filtered[:per_leaf]:
                        pair = (exp_leaf.id, cand.node_id)
                        if pair in existing_pairs:
                            continue

                        gap_type = RelationType.CONCEPTUAL_GAP.value
                        if exp_leaf.attributes:
                            gap_type = RelationType.TACIT_GAP.value
                        elif path_mismatch:
                            gap_type = RelationType.PROCESS_GAP.value
                        elif lca_layer == Layer.ROOT.value:
                            gap_type = RelationType.CONCEPTUAL_GAP.value
                        elif lca_layer == Layer.L1.value:
                            gap_type = RelationType.SCOPE_GAP.value

                        alignments.leaf_alignments.append(NodeAlignment(
                            expert_node_id=exp_leaf.id,
                            researcher_node_id=cand.node_id,
                            is_aligned=False,
                            reason=f"{gap_type}: fallback mismatch (sim={sim:.2f}, lca={lca_layer})",
                            semantic_similarity=sim,
                            relation_type=gap_type,
                        ))
                        existing_pairs.add(pair)
                        misaligned_expert.add(exp_leaf.id)
                        misaligned_pairs.append(alignments.leaf_alignments[-1])
                        added_pairs += 1

                gap_coverage_after = len(misaligned_expert) / total_expert_leaves
                print(
                    f"[GraphFactory] Fallback added {added_pairs} pairs. "
                    f"Gap coverage {gap_coverage:.2f} -> {gap_coverage_after:.2f}."
                )
        
        # Step 2: Math algorithm builds links
        links = build_links_from_alignments(
            expert_tree=expert_tree,
            researcher_tree=researcher_tree,
            alignments=alignments,
        )
        
        # Step 3: Generate type-specific assistance payloads (optional)
        if generate_templates and links:
            links = generate_all_assistance_payloads(
                links=links,
                expert_tree=expert_tree,
                researcher_tree=researcher_tree,
                alignments=alignments,
            )
        
        return links, alignments
    
    def generate_links_legacy(
        self,
        topic: str,
        expert_tree: TaxonomyTree,
        researcher_tree: TaxonomyTree,
    ) -> List[GapLink]:
        """[LEGACY] Agent C: Analyze trees and generate gap links directly."""
        llm = self._get_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("user", GRAPH_ARCHITECT_PROMPT_LEGACY)
        ])
        chain = prompt | llm
        
        # Prepare tree summaries for the prompt
        expert_tree_json = json.dumps(
            self._tree_to_summary(expert_tree),
            indent=2,
            ensure_ascii=False
        )
        researcher_tree_json = json.dumps(
            self._tree_to_summary(researcher_tree),
            indent=2,
            ensure_ascii=False
        )
        
        response = chain.invoke({
            "topic": topic,
            "expert_tree_json": expert_tree_json,
            "researcher_tree_json": researcher_tree_json,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        return _build_links_legacy(parsed, expert_tree, researcher_tree)
    
    def _tree_to_summary(self, tree: TaxonomyTree) -> Dict[str, Any]:
        """Convert tree to a summary format for the architect prompt."""
        summary = {
            "tower": tree.tower,
            "nodes": []
        }
        
        for node in tree.nodes.values():
            node_info = {
                "id": node.id,
                "layer": node.layer,
                "label": node.label,
                "description": node.description,
                "parent_id": node.parent_id,
            }
            if node.attributes:
                node_info["attributes"] = node.attributes
            if node.aliases:
                node_info["aliases"] = node.aliases
            summary["nodes"].append(node_info)
        
        return summary
    
    def generate_graph(
        self,
        topic: str,
        researcher_bg: str,
        expert_bg: str,
        questionnaire: str = "",
        generate_templates: bool = True,
    ) -> DSAGGraph:
        """
        Generate complete DSAG graph using math-based link construction.
        
        New Flow:
        1. Generate expert tree (Agent A)
        2. Generate researcher tree (Agent B) - can be parallel with step 1
        3. Agent C judges alignments between trees (LLM)
        4. Math algorithm builds GapLinks from alignments (tree traversal)
        5. Generate bridge templates (LLM, optional)
        
        Args:
            topic: Interview topic
            researcher_bg: Researcher background
            expert_bg: Expert background
            generate_templates: Whether to generate bridge templates (default True)
        
        Returns:
            Complete DSAGGraph
        """
        print(f"[GraphFactory] Starting DSAG generation for topic: {topic}")
        
        # Step 1 & 2: Generate both trees
        # In production, these could be parallelized
        print("[GraphFactory] Agent A: Generating expert tree...")
        expert_tree = self.generate_expert_tree(topic, expert_bg)
        print(f"[GraphFactory] Expert tree generated: {len(expert_tree.get_leaves())} leaves")
        
        print("[GraphFactory] Agent B: Generating researcher tree...")
        researcher_tree = self.generate_researcher_tree(topic, researcher_bg, questionnaire)
        print(f"[GraphFactory] Researcher tree generated: {len(researcher_tree.get_leaves())} leaves")
        
        # Step 3-5: Generate links using new flow
        # (Agent C judges alignments -> Math algorithm builds links -> Generate templates)
        print("[GraphFactory] Generating gap links (math-based approach)...")
        links, alignments = self.generate_links(
            topic, 
            expert_tree, 
            researcher_tree,
            generate_templates=generate_templates,
        )
        print(f"[GraphFactory] Generated {len(links)} gap links")
        
        # Build complete graph
        graph = DSAGGraph(
            topic=topic,
            researcher_bg=researcher_bg,
            expert_bg=expert_bg,
            expert_tree=expert_tree,
            researcher_tree=researcher_tree,
            links=links,
            metadata={
                "created_at": datetime.utcnow().isoformat(),
                "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
                "expert_leaves": len(expert_tree.get_leaves()),
                "researcher_leaves": len(researcher_tree.get_leaves()),
                "link_count": len(links),
                "alignment_stats": {
                    "leaf_alignments": len(alignments.leaf_alignments),
                    "concept_alignments": len(alignments.concept_alignments),
                    "misaligned_leaves": len(alignments.get_misaligned_leaf_pairs()),
                },
                "construction_method": "math-based",
            },
        )
        
        # Validate
        errors = validate_graph(graph)
        if errors:
            print(f"[GraphFactory] Validation warnings: {errors}")
        
        print("[GraphFactory] DSAG generation complete")
        return graph


# ============== Convenience Function ==============

def create_dsag_graph(
    topic: str,
    researcher_bg: str,
    expert_bg: str,
    questionnaire: str = "",
) -> DSAGGraph:
    """
    Convenience function to create a DSAG graph.
    
    Args:
        topic: Interview topic (becomes shared root)
        researcher_bg: Researcher background description
        expert_bg: Expert background description
    
    Returns:
        Complete DSAGGraph
    """
    factory = GraphFactory()
    return factory.generate_graph(topic, researcher_bg, expert_bg, questionnaire=questionnaire)
