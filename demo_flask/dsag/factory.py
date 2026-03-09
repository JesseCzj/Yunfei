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
import concurrent.futures
import time
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
- "LexicalGap": The two sides use different terms, jargon, or labels for the same concept.
- "ConceptualGap": The two sides hold different mental models, analogies, or interpretations of the concept.
- "TacitGap": The expert relies on intuition, implicit knowledge, or experience that the researcher cannot easily observe or quantify. This includes methodology-level tacit knowledge — when the expert's working approach is an intuitive blend of multiple frameworks rather than a nameable paradigm, and they cannot articulate a structured workflow because their practice is driven by experiential pattern recognition.
- "ScopeGap": The two sides differ in purpose or expectations — the expert focuses on practical utility while the researcher focuses on research value, leading to inconsistent goals about "what to do."
- "ProcessGap": A mismatch that risks disrupting the interview process. Two sub-types:
  (a) Factual Risk — The researcher's preparation contains assumptions or claims that may conflict with established domain practice. If stated as fact during the interview, the expert will correct the error, derailing the original discussion flow.
  (b) Methodology Conflict — The domain has multiple accepted paradigms or approaches for the same task, but the researcher only anticipates one. Using confirmatory questions ("You use X, right?") would miss the expert's actual practice. KEY BOUNDARY: This applies only when the expert CAN name their approach (e.g., "I use rolling CV, not k-fold"). If the expert's methodology is an unarticulated intuitive blend that resists being named or structured, classify as TacitGap instead — the problem is articulability, not paradigm selection.

Classification hint: Expert leaf nodes that contain a non-empty "attributes" list indicate tacit, intuition-based knowledge (Tacit Knowledge Facets). When such a leaf is part of a misaligned pair, strongly prefer "TacitGap" as the relation_type.

---

**One canonical example per gap type** (node format: label — description):

LexicalGap example:
  Expert:     "EHR audit trail" — The log of all actions performed on a patient's record, used by clinicians for accountability.
  Researcher: "Activity log" — A timestamped record of system interactions stored in the database, used for access-control analysis.
  → LexicalGap: Both sides describe the same artifact. The difference is purely terminological — adopting a shared label would fully close the gap.

ConceptualGap example:
  Expert:     "Clinical reliability" — A system that is always available when I need it during rounds — no downtime, no lag.
  Researcher: "System reliability" — The degree to which a system produces consistent, reproducible outputs under repeated testing conditions.
  → ConceptualGap: Both use "reliability" but hold fundamentally different mental models — uptime/availability vs. statistical reproducibility. Agreeing on a term would not resolve the disagreement; the underlying constructs differ.

TacitGap example:
  Expert:     "Deterioration judgment" — Recognizing when a patient is about to decline based on subtle cues — skin tone, breathing rhythm, demeanor — before vitals change. [attributes: "skin color change", "respiratory pattern shift", "patient demeanor", "nurse intuition threshold"]
  Researcher: "Early deterioration signal" — A quantifiable indicator that precedes a clinical deterioration event, suitable for inclusion in a predictive model.
  → TacitGap: The expert possesses valid, structured knowledge (confirmed by non-empty attributes list) but it exists as implicit pattern recognition. The knowledge is real — it has not yet been surfaced and made explicit.

ScopeGap example:
  Expert:     "Alarm threshold tuning" — Adjusting alert thresholds to reduce false positives and make the alarm system practical for daily clinical use.
  Researcher: "Alarm fatigue measurement" — Quantifying the systemic cognitive burden that high alarm volumes impose on clinical staff, as a variable in a human-factors study.
  → ScopeGap: The expert's goal is practical utility (make alarms less disruptive); the researcher's goal is research value (measure alarm fatigue as a construct). They disagree on what this work should produce — not on what the concept means.

ProcessGap example (Factual Risk):
  Expert:     "Patient discharge bottleneck" — The final pharmacy reconciliation review takes 30-60 minutes per patient and is the rate-limiting step before discharge.
  Researcher: "Discharge workflow delay" — The primary bottleneck in the discharge process is waiting for the attending physician's sign-off, which could be streamlined through automation.
  → ProcessGap: The researcher's preparation assumes physician sign-off is the bottleneck, but in this expert's domain, pharmacy reconciliation is the actual rate-limiting step. If the researcher states "the main delay is physician sign-off" as fact, the expert will correct the error, consuming interview time on factual correction instead of sharing deeper workflow insights. The gap is about a factual vulnerability in the researcher's preparation, not about differing mental models.

ProcessGap example (Methodology Conflict):
  Expert:     "Model validation practice" — We validate using rolling cross-validation tailored to our temporal clinical data; standard k-fold would leak future information into training.
  Researcher: "Cross-validation protocol" — Applying k-fold cross-validation as the standard evaluation methodology for predictive model performance.
  → ProcessGap: The domain has multiple valid validation approaches depending on data structure. The researcher only anticipates standard k-fold, but the expert uses a domain-specific variant that avoids temporal data leakage. If the researcher asks "How do you handle your k-fold splits?", the expert may either correct the premise or answer within the wrong framework. The gap is about procedural multiplicity — the researcher must use open-ended process questions ("How do you validate your models?") instead of confirmatory ones.

---

**One boundary edge case per pair of types** (what looks like X but is actually Y):

[LexicalGap vs ConceptualGap]
  Expert:     "Patient risk score" — The number the system gives a patient; high means they need attention soon.
  Researcher: "Risk stratification" — Classifying patients into probabilistic risk tiers based on weighted feature combinations in a predictive model.
  → ConceptualGap (NOT LexicalGap): The expert's concept is a simple output number for triage priority; the researcher's concept is a multi-step classification process. These are different mental models of what risk assessment IS — not the same concept under different labels. A shared label would not close the gap.

[LexicalGap vs TacitGap]
  Expert:     "Clinical intuition" — The sense that something is wrong with a patient even when numbers look normal — experienced nurses develop this over years.
  Researcher: "Non-quantifiable clinical signal" — An expert-identified indicator that lacks a formal operational definition.
  → TacitGap (NOT LexicalGap): The researcher's label is an attempt to formalize what the expert experiences as intuition. Simply agreeing on a shared term would not help the researcher extract the actual knowledge — the gap is about surfacing implicit experiential knowledge, not about aligning vocabulary.

[LexicalGap vs ScopeGap]
  Expert:     "Medication reconciliation" — Checking that drugs listed in the system match what the patient is actually taking at every care transition.
  Researcher: "Drug list validation" — Verifying the completeness and accuracy of the pharmacological record in the EHR.
  → LexicalGap (NOT ScopeGap): Both sides describe the same process and share the same goal — understand this verification step. The difference is vocabulary only ("medication reconciliation" vs. "drug list validation"). There is no divergence in purpose or expectations.

[ConceptualGap vs TacitGap]
  Expert:     "Patient severity assessment" — I look at labs, vitals, how the patient looks, and just know whether they're sick. Hard to explain, but you learn it. [attributes: "lab trend pattern", "vital sign cluster", "patient appearance", "gut feeling threshold"]
  Researcher: "Acuity scoring" — A structured composite score based on weighted physiological parameters that classifies patient severity on a standardized scale.
  → TacitGap (NOT ConceptualGap): The expert is not operating on a different model of what "severity" means — they agree it involves labs, vitals, and patient state. The problem is that their assessment process is experiential and unarticulated, confirmed by the non-empty attributes list. A ConceptualGap would require fundamental disagreement about what "severity" IS.

[ConceptualGap vs ScopeGap]
  Expert:     "Workflow efficiency" — Getting through the patient list faster by reducing the number of steps and screens needed to document each encounter.
  Researcher: "Workflow optimization" — Identifying cognitive bottlenecks in clinical workflows that can be modeled and reduced through system redesign.
  → ConceptualGap (NOT ScopeGap): Both sides want to improve workflows — their interview goals are aligned. The gap is in their mental models of what "optimization" means: reducing manual steps vs. modeling cognitive bottlenecks. This is a disagreement about how the concept works, not about what the interview should produce.

[TacitGap vs ScopeGap]
  Expert:     "Drug dosing judgment" — Adjusting doses based on weight, age, renal function, and what's worked before — I know when to deviate from the protocol. [attributes: "patient weight", "renal function marker", "prior drug response", "experience threshold"]
  Researcher: "Dosing parameter study" — Identifying which patient variables most influence clinician deviation from standard protocols, to inform future guideline development.
  → TacitGap (NOT ScopeGap): The researcher's goal — extract dosing variables — is fully aligned with the interview purpose. The problem is that the expert's decisions are driven by implicit pattern recognition (confirmed by attributes). There is no divergence in expectations; there is an articulability problem. A ScopeGap would require the expert to want the interview to produce something entirely different.

[ConceptualGap vs ProcessGap]
  Expert:     "Feature engineering" — Manually crafting domain-specific input features based on years of clinical knowledge before feeding them to the model.
  Researcher: "Feature extraction" — Using automated methods (PCA, autoencoders) to derive discriminative features from raw data.
  → ConceptualGap (NOT ProcessGap): The difference is in their mental models of what "feature creation" means — manual domain knowledge vs. automated computation. Neither side's approach is factually wrong, and the domain does not have "multiple paradigms for the same step" — these are genuinely different concepts based on different philosophies. A ProcessGap would require either a factual error in the researcher's claim or the existence of multiple valid approaches to the SAME task that the researcher fails to anticipate.

[ScopeGap vs ProcessGap]
  Expert:     "Clinical trial endpoint selection" — We pick endpoints based on what is achievable in our patient population and what regulators accept.
  Researcher: "Outcome measure selection" — Selecting outcomes that maximize statistical power and align with the study's theoretical framework.
  → ScopeGap (NOT ProcessGap): Both sides describe the same procedural step (selecting endpoints/outcomes). The difference is in their priorities — practical feasibility vs. research rigor. There is no factual error in the researcher's approach, and both accept that endpoint selection is necessary. The gap is about purpose/focus divergence, not about the researcher misunderstanding domain practice or missing alternative methodologies.

[TacitGap vs ProcessGap]
  Expert:     "Therapeutic approach selection" — I draw from CBT, psychodynamic, and mindfulness techniques depending on what the patient presents — it's not one framework, I just know what fits after years of practice. [attributes: "patient presentation pattern", "therapeutic rapport cues", "intervention selection intuition", "session pacing judgment"]
  Researcher: "Therapeutic methodology" — Identifying which evidence-based therapeutic framework (CBT, psychodynamic, DBT) the clinician primarily uses, to document a step-by-step treatment protocol.
  → TacitGap (NOT ProcessGap): This looks like methodology_conflict because the domain has multiple paradigms and the researcher assumes one. But the expert CANNOT name a single method — their practice is an intuitive blend driven by experiential pattern recognition (confirmed by non-empty attributes). The core problem is not "the researcher picked the wrong paradigm" but "the expert's methodology is itself tacit and resists structured-workflow extraction." An open-process question alone cannot solve this; the researcher needs attribute decomposition and probes to surface the implicit decision factors.

---

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
        "process", "workflow", "procedure", "method", "methodology",
        "pipeline", "protocol", "paradigm", "approach",
        "factual", "incorrect", "error", "wrong", "inaccurate",
        "assumption", "assumes", "presuppose",
        # -- Tier 2 (2pt) --
        "factual risk", "factual error", "factual vulnerability",
        "incorrect assumption", "wrong assumption",
        "multiple approaches", "multiple paradigms", "multiple methods",
        "methodology conflict", "conflicting methods",
        "different procedure", "procedural mismatch",
        "domain practice", "standard practice",
        "confirmatory question", "presupposes a method",
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

TACIT_GAP_PROMPT = """You are generating an exhaustive probe arsenal to surface tacit knowledge from a domain expert during an interview.

**Expert Concept:**
- Label: {expert_label}
- Description: {expert_description}
- Seed Attributes (from expert tree): {attributes}

**Researcher Concept:**
- Label: {researcher_label}
- Description: {researcher_description}

**Misalignment Reason:** {misalignment_reason}

The expert holds intuitive, experience-based knowledge that is hard to articulate.
The researcher needs explicit, quantifiable design parameters.

Follow a strict 3-step pipeline:

Step 1: EXPAND the attribute list.
  The seed attributes above are Agent A's initial 2-4 facets. Your job is to generate MORE possible/potential related attributes that may underlie the expert's tacit knowledge for this concept. Consider:
  - Decision heuristics the expert might implicitly use
  - Contextual cues or environmental constraints
  - Experience-based thresholds or patterns
  - Variables the expert might weigh subconsciously
  Include the original seed attributes AND your newly generated ones.
  Aim for 6-10 total attributes to build a comprehensive arsenal.

Step 2: For EACH attribute (both seed and expanded), generate a MULTIPLE-CHOICE probe question to guide the expert to dismantle factors through quantifiable indicators.
  RULE: NEVER ask open-ended "Why" questions. Instead, formulate as:
  "Is your concern related to [Attribute A] or [Attribute B]?" or
  "Which matters more: A, B, or C?"

Step 3: Generate 2-3 hypothetical scenarios for optional checkout.
  RULE: Each scenario alters only ONE variable at a time to test the boundaries of their intuition.
  Cover variables from DIFFERENT attributes to maximize diagnostic breadth.

Return ONLY valid JSON:
{{
  "attributes": ["seed_attr1", "seed_attr2", "expanded_attr3", "expanded_attr4", "..."],
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

PROCESS_GAP_PROMPT = """You are generating a preventive assistance payload for a ProcessGap — a mismatch that risks disrupting the interview process.

**Expert Concept:**
- Path: {expert_path}
- Label: {expert_label}
- Description: {expert_description}

**Researcher Concept:**
- Path: {researcher_path}
- Label: {researcher_label}
- Description: {researcher_description}

**Misalignment Reason (from alignment judge):** {misalignment_reason}

ProcessGap has two sub-types. Read the misalignment reason carefully and determine which applies:

(a) **Factual Risk**: The researcher's preparation contains an assumption or claim that may conflict with established domain practice. If stated as fact, the expert will correct it, consuming interview time on error correction.

(b) **Methodology Conflict**: The domain has multiple accepted approaches for this task, but the researcher only anticipates one. Confirmatory questions would miss the expert's actual practice.

Determine the sub-type, then generate the appropriate fields.

Return ONLY valid JSON matching ONE of these two schemas:

Schema A (Factual Risk):
{{
  "sub_type": "factual_risk",
  "vulnerable_assumption": "The specific claim/assumption in the researcher's preparation that may be incorrect",
  "domain_correction": "What experts in this domain actually believe or practice",
  "safe_phrasing": "A suggested open question the researcher can ask instead of asserting the assumption"
}}

Schema B (Methodology Conflict):
{{
  "sub_type": "methodology_conflict",
  "known_approaches": ["Approach/paradigm A", "Approach/paradigm B"],
  "researcher_assumed_approach": "The single approach the researcher's preparation implies",
  "open_process_question": "A suggested open-ended process question that does not presuppose any specific method"
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
        if not data:
            data = {}
        if link.relation_type == RelationType.PROCESS_GAP.value:
            data["misalignment_reason"] = misalignment_reason
        return data
    except Exception as e:
        print(f"[GraphFactory] Error generating payload for {link.relation_type}: {e}")
        if link.relation_type == RelationType.PROCESS_GAP.value:
            return {"misalignment_reason": misalignment_reason}
        return {}


def generate_all_assistance_payloads(
    links: List[GapLink],
    expert_tree: TaxonomyTree,
    researcher_tree: TaxonomyTree,
    alignments: TreeAlignments,
) -> List[GapLink]:
    """Generate type-specific assistance payloads for all GapLinks concurrently."""
    print(f"[GraphFactory] Generating assistance payloads for {len(links)} links concurrently...")

    # Keep this configurable to match provider rate limits.
    max_workers = max(1, int(os.getenv("DSAG_PAYLOAD_MAX_WORKERS", "5")))

    # Build lookup for misalignment reasons
    reason_lookup = {}
    for align in alignments.leaf_alignments:
        key = (align.expert_node_id, align.researcher_node_id)
        reason_lookup[key] = align.reason

    def process_single_link(link: GapLink) -> GapLink:
        # Use a per-thread/per-task LLM client to avoid shared-client race issues.
        local_llm = _build_llm(temperature=0.7)
        key = (link.expert_leaf_id, link.researcher_leaf_id)
        reason = reason_lookup.get(key, "")

        payload = generate_assistance_payload_for_link(
            link=link,
            expert_tree=expert_tree,
            researcher_tree=researcher_tree,
            misalignment_reason=reason,
            llm=local_llm,
        )
        link.assistance_payload = payload
        return link

    # Process links concurrently. map preserves input order.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process_single_link, links))

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

    def _invoke_tree_prompt_with_retry(
        self,
        prompt_text: str,
        variables: Dict[str, Any],
        tree_name: str,
    ) -> Dict[str, Any]:
        """
        Invoke a tree-generation prompt with retries and strict JSON validation.
        Uses a fresh LLM client per attempt for thread-safety in concurrent mode.
        """
        max_attempts = max(1, int(os.getenv("DSAG_TREE_MAX_RETRIES", "3")))
        last_content = ""
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                llm = _build_llm(temperature=0.3)
                prompt = ChatPromptTemplate.from_messages([("user", prompt_text)])
                chain = prompt | llm
                response = chain.invoke(variables)
                content = getattr(response, "content", str(response))
                last_content = content
                parsed = _parse_json(content)
                perspectives = parsed.get("perspectives")
                if isinstance(perspectives, list) and perspectives:
                    return parsed

                print(
                    f"[GraphFactory] {tree_name} attempt {attempt}/{max_attempts}: "
                    "no valid perspectives in response"
                )
            except Exception as exc:
                last_error = exc
                print(
                    f"[GraphFactory] {tree_name} attempt {attempt}/{max_attempts} failed: {exc}"
                )

            # Small exponential backoff: 0.8s, 1.6s, 2.4s...
            if attempt < max_attempts:
                time.sleep(0.8 * attempt)

        if last_error is not None:
            raise ValueError(
                f"Failed to generate {tree_name.lower()} after {max_attempts} attempts: {last_error}"
            )
        raise ValueError(
            f"Failed to generate {tree_name.lower()}: no perspectives returned. "
            f"Last raw response preview: {str(last_content)[:240]}"
        )
    
    def generate_expert_tree(self, topic: str, expert_bg: str) -> TaxonomyTree:
        """Agent A: Generate expert taxonomy tree."""
        parsed = self._invoke_tree_prompt_with_retry(
            prompt_text=EXPERT_PERSONA_PROMPT,
            variables={
            "topic": topic,
            "expert_bg": expert_bg,
            },
            tree_name="Expert tree",
        )
        return _build_expert_tree(parsed, topic)
    
    def generate_researcher_tree(
        self,
        topic: str,
        researcher_bg: str,
        questionnaire: str = "",
    ) -> TaxonomyTree:
        """Agent B: Generate researcher taxonomy tree."""
        parsed = self._invoke_tree_prompt_with_retry(
            prompt_text=RESEARCHER_PERSONA_PROMPT,
            variables={
            "topic": topic,
            "researcher_bg": researcher_bg,
            "questionnaire": questionnaire or "",
            },
            tree_name="Researcher tree",
        )
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

        # Step 1 & 2: Generate both trees (parallel preferred, serial fallback)
        tree_concurrent = os.getenv("DSAG_TREE_CONCURRENT", "1").strip().lower() not in (
            "0", "false", "no"
        )
        tree_workers = max(1, int(os.getenv("DSAG_TREE_MAX_WORKERS", "2")))

        expert_tree: TaxonomyTree
        researcher_tree: TaxonomyTree

        if tree_concurrent and tree_workers > 1:
            print(f"[GraphFactory] Agent A & B: Generating trees concurrently (workers={tree_workers})...")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=tree_workers) as executor:
                    future_expert = executor.submit(self.generate_expert_tree, topic, expert_bg)
                    future_researcher = executor.submit(
                        self.generate_researcher_tree,
                        topic,
                        researcher_bg,
                        questionnaire,
                    )
                    expert_tree = future_expert.result()
                    researcher_tree = future_researcher.result()
            except Exception as exc:
                # Weak networks/proxies may fail under simultaneous long requests.
                # Fall back to serial generation for higher success rate.
                print(f"[GraphFactory] Concurrent tree generation failed: {exc}")
                print("[GraphFactory] Falling back to serial tree generation...")
                expert_tree = self.generate_expert_tree(topic, expert_bg)
                researcher_tree = self.generate_researcher_tree(topic, researcher_bg, questionnaire)
        else:
            print("[GraphFactory] Agent A: Generating expert tree (serial mode)...")
            expert_tree = self.generate_expert_tree(topic, expert_bg)
            print("[GraphFactory] Agent B: Generating researcher tree (serial mode)...")
            researcher_tree = self.generate_researcher_tree(topic, researcher_bg, questionnaire)

        print(
            "[GraphFactory] Trees generated. "
            f"Expert leaves: {len(expert_tree.get_leaves())}, "
            f"Researcher leaves: {len(researcher_tree.get_leaves())}"
        )
        
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
