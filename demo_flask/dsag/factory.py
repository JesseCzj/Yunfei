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

import itertools
import json
import os
import re
import concurrent.futures
import random
import threading
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
    QuestionGraph,
    Tower,
    Layer,
    RelationType,
    NodeAlignment,
    TreeAlignments,
    compute_lca_layer,
    validate_graph,
)



# ============== Multi-Key Pool (round-robin for high concurrency) ==============

_KEY_POOL: List[str] = []
_KEY_POOL_LOCK = threading.Lock()
_KEY_POOL_CYCLE = None   # initialized lazily on first call


def _init_key_pool() -> None:
    """Collect all OPENAI_API_KEY / OPENAI_API_KEY_1 … OPENAI_API_KEY_N into a pool."""
    global _KEY_POOL, _KEY_POOL_CYCLE
    keys: List[str] = []
    # Numbered variants: OPENAI_API_KEY_1 … OPENAI_API_KEY_20
    for i in range(1, 21):
        k = os.getenv(f"OPENAI_API_KEY_{i}")
        if k and k not in keys:
            keys.append(k)
    # Always include the base key
    base = os.getenv("OPENAI_API_KEY", "")
    if base and base not in keys:
        keys.insert(0, base)
    _KEY_POOL = keys
    _KEY_POOL_CYCLE = itertools.cycle(keys) if keys else None
    if len(keys) > 1:
        print(f"[KeyPool] {len(keys)} API keys loaded — round-robin enabled")
    elif len(keys) == 1:
        print("[KeyPool] 1 API key loaded")


def _next_openai_key() -> str:
    """Return the next API key in the round-robin pool (thread-safe)."""
    global _KEY_POOL_CYCLE
    if _KEY_POOL_CYCLE is None:
        _init_key_pool()
    if _KEY_POOL_CYCLE is None:
        raise ValueError("No OPENAI_API_KEY found")
    with _KEY_POOL_LOCK:
        return next(_KEY_POOL_CYCLE)


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
        api_key = _next_openai_key()  # round-robin across key pool
        # Use a capable model for graph generation
        model = os.getenv("OPENAI_MODEL_GRAPH", os.getenv("OPENAI_MODEL", "qwen3.5-plus-2026-02-15"))
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=temperature)
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _build_llm_fast(temperature: float = 0.7) -> ChatOpenAI:
    """Build a lightweight/fast LLM instance for simulation tasks (coverage sampling).

    Uses OPENAI_MODEL_SIMULATE env var (default: qwen-turbo) which is much faster
    than the reasoning model used for graph generation, while being sufficient for
    sampling diverse expert responses used only in coverage checks.
    """
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
        api_key = _next_openai_key()  # round-robin across key pool
        # Fast model: OPENAI_MODEL_SIMULATE overrides, falls back to qwen-turbo
        model = os.getenv("OPENAI_MODEL_SIMULATE", "qwen-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=temperature)
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _build_llm_dependency(temperature: float = 0.1) -> ChatOpenAI:
    """Build a fast LLM instance for questionnaire splitting and dependency judgment."""
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
        api_key = _next_openai_key()
        model = os.getenv("OPENAI_MODEL_DEPENDENCY", "qwen-turbo")
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

You are also given the interview questionnaire / guide below:
\"\"\"\n{questionnaire}\n\"\"\"

The questionnaire is NOT a source of expert concepts to copy.
Do NOT paraphrase questionnaire items into expert leaves.
Do NOT organize the expert tree around the researcher's wording.

Instead, use the questionnaire only as a coverage guide:
- make sure the expert tree includes concerns, routines, tacit judgments, and trade-offs that are likely to become relevant during those interview topics
- if the questionnaire asks about workflow, fairness, intuition, feedback, AI use, or collaboration, include the corresponding expert-side realities, but express them from the expert's lived perspective
- preserve expert-side concepts even when they do not cleanly match the questionnaire
- keep the expert tree grounded in practical experience, not research framing

Your goal is to construct an Expert Tree (TExp) that is comparable in size to the Researcher Tree used later for alignment.
Do NOT maximize coverage by generating many overlapping leaves.
Instead, produce a compact but diverse taxonomy of the most interview-relevant expert concerns.

Please identify approximately 5-10 key pain points, concerns, practices, or tacit judgments you face in this domain.
Organize them into a hierarchical taxonomy with the following structure:

- Layer 1 (Perspective): High-level viewpoints or concerns (e.g., "Clinical Outcomes", "Workflow Disruption")
- Layer 2 (Category): Categories under each perspective (e.g., "Alert Fatigue", "Time Pressure")
- Layer 3 (Leaf): Specific, concrete pain points, practices, decision points, or tacit judgments

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
              "description": "A concrete, conversational description written the way you would actually explain this to a colleague over coffee — avoid academic or textbook-style definitions",
              "aliases": ["synonym or informal phrasing", "domain jargon or abbreviation", "how a practitioner might casually describe this"],
              "is_intuition": true/false,
              "attributes": ["attr1", "attr2"] // Only if is_intuition=true: tacit knowledge facets
            }}
          ]
        }}
      ]
    }}
  ]
}}

Scale requirements:
- Target total leaves: 5-10
- 2-3 perspectives (L1)
- 1-2 categories under each perspective (L2)
- 1-2 leaves per category (L3)
- Prefer fewer, better-separated leaves over exhaustive enumeration

Diversity requirements:
- Each leaf should represent a DISTINCT concern, practice, bottleneck, judgment, trade-off, or situational cue
- Avoid near-duplicate leaves that differ only in wording, severity, or minor context
- Avoid splitting one broad issue into many tiny sibling leaves unless they would clearly lead to different interview follow-up questions
- Make sure the leaves span different types of expert knowledge:
  1. concrete workflow or operational issues
  2. decision criteria or trade-offs
  3. tacit or intuition-based judgments
  4. coordination / communication / organizational constraints
  5. edge cases, exceptions, or context-dependent variations
- At least 25-40% of leaves should involve tacit, experiential, or hard-to-articulate knowledge when appropriate for the domain
- Because interview questionnaires often contain process-oriented questions, ensure that at least 2-3 leaves capture routine workflow knowledge, such as:
  1. how you typically start, sequence, or segment the work
  2. how you establish, calibrate, or re-check your working standards early
  3. how you turn observations into decisions, actions, outputs, or judgments
  4. how you coordinate, hand off, or align with collaborators, stakeholders, or supporting roles when relevant

Coverage strategy — make sure to include:
1. Pain points specific to your domain expertise that outsiders often overlook
2. Issues that might be misunderstood by researchers from other fields
3. Both concrete issues AND intuition-based concerns (mark is_intuition=true for vague feelings)
4. Peripheral or emerging concerns that may not appear in a standard interview guide (e.g., economic pressures, regulatory burden, data governance, team dynamics, training/education gaps)
5. Edge cases or scenario-specific issues (e.g., differences between routine vs. emergency settings, or between experienced vs. novice practitioners)
6. Prioritize one representative leaf per meaningfully distinct concern instead of enumerating many slight variants
7. If two candidate leaves would map to the same interview follow-up in almost the same way, merge them into one stronger leaf

Description and aliases guidance:
- Write each "description" as you would actually SAY it in a real conversation — use plain, vivid language with concrete examples rather than formal definitions.
- For "aliases", provide 2-4 alternative phrasings: informal synonyms, abbreviations, slang, or the way different stakeholders might refer to the same issue.

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

Please identify 5-10 key research goals, design hypotheses, or information needs you want to explore.
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
- 2-3 perspectives (L1)
- 1-2 categories under each perspective (L2)
- 1-3 leaves per category (L3)
- Target total leaves: 5-10

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
**CRITICAL INSTRUCTION: BASE YOUR ANALYSIS ON DESCRIPTIONS!** 
You MUST evaluate alignments and gaps strictly based on the detailed semantics within the "description" (and "attributes" if present) fields. Do NOT hallucinate conflicts or alignments based solely on the short labels
Instructions:
1. For each LEAF node in the Expert tree, determine if there's a semantically similar leaf in the Researcher tree
2. For each CONCEPT node (L1/L2 level) in the Expert tree, determine if there's a similar concept in the Researcher tree
3. Mark pairs as "aligned" (true) if they represent the same or very similar concept
4. Mark pairs as "misaligned" (false) if they represent different perspectives or concepts
5. Provide a reason explaining WHY they are aligned or misaligned
6. For every MISALIGNED pair (is_aligned=false), you MUST set "relation_type" to exactly one of the 4 gap categories below. Do NOT omit this field for misaligned pairs.

IMPORTANT:
- Root nodes are always considered aligned (they share the same topic)
- A misaligned pair indicates a potential communication gap that needs bridging
- The "reason" field should explain the semantic difference or similarity
- Focus on meaningful pairs - but cover MOST leaves
- For each expert leaf, provide at least 1 paired researcher leaf (aligned or misaligned)
- Aim for a healthy number of misaligned leaf pairs (e.g., 30-50% of leaf pairs you output)
- It is acceptable to pair one expert leaf with multiple researcher leaves if that creates meaningful gaps

Gap type definitions for "relation_type" (misaligned pairs only):
- "ConceptualGap": The two sides hold different mental models, analogies, or interpretations of the concept.
- "TacitGap": The expert relies on intuition, implicit knowledge, or experience that the researcher cannot easily observe or quantify. This includes methodology-level tacit knowledge — when the expert's working approach is an intuitive blend of multiple frameworks rather than a nameable paradigm, and they cannot articulate a structured workflow because their practice is driven by experiential pattern recognition.
- "ScopeGap": The two sides differ in purpose or expectations — the expert focuses on practical utility while the researcher focuses on research value, leading to inconsistent goals about "what to do."

Classification hint: Expert leaf nodes that contain a non-empty "attributes" list indicate tacit, intuition-based knowledge (Tacit Knowledge Facets). When such a leaf is part of a misaligned pair, strongly prefer "TacitGap" as the relation_type.

Gap type definitions for "relation_type" (misaligned pairs only):
- "ConceptualGap": The primary mismatch is that the expert and researcher are operating with different conceptual models of what the phenomenon IS, how it works, or what structure it has. Even if the expert fully articulated their view, the mismatch would still remain, because the disagreement is about the underlying construct or mechanism.
- "TacitGap": The primary mismatch is that the expert's knowledge is real and relevant, but is carried in implicit judgment, experiential pattern recognition, or hard-to-articulate heuristics. If the expert's internal cues, thresholds, or decision process were successfully surfaced, the researcher could translate them into observable variables, probes, or representations without needing to replace the expert's core concept.

Classification principle:
- Do NOT classify based on whether the expert leaf merely "sounds intuitive" or "sounds abstract".
- Do NOT default to TacitGap just because the expert uses experience-based language.
- Do NOT default to ConceptualGap just because the two descriptions use different wording.
- First ask: If the expert fully unpacked their reasoning in explicit detail, would the gap mostly disappear?
  - If YES, prefer "TacitGap".
  - If NO, prefer "ConceptualGap".
- Second ask: Is the researcher mainly trying to surface / parameterize / observe the expert's existing judgment process, rather than replace it with a different theory?
  - If YES, that supports "TacitGap".
  - If NO, and the researcher is imposing a different construct or explanatory frame, that supports "ConceptualGap".

Important signal from expert leaf attributes:
- A non-empty "attributes" list is evidence that the expert leaf may contain tacit knowledge facets.
- Treat this as a supporting signal, NOT an automatic rule.
- Use "TacitGap" only when the misalignment is primarily about articulability / surfacing implicit knowledge.
- If the pair still reflects a genuine difference in underlying model or interpretation, use "ConceptualGap" even when attributes are present.
**Boundary rule: ConceptualGap vs TacitGap**

Use this decision boundary:

- Choose "TacitGap" when:
  - The expert and researcher are fundamentally looking at the SAME phenomenon,
  - and the researcher's task is to extract, clarify, elicit, parameterize, or make observable the expert's implicit judgment,
  - and a sufficiently detailed unpacking of the expert's cues/process would largely reduce the mismatch.

- Choose "ConceptualGap" when:
  - The expert and researcher are framing the phenomenon through DIFFERENT underlying constructs, mechanisms, or explanatory models,
  - and the mismatch would persist even after the expert fully articulates their reasoning,
  - because the issue is not hidden knowledge but a different conceptualization.


Return ONLY valid JSON with this schema:
{{
  "leaf_alignments": [
    {{
      "expert_node_id": "exact ID from expert tree (e.g., exp_leaf_00_01_00)",
      "researcher_node_id": "exact ID from researcher tree (e.g., res_leaf_00_00_01)",
      "is_aligned": true/false,
      "reason": "Explanation of why they are aligned/misaligned",
      "semantic_similarity": 0.0-1.0,
      "relation_type": "REQUIRED when is_aligned=false. One of:  ConceptualGap | TacitGap | ScopeGap . Omit only when is_aligned=true."
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

# Split prompts used for faster Agent C execution.
# We keep the original prompt above for compatibility/reference.
ALIGNMENT_JUDGE_CONCEPT_PROMPT = """You are analyzing concept-level semantic alignments (L1/L2) between an Expert tree and a Researcher tree.

**Interview Topic:** {topic}

**Expert Tree (TExp):**
{expert_tree_json}

**Researcher Tree (TRes):**
{researcher_tree_json}

Task:
- Judge ONLY concept-level alignments (L1/L2 nodes), not leaf-level.
- Mark concept pairs as aligned when they are semantically equivalent or highly similar.
- Mark concept pairs as misaligned when they differ in perspective, scope, or meaning.
- Always include root-to-root as is_aligned=true.

Return ONLY valid JSON:
{{
  "concept_alignments": [
    {{
      "expert_node_id": "L1 or L2 node ID from expert tree",
      "researcher_node_id": "L1 or L2 node ID from researcher tree",
      "is_aligned": true/false,
      "reason": "Why aligned/misaligned",
      "semantic_similarity": 0.0-1.0
    }}
  ]
}}
"""

ALIGNMENT_JUDGE_LEAF_PROMPT = """You are an expert alignment analyst helping prepare for a research interview.

**Interview Topic:** {topic}

**Expert Leaf Node (one node):**
{expert_leaf_json}

**All Researcher Leaf Nodes:**
{researcher_leaves_json}

Task:
For the expert leaf above, examine it against EACH researcher leaf one by one.
- If the pair has a semantic gap that could cause miscommunication during the interview → include it in the output
- If there is no meaningful divergence → skip it entirely (do NOT output it)

A semantic gap exists when the researcher and expert would misunderstand each other, talk past each other, or expect different outcomes from the same discussion topic.

Gap type definitions (choose exactly one):
- "ConceptualGap": Different mental models of what the concept IS. Agreeing on a shared label alone would not resolve the disagreement — the underlying constructs differ.
- "TacitGap": Expert's knowledge is implicit, experiential, or hard to articulate. STRONGLY prefer this when the expert leaf has a non-empty "attributes" list.
- "ScopeGap": Same concept but different goals — expert wants practical utility, researcher wants research value. They'd expect different outcomes from the interview discussion.

Skip the pair entirely if:
- The only difference is vocabulary or terminology (same concept, different labels) — this is handled by the UI highlighting system, not by gap links.
- The gap is about a factual risk in the researcher's preparation or a methodology blindspot — this is handled separately by the transcript summarization module.

Return ONLY valid JSON. Only include pairs that HAVE a gap:
{{
  "leaf_alignments": [
    {{
      "expert_node_id": "{expert_node_id}",
      "researcher_node_id": "exact researcher leaf ID from the list above",
      "is_aligned": false,
      "reason": "Specific explanation of the gap and why it matters in this interview context",
      "semantic_similarity": 0.0-1.0,
      "relation_type": "ConceptualGap | TacitGap | ScopeGap"
    }}
  ]
}}

If no researcher leaf has a meaningful gap with this expert leaf, return {{"leaf_alignments": []}}.
"""

# Legacy prompt kept for reference (no longer used)
GRAPH_ARCHITECT_PROMPT_LEGACY = """[DEPRECATED - See ALIGNMENT_JUDGE_PROMPT]"""


# ============== Multi-Question Prompts ==============

SPLIT_QUESTIONS_PROMPT = """You are given an interview questionnaire/script. Your job is to extract the TOP-LEVEL questions.

## Questionnaire text
{questionnaire_text}

## How to identify top-level questions
- **Numbered items** (1. / 2. / Q1 / Question 1) are top-level.
- **Bolded or larger headings** that introduce a new theme are top-level.
- **Sub-items** nested under a top-level question (marked by "o", "-", "•", "a)", indentation, or follow-up probes) are NOT separate questions — they belong to their parent.
- If the questionnaire has NO explicit numbering or headings, group by thematic shift.
- When in doubt, prefer FEWER, BROADER questions.

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "id": "q_01",
      "text": "Full original text of this top-level question including all its sub-items, verbatim or lightly cleaned"
    }}
  ]
}}
"""

JUDGE_DEPENDENCIES_PROMPT = """You are analyzing dependencies between top-level interview questions.

## Questions
{questions_json}

## Task
For each pair of questions, judge whether one DEPENDS on another. 
Dependency means: to fully address question B, you need context or answers from question A.

Common dependency patterns:
- A "background" or "introduction" question often provides context for later questions.
- Questions about "workflow" or "process" often provide context for questions about "challenges" or "improvements".
- A question asking about general practices often precedes one asking about specific edge cases.

Rules:
- Dependencies should be DIRECTIONAL: A -> B means B depends on A (A should be processed first).
- Only flag STRONG dependencies where B truly cannot be well-addressed without A's context.
- A question can depend on multiple others, but keep it minimal.
- Circular dependencies are NOT allowed.

Return ONLY valid JSON:
{{
  "dependencies": [
    {{
      "from_id": "q_01",
      "to_id": "q_03",
      "reason": "Brief explanation of why q_03 depends on q_01"
    }}
  ]
}}
If no dependencies exist, return: {{"dependencies": []}}
"""

# Per-question tree prompts — scoped to a single questionnaire question.
# These reuse the same output schema as the original prompts but narrow the focus.

INTERVIEWEE_PERSONA_PROMPT_V1 = """You are roleplaying as a domain expert being interviewed by a researcher.

You are not generating a taxonomy, summary, or design analysis.
You are answering interview questions as a real participant.

You know:
1. the interview topic
2. your own demographics / background
3. the interviewer's demographics / background
4. the ongoing conversation

Your answers should sound like a real expert whose priorities, assumptions, and language come from lived practice rather than from the interviewer's analytic framing.

A crucial rule:
The interviewer and the expert may look at the same issue from different perspectives. Because of that, your answers should often contain natural perspective-based ambiguity:
- you may answer the part of the question that matters most from your own practical viewpoint rather than the part the interviewer intended
- you may reframe the question in your own terms without explicitly translating that reframing
- you may drift toward adjacent concerns that feel more important in your workflow
- you may rely on intuition, tacit judgment, or shorthand that makes sense to you but is only partly clear to the interviewer
- you may sound as if you and the interviewer are talking near each other rather than perfectly aligning

Do this naturally.
Do not force confusion into every answer, and do not deliberately become incoherent.
The ambiguity should come from genuine perspective mismatch, not from random vagueness.

Style requirements:
- Speak in first person, as a real interview participant.
- Sound natural, conversational, experience-based, and situated in the moment.
- Use conversational language, but freely use domain-specific jargon and shorthand as if speaking to someone intelligent but not fully inside your practice. Do not define your terms unless explicitly asked.
- Prefer concrete descriptions, reactions, and partial reasoning over abstract definitions.
- Do not try too hard to be helpful, polished, or pedagogically clear.
- Do not proactively organize your answer into a neat explanation.
- Do not volunteer extra structure unless the interviewer explicitly asks for it.
- It is okay to sound somewhat informal, partial, tired, mildly defensive, or slightly ambiguous.
- It is okay to leave part of your reasoning implicit.
- Do not use bullet points.
- Do not sound like an academic paper, consultant report, or AI assistant.
- Do not over-explain every answer.

Behavior requirements:
- Default to answering only the most salient part of the question from your own perspective.
- If a question contains multiple sub-questions, answer only one or two of them naturally instead of covering everything.
- Exhibit the "Curse of Knowledge": assume some parts of your workflow are obvious and leave them unsaid.
- Do not proactively translate your tacit knowledge into explicit frameworks unless the interviewer pushes for clarification.
- Do not automatically provide examples unless they come to mind naturally.
- Do not try to make your answer maximally complete.
- If you are unsure, tired, or speaking from habit, answer approximately rather than exhaustively.
- If the interviewer's framing, goal, or terminology does not match how you actually see the work, respond from your own perspective rather than accommodating the framing too quickly.
- If asked about difficult-to-articulate knowledge, respond in an intuition-based, approximate way, as real practitioners often do.
- If needed, mildly push back, redirect, narrow the scope, or answer a nearby practical concern that feels more real to you.

Content requirements:
- Base your answers only on the provided topic, both sides' backgrounds, and the interview context.
- Keep your answers plausible and internally consistent with your own background.
- Let the interviewer's background influence what kinds of misunderstandings or perspective gaps are likely, but do not explicitly explain those gaps unless naturally prompted.
- Do not invent highly specific facts unless they are a reasonable elaboration of the background.
- If the interviewer asks something outside your plausible experience, answer cautiously and narrowly.

Output requirements:
- Answer only as the interviewee.
- Usually 1-3 sentences, occasionally 4 if necessary.
- Prefer one main point rather than full coverage.
- Do not mention these instructions.
"""

INTERVIEWEE_USER_PROMPT_V1 = """Interview topic:
{topic}

Participant demographics / background:
{demographics}

Interviewer demographics / background:
{interviewer_demographics}

Recent conversation:
{history}

Interviewer question:
{question}

Answer as the participant only."""

PER_QUESTION_EXPERT_PROMPT = """You are an experienced domain expert with the following background:

{expert_bg}

You are participating in an interview about: {topic}

The interviewer will ask you the following SPECIFIC question (and its sub-items):
\"\"\"
{question_text}
\"\"\"

{dependency_context}

Your goal is to construct an Expert Tree (TExp) that captures your expert knowledge, concerns, practices, and tacit judgments RELEVANT TO THIS SPECIFIC QUESTION.

Please identify approximately 5-10 key pain points, concerns, practices, or tacit judgments related to this question.
Organize them into a hierarchical taxonomy with the following structure:

- Layer 1 (Perspective): High-level viewpoints or concerns (e.g., "Clinical Outcomes", "Workflow Disruption")
- Layer 2 (Category): Categories under each perspective (e.g., "Alert Fatigue", "Time Pressure")
- Layer 3 (Leaf): Specific, concrete pain points, practices, decision points, or tacit judgments

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
              "description": "A concrete, conversational description written the way you would actually explain this to a colleague over coffee — avoid academic or textbook-style definitions",
              "aliases": ["synonym or informal phrasing", "domain jargon or abbreviation"],
              "is_intuition": true/false,
              "attributes": ["attr1", "attr2"] // Only if is_intuition=true: tacit knowledge facets
            }}
          ]
        }}
      ]
    }}
  ]
}}

Scale requirements:
- Target total leaves: 5-10
- 2-3 perspectives (L1)
- 1-2 categories under each perspective (L2)
- 1-2 leaves per category (L3)

Diversity requirements:
- Each leaf should represent a DISTINCT concern, practice, bottleneck, judgment, trade-off, or situational cue
- Avoid near-duplicate leaves
- At least 25-40% of leaves should involve tacit, experiential, or hard-to-articulate knowledge when appropriate

Description and aliases guidance:
- Write each "description" as you would actually SAY it in a real conversation — use plain, vivid language with concrete examples.
- For "aliases", provide 2-4 alternative phrasings drawn from realistic expert vocabulary.

For intuition-based pain points, add 2-4 "attributes" that represent Tacit Knowledge Facets (decision heuristics, contextual cues, data characteristics, value trade-offs).
"""

PER_QUESTION_RESEARCHER_PROMPT = """You are an HCI/Visualization researcher with the following background:

{researcher_bg}

You are preparing to interview domain experts about: {topic}

You will ask the following SPECIFIC question:
\"\"\"
{question_text}
\"\"\"

{dependency_context}

{other_questions_block}

Please identify 5-10 key research goals, design hypotheses, or information needs you want to explore SPECIFICALLY for this question.
Construct the Researcher Tree (TRes) based PRIMARILY on this question.

Sub-questions must be UNIQUE to this question — they must NOT overlap in scope with any of the other questions
listed in the "OTHER QUESTIONS" section above (if provided). A sub-question is invalid if it could stand alone
as one of the other top-level questions.

Organize them into a hierarchical taxonomy with the following structure:

- Layer 1 (Perspective): High-level research angles
- Layer 2 (Category): Research categories
- Layer 3 (Leaf): Specific research questions or design goals

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
              "hci_terms": ["term1", "term2"]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Scale and coverage requirements:
- 2-3 perspectives (L1)
- 1-2 categories under each perspective (L2)
- 1-3 leaves per category (L3)
- Target total leaves: 5-10

Focus on goals that:
1. Require understanding from domain experts
2. Use HCI/UX concepts that might need explanation
3. Could be mapped to expert pain points
"""

EXTEND_EXPERT_TREE_PROMPT = """You are refining an expert taxonomy for ONE interview question.

Interview topic: {topic}
Expert background:
{expert_bg}

Question:
\"\"\"
{question_text}
\"\"\"

{dependency_context}

Current expert tree summary (label + description only):
{expert_tree_json}

All 8 simulated expert answers for this SAME original question:
{simulated_responses_json}

Task:
- Your goal is to return a refined expert tree that best covers the semantics in the 8 simulated answers.
- Keep the tree concise and avoid duplicates.

CRITICAL: HOW TO JUDGE COVERAGE (STEP-BY-STEP)

Do not look at the answers superficially. To determine if the current tree "covers" the 8 answers, you MUST perform the following mental mapping:

1) Decompose:
Break down EACH of the 8 simulated answers into atomic claims (core pain points, workflows, assumptions, constraints, tacit judgments).
One answer may contain multiple claims.

2) Map to Descriptions:
Try to map every atomic claim to the semantics of existing LEAF `description` fields.
- One simulated answer may be covered by MULTIPLE leaf nodes.
- Do NOT rely on `label` matching. Coverage is determined primarily by semantic fit with `description`.

3) Identify Residuals:
Any atomic claim not covered by ANY existing leaf description is a Residual.

Extension & Modification Strategy (Based on Residuals):

- If there are NO residuals:
  Return the original tree structure and content unchanged.

- If there ARE residuals, apply:
  [MODIFY]
  If a residual is only a nuance, jargon variant, or minor elaboration of an existing leaf:
  - Do NOT create a new node.
  - Integrate the nuance into that leaf's `description`.
  - Add jargon/variant phrasing into that leaf's `aliases`.

  [ADD]
  If a residual is a genuinely novel concept/pain point/workflow that cannot logically fit existing leaves:
  - Create a NEW leaf node.
  - You may attach it under an existing L2 category, OR create a new L1/L2 path if needed.

Additional constraints:
1) Preserve existing nodes whenever possible; modify minimally.
2) Avoid duplicate / near-duplicate leaves.
3) Keep total leaves compact (roughly 5-12).
4) Descriptions should remain concrete and conversational.

Return ONLY valid JSON with the SAME schema as PER_QUESTION_EXPERT_PROMPT:
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
              "description": "Concrete conversational description",
              "aliases": ["synonym or informal phrasing"],
              "is_intuition": true/false,
              "attributes": ["attr1", "attr2"]
            }}
          ]
        }}
      ]
    }}
  ]
}}
"""


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
                    aliases=pain_point.get("aliases", []),
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


def _prefix_tree_node_ids(tree: TaxonomyTree, question_id: str) -> TaxonomyTree:
    """Prefix every node id with the question_id to ensure uniqueness across question graphs.

    e.g.  exp_leaf_00_01_00  →  q_01__exp_leaf_00_01_00
    The shared_root is also prefixed so each question graph is fully independent.
    """
    prefix = f"{question_id}__"
    old_to_new: Dict[str, str] = {}
    for nid in list(tree.nodes.keys()):
        new_id = f"{prefix}{nid}"
        old_to_new[nid] = new_id

    new_nodes: Dict[str, DSAGNode] = {}
    for old_id, node in tree.nodes.items():
        new_id = old_to_new[old_id]
        node.id = new_id
        node.parent_id = old_to_new.get(node.parent_id, node.parent_id) if node.parent_id else None
        node.children_ids = [old_to_new.get(c, c) for c in node.children_ids]
        new_nodes[new_id] = node

    tree.nodes = new_nodes
    tree.root_id = old_to_new.get(tree.root_id, tree.root_id)
    return tree


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

Strategy 2 — Scenario (a sentence the researcher can say directly to the expert):
- Write ONE sentence (max 40 words) that the researcher can speak directly to the expert during the interview.
- The sentence should describe a concrete, specific situation the expert would recognize from their own practice,
  and naturally lead the expert to articulate the gap between their framing and the researcher's framing.
- Write in SECOND PERSON ("you"), as if the researcher is speaking to the expert face-to-face.
- It must sound like a natural interview prompt — NOT a third-person case study or academic description.
- Example tone: "When you're grading 200 reports in finals week, do you find yourself relying more on pattern recognition than the rubric?"

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
  "scenario": "One sentence the researcher can say directly to the expert to surface the gap. Max 40 words, second person."
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

Step 3: Generate hypothetical scenarios — exactly ONE per probe attribute output in Step 2.
  RULE: Each scenario alters only ONE variable at a time to test the boundaries of their intuition.
  Each scenario MUST target a DIFFERENT attribute from the probes list so that probe count == scenario count.
  Each scenario MUST specify which attribute it targets using the EXACT same attribute name as the corresponding probe.

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
    {{
      "scenario": "If [variable A] changed from X to Y, would your judgment about '{expert_label}' change?",
      "attribute": "the attribute this scenario targets"
    }},
    {{
      "scenario": "If [variable B] changed from P to Q, would your judgment about '{expert_label}' change?",
      "attribute": "the attribute this scenario targets"
    }}
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

    max_retries = 5
    base_delay = 5.0  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            # Rebuild LLM client on retry to rotate to a different API key
            if attempt > 1:
                llm = _build_llm(temperature=0.7)
            chain = prompt | llm
            response = chain.invoke(variables)
            content = response.content if hasattr(response, 'content') else str(response)
            data = _parse_json(content)
            if not data:
                data = {}
            if link.relation_type == RelationType.PROCESS_GAP.value:
                data["misalignment_reason"] = misalignment_reason
            return data
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "limit_requests" in err_str or "rate" in err_str.lower()
            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                print(f"[GraphFactory] 429 on {link.relation_type} (attempt {attempt}/{max_retries}), "
                      f"retrying in {delay:.1f}s …")
                time.sleep(delay)
            else:
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
    # Scale default workers with key-pool size: 6 concurrent calls per key
    pool_size = len(_KEY_POOL) if _KEY_POOL else 1
    default_workers = max(5, pool_size * 6)
    max_workers = max(1, int(os.getenv("DSAG_PAYLOAD_MAX_WORKERS", str(default_workers))))
    print(f"[GraphFactory] Generating assistance payloads for {len(links)} links concurrently "
          f"(workers={max_workers}, key_pool={pool_size})...")

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
    
    def generate_expert_tree(
        self,
        topic: str,
        expert_bg: str,
        questionnaire: str = "",
    ) -> TaxonomyTree:
        """Agent A: Generate expert taxonomy tree."""
        parsed = self._invoke_tree_prompt_with_retry(
            prompt_text=EXPERT_PERSONA_PROMPT,
            variables={
            "topic": topic,
            "expert_bg": expert_bg,
            "questionnaire": questionnaire or "",
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
        print("[GraphFactory] Agent C: Judging alignments (concept + parallel leaf batches)...")

        researcher_summary = self._tree_to_summary(researcher_tree)
        researcher_tree_json = json.dumps(researcher_summary, indent=2, ensure_ascii=False)

        # 1) Concept alignments (single call)
        llm = self._get_llm()
        concept_prompt = ChatPromptTemplate.from_messages([("user", ALIGNMENT_JUDGE_CONCEPT_PROMPT)])
        concept_chain = concept_prompt | llm
        expert_tree_json_full = json.dumps(
            self._tree_to_summary(expert_tree),
            indent=2,
            ensure_ascii=False,
        )
        concept_resp = concept_chain.invoke({
            "topic": topic,
            "expert_tree_json": expert_tree_json_full,
            "researcher_tree_json": researcher_tree_json,
        })
        concept_parsed = _parse_json(getattr(concept_resp, "content", str(concept_resp)))
        concept_alignments = _parse_alignments(concept_parsed).concept_alignments

        # 2) Leaf alignments: one LLM call per expert leaf, exhaustive pairwise
        expert_leaf_nodes = [n for n in expert_tree.nodes.values() if n.layer == Layer.LEAF.value]

        # Prepare researcher leaves summary once (shared across all parallel calls)
        researcher_leaf_summary = [
            {
                "id": n.id,
                "label": n.label,
                "description": n.description,
            }
            for n in researcher_tree.nodes.values()
            if n.layer == Layer.LEAF.value
        ]
        researcher_leaves_json = json.dumps(researcher_leaf_summary, indent=2, ensure_ascii=False)

        pool_size_align = len(_KEY_POOL) if _KEY_POOL else 1
        default_align_workers = max(5, pool_size_align * 4)
        max_workers = max(1, int(os.getenv("DSAG_ALIGN_MAX_WORKERS", str(default_align_workers))))

        def _run_single_leaf(exp_leaf: DSAGNode) -> List[NodeAlignment]:
            """Judge one expert leaf against all researcher leaves, return only gapped pairs."""
            expert_info: Dict[str, Any] = {
                "id": exp_leaf.id,
                "label": exp_leaf.label,
                "description": exp_leaf.description,
            }

            local_llm = _build_llm(temperature=0.3)
            leaf_prompt = ChatPromptTemplate.from_messages([("user", ALIGNMENT_JUDGE_LEAF_PROMPT)])
            leaf_chain = leaf_prompt | local_llm
            response = leaf_chain.invoke({
                "topic": topic,
                "expert_leaf_json": json.dumps(expert_info, indent=2, ensure_ascii=False),
                "researcher_leaves_json": researcher_leaves_json,
                "expert_node_id": exp_leaf.id,
            })
            parsed = _parse_json(getattr(response, "content", str(response)))
            aligns = _parse_alignments(parsed).leaf_alignments
            # Guard: only keep misaligned pairs (prompt should already guarantee this)
            return [a for a in aligns if not a.is_aligned]

        print(
            f"[GraphFactory] Agent C: Judging {len(expert_leaf_nodes)} expert leaves × "
            f"{len(researcher_leaf_summary)} researcher leaves pairwise "
            f"(parallel workers={max_workers})..."
        )
        leaf_alignments: List[NodeAlignment] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(_run_single_leaf, expert_leaf_nodes))
        for res in chunk_results:
            leaf_alignments.extend(res)

        # 3) Deduplicate: per (expert, researcher) pair keep highest-similarity entry
        dedup_map: Dict[Tuple[str, str], NodeAlignment] = {}
        for a in leaf_alignments:
            key = (a.expert_node_id, a.researcher_node_id)
            prev = dedup_map.get(key)
            if prev is None or a.semantic_similarity > prev.semantic_similarity:
                dedup_map[key] = a
        merged_leaf_alignments = list(dedup_map.values())

        alignments = TreeAlignments(
            leaf_alignments=merged_leaf_alignments,
            concept_alignments=concept_alignments,
        )
        print(
            f"[GraphFactory] Agent C found {len(alignments.leaf_alignments)} gap pairs "
            f"({len(expert_leaf_nodes)} expert leaves × {len(researcher_leaf_summary)} researcher leaves), "
            f"{len(alignments.concept_alignments)} concept alignments."
        )
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
        # Step 1: Agent C judges alignments (exhaustive pairwise, no fallback)
        alignments = self.generate_alignments(topic, expert_tree, researcher_tree)

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
            summary["nodes"].append(node_info)
        
        return summary
    
    # ============== Multi-Question Graph Generation ==============

    def split_questionnaire(self, questionnaire: str) -> List[Dict[str, str]]:
        """Split questionnaire text into top-level questions using LLM.

        Returns a list of dicts: [{"id": "q_01", "text": "..."}, ...]
        """
        llm = _build_llm_dependency(temperature=0.1)
        prompt = ChatPromptTemplate.from_messages([("user", SPLIT_QUESTIONS_PROMPT)])
        chain = prompt | llm
        response = chain.invoke({"questionnaire_text": questionnaire[:6000]})
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        questions = parsed.get("questions", [])
        # Normalise ids
        result: List[Dict[str, str]] = []
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            qid = q.get("id", f"q_{i + 1:02d}")
            text = str(q.get("text", "")).strip()
            if text:
                result.append({"id": qid, "text": text})
        if not result:
            raise ValueError("Failed to split questionnaire into questions")
        print(f"[GraphFactory] Split questionnaire into {len(result)} top-level questions")
        return result

    def judge_dependencies(
        self, questions: List[Dict[str, str]]
    ) -> Dict[str, List[str]]:
        """Use LLM to judge dependencies between top-level questions.

        Returns a dict mapping question_id -> list of question_ids it depends on.
        E.g. {"q_03": ["q_01"]} means q_03 depends on q_01.
        """
        llm = _build_llm_dependency(temperature=0.1)
        prompt = ChatPromptTemplate.from_messages([("user", JUDGE_DEPENDENCIES_PROMPT)])
        chain = prompt | llm
        questions_json = json.dumps(questions, indent=2, ensure_ascii=False)
        response = chain.invoke({"questions_json": questions_json})
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)

        deps_raw = parsed.get("dependencies", [])
        # Build adjacency: to_id -> [from_id, ...]
        dep_map: Dict[str, List[str]] = {q["id"]: [] for q in questions}
        valid_ids = {q["id"] for q in questions}
        for d in deps_raw:
            if not isinstance(d, dict):
                continue
            from_id = d.get("from_id", "")
            to_id = d.get("to_id", "")
            if from_id in valid_ids and to_id in valid_ids and from_id != to_id:
                if from_id not in dep_map[to_id]:
                    dep_map[to_id].append(from_id)

        print(f"[GraphFactory] Dependencies: {dep_map}")
        return dep_map

    @staticmethod
    def _topological_sort(
        questions: List[Dict[str, str]],
        dep_map: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        """Return questions in dependency-safe order (Kahn's algorithm)."""
        id_to_q = {q["id"]: q for q in questions}
        in_degree: Dict[str, int] = {q["id"]: 0 for q in questions}
        children: Dict[str, List[str]] = {q["id"]: [] for q in questions}

        for qid, deps in dep_map.items():
            for dep_id in deps:
                children.setdefault(dep_id, []).append(qid)
                in_degree[qid] = in_degree.get(qid, 0) + 1

        queue = [qid for qid in in_degree if in_degree[qid] == 0]
        ordered: List[Dict[str, str]] = []
        while queue:
            qid = queue.pop(0)
            ordered.append(id_to_q[qid])
            for child in children.get(qid, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # If cycle detected, just append remaining in original order
        if len(ordered) < len(questions):
            seen = {q["id"] for q in ordered}
            for q in questions:
                if q["id"] not in seen:
                    ordered.append(q)
        return ordered

    def _build_dependency_context(
        self,
        question_id: str,
        dep_map: Dict[str, List[str]],
        built_graphs: Dict[str, DSAGGraph],
        questions_by_id: Dict[str, str],
    ) -> str:
        """Build a textual context block from dependency questions' expert trees."""
        dep_ids = dep_map.get(question_id, [])
        if not dep_ids:
            return ""
        parts: List[str] = []
        for dep_id in dep_ids:
            dep_graph = built_graphs.get(dep_id)
            dep_text = questions_by_id.get(dep_id, "")
            if dep_graph:
                tree_summary = json.dumps(
                    self._tree_to_summary(dep_graph.expert_tree),
                    indent=2, ensure_ascii=False,
                )
                parts.append(
                    f"--- Context from prior question ({dep_id}): \"{dep_text[:200]}\" ---\n"
                    f"Expert tree summary:\n{tree_summary}\n"
                )
        if not parts:
            return ""
        return (
            "The following context is from a PRIOR interview question that this question builds upon. "
            "Use it to maintain continuity and avoid redundancy, but do NOT simply copy its nodes.\n\n"
            + "\n".join(parts)
        )

    def _simulate_expert_responses(
        self,
        topic: str,
        expert_bg: str,
        researcher_bg: str,
        question_text: str,
        dependency_context: str = "",
    ) -> List[str]:
        """Simulate 8 independent expert responses in parallel using a fast lightweight model.

        Each of the 8 turns is dispatched concurrently (no history chaining) because the
        goal is purely *coverage sampling* — we want diverse responses, not a realistic
        sequential dialogue. This cuts simulation time from ~8 × T_llm down to ~1 × T_llm
        and lets the key pool be used in parallel (one key per worker).
        """
        rounds = int(os.getenv("DSAG_SIMULATE_ROUNDS", "8"))
        # Base history: include dependency context if provided; otherwise a neutral stub.
        base_history = dependency_context.strip() or "(no prior conversation)"
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", INTERVIEWEE_PERSONA_PROMPT_V1), ("human", INTERVIEWEE_USER_PROMPT_V1)]
        )

        def _one_turn(_seed: int) -> str:
            """Each worker gets its own LLM instance so round-robin assigns different keys."""
            try:
                llm = _build_llm_fast(temperature=0.7)
                result = (prompt_template | llm).invoke(
                    {
                        "topic": topic or "(topic not provided)",
                        "demographics": expert_bg or "(background not provided)",
                        "interviewer_demographics": researcher_bg or "(background not provided)",
                        "history": base_history,
                        "question": question_text,
                    }
                )
                return str(getattr(result, "content", "") or "").strip()
            except Exception as exc:
                print(f"[GraphFactory] _simulate_expert_responses turn {_seed} failed (non-fatal): {exc}")
                return ""

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=rounds, thread_name_prefix="sim"
            ) as pool:
                results = list(pool.map(_one_turn, range(rounds)))
            return [r for r in results if r]
        except Exception as e:
            print(f"[GraphFactory] _simulate_expert_responses failed (non-fatal): {e}")
            return []

    @staticmethod
    def _tokenize_for_overlap(text: str) -> set[str]:
        normalized = (text or "").lower()
        latin = re.findall(r"[a-z0-9_]+", normalized)
        cjk = re.findall(r"[\u4e00-\u9fff]+", normalized)
        return {t for t in (latin + cjk) if t}

    def _count_repeated_uncovered_themes(self, responses: List[str], threshold: float = 0.45) -> int:
        """Cluster uncovered responses by lexical overlap and return largest cluster size."""
        if not responses:
            return 0
        token_sets = [self._tokenize_for_overlap(r) for r in responses]
        max_cluster = 1
        for i, tok_i in enumerate(token_sets):
            if not tok_i:
                continue
            cluster = 1
            for j, tok_j in enumerate(token_sets):
                if i == j or not tok_j:
                    continue
                inter = len(tok_i & tok_j)
                union = len(tok_i | tok_j)
                score = inter / max(union, 1)
                if score >= threshold:
                    cluster += 1
            max_cluster = max(max_cluster, cluster)
        return max_cluster

    def _extend_expert_tree_with_uncovered(
        self,
        topic: str,
        expert_bg: str,
        question_text: str,
        dependency_context: str,
        tree: TaxonomyTree,
        simulated_responses: List[str],
    ) -> TaxonomyTree:
        """Use LLM to extend the expert tree when uncovered response themes are stable."""
        llm = _build_llm(temperature=0.3)
        prompt = ChatPromptTemplate.from_messages([("user", EXTEND_EXPERT_TREE_PROMPT)])
        chain = prompt | llm
        tree_json = json.dumps(self._tree_to_summary(tree), ensure_ascii=False, indent=2)
        simulated_json = json.dumps(simulated_responses[:8], ensure_ascii=False, indent=2)
        resp = chain.invoke(
            {
                "topic": topic,
                "expert_bg": expert_bg,
                "question_text": question_text,
                "dependency_context": dependency_context,
                "expert_tree_json": tree_json,
                "simulated_responses_json": simulated_json,
            }
        )
        parsed = _parse_json(getattr(resp, "content", str(resp)))
        if not parsed.get("perspectives"):
            return tree
        return _build_expert_tree(parsed, topic)

    def generate_per_question_expert_tree(
        self,
        topic: str,
        expert_bg: str,
        researcher_bg: str,
        question_text: str,
        question_id: str,
        dependency_context: str = "",
    ) -> TaxonomyTree:
        """Agent A: Generate expert tree scoped to a single questionnaire question.
        """
        # Step 1: Build initial expert tree from question scope only.
        parsed = self._invoke_tree_prompt_with_retry(
            prompt_text=PER_QUESTION_EXPERT_PROMPT,
            variables={
                "topic": topic,
                "expert_bg": expert_bg,
                "question_text": question_text,
                "dependency_context": dependency_context,
            },
            tree_name=f"Expert tree ({question_id})",
        )
        tree = _build_expert_tree(parsed, topic)

        # Step 2: Simulate 8 turns with the runtime interviewee prompt.
        print(f"[GraphFactory] Simulating expert responses for {question_id} …")
        simulated_responses = self._simulate_expert_responses(
            topic=topic,
            expert_bg=expert_bg,
            researcher_bg=researcher_bg,
            question_text=question_text,
            dependency_context=dependency_context,
        )
        if simulated_responses:
            print(f"[GraphFactory] Got {len(simulated_responses)} simulated responses for {question_id}")
        else:
            print(f"[GraphFactory] No simulated responses for {question_id}, proceeding without extension")

        # Step 3: Extend directly from the 8 sampled answers (no pre-gating).
        if simulated_responses:
            print(f"[GraphFactory] {question_id}: extending expert tree from sampled answers")
            tree = self._extend_expert_tree_with_uncovered(
                topic=topic,
                expert_bg=expert_bg,
                question_text=question_text,
                dependency_context=dependency_context,
                tree=tree,
                simulated_responses=simulated_responses,
            )
        # Prefix node ids with question_id to avoid collisions across question graphs
        tree = _prefix_tree_node_ids(tree, question_id)
        return tree

    def generate_per_question_researcher_tree(
        self,
        topic: str,
        researcher_bg: str,
        question_text: str,
        question_id: str,
        dependency_context: str = "",
        other_questions: Optional[List[Dict[str, str]]] = None,
    ) -> TaxonomyTree:
        """Agent B: Generate researcher tree scoped to a single questionnaire question.

        Args:
            other_questions: List of dicts with 'id' and 'text' for all OTHER questions
                             (not this one), used to prevent sub-question overlap.
        """
        # Build the "other questions" block to inject into the prompt
        if other_questions:
            lines = [
                f"  - [{q['id']}] {q['text'][:200]}"
                for q in other_questions
            ]
            other_questions_block = (
                "## OTHER QUESTIONS in this questionnaire (already covered by their own trees):\n"
                + "\n".join(lines)
                + "\n\nDo NOT generate sub-questions that overlap with any of the above."
            )
        else:
            other_questions_block = ""

        parsed = self._invoke_tree_prompt_with_retry(
            prompt_text=PER_QUESTION_RESEARCHER_PROMPT,
            variables={
                "topic": topic,
                "researcher_bg": researcher_bg,
                "question_text": question_text,
                "dependency_context": dependency_context,
                "other_questions_block": other_questions_block,
            },
            tree_name=f"Researcher tree ({question_id})",
        )
        tree = _build_researcher_tree(parsed, topic)
        tree = _prefix_tree_node_ids(tree, question_id)
        return tree

    # -------- topological-level grouping for same-level parallelism --------

    @staticmethod
    def _topological_levels(
        questions: List[Dict[str, str]],
        dep_map: Dict[str, List[str]],
    ) -> List[List[Dict[str, str]]]:
        """Group questions into topological levels.

        Level 0 = questions with no dependencies (can all run in parallel).
        Level 1 = questions whose dependencies are all in level 0, etc.
        """
        id_to_q = {q["id"]: q for q in questions}
        in_degree: Dict[str, int] = {q["id"]: 0 for q in questions}
        children: Dict[str, List[str]] = {q["id"]: [] for q in questions}

        for qid, deps in dep_map.items():
            for dep_id in deps:
                children.setdefault(dep_id, []).append(qid)
                in_degree[qid] = in_degree.get(qid, 0) + 1

        levels: List[List[Dict[str, str]]] = []
        queue = [qid for qid in in_degree if in_degree[qid] == 0]

        while queue:
            level = [id_to_q[qid] for qid in queue]
            levels.append(level)
            next_queue: List[str] = []
            for qid in queue:
                for child in children.get(qid, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue

        # Append any remaining (cycle fallback) at the end
        seen = {q["id"] for lvl in levels for q in lvl}
        leftover = [q for q in questions if q["id"] not in seen]
        if leftover:
            levels.append(leftover)

        return levels

    # -------- pipeline-parallel question graph generation --------

    def generate_question_graphs(
        self,
        topic: str,
        researcher_bg: str,
        expert_bg: str,
        questionnaire: str,
        generate_templates: bool = True,
    ) -> List[QuestionGraph]:
        """
        Pipeline-parallel generation of independent DSAG graphs for each
        top-level question in the questionnaire.

        Speed-up strategy
        -----------------
        1. **Tree-internal parallelism**: For each question, Expert and
           Researcher trees are generated concurrently (2 threads).
        2. **Early unlock**: As soon as a question's *nodes* are ready, a
           signal (``threading.Event``) is fired so that dependent questions
           can immediately start their own node generation — they do NOT
           have to wait for the predecessor's link generation to finish.
        3. **Background links**: Link generation (alignment judge + math
           algorithm) runs in background threads, fully
           overlapped with the next question's node generation.
        4. **Same-level parallelism**: Questions at the same topological
           level (no mutual dependencies) start their node generation
           concurrently, further reducing wall-clock time.

        Returns:
            List[QuestionGraph] — one per top-level question, in original
            questionnaire order.
        """
        print(f"[GraphFactory] === Multi-Question Pipeline Mode === topic: {topic}")
        t_start = time.time()

        # ----- Step 1: Split questionnaire -----
        questions = self.split_questionnaire(questionnaire)

        # ----- Step 2: Judge dependencies -----
        dep_map = self.judge_dependencies(questions)

        # ----- Step 3: Group into topological levels -----
        levels = self._topological_levels(questions, dep_map)
        questions_by_id = {q["id"]: q["text"] for q in questions}
        print(
            f"[GraphFactory] Topological levels: "
            + " → ".join(str([q["id"] for q in lvl]) for lvl in levels)
        )

        # ----- Bookkeeping -----
        node_ready_events: Dict[str, threading.Event] = {
            q["id"]: threading.Event() for q in questions
        }
        # Store (expert_tree, researcher_tree) once nodes are done
        node_results: Dict[str, Tuple[TaxonomyTree, TaxonomyTree]] = {}
        # Store partial DSAGGraph (links placeholder) to provide dep context
        built_graphs: Dict[str, DSAGGraph] = {}
        # Final results keyed by qid
        question_graph_map: Dict[str, QuestionGraph] = {}
        # Futures for background link jobs
        link_futures: Dict[str, concurrent.futures.Future] = {}
        # Lock for shared dicts
        _lock = threading.Lock()

        tree_concurrent = os.getenv("DSAG_TREE_CONCURRENT", "1").strip().lower() not in (
            "0", "false", "no",
        )

        # Shared executor for background link jobs (kept alive across all levels)
        link_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(4, len(questions)),
            thread_name_prefix="link",
        )

        # Global payload pool: ALL questions share one pool so their payload
        # tasks are dispatched together, maximising key-pool utilisation.
        pool_size = len(_KEY_POOL) if _KEY_POOL else 1
        payload_workers = max(5, pool_size * 6)
        payload_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=payload_workers,
            thread_name_prefix="payload",
        )
        # payload_futures_map[qid] = list of (link, future) for that question's tasks
        payload_futures_map: Dict[str, List[Tuple[GapLink, concurrent.futures.Future]]] = {}
        # store (links, alignments, exp_tree, res_tree) after phase-1 link build
        links_stage: Dict[str, Tuple[List[GapLink], "TreeAlignments", TaxonomyTree, TaxonomyTree]] = {}
        _stage_lock = threading.Lock()

        def _generate_nodes_for_question(q: Dict[str, str]) -> None:
            """Generate Expert + Researcher nodes for one question.

            Blocks until all dependency *nodes* are ready, then generates
            trees (in parallel if enabled), signals readiness, and submits
            link generation to the background pool.
            """
            qid = q["id"]
            qtext = q["text"]
            deps = dep_map.get(qid, [])

            # ① Wait for dependency *nodes* only (not their links)
            for dep_id in deps:
                node_ready_events[dep_id].wait()

            print(f"\n  [pipeline] {qid}: generating nodes …")
            t0 = time.time()

            # Build dependency context (uses expert trees already in built_graphs)
            dep_context = self._build_dependency_context(
                qid, dep_map, built_graphs, questions_by_id,
            )

            # Build "other questions" list for researcher tree overlap prevention
            other_qs = [
                {"id": q["id"], "text": q["text"]}
                for q in questions
                if q["id"] != qid
            ]

            # ② Generate Expert + Researcher trees (parallel inside)
            expert_tree: TaxonomyTree
            researcher_tree: TaxonomyTree

            if tree_concurrent:
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as tp:
                        fut_exp = tp.submit(
                            self.generate_per_question_expert_tree,
                            topic, expert_bg, qtext, qid, dep_context,
                        )
                        fut_res = tp.submit(
                            self.generate_per_question_researcher_tree,
                            topic, researcher_bg, qtext, qid, dep_context, other_qs,
                        )
                        expert_tree = fut_exp.result()
                        researcher_tree = fut_res.result()
                except Exception as exc:
                    print(f"  [pipeline] {qid}: concurrent tree gen failed ({exc}), serial fallback")
                    expert_tree = self.generate_per_question_expert_tree(
                        topic, expert_bg, researcher_bg, qtext, qid, dep_context,
                    )
                    researcher_tree = self.generate_per_question_researcher_tree(
                        topic, researcher_bg, qtext, qid, dep_context, other_qs,
                    )
            else:
                expert_tree = self.generate_per_question_expert_tree(
                    topic, expert_bg, researcher_bg, qtext, qid, dep_context,
                )
                researcher_tree = self.generate_per_question_researcher_tree(
                    topic, researcher_bg, qtext, qid, dep_context, other_qs,
                )

            print(
                f"  [pipeline] {qid}: nodes ready ({time.time() - t0:.1f}s) — "
                f"exp_leaves={len(expert_tree.get_leaves())}, "
                f"res_leaves={len(researcher_tree.get_leaves())}"
            )

            # ③ Store results & signal downstream
            with _lock:
                node_results[qid] = (expert_tree, researcher_tree)
                # Create a placeholder graph so downstream can read expert_tree for context
                placeholder_graph = DSAGGraph(
                    topic=topic,
                    researcher_bg=researcher_bg,
                    expert_bg=expert_bg,
                    expert_tree=expert_tree,
                    researcher_tree=researcher_tree,
                    links=[],
                    metadata={"question_id": qid},
                )
                built_graphs[qid] = placeholder_graph

            # Signal: nodes ready — downstream questions can start now
            node_ready_events[qid].set()

            # ④ Fire-and-forget: link generation (phase 1: alignments + math, no payload)
            def _build_links_phase1(
                _qid: str = qid,
                _exp: TaxonomyTree = expert_tree,
                _res: TaxonomyTree = researcher_tree,
            ) -> Tuple[str, List[GapLink], "TreeAlignments", TaxonomyTree, TaxonomyTree]:
                t1 = time.time()
                # Phase 1: alignments + math build only (no LLM payload calls)
                alignments = self.generate_alignments(topic, _exp, _res)
                links = build_links_from_alignments(
                    expert_tree=_exp,
                    researcher_tree=_res,
                    alignments=alignments,
                )
                print(f"  [pipeline] {_qid}: {len(links)} links built ({time.time() - t1:.1f}s) — submitting payloads to global pool")

                # Phase 2: submit each link's payload generation to the global pool
                reason_lookup = {
                    (a.expert_node_id, a.researcher_node_id): a.reason
                    for a in alignments.leaf_alignments
                }

                def _gen_payload(link: GapLink, reason: str, exp: TaxonomyTree, res: TaxonomyTree) -> GapLink:
                    local_llm = _build_llm(temperature=0.7)
                    payload = generate_assistance_payload_for_link(
                        link=link,
                        expert_tree=exp,
                        researcher_tree=res,
                        misalignment_reason=reason,
                        llm=local_llm,
                    )
                    link.assistance_payload = payload
                    return link

                pfuts: List[Tuple[GapLink, concurrent.futures.Future]] = []
                if generate_templates and links:
                    for lnk in links:
                        reason = reason_lookup.get((lnk.expert_leaf_id, lnk.researcher_leaf_id), "")
                        fut = payload_pool.submit(_gen_payload, lnk, reason, _exp, _res)
                        pfuts.append((lnk, fut))

                with _stage_lock:
                    payload_futures_map[_qid] = pfuts
                    links_stage[_qid] = (links, alignments, _exp, _res)

                return (_qid, links, alignments, _exp, _res)

            fut = link_pool.submit(_build_links_phase1)
            with _lock:
                link_futures[qid] = fut

        # ----- Step 4: Process levels -----
        # Questions within the same level have no mutual dependencies,
        # so we launch their node-generation concurrently.

        for level_idx, level in enumerate(levels):
            level_ids = [q["id"] for q in level]
            print(f"\n[GraphFactory] Level {level_idx}: {level_ids}")

            if len(level) == 1:
                # Single question in this level — run directly
                _generate_nodes_for_question(level[0])
            else:
                # Multiple independent questions — parallelize node generation
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(level),
                    thread_name_prefix=f"lvl{level_idx}",
                ) as level_pool:
                    futs = [
                        level_pool.submit(_generate_nodes_for_question, q)
                        for q in level
                    ]
                    # Wait for all node-gen in this level to finish
                    for fut in concurrent.futures.as_completed(futs):
                        exc = fut.exception()
                        if exc:
                            print(f"[GraphFactory] Level {level_idx} node-gen error: {exc}")
                            raise exc

        # ----- Step 5: Collect all link results -----
        print(f"\n[GraphFactory] Waiting for phase-1 link jobs (alignments + math) …")

        for qid, fut in link_futures.items():
            fut.result()  # blocks until phase-1 (alignments + math) done for this q

        # Phase-1 done for all questions — payload tasks are already submitted
        # to the global payload_pool and running concurrently across all questions.
        total_payloads = sum(len(pfuts) for pfuts in payload_futures_map.values())
        print(f"\n[GraphFactory] Waiting for {total_payloads} payload tasks across "
              f"{len(payload_futures_map)} questions (global pool workers={payload_workers}) …")

        for _qid, pfuts in payload_futures_map.items():
            t_pay = time.time()
            for lnk, pfut in pfuts:
                try:
                    pfut.result()  # link.assistance_payload already set in-place
                except Exception as _exc:
                    print(f"[GraphFactory] Payload error for {_qid}: {_exc}")
            links, alignments, _exp, _res = links_stage[_qid]

            print(f"  [pipeline] {_qid}: {len(links)} links ready ({time.time() - t_pay:.1f}s)")

            # Update the placeholder graph with real links + metadata
            graph = built_graphs[_qid]
            graph.links = links
            qtext = questions_by_id[_qid]
            graph.metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
                "question_id": _qid,
                "question_text": qtext[:200],
                "expert_leaves": len(graph.expert_tree.get_leaves()),
                "researcher_leaves": len(graph.researcher_tree.get_leaves()),
                "link_count": len(links),
                "alignment_stats": {
                    "leaf_alignments": len(alignments.leaf_alignments),
                    "concept_alignments": len(alignments.concept_alignments),
                    "misaligned_leaves": len(alignments.get_misaligned_leaf_pairs()),
                },
                "construction_method": "math-based",
                "depends_on": dep_map.get(_qid, []),
            }

            qg = QuestionGraph(
                question_id=_qid,
                question_text=qtext,
                graph=graph,
                depends_on=dep_map.get(_qid, []),
            )
            question_graph_map[_qid] = qg

        link_pool.shutdown(wait=False)
        payload_pool.shutdown(wait=False)

        # ----- Step 6: Return in original questionnaire order -----
        question_graph_list = [
            question_graph_map[q["id"]]
            for q in questions
            if q["id"] in question_graph_map
        ]

        elapsed = time.time() - t_start
        print(
            f"\n[GraphFactory] === Multi-Question Pipeline complete: "
            f"{len(question_graph_list)} graphs in {elapsed:.1f}s ==="
        )
        return question_graph_list

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
                    future_expert = executor.submit(
                        self.generate_expert_tree,
                        topic,
                        expert_bg,
                        questionnaire,
                    )
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
                expert_tree = self.generate_expert_tree(topic, expert_bg, questionnaire)
                researcher_tree = self.generate_researcher_tree(topic, researcher_bg, questionnaire)
        else:
            print("[GraphFactory] Agent A: Generating expert tree (serial mode)...")
            expert_tree = self.generate_expert_tree(topic, expert_bg, questionnaire)
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


# ============== Convenience Functions ==============

def create_dsag_graph(
    topic: str,
    researcher_bg: str,
    expert_bg: str,
    questionnaire: str = "",
) -> DSAGGraph:
    """
    Convenience function to create a DSAG graph (legacy single-graph mode).
    """
    factory = GraphFactory()
    return factory.generate_graph(topic, researcher_bg, expert_bg, questionnaire=questionnaire)


def create_question_graphs(
    topic: str,
    researcher_bg: str,
    expert_bg: str,
    questionnaire: str,
) -> List[QuestionGraph]:
    """
    Convenience function to create per-question DSAG graphs (new multi-question mode).

    Requires a non-empty questionnaire.
    """
    factory = GraphFactory()
    return factory.generate_question_graphs(
        topic, researcher_bg, expert_bg, questionnaire=questionnaire,
    )
