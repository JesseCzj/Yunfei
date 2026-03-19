"""
DSAG Runtime Engine

Provides real-time, type-specific analysis of interview turns using the DSAG graph.
Each of the 5 mismatch types (Lexical, Conceptual, Tacit, Scope, Process) produces
a different assistance structure following its own rules and strategies.

Key functions:
- Locate current positions in both trees
- Query relevant links
- Generate type-specific assistance
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .schema import (
    DSAGGraph,
    DSAGNode,
    GapLink,
    TaxonomyTree,
    Layer,
    RelationType,
    compute_lca_layer,
)


# ============== Search Result ==============

@dataclass
class SearchResult:
    """Result of a leaf-matching search (via LLM)."""
    node_id: str
    node: DSAGNode
    score: float
    snapped: bool = False
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


# ============== LLM for Template Filling ==============

def _build_light_llm() -> ChatOpenAI:
    """Build a lightweight LLM for template filling (fast, cheap)."""
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
            temperature=0.3,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        model = os.getenv("OPENAI_MODEL_RUNTIME", "qwen3-max-2026-01-23")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=0.3)
        return ChatOpenAI(api_key=api_key, model=model, temperature=0.3)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _build_match_llm() -> ChatOpenAI:
    """Build a low-latency LLM dedicated to leaf matching."""
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
            temperature=0.3,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        model = os.getenv("OPENAI_MODEL_MATCH", "qwen-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=0.3)
        return ChatOpenAI(api_key=api_key, model=model, temperature=0.3)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _build_polish_llm() -> ChatOpenAI:
    """Build a low-latency LLM for assistance polishing."""
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
            temperature=0.3,
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        model = os.getenv("OPENAI_MODEL_POLISH", "qwen-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=0.3)
        return ChatOpenAI(api_key=api_key, model=model, temperature=0.3)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with simple fallbacks."""
    try:
        return json.loads(text)
    except Exception:
        pass
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except Exception:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


# ============== Leaf Matching via LLM ==============

LEAF_MATCH_PROMPT = """You are a semantic matching engine for an interview analysis system.

Given a list of leaf nodes (each with an id, label, and description) and an utterance from a conversation, determine which leaf nodes best match the utterance semantically.

## Leaf nodes
{leaves_json}

## Utterance
"{utterance}"

## Task
Identify the top 3 leaf nodes that are most semantically relevant to the utterance.
For each match, provide:
- node_id: the exact id from the leaf list
- score: a similarity score from 0.0 (completely unrelated) to 1.0 (perfect match)

Scoring guidelines:
- 0.8-1.0: The utterance directly discusses this exact concept
- 0.6-0.8: The utterance is closely related to this concept
- 0.4-0.6: The utterance has some connection to this concept
- 0.0-0.4: The utterance is mostly unrelated to this concept

Return ONLY valid JSON:
{{
  "matches": [
    {{"node_id": "...", "score": 0.0}}
  ]
}}
"""


UNCERTAIN_INTERPRETATION_FOLLOWUPS_PROMPT = """You are writing disambiguation follow-up questions for a live interview.

The system detected possible expert concepts that might match the expert's latest answer, but cannot confidently determine which one (if either) the expert is discussing.
Your job: help the researcher figure this out with natural follow-up questions.

## Conversation context
- Researcher's latest question: "{researcher_question}"
- Expert's latest answer: "{expert_answer}"
- Recent conversation summary: "{context_summary}"

## Candidate concepts
{candidates_json}

## Task

Step 1 — Identify the CORE semantic distinction between the candidates: what specific aspect of the expert's answer could point to one versus the other?

Step 2 — For EACH candidate, generate exactly 1 follow-up question that:
1. Sounds natural and directly speakable by the researcher.
2. Targets the semantic DIFFERENCE between candidates — a good question should help reveal which concept the expert means, or whether the expert means neither.
3. Uses the expert's own wording when possible.
4. Does NOT mention system internals (scores, confidence, candidates, uncertainty).
5. Does NOT simply repeat the concept label as a yes/no question — instead probes the distinguishing aspect.
6. Does NOT invent new facts or technical claims.
7. Keeps to a single sentence.

## Output format
Return ONLY valid JSON with this exact top-level structure:
{{
  "semantic_distinction": "one sentence describing the key difference between the candidates",
  "candidate_followups": [
    {{
      "node_id": "candidate node id",
      "followup_questions": ["one disambiguation question"]
    }}
  ]
}}
"""


# ============== Result Data Classes ==============

@dataclass
class LocatedPosition:
    """Result of locating a position in the DSAG trees."""
    expert_results: List[SearchResult] = field(default_factory=list)
    researcher_results: List[SearchResult] = field(default_factory=list)
    best_expert_leaf_id: Optional[str] = None
    best_researcher_leaf_id: Optional[str] = None
    expert_confidence: float = 0.0
    researcher_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert": {
                "best_leaf_id": self.best_expert_leaf_id,
                "confidence": round(self.expert_confidence, 4),
                "top_results": [r.to_dict() for r in self.expert_results[:3]],
            },
            "researcher": {
                "best_leaf_id": self.best_researcher_leaf_id,
                "confidence": round(self.researcher_confidence, 4),
                "top_results": [r.to_dict() for r in self.researcher_results[:3]],
            },
        }


@dataclass
class DivergenceInfo:
    """Information about the divergence between expert and researcher."""
    lca_layer: str = "root"
    expert_branch: str = ""
    researcher_branch: str = ""
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lca_layer": self.lca_layer,
            "expert_branch": self.expert_branch,
            "researcher_branch": self.researcher_branch,
            "explanation": self.explanation,
        }


@dataclass
class Assistance:
    """
    Type-specific assistance payload.
    The structure of `payload` depends on `relation_type`.

    Payload shapes by type:
    - LexicalGap:    {"term_mapping": {"expert_term", "researcher_term", "explanation"}}
    - ConceptualGap: {"analogy": {"source_concept", "source_evidence": {"quote", "start_char", "end_char"},
                      "structural_mapping": {inputs, logic, outputs}, "explanation"},
                      "scenario": "single prose paragraph (situation → researcher lens → tension point, max 60 words)"}
    - TacitGap:      {"attributes": [...], "probes": [{attribute, question, choices}],
                      "hypothetical_scenarios": ["...", "..."],
                      "extracted_attributes": [...], "mentioned_attributes": [...]}  // extracted/mentioned attrs are displayed for transparency
    - ScopeGap:      {"validate_focus": "...", "pivot": {limitation, research_goal,
                      compelling_reason, coarse_scenario}}
    - ProcessGap:    Offline prevention payload with two sub-types:
                      factual_risk: {"sub_type", "vulnerable_assumption", "domain_correction",
                                     "safe_phrasing", "misalignment_reason"}
                      methodology_conflict: {"sub_type", "known_approaches", "researcher_assumed_approach",
                                             "open_process_question", "misalignment_reason"}
                      (factual risk and methodology conflict prevention payloads)
    """
    relation_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "payload": self.payload,
        }


@dataclass
class RuntimeAnalysis:
    """Complete analysis result for a turn."""
    located: LocatedPosition = field(default_factory=LocatedPosition)
    divergence: Optional[DivergenceInfo] = None
    assistance: Optional[Assistance] = None
    selected_link: Optional[GapLink] = None
    confidence_warning: str = ""
    uncertain_interpretation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "located": self.located.to_dict(),
            "confidence_warning": self.confidence_warning,
            "uncertain_interpretation": self.uncertain_interpretation,
        }
        if self.divergence:
            result["divergence"] = self.divergence.to_dict()
        if self.assistance and self.assistance.relation_type != RelationType.PROCESS_GAP.value:
            result["assistance"] = self.assistance.to_dict()
        if self.selected_link and self.selected_link.relation_type != RelationType.PROCESS_GAP.value:
            result["selected_link"] = {
                "expert_leaf_id": self.selected_link.expert_leaf_id,
                "researcher_leaf_id": self.selected_link.researcher_leaf_id,
                "relation_type": self.selected_link.relation_type,
                "weight": self.selected_link.weight,
            }
        return result


# ============== Runtime Engine ==============

class RuntimeEngine:
    """
    Runtime engine for DSAG-based interview navigation.
    Generates type-specific assistance based on the mismatch type of
    the selected GapLink.
    """

    # Confidence threshold for leaf matching (below this → "no mismatch" fallback)
    MATCH_THRESHOLD = 0.4
    TACIT_PRIORITY_MIN_CONFIDENCE = 0.6
    TACIT_HINT_KEYWORDS = (
        "intuition",
        "intuitive",
        "experience",
        "experienced",
        "gut feeling",
        "instinct",
        "tacit",
        "know it when i see it",
    )

    def __init__(self, graph: DSAGGraph):
        self.graph = graph
        self._llm = None
        self._match_llm = None

    @staticmethod
    def _has_non_empty_payload(payload: Optional[Dict[str, Any]]) -> bool:
        """Treat payload as usable only when it has at least one non-empty value."""
        if not isinstance(payload, dict) or not payload:
            return False
        return any(value not in (None, "", [], {}) for value in payload.values())

    @classmethod
    def _contains_tacit_cues(cls, text: str) -> bool:
        """Detect simple tacit-knowledge cues in the current QA turn."""
        normalized = (text or "").lower()
        return any(keyword in normalized for keyword in cls.TACIT_HINT_KEYWORDS)

    @staticmethod
    def _build_turn_match_text(researcher_question: str, expert_answer: str) -> str:
        """Combine both sides of the current QA turn for cue detection."""
        parts = [part.strip() for part in (researcher_question, expert_answer) if part and part.strip()]
        return "\n".join(parts)

    @staticmethod
    def _promote_search_result(
        results: List[SearchResult],
        preferred_node_id: str,
    ) -> List[SearchResult]:
        """Move the preferred candidate to the top while preserving the remaining order."""
        if not results:
            return results
        promoted = next((item for item in results if item.node_id == preferred_node_id), None)
        if promoted is None:
            return results
        return [promoted] + [item for item in results if item.node_id != preferred_node_id]

    def _select_tacitgap_priority_pair(
        self,
        expert_results: List[SearchResult],
        researcher_results: List[SearchResult],
    ) -> Optional[tuple[SearchResult, SearchResult]]:
        """Prefer a linked TacitGap pair when both sides are confidently matched."""
        expert_by_id = {
            item.node_id: item
            for item in expert_results
            if item.score >= self.TACIT_PRIORITY_MIN_CONFIDENCE
        }
        researcher_by_id = {
            item.node_id: item
            for item in researcher_results
            if item.score >= self.TACIT_PRIORITY_MIN_CONFIDENCE
        }
        if not expert_by_id or not researcher_by_id:
            return None

        best_pair: Optional[tuple[tuple[float, float], SearchResult, SearchResult]] = None
        for link in self.graph.links:
            if link.relation_type != RelationType.TACIT_GAP.value:
                continue
            expert_match = expert_by_id.get(link.expert_leaf_id)
            researcher_match = researcher_by_id.get(link.researcher_leaf_id)
            if expert_match is None or researcher_match is None:
                continue

            pair_score = (
                min(expert_match.score, researcher_match.score),
                (expert_match.score + researcher_match.score) / 2.0,
            )
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, expert_match, researcher_match)

        if best_pair is None:
            return None
        return best_pair[1], best_pair[2]

    def _build_minimal_payload(self, link: GapLink) -> Dict[str, Any]:
        """Build a deterministic fallback payload so a selected mismatch is always actionable."""
        exp_node = self.graph.expert_tree.get_node(link.expert_leaf_id)
        res_node = self.graph.researcher_tree.get_node(link.researcher_leaf_id)
        expert_label = exp_node.label if exp_node else (link.conflict.get("expert_branch", "") or "expert concept")
        researcher_label = res_node.label if res_node else (link.conflict.get("researcher_branch", "") or "research goal")
        relation = (link.relation_type or RelationType.CONCEPTUAL_GAP.value).strip()

        if relation == RelationType.LEXICAL_GAP.value:
            return {
                "term_mapping": {
                    "expert_term": expert_label,
                    "researcher_term": researcher_label,
                    "explanation": "These terms likely refer to related ideas but use different wording.",
                }
            }
        if relation == RelationType.TACIT_GAP.value:
            attrs = list(getattr(exp_node, "attributes", []) or [])
            return {
                "attributes": attrs[:3] if attrs else [expert_label],
                "probes": [],
                "hypothetical_scenarios": [],
                "extracted_attributes": [],
                "mentioned_attributes": [],
            }
        if relation == RelationType.SCOPE_GAP.value:
            return {
                "pivot": {
                    "condensed_explanation": (
                        f"Expert answer focuses on '{expert_label}', while your question targets "
                        f"'{researcher_label}'. A bridging question can connect these scopes."
                    )
                }
            }
        if relation == RelationType.PROCESS_GAP.value:
            return {
                "sub_type": "factual_risk",
                "vulnerable_assumption": f"Assuming '{researcher_label}' directly maps to expert workflow.",
                "domain_correction": f"The expert framing appears closer to '{expert_label}'.",
                "safe_phrasing": "Could you walk me through how this works in your actual grading process?",
                "misalignment_reason": "Runtime fallback payload generated due missing offline payload.",
            }
        return {
            "analogy": {
                "source_concept": expert_label,
                "target_concept": researcher_label,
                "structural_mapping": {"inputs": "", "logic": "", "outputs": ""},
                "explanation": "Use the expert concept to explain the research target in the same task context.",
            }
        }

    def _ensure_actionable_link(self, link: GapLink) -> GapLink:
        """
        Ensure selected links are always usable at runtime:
        relation_type is set and assistance_payload is non-empty.
        """
        if not link.relation_type:
            link.relation_type = RelationType.CONCEPTUAL_GAP.value
        if not self._has_non_empty_payload(link.assistance_payload):
            link.assistance_payload = self._build_minimal_payload(link)
        return link

    def _build_runtime_fallback_link(self, located: LocatedPosition) -> Optional[GapLink]:
        """Create a minimal mismatch link when graph links are missing for a matched expert leaf."""
        expert_leaf_id = located.best_expert_leaf_id
        if not expert_leaf_id:
            return None

        researcher_leaf_id = located.best_researcher_leaf_id
        if not researcher_leaf_id and located.researcher_results:
            researcher_leaf_id = located.researcher_results[0].node_id
        if not researcher_leaf_id:
            researcher_leaves = self.graph.researcher_tree.get_leaves()
            if researcher_leaves:
                researcher_leaf_id = researcher_leaves[0].id
        if not researcher_leaf_id:
            return None

        expert_path = self.graph.expert_tree.get_aligned_path(expert_leaf_id)
        researcher_path = self.graph.researcher_tree.get_aligned_path(researcher_leaf_id)
        lca_layer = compute_lca_layer(expert_path, researcher_path)
        exp_node = self.graph.expert_tree.get_node(expert_leaf_id)
        res_node = self.graph.researcher_tree.get_node(researcher_leaf_id)
        relation = RelationType.TACIT_GAP.value if (exp_node and exp_node.attributes) else RelationType.CONCEPTUAL_GAP.value

        link = GapLink(
            expert_leaf_id=expert_leaf_id,
            researcher_leaf_id=researcher_leaf_id,
            aligned_path_expert=expert_path,
            aligned_path_researcher=researcher_path,
            lca_layer=lca_layer,
            conflict={
                "expert_branch": (exp_node.label if exp_node else (expert_path[-1] if expert_path else "")),
                "researcher_branch": (res_node.label if res_node else (researcher_path[-1] if researcher_path else "")),
            },
            relation_type=relation,
            assistance_payload={},
            weight=0.01,
        )
        return self._ensure_actionable_link(link)

    def _get_llm(self) -> ChatOpenAI:
        """Lazy load LLM for template filling."""
        if self._llm is None:
            self._llm = _build_light_llm()
        return self._llm

    # ---- Type-specific polishing instructions ----
    # Each entry specifies which fields are FROZEN (keep verbatim) vs POLISHABLE
    # (make more natural, weave in expert's wording).

    _POLISH_RULES_BY_TYPE = {
        RelationType.LEXICAL_GAP.value: """## LexicalGap polishing rules
CONTEXTUALIZABLE fields (you MAY rephrase to match the expert's actual wording,
but must preserve the core meaning — e.g. if the expert said "latency" instead
of "response time", use "latency"):
  - term_mapping.expert_term
  - term_mapping.researcher_term

POLISHABLE fields:
  - term_mapping.explanation — make it a crisp, natural-sounding sentence
    that a researcher could say aloud. Weave in the expert's phrasing when
    possible (e.g. "When the expert says '…', it corresponds to '…'").
    Reference the actual Q&A exchange so the explanation feels specific to
    this conversation moment.

No follow-up questions expected for this type.""",

        RelationType.CONCEPTUAL_GAP.value: """## ConceptualGap polishing rules
CONTEXTUALIZABLE fields (you MAY rephrase to match the expert's actual wording
and the conversation context, but must preserve the core semantic meaning):
  - scenario — this is a sentence the researcher will SAY DIRECTLY to the expert
    during the interview. Rewrite it so that:
    (a) It uses SECOND PERSON ("you") addressing the expert face-to-face.
    (b) It references a concrete situation from the expert's practice that just
        came up in the conversation (use the expert's own words/examples).
    (c) It naturally leads the expert to articulate the gap between their framing
        and the researcher's framing.
    (d) It sounds like a natural interview question or prompt — NOT a third-person
        case study, NOT an academic description, NOT a narration about "the researcher".
    (e) Max 1 sentence, <= 40 words.
    Example tone: "When you mentioned [expert's own phrase], does that pattern
    hold even when [edge case from researcher's model]?"

STRUCTURAL fields (keep the structure, but you may lightly rephrase for clarity):
  - analogy.structural_mapping.inputs
  - analogy.structural_mapping.logic
  - analogy.structural_mapping.outputs

POLISHABLE fields:
  - analogy.explanation — make it a fluent sentence the researcher could speak
    naturally. Reference the expert's actual vocabulary from their answer.
    Connect it to what was just discussed in the Q&A exchange.

SOURCE GROUNDING (STRICT):
  - analogy.source_concept MUST be an exact substring copied from the expert's latest answer.
  - You MUST also output analogy.source_evidence with:
      * quote: exact same substring as source_concept
      * start_char: start index in expert answer (0-based)
      * end_char: end index (exclusive)
  - If you cannot find an exact quote substring from the expert answer, return:
      {"payload": {}}
    (do NOT output a conceptual gap payload without exact quote grounding).

IMPORTANT: Adapt the entire assistance so it feels written specifically for this
conversation moment, not as a generic pre-written template. Use the expert's own
words and examples wherever possible.

The analogy and scenario are PARALLEL strategies (no required order).
Do not merge them or remove either one.

ADDITIONAL PRIORITY:
  - Make both the analogy and the scenario maximally useful for resolving the
    CURRENT framing ambiguity between the researcher and the expert.
  - Prefer wording that helps the user see what the expert likely means in
    practice right now, not just what the taxonomy says abstractly.""",

        RelationType.TACIT_GAP.value: """## TacitGap polishing rules
The offline payload contains a source inventory of attributes, probes, and hypothetical scenarios.
Your PRIMARY task is to read the expert's latest answer and identify which source
attributes the expert has ALREADY expressed, even if the wording is indirect.

### Step A — Extract already-mentioned attributes
- Look ONLY at the source inventory in payload.attributes.
- Determine which of those source attributes are already present in the expert's
  latest answer, either explicitly or implicitly.
- Return those attributes in "extracted_attributes" and "mentioned_attributes".
- Also return "extracted_quotes": a list of short exact substrings copied
  verbatim from the expert's latest answer that best justify those already-mentioned attributes.

IMPORTANT:
- Extract only the 1 to 3 STRONGEST, most independent already-mentioned points.
- Do NOT split one underlying idea into multiple tiny quote fragments.
- Prefer one quote per major point the expert clearly expressed.

CRITICAL:
- Every item in "extracted_attributes" MUST be copied EXACTLY from the original
  payload.attributes strings.
- Do NOT invent new attribute names.
- Do NOT paraphrase attribute names inside extracted_attributes.
- Do NOT return concepts that are not in the source inventory.
- Every item in "extracted_quotes" must be an EXACT substring of the expert's latest answer.
- Keep each quote short, ideally 1 to 6 words.
- If multiple attributes map to the same quote, you may return the quote once.

### Step B — Contextual polishing for what remains unclear
You may polish the wording of probes[].question and
hypothetical_scenarios[].scenario so they feel grounded in the current exchange.
When polishing, optimize for helping the user resolve the CURRENT ambiguity.
That means the wording should make it easier to see:
- what the expert likely means right now,
- what is still underspecified,
- and what concrete contrast would clarify the issue fastest.

FROZEN fields (copy verbatim — do NOT alter):
  - attributes entries
  - probes[].attribute
  - probes[].choices
  - hypothetical_scenarios[].attribute

POLISHABLE fields:
  - probes[].question
  - hypothetical_scenarios[].scenario

### Output requirements
- Keep the same top-level keys: attributes, probes, hypothetical_scenarios,
  extracted_attributes, mentioned_attributes.
- You may additionally include "extracted_quotes" for UI display only.
- The runtime will decide which remaining attributes to surface as "Still to Explore".
- Your most important output is accurate extraction of already-mentioned attributes.
- If none of the source attributes seem already mentioned, return empty lists for
  extracted_attributes, mentioned_attributes, and extracted_quotes.
- IMPORTANT: The entire output should feel grounded in the current interview
  moment, not like a generic template.""",

        RelationType.SCOPE_GAP.value: """## ScopeGap polishing rules
FROZEN fields (copy verbatim — do NOT alter):
  - pivot.limitation (the factual limitation statement)
  - pivot.research_goal (the factual research goal)

POLISHABLE fields:
  - validate_focus — this is the empathetic acknowledgment of the expert's
    practical concern. Rewrite it to echo the expert's OWN words and tone.
    It should feel genuinely validating, not formulaic.
  - pivot.compelling_reason — make it persuasive and concrete. Show what the
    expert GAINS. Do NOT just say "it's a necessary step."
  - pivot.coarse_scenario — a 2-3 sentence "day after adoption" narrative.
    Keep it at story level, NO technical details, NO budget arguments.
    Make it vivid and relatable to the expert's world.

NEW REQUIRED field:
  - pivot.condensed_explanation — produce a concise 2-3 sentence summary that
    combines the essence of limitation, research_goal, and compelling_reason
    into a single flowing paragraph. This is what the user sees first (the
    detailed fields are hidden). Make it natural and persuasive, grounded in
    the expert's actual words from the conversation.

ADDITIONAL PRIORITY:
  - Make the explanation maximally helpful for showing the user why the
    expert's current concern and the researcher's current goal are misaligned
    in this moment.
  - Prefer wording that helps the user understand how to pivot without losing
    the expert.
  - Make pivot.condensed_explanation directly explain:
      1. what the expert is optimizing for right now,
      2. what the researcher is trying to obtain,
      3. why they are misaligned in this exact exchange,
      4. what conversational bridge would help next.
  - Make pivot.coarse_scenario feel like the user's next move in THIS interview,
    not a generic adoption story.""",

        RelationType.PROCESS_GAP.value: """## ProcessGap polishing rules
ProcessGap has two sub-types. Check the "sub_type" field to determine which.

### Sub-type A: factual_risk
FROZEN fields (copy verbatim — do NOT alter):
  - sub_type
  - vulnerable_assumption (the factual claim at risk)
  - domain_correction (what experts actually practice)

POLISHABLE fields:
  - safe_phrasing — rewrite the suggested question so it sounds natural and
    uses vocabulary the expert would recognize. Reference the expert's recent
    answer if it provides context. The question MUST remain open-ended and
    MUST NOT assert the vulnerable assumption.

### Sub-type B: methodology_conflict
FROZEN fields (copy verbatim — do NOT alter):
  - sub_type
  - known_approaches (the list of domain paradigms)
  - researcher_assumed_approach

POLISHABLE fields:
  - open_process_question — rewrite the suggested question so it sounds
    natural and conversational. It MUST remain open-ended and MUST NOT
    presuppose any specific methodology. Reference the expert's recent
    answer if relevant.""",
    }

    def _polish_assistance(
        self,
        assistance: Assistance,
        researcher_question: str,
        expert_answer: str,
        context_summary: str,
    ) -> Assistance:
        """
        Polish assistance text with a low-latency LLM.

        Each of the 5 mismatch types gets type-specific polishing instructions
        that specify which fields are FROZEN (structural / factual data) and
        which are POLISHABLE (prose that should sound natural and reference
        the expert's actual wording).
        """
        if not assistance or not assistance.payload:
            return assistance

        relation = assistance.relation_type
        type_rules = self._POLISH_RULES_BY_TYPE.get(relation, "")
        if not type_rules:
            # Unknown type — return unpolished to avoid corruption
            return assistance

        tacit_source_payload = dict(assistance.payload) if assistance.relation_type == RelationType.TACIT_GAP.value else {}
        llm = _build_polish_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("user", """You are polishing interview assistance so it sounds natural, directly speakable by the researcher, and deeply grounded in the actual conversation happening right now.

## Mismatch type: {relation_type}

## Current conversation exchange
- Researcher's latest question: "{researcher_question}"
- Expert's latest answer: "{expert_answer}"

## Recent conversation context (last 3 turns)
{context_summary}

## Assistance to polish
{assistance_json}

## Primary goal
Your job is NOT just to rewrite the assistance more fluently.
Your primary goal is to make the assistance maximally useful for resolving the
CURRENT ambiguity between the researcher and the expert.

This means:
- identify what is most likely being misunderstood, talked past, or left
  implicit in this exact moment
- express the assistance in the form that would be easiest for the user to
  understand and act on immediately
- prefer the clearest and most ambiguity-resolving presentation of the same
  underlying payload
- make the assistance help the researcher recover alignment with the expert's
  perspective

## General rules (apply to ALL types)
1. Keep the EXACT same JSON keys and nesting structure.
2. Do NOT add, remove, rename, or reorder the top-level strategy blocks used by the UI.
3. Make polishable text natural, concise, directly speakable, and easy for a non-expert user of the UI to understand.
4. Reference the expert's OWN vocabulary and phrasing where possible.
5. Do NOT invent new facts or concepts.
6. Preserve the underlying semantic intent of the original payload.
7. Questions MUST remain questions.
8. Output ONLY valid JSON with top-level key "payload" (same structure as the input).
9. CRITICAL: Adapt the assistance so it directly addresses the ambiguity in the current Q&A exchange above. The output should feel written for THIS exact moment, not as a generic template.
10. If several phrasings are possible, choose the one that best helps the user understand what the expert likely means, why the mismatch happened, and what to ask or notice next.
11. Language constraint: ALL generated text values in payload must be English only.
    Do NOT output Chinese or mixed-language text.

{type_specific_rules}
""")
        ])
        chain = prompt | llm
        response = chain.invoke({
            "relation_type": relation,
            "expert_answer": expert_answer[:800],
            "researcher_question": researcher_question[:500],
            "context_summary": context_summary[:1200],
            "assistance_json": json.dumps(
                {"payload": assistance.payload},
                ensure_ascii=False,
            ),
            "type_specific_rules": type_rules,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json_from_text(content)
        force_context_types = {
            RelationType.CONCEPTUAL_GAP.value,
            RelationType.SCOPE_GAP.value,
            RelationType.TACIT_GAP.value,
        }
        if not parsed:
            if assistance.relation_type in force_context_types:
                fallback = Assistance(relation_type=assistance.relation_type)
                fallback_payload = self._must_contextualize_payload(
                    relation=assistance.relation_type,
                    payload=dict(assistance.payload),
                    researcher_question=researcher_question,
                    expert_answer=expert_answer,
                )
                if assistance.relation_type == RelationType.TACIT_GAP.value:
                    fallback_payload["_source_attributes"] = list(tacit_source_payload.get("attributes", []) or [])
                    fallback_payload["_source_probes"] = list(tacit_source_payload.get("probes", []) or [])
                    fallback_payload["_source_hypothetical_scenarios"] = list(
                        tacit_source_payload.get("hypothetical_scenarios", []) or []
                    )
                    fallback_payload["_expert_answer_hint"] = expert_answer
                    fallback_payload = self._postprocess_tacit_payload(fallback_payload)
                fallback.payload = fallback_payload
                return fallback
            if assistance.relation_type == RelationType.TACIT_GAP.value:
                fallback = Assistance(relation_type=assistance.relation_type)
                fallback_payload = dict(assistance.payload)
                fallback_payload["_source_attributes"] = list(tacit_source_payload.get("attributes", []) or [])
                fallback_payload["_source_probes"] = list(tacit_source_payload.get("probes", []) or [])
                fallback_payload["_source_hypothetical_scenarios"] = list(
                    tacit_source_payload.get("hypothetical_scenarios", []) or []
                )
                fallback_payload["_expert_answer_hint"] = expert_answer
                fallback.payload = self._postprocess_tacit_payload(fallback_payload)
                return fallback
            return assistance

        polished = Assistance(relation_type=assistance.relation_type)
        polished.payload = parsed.get("payload", assistance.payload)
        if polished.relation_type in force_context_types:
            polished.payload = self._must_contextualize_payload(
                relation=polished.relation_type,
                payload=polished.payload,
                researcher_question=researcher_question,
                expert_answer=expert_answer,
            )
        if polished.relation_type == RelationType.TACIT_GAP.value:
            polished.payload["_source_attributes"] = list(tacit_source_payload.get("attributes", []) or [])
            polished.payload["_source_probes"] = list(tacit_source_payload.get("probes", []) or [])
            polished.payload["_source_hypothetical_scenarios"] = list(
                tacit_source_payload.get("hypothetical_scenarios", []) or []
            )
            polished.payload["_expert_answer_hint"] = expert_answer
            polished.payload = self._postprocess_tacit_payload(polished.payload)
            polished.payload = self._contextualize_tacit_remaining_payload(
                polished.payload,
                researcher_question=researcher_question,
                expert_answer=expert_answer,
            )
        if polished.relation_type == RelationType.CONCEPTUAL_GAP.value:
            grounded_payload = self._enforce_conceptual_source_quote(
                payload=polished.payload,
                expert_answer=expert_answer,
            )
            if not grounded_payload:
                return Assistance()
            polished.payload = grounded_payload
        return polished

    @staticmethod
    def _find_exact_quote_span(text: str, quote: str) -> Optional[tuple]:
        text_s = str(text or "")
        quote_s = str(quote or "")
        if not text_s or not quote_s:
            return None
        start = text_s.find(quote_s)
        if start < 0:
            return None
        return (start, start + len(quote_s))

    def _enforce_conceptual_source_quote(
        self,
        payload: Dict[str, Any],
        expert_answer: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Enforce conceptual analogy source grounding:
        - source_concept must be an exact substring of expert_answer
        - source_evidence.quote must exist and match the same exact substring
        If grounding fails, return None to drop conceptual assistance.
        """
        out = dict(payload or {})
        analogy = dict(out.get("analogy", {}) or {})
        if not analogy:
            return None

        source_evidence = dict(analogy.get("source_evidence", {}) or {})
        quote = str(source_evidence.get("quote", "")).strip()
        source_concept = str(analogy.get("source_concept", "")).strip()

        # Prefer explicit source_evidence.quote; fallback to source_concept.
        candidate = quote or source_concept
        span = self._find_exact_quote_span(expert_answer, candidate)
        if not span:
            return None

        start_char, end_char = span
        exact_quote = str(expert_answer or "")[start_char:end_char]
        analogy["source_concept"] = exact_quote
        analogy["source_evidence"] = {
            "quote": exact_quote,
            "start_char": start_char,
            "end_char": end_char,
        }
        out["analogy"] = analogy
        return out

    def _must_contextualize_payload(
        self,
        relation: str,
        payload: Dict[str, Any],
        researcher_question: str,
        expert_answer: str,
    ) -> Dict[str, Any]:
        """
        Deterministic context grounding fallback:
        ensures key fields explicitly reference the current turn.
        """
        out = dict(payload or {})
        rq = str(researcher_question or "").strip()
        ea = str(expert_answer or "").strip()
        rq_short = rq[:160] if rq else "the current researcher question"
        ea_short = ea[:220] if ea else "the expert's latest answer"

        if relation == RelationType.CONCEPTUAL_GAP.value:
            analogy = dict(out.get("analogy", {}))
            if not analogy.get("source_concept"):
                analogy["source_concept"] = "expert-side concept in this answer"
            if not analogy.get("target_concept"):
                analogy["target_concept"] = "researcher-side target in this question"
            explanation = str(analogy.get("explanation", "")).strip()
            if not explanation or "current" in explanation.lower() or "template" in explanation.lower():
                analogy["explanation"] = (
                    f'When you asked "{rq_short}", the expert emphasized "{ea_short}". '
                    "Bridge from the expert's framing first, then map it to your research target concept."
                )
            out["analogy"] = analogy
            if not str(out.get("scenario", "")).strip():
                out["scenario"] = (
                    f'For example, if we keep the expert framing "{analogy.get("source_concept", ea_short)}" '
                    f'while your question targets "{analogy.get("target_concept", rq_short)}", '
                    "we can test where the boundary conditions diverge in a concrete case."
                )
            return out

        if relation == RelationType.SCOPE_GAP.value:
            pivot = dict(out.get("pivot", {}))
            condensed = str(pivot.get("condensed_explanation", "")).strip()
            if not condensed:
                pivot["condensed_explanation"] = (
                    f'Your current question focuses on "{rq_short}", while this answer stresses "{ea_short}". '
                    "A better bridge is to acknowledge the expert's immediate concern first, then pivot to your target scope in a way that still sounds useful to them."
                )
            coarse = str(pivot.get("coarse_scenario", "")).strip()
            if not coarse:
                pivot["coarse_scenario"] = (
                    f'In the next turn, first reflect back the expert concern in "{ea_short}", '
                    f'then connect it to what you were trying to learn in "{rq_short}". '
                    "That bridge should make the pivot feel like a continuation of the expert's priorities, not a topic change."
                )
            out["pivot"] = pivot
            return out

        if relation == RelationType.TACIT_GAP.value:
            out.setdefault("attributes", [])
            out.setdefault("probes", [])
            out.setdefault("hypothetical_scenarios", [])
            out.setdefault("extracted_attributes", [])
            out.setdefault("mentioned_attributes", [])
            out.setdefault("extracted_quotes", [])

            scenarios = out.get("hypothetical_scenarios", [])
            if isinstance(scenarios, list) and not scenarios and out["attributes"]:
                first_attr = str(out["attributes"][0]).strip() or "key judgment factor"
                out["hypothetical_scenarios"] = [{
                    "scenario": (
                        f'Given the expert just highlighted "{ea_short}", if "{first_attr}" shifts under the same task, '
                        "which decision step would you adjust first?"
                    ),
                    "attribute": first_attr,
                }]
            return out

        return out

    @staticmethod
    def _normalize_attr_key(value: str) -> str:
        """Normalize attribute labels for best-effort matching."""
        text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
        return " ".join(tokens)

    def _map_attr_to_inventory(self, raw_attr: str, inventory: List[str]) -> str:
        """Map a potentially rephrased attribute label back to the source inventory."""
        raw_key = self._normalize_attr_key(raw_attr)
        if not raw_key:
            return ""

        best_attr = ""
        best_score = 0.0
        raw_tokens = set(raw_key.split())
        for attr in inventory:
            attr_key = self._normalize_attr_key(attr)
            if not attr_key:
                continue
            if raw_key == attr_key:
                return attr
            if raw_key in attr_key or attr_key in raw_key:
                score = 0.95
            else:
                attr_tokens = set(attr_key.split())
                if not raw_tokens or not attr_tokens:
                    score = 0.0
                else:
                    score = len(raw_tokens & attr_tokens) / max(len(raw_tokens | attr_tokens), 1)
            if score > best_score:
                best_score = score
                best_attr = attr

        return best_attr if best_score >= 0.34 else ""

    def _attr_similarity(self, left: str, right: str) -> float:
        """Cheap lexical similarity for selecting diverse TacitGap attributes."""
        left_tokens = set(self._normalize_attr_key(left).split())
        right_tokens = set(self._normalize_attr_key(right).split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)

    def _select_tacit_remaining_attrs(
        self,
        remaining_pool: List[str],
        extracted: List[str],
        limit: int = 2,
    ) -> List[str]:
        """Select up to `limit` remaining attributes that are lexically distinct."""
        if not remaining_pool or limit <= 0:
            return []

        selected: List[str] = []
        anchors = list(extracted)
        pool = list(remaining_pool)
        while pool and len(selected) < limit:
            best_attr = ""
            best_score = float("inf")
            for candidate in pool:
                comparison_set = anchors + selected
                if comparison_set:
                    score = max(self._attr_similarity(candidate, other) for other in comparison_set)
                else:
                    score = 0.0
                if score < best_score:
                    best_score = score
                    best_attr = candidate
            if not best_attr:
                best_attr = pool[0]
            selected.append(best_attr)
            pool = [attr for attr in pool if attr != best_attr]
        return selected

    @staticmethod
    def _humanize_attr_label(attr: str) -> str:
        text = str(attr or "").strip().replace("_", " ").replace("-", " ")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_probe_choices(raw_choices: Any) -> List[str]:
        """Normalize probe choices into a deduplicated, non-empty string list."""
        if not isinstance(raw_choices, list):
            return []
        out: List[str] = []
        for item in raw_choices:
            choice = str(item).strip()
            if choice and choice not in out:
                out.append(choice)
        return out

    @staticmethod
    def _clean_inferred_choice(text: str) -> str:
        """Trim punctuation and leading connectors from inferred choice labels."""
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^[\s:;,\-]+", "", cleaned)
        cleaned = re.sub(r"[\s\?\.\!;,:]+$", "", cleaned)
        cleaned = re.sub(r"^(?:more about|mainly about|mostly about|related to)\s+", "", cleaned, flags=re.I)
        cleaned = re.sub(r"^(?:是|更像是|主要是)\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _infer_choices_from_probe_question(self, question: str) -> List[str]:
        """Infer 2-4 multiple-choice options from common probe phrasings."""
        text = str(question or "").strip()
        if not text:
            return []

        patterns = [
            r"(?:which matters more|which is more important|is the unresolved issue mainly about|is your concern related to)\s+(.+?)\s*,\s*(.+?)\s*,\s*or\s+(.+?)(?:\?|$)",
            r"(?:which matters more|which is more important|is the unresolved issue mainly about|is your concern related to)\s+(.+?)\s+or\s+(.+?)(?:\?|$)",
            r"(?:主要是|更像是|是因为)(.+?)[、,，](.+?)[、,，]还是(.+?)(?:[？?]|$)",
            r"(?:主要是|更像是|是因为)(.+?)还是(.+?)(?:[？?]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if not match:
                continue
            choices = [self._clean_inferred_choice(group) for group in match.groups()]
            choices = [choice for choice in choices if choice]
            if len(choices) >= 2:
                return list(dict.fromkeys(choices))[:4]
        return []

    def _fallback_probe_choices(self, attr: str) -> List[str]:
        """Provide a stable fallback set when a probe is missing explicit choices."""
        attr_label = self._humanize_attr_label(attr) or "this factor"
        return [
            f"Mostly {attr_label}",
            f"{attr_label} plus other cues",
            "Something else",
        ]

    def _ensure_probe_has_choices(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        """Guarantee each TacitGap probe has 2-4 concise answer choices."""
        probe_out = dict(probe)
        choices = self._normalize_probe_choices(probe_out.get("choices", []))
        if len(choices) >= 2:
            probe_out["choices"] = choices[:4]
            return probe_out

        inferred = self._infer_choices_from_probe_question(probe_out.get("question", ""))
        if len(inferred) >= 2:
            probe_out["choices"] = inferred[:4]
            return probe_out

        probe_out["choices"] = self._fallback_probe_choices(probe_out.get("attribute", ""))
        return probe_out

    def _contextualize_tacit_remaining_payload(
        self,
        payload: Dict[str, Any],
        researcher_question: str,
        expert_answer: str,
    ) -> Dict[str, Any]:
        """Rewrite the final 1-2 TacitGap remaining items so they are easy to act on now."""
        out = dict(payload or {})
        attrs = [str(a).strip() for a in out.get("attributes", []) if str(a).strip()]
        probes = [dict(p) for p in out.get("probes", []) if isinstance(p, dict)]
        scenarios = [dict(s) for s in out.get("hypothetical_scenarios", []) if isinstance(s, dict)]
        if not attrs:
            return out

        rq = str(researcher_question or "").strip()
        ea = str(expert_answer or "").strip()
        rq_short = rq[:180] if rq else "the current question"
        ea_short = ea[:220] if ea else "the expert's latest answer"

        prompt = ChatPromptTemplate.from_messages([
            ("user", """You are refining the final "Still to Explore" section for a TacitGap in an interview assistant.

## Current researcher question
"{researcher_question}"

## Current expert answer
"{expert_answer}"

## Items to refine
{payload_json}

## Task
Rewrite ONLY the remaining TacitGap items so they are maximally useful for resolving the CURRENT ambiguity.

Requirements:
- Keep the same number and order of attributes, probes, and hypothetical_scenarios.
- Each attribute should become a short, human-readable label in English.
- Each probe question must remain multiple-choice and directly help the user clarify what the expert means right now.
- Every probe MUST include a "choices" array with 2 to 4 concise options.
- If the incoming probe already has choices and they still fit the rewritten question, preserve them.
- If the incoming probe is missing choices or the choices no longer fit the rewritten question, regenerate a better-fitting set of 2 to 4 concise choices.
- Each hypothetical scenario should test a concrete contrast the expert would recognize in this exact conversation.
- Keep the JSON structure the same.

Return ONLY valid JSON with top-level key "payload".""")
        ])

        try:
            llm = _build_polish_llm()
            chain = prompt | llm
            response = chain.invoke({
                "researcher_question": rq[:500],
                "expert_answer": ea[:800],
                "payload_json": json.dumps({
                    "payload": {
                        "attributes": attrs,
                        "probes": probes,
                        "hypothetical_scenarios": scenarios,
                    }
                }, ensure_ascii=False),
            })
            parsed = _parse_json_from_text(getattr(response, "content", str(response)))
            candidate = parsed.get("payload", {}) if isinstance(parsed, dict) else {}
            cand_attrs = [str(a).strip() for a in candidate.get("attributes", []) if str(a).strip()]
            cand_probes = [
                self._ensure_probe_has_choices(dict(p))
                for p in candidate.get("probes", [])
                if isinstance(p, dict)
            ]
            cand_scenarios = [dict(s) for s in candidate.get("hypothetical_scenarios", []) if isinstance(s, dict)]
            if len(cand_attrs) == len(attrs) and len(cand_probes) == len(attrs) and len(cand_scenarios) == len(attrs):
                out["attributes"] = cand_attrs
                out["probes"] = cand_probes
                out["hypothetical_scenarios"] = cand_scenarios
                return out
        except Exception:
            pass

        # Deterministic fallback if the LLM refinement fails.
        humanized_attrs = [self._humanize_attr_label(attr) for attr in attrs]
        attr_map = {old: new for old, new in zip(attrs, humanized_attrs)}
        for idx, probe in enumerate(probes):
            old_attr = str(probe.get("attribute", "")).strip()
            new_attr = attr_map.get(old_attr, humanized_attrs[min(idx, len(humanized_attrs) - 1)])
            probe["attribute"] = new_attr
            if not str(probe.get("question", "")).strip():
                probe["question"] = (
                    f'When the expert says "{ea_short}", is the unresolved issue mainly about {new_attr} '
                    "or about something else in the same answer?"
                )
            probes[idx] = self._ensure_probe_has_choices(probe)
        for idx, scenario in enumerate(scenarios):
            old_attr = str(scenario.get("attribute", "")).strip()
            new_attr = attr_map.get(old_attr, humanized_attrs[min(idx, len(humanized_attrs) - 1)])
            scenario["attribute"] = new_attr
            scenario["scenario"] = (
                f'If you followed up on "{rq_short}" and changed only {new_attr} while keeping the rest of the answer similar, '
                "would the expert's judgment still sound the same?"
            )
        out["attributes"] = humanized_attrs
        out["probes"] = probes
        out["hypothetical_scenarios"] = scenarios
        return out

    def _postprocess_tacit_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize TacitGap payload so extracted attributes are explicitly displayed
        and never accompanied by probes/scenarios.
        """
        if not isinstance(payload, dict):
            return {
                "attributes": [],
                "probes": [],
                "hypothetical_scenarios": [],
                "extracted_attributes": [],
                "mentioned_attributes": [],
            }

        def _norm_list(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            out: List[str] = []
            for item in value:
                s = str(item).strip()
                if s and s not in out:
                    out.append(s)
            return out

        source_attributes = _norm_list(payload.get("_source_attributes", []))
        attributes = source_attributes or _norm_list(payload.get("attributes", []))
        if not attributes and isinstance(payload.get("_source_probes", []), list):
            for probe in payload.get("_source_probes", []):
                if isinstance(probe, dict):
                    probe_attr = str(probe.get("attribute", "")).strip()
                    if probe_attr and probe_attr not in attributes:
                        attributes.append(probe_attr)
        if not attributes and isinstance(payload.get("probes", []), list):
            for probe in payload.get("probes", []):
                if isinstance(probe, dict):
                    probe_attr = str(probe.get("attribute", "")).strip()
                    if probe_attr and probe_attr not in attributes:
                        attributes.append(probe_attr)

        raw_extracted = _norm_list(
            payload.get("extracted_attributes", payload.get("mentioned_attributes", []))
        )
        extracted_mapped: List[str] = []
        for attr in raw_extracted:
            mapped = self._map_attr_to_inventory(attr, attributes) if attributes else attr
            chosen = mapped or attr
            if chosen and chosen not in extracted_mapped:
                extracted_mapped.append(chosen)
        expert_answer = str(payload.get("_expert_answer_hint", "")).strip().lower()
        if not extracted_mapped and expert_answer and attributes:
            for attr in attributes:
                attr_lower = attr.lower().strip()
                if not attr_lower:
                    continue
                if attr_lower in expert_answer:
                    extracted_mapped.append(attr)
                    continue
                tokens = [t for t in re.findall(r"[a-z0-9]+", attr_lower) if len(t) >= 4]
                if not tokens:
                    continue
                hit_count = sum(1 for t in tokens if t in expert_answer)
                # Relaxed threshold: for short attrs (<=2 tokens) any 1 hit counts;
                # for longer attrs, at least half the tokens must match.
                threshold = 1 if len(tokens) <= 2 else max(1, len(tokens) // 2)
                if hit_count >= threshold:
                    extracted_mapped.append(attr)
        extracted = [attr for attr in attributes if attr in extracted_mapped]
        max_extracted = max(0, len(attributes) - 2) if len(attributes) >= 3 else len(attributes)
        if len(extracted) > max_extracted:
            extracted = extracted[:max_extracted]
        if len(extracted) > 3:
            extracted = extracted[:3]
        extracted_lower = {a.lower() for a in extracted}

        def _iter_probes(raw_items: Any) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if not isinstance(raw_items, list):
                return out
            for probe in raw_items:
                if not isinstance(probe, dict):
                    continue
                probe_attr = str(probe.get("attribute", "")).strip()
                mapped = self._map_attr_to_inventory(probe_attr, attributes) if attributes else probe_attr
                if not mapped:
                    continue
                probe_copy = dict(probe)
                probe_copy["attribute"] = mapped
                out.append(probe_copy)
            return out

        def _iter_scenarios(raw_items: Any) -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            if not isinstance(raw_items, list):
                return out
            for sc in raw_items:
                if isinstance(sc, str):
                    sc_obj: Dict[str, str] = {"scenario": sc, "attribute": ""}
                elif isinstance(sc, dict):
                    sc_obj = {
                        "scenario": str(sc.get("scenario", "")),
                        "attribute": str(sc.get("attribute", "")),
                    }
                else:
                    continue
                mapped = self._map_attr_to_inventory(sc_obj["attribute"], attributes) if attributes else sc_obj["attribute"]
                if mapped:
                    sc_obj["attribute"] = mapped
                out.append(sc_obj)
            return out

        source_probes = _iter_probes(payload.get("_source_probes", []))
        polished_probes = _iter_probes(payload.get("probes", []))
        source_scenarios = _iter_scenarios(payload.get("_source_hypothetical_scenarios", []))
        polished_scenarios = _iter_scenarios(payload.get("hypothetical_scenarios", []))

        probe_by_attr: Dict[str, Dict[str, Any]] = {}
        for probe in polished_probes + source_probes:
            attr = str(probe.get("attribute", "")).strip()
            if not attr or attr.lower() in extracted_lower or attr in probe_by_attr:
                continue
            probe_by_attr[attr] = probe

        scenario_by_attr: Dict[str, Dict[str, str]] = {}
        remaining_attr_set = {a.lower() for a in attributes if a.lower() not in extracted_lower}
        for sc_obj in polished_scenarios + source_scenarios:
            scenario_text = str(sc_obj.get("scenario", "")).strip()
            if not scenario_text:
                continue
            attr = str(sc_obj.get("attribute", "")).strip()
            if not attr:
                lower_scenario = scenario_text.lower()
                inferred = ""
                best_hits = 0
                for candidate in attributes:
                    if candidate.lower() not in remaining_attr_set:
                        continue
                    tokens = [t for t in re.findall(r"[a-z0-9]+", candidate.lower()) if len(t) >= 3]
                    hits = sum(1 for t in tokens if t in lower_scenario)
                    if hits > best_hits:
                        best_hits = hits
                        inferred = candidate
                if inferred and best_hits >= 1:
                    attr = inferred
            if not attr or attr.lower() in extracted_lower or attr in scenario_by_attr:
                continue
            sc_copy = dict(sc_obj)
            sc_copy["attribute"] = attr
            scenario_by_attr[attr] = sc_copy

        # Keep only not-yet-mentioned attributes in the actionable list, then
        # pick up to 2 that are lexically distinct from the already-mentioned set.
        remaining_pool = [a for a in attributes if a.lower() not in extracted_lower]
        remaining_attrs = self._select_tacit_remaining_attrs(
            remaining_pool=remaining_pool,
            extracted=extracted,
            limit=2,
        )

        probes_out: List[Dict[str, Any]] = []
        scenarios_out: List[Dict[str, str]] = []
        answer_hint = str(payload.get("_expert_answer_hint", "")).strip()
        for attr in remaining_attrs:
            probe = probe_by_attr.get(attr)
            if not probe:
                probe = {
                    "attribute": attr,
                    "question": f"When judging {attr.replace('_', ' ')}, is it mostly clearly visible or still something you infer indirectly?",
                    "choices": ["Clearly visible", "Inferred from the overall pattern", "Something else"],
                }
            probes_out.append(self._ensure_probe_has_choices(probe))

            scenario = scenario_by_attr.get(attr)
            if not scenario:
                scenario = {
                    "attribute": attr,
                    "scenario": (
                        f'Given the expert just highlighted "{answer_hint[:120] or "this answer"}", '
                        f'if {attr.replace("_", " ")} changed while the rest stayed similar, '
                        "would your judgment shift immediately or only after a fuller read?"
                    ),
                }
            scenarios_out.append(scenario)

        expert_answer_full = str(payload.get("_expert_answer_hint", "")).strip()
        raw_quotes = _norm_list(payload.get("extracted_quotes", []))
        display_quotes: List[str] = []
        for quote in raw_quotes:
            span = self._find_exact_quote_span(expert_answer_full, quote)
            if not span:
                continue
            exact_quote = expert_answer_full[span[0]:span[1]].strip()
            if exact_quote and exact_quote not in display_quotes:
                display_quotes.append(exact_quote)
        if len(display_quotes) > len(extracted):
            display_quotes = display_quotes[:len(extracted)]
        if len(display_quotes) > 3:
            display_quotes = display_quotes[:3]

        payload["attributes"] = remaining_attrs
        payload["probes"] = probes_out
        payload["hypothetical_scenarios"] = scenarios_out
        payload["extracted_attributes"] = extracted
        payload["extracted_quotes"] = display_quotes
        # Keep old field for compatibility with existing clients/docs.
        payload["mentioned_attributes"] = extracted
        payload.pop("_expert_answer_hint", None)
        payload.pop("_source_attributes", None)
        payload.pop("_source_probes", None)
        payload.pop("_source_hypothetical_scenarios", None)
        return payload

    def _get_match_llm(self) -> ChatOpenAI:
        """Lazy load LLM for leaf matching."""
        if self._match_llm is None:
            self._match_llm = _build_match_llm()
        return self._match_llm

    def _llm_match_leaves(
        self,
        tree: TaxonomyTree,
        utterance: str,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """Use LLM to match an utterance against leaf nodes (label + description)."""
        leaves = tree.get_leaves()
        if not leaves or not utterance.strip():
            return []

        leaves_data = [
            {"id": leaf.id, "label": leaf.label, "description": leaf.description}
            for leaf in leaves
        ]

        try:
            llm = self._get_match_llm()
            prompt = ChatPromptTemplate.from_messages([("user", LEAF_MATCH_PROMPT)])
            chain = prompt | llm
            response = chain.invoke({
                "leaves_json": json.dumps(leaves_data, ensure_ascii=False),
                "utterance": utterance[:1000],
            })
            content = getattr(response, "content", str(response))
            parsed = _parse_json_from_text(content)
            matches = parsed.get("matches", []) if isinstance(parsed, dict) else []
        except Exception as exc:
            print(f"[RuntimeEngine] LLM leaf matching failed: {exc}")
            matches = []

        results: List[SearchResult] = []
        for match in matches[:top_k]:
            if not isinstance(match, dict):
                continue
            node_id = str(match.get("node_id", "")).strip()
            score = float(match.get("score", 0.0))
            node = tree.get_node(node_id)
            if node is None:
                continue
            snapped = score < 0.75
            snap_reason = ""
            if snapped and score >= self.MATCH_THRESHOLD:
                snap_reason = f"Low confidence match (score={score:.2f})"
            elif snapped and score < self.MATCH_THRESHOLD:
                snap_reason = f"Very low confidence (score={score:.2f}), may not be accurate"
            results.append(SearchResult(
                node_id=node_id,
                node=node,
                score=score,
                snapped=snapped,
                snap_reason=snap_reason,
            ))
        return results

    def locate_positions(
        self,
        researcher_question: str,
        expert_answer: str,
    ) -> LocatedPosition:
        """Locate positions in both trees based on the current turn using LLM matching."""
        result = LocatedPosition()

        # Match expert tree with expert's answer
        if expert_answer.strip():
            expert_results = self._llm_match_leaves(
                self.graph.expert_tree, expert_answer, top_k=3,
            )
            result.expert_results = expert_results
            if expert_results:
                result.best_expert_leaf_id = expert_results[0].node_id
                result.expert_confidence = expert_results[0].score

        # Match researcher tree with researcher's question
        if researcher_question.strip():
            researcher_results = self._llm_match_leaves(
                self.graph.researcher_tree, researcher_question, top_k=3,
            )
            result.researcher_results = researcher_results
            if researcher_results:
                result.best_researcher_leaf_id = researcher_results[0].node_id
                result.researcher_confidence = researcher_results[0].score

        turn_match_text = self._build_turn_match_text(researcher_question, expert_answer)
        if (
            self._contains_tacit_cues(turn_match_text)
            and result.expert_results
            and result.researcher_results
        ):
            tacit_pair = self._select_tacitgap_priority_pair(
                result.expert_results,
                result.researcher_results,
            )
            if tacit_pair is not None:
                expert_match, researcher_match = tacit_pair
                result.expert_results = self._promote_search_result(
                    result.expert_results,
                    expert_match.node_id,
                )
                result.researcher_results = self._promote_search_result(
                    result.researcher_results,
                    researcher_match.node_id,
                )
                result.best_expert_leaf_id = expert_match.node_id
                result.expert_confidence = expert_match.score
                result.best_researcher_leaf_id = researcher_match.node_id
                result.researcher_confidence = researcher_match.score

        return result

    def _build_uncertain_interpretation(self, located: LocatedPosition) -> Dict[str, Any]:
        """Build a structured low-confidence response without changing gate logic."""
        candidates = []
        for result in located.expert_results[:2]:
            candidates.append({
                "node_id": result.node_id,
                "label": result.node.label,
                "description": result.node.description,
                "score": round(result.score, 4),
            })

        return {
            "status": (
                "Top 2 possible expert concepts are shown below."
                if len(candidates) >= 2
                else "A possible expert concept is shown below."
                if len(candidates) == 1
                else "The current expert response could not be mapped confidently."
            ),
            "candidates": candidates,
        }

    def _build_no_mismatch_interpretation(self) -> Dict[str, Any]:
        """Return the product fallback state for turns with no actionable mismatch."""
        return {
            "status": "No mismatch requiring system intervention has been found yet.",
            "candidates": [],
        }

    def _fallback_uncertainty_followups(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate deterministic fallback questions when LLM call fails.

        Questions probe each candidate's distinguishing description rather than
        asking a yes/no confirmation on the label, consistent with the prompt's
        contrastive design.
        """
        followups: List[Dict[str, Any]] = []

        for i, candidate in enumerate(candidates):
            label = candidate.get("label", "").strip()
            desc = candidate.get("description", "").strip()
            short_desc = desc[:80].rsplit(" ", 1)[0] if len(desc) > 80 else desc

            if len(candidates) >= 2 and short_desc:
                question = (
                    f"Could you tell me more about how {short_desc.lower()} "
                    f"plays into what you're describing?"
                )
            elif short_desc:
                question = (
                    f"Could you elaborate on whether {short_desc.lower()} "
                    f"relates to what you're getting at?"
                )
            elif label:
                question = (
                    f"Could you elaborate on what you mean? "
                    f"I want to understand how {label.lower()} fits into your thinking."
                )
            else:
                question = "Could you elaborate a bit more on what you mean by that?"

            followups.append({
                "node_id": candidate.get("node_id", ""),
                "followup_questions": [question],
            })
        return followups

    def _polish_uncertain_interpretation(
        self,
        uncertain: Dict[str, Any],
        researcher_question: str,
        expert_answer: str,
        context_summary: str,
    ) -> Dict[str, Any]:
        """Add candidate-specific follow-up questions using the polish model."""
        candidates = uncertain.get("candidates", [])
        if not candidates:
            return uncertain

        fallback_followups = self._fallback_uncertainty_followups(candidates)
        fallback_map = {
            item.get("node_id", ""): item.get("followup_questions", [])
            for item in fallback_followups
        }

        try:
            candidates_for_llm = [
                {"node_id": c.get("node_id", ""), "label": c.get("label", ""),
                 "description": c.get("description", "")}
                for c in candidates
            ]
            llm = _build_polish_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("user", UNCERTAIN_INTERPRETATION_FOLLOWUPS_PROMPT)
            ])
            chain = prompt | llm
            response = chain.invoke({
                "researcher_question": researcher_question[:300],
                "expert_answer": expert_answer[:500],
                "context_summary": context_summary[:600],
                "candidates_json": json.dumps(
                    {"candidates": candidates_for_llm}, ensure_ascii=False,
                ),
            })
            content = getattr(response, "content", str(response))
            parsed = _parse_json_from_text(content)
            generated = parsed.get("candidate_followups", []) if isinstance(parsed, dict) else []
        except Exception:
            generated = []

        generated_map: Dict[str, List[str]] = {}
        for item in generated:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id", "")).strip()
            questions = item.get("followup_questions", [])
            cleaned: List[str] = []
            if isinstance(questions, list):
                for question in questions:
                    text = str(question).strip()
                    if text:
                        cleaned.append(text)
            if node_id and cleaned:
                generated_map[node_id] = cleaned[:1]

        polished_candidates = []
        for candidate in candidates:
            node_id = candidate.get("node_id", "")
            question_list = generated_map.get(node_id) or fallback_map.get(node_id, [])
            candidate_out = dict(candidate)
            candidate_out["followup_questions"] = question_list
            polished_candidates.append(candidate_out)

        uncertain_out = dict(uncertain)
        uncertain_out["candidates"] = polished_candidates
        return uncertain_out

    def find_best_link(
        self,
        located: LocatedPosition,
    ) -> Optional[GapLink]:
        """
        Find the link between the top-1 expert leaf and top-1 researcher leaf.
        If no direct link exists between them, return None (show 'No mismatch').
        """
        expert_leaf_id = located.best_expert_leaf_id
        researcher_leaf_id = located.best_researcher_leaf_id
        if not expert_leaf_id or not researcher_leaf_id:
            return None

        link = self.graph.get_link(expert_leaf_id, researcher_leaf_id)
        if link and link.relation_type != RelationType.PROCESS_GAP.value:
            return self._ensure_actionable_link(link)

        return None

    def compute_divergence(self, link: GapLink) -> DivergenceInfo:
        """Compute divergence information from a link."""
        divergence = DivergenceInfo(
            lca_layer=link.lca_layer,
            expert_branch=link.conflict.get("expert_branch", ""),
            researcher_branch=link.conflict.get("researcher_branch", ""),
        )

        if link.lca_layer == Layer.ROOT.value:
            divergence.explanation = (
                f"Major perspective difference: Expert focuses on '{divergence.expert_branch}' "
                f"while you focus on '{divergence.researcher_branch}'. "
                "You need high-level bridging."
            )
        elif link.lca_layer == Layer.L1.value:
            divergence.explanation = (
                f"Category-level difference within the same perspective. "
                f"Expert: '{divergence.expert_branch}', You: '{divergence.researcher_branch}'. "
                "Try connecting through shared workflow concepts."
            )
        else:
            divergence.explanation = (
                f"Specific difference at detail level. "
                f"Expert: '{divergence.expert_branch}', You: '{divergence.researcher_branch}'. "
                "You can use concrete examples to bridge."
            )

        return divergence

    def _polish_divergence(
        self,
        divergence: DivergenceInfo,
        researcher_question: str,
        expert_answer: str,
        context_summary: str,
    ) -> DivergenceInfo:
        """
        Rewrite divergence branch labels so they are grounded in the expert's
        actual wording while preserving the semantic meaning of the original
        taxonomy label.  Keeps the output concise (short phrase, not a sentence).
        """
        if not divergence or (not divergence.expert_branch and not divergence.researcher_branch):
            return divergence

        try:
            llm = _build_polish_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("user", """You are contextualizing divergence labels for a live interview assistant.

## Current conversation
- Researcher's latest question: "{researcher_question}"
- Expert's latest answer: "{expert_answer}"
- Recent conversation (last 3 turns): "{context_summary}"

## Original divergence labels (from the offline taxonomy)
- Expert branch label: "{expert_branch}"
- Researcher branch label: "{researcher_branch}"

## Task
Rewrite BOTH labels so they:
1. Use vocabulary and phrasing the expert actually used in their answer or that appeared in the recent conversation.
2. Preserve the core semantic meaning of the original label.
3. Stay concise — a short noun phrase (2-6 words), NOT a full sentence.
4. Feel natural and recognizable to someone reading the live conversation.
5. Do not invent concepts or terms (e.g. specific entities or scores) that the expert did not mention. If the expert's answer does not give you wording to rephrase with, or you cannot rephrase using their words or close synonyms, return the original label unchanged.

Return ONLY valid JSON:
{{
  "expert_branch": "rewritten expert branch label",
  "researcher_branch": "rewritten researcher branch label"
}}
""")
            ])
            chain = prompt | llm
            response = chain.invoke({
                "researcher_question": researcher_question[:500],
                "expert_answer": expert_answer[:800],
                "context_summary": context_summary[:1200],
                "expert_branch": divergence.expert_branch,
                "researcher_branch": divergence.researcher_branch,
            })
            content = getattr(response, "content", str(response))
            parsed = _parse_json_from_text(content)
            if parsed:
                new_expert = str(parsed.get("expert_branch", "")).strip()
                new_researcher = str(parsed.get("researcher_branch", "")).strip()
                if new_expert:
                    divergence.expert_branch = new_expert
                if new_researcher:
                    divergence.researcher_branch = new_researcher
        except Exception as e:
            print(f"[RuntimeEngine] Divergence polish failed (keeping originals): {e}")

        return divergence

    def generate_assistance(
        self,
        expert_leaf_id: Optional[str],
        selected_link: Optional[GapLink],
        expert_answer: str = "",
        researcher_question: str = "",
    ) -> Assistance:
        """Generate type-specific assistance based on the selected link's relation_type."""
        if not selected_link:
            return Assistance()

        selected_link = self._ensure_actionable_link(selected_link)
        relation = selected_link.relation_type
        if relation == RelationType.PROCESS_GAP.value:
            # ProcessGap is deprecated in UI/product flow and replaced by summarization.
            return Assistance()
        payload = dict(selected_link.assistance_payload)
        assistance = Assistance(relation_type=relation, payload=payload)

        # ---- LexicalGap ----
        if relation == RelationType.LEXICAL_GAP.value:
            pass

        # ---- ConceptualGap ----
        elif relation == RelationType.CONCEPTUAL_GAP.value:
            exp_node = self.graph.expert_tree.get_node(expert_leaf_id) if expert_leaf_id else None
            res_node = self.graph.researcher_tree.get_node(selected_link.researcher_leaf_id)

            ea = str(expert_answer or "").strip()
            rq = str(researcher_question or "").strip()
            ea_short = (ea[:160].rsplit(" ", 1)[0] if len(ea) > 160 else ea) or "expert-side concept"
            rq_short = (rq[:160].rsplit(" ", 1)[0] if len(rq) > 160 else rq) or "researcher-side target"

            analogy = dict(payload.get("analogy", {}) or {})
            analogy.setdefault(
                "source_concept",
                (exp_node.label if exp_node and exp_node.label else ea_short),
            )
            analogy.setdefault(
                "target_concept",
                (res_node.label if res_node and res_node.label else rq_short),
            )

            structural = dict(analogy.get("structural_mapping", {}) or {})
            structural.setdefault("inputs", f'Expert framing: "{analogy["source_concept"]}"')
            structural.setdefault(
                "logic",
                ea_short if ea_short else "Expert reasoning path from current answer context.",
            )
            structural.setdefault("outputs", f'Research goal focus: "{analogy["target_concept"]}"')
            analogy["structural_mapping"] = structural

            explanation = str(analogy.get("explanation", "")).strip()
            if not explanation:
                analogy["explanation"] = (
                    f'This turn emphasizes "{analogy["source_concept"]}" on the expert side, '
                    f'while your question targets "{analogy["target_concept"]}". '
                    "Use the former as a bridge to probe the latter."
                )

            assistance.payload = dict(payload or {})
            assistance.payload["analogy"] = analogy

        # ---- TacitGap ----
        elif relation == RelationType.TACIT_GAP.value:
            assistance.payload = payload

        # ---- ScopeGap ----
        elif relation == RelationType.SCOPE_GAP.value:
            pass

        return assistance

    def analyze_turn(
        self,
        researcher_question: str,
        expert_answer: str,
        context_summary: str = "",
    ) -> RuntimeAnalysis:
        """
        Analyze a conversation turn and provide type-specific navigation guidance.
        """
        analysis = RuntimeAnalysis()
        match_threshold = self.MATCH_THRESHOLD

        # 1. Locate positions
        analysis.located = self.locate_positions(researcher_question, expert_answer)

        # Product rule: "No mismatch" is shown whenever the current turn cannot
        # be mapped confidently enough to an actionable expert/researcher leaf pair.
        if not analysis.located.best_expert_leaf_id:
            analysis.confidence_warning = (
                "No expert concept was matched for this turn; showing the no-mismatch fallback."
            )
            analysis.uncertain_interpretation = self._build_no_mismatch_interpretation()
            return analysis

        if not analysis.located.best_researcher_leaf_id:
            analysis.confidence_warning = (
                "No researcher concept was matched for this turn; showing the no-mismatch fallback."
            )
            analysis.uncertain_interpretation = self._build_no_mismatch_interpretation()
            return analysis

        if analysis.located.expert_confidence < match_threshold:
            analysis.confidence_warning = (
                f"Expert match confidence {analysis.located.expert_confidence:.2f} "
                f"is below threshold {match_threshold:.2f}; showing the no-mismatch fallback."
            )
            analysis.uncertain_interpretation = self._build_no_mismatch_interpretation()
            return analysis

        if analysis.located.researcher_confidence < match_threshold:
            analysis.confidence_warning = (
                f"Researcher match confidence {analysis.located.researcher_confidence:.2f} "
                f"is below threshold {match_threshold:.2f}; showing the no-mismatch fallback."
            )
            analysis.uncertain_interpretation = self._build_no_mismatch_interpretation()
            return analysis

        # 2. Find best link
        analysis.selected_link = self.find_best_link(
            analysis.located,
        )
        if (
            analysis.selected_link
            and analysis.selected_link.relation_type == RelationType.PROCESS_GAP.value
        ):
            analysis.selected_link = None

        if not analysis.selected_link:
            analysis.confidence_warning = (
                "No direct mismatch link was found for the top matched expert/researcher leaves; "
                "showing the no-mismatch fallback."
            )
            analysis.uncertain_interpretation = self._build_no_mismatch_interpretation()
            return analysis

        # 3. Compute divergence
        analysis.divergence = self.compute_divergence(analysis.selected_link)

        # 3b. Polish divergence labels to match live conversation
        analysis.divergence = self._polish_divergence(
            analysis.divergence,
            researcher_question,
            expert_answer,
            context_summary,
        )

        # 4. Generate type-specific assistance
        analysis.assistance = self.generate_assistance(
            analysis.located.best_expert_leaf_id,
            analysis.selected_link,
            expert_answer=expert_answer,
            researcher_question=researcher_question,
        )

        if analysis.assistance and analysis.assistance.payload:
            analysis.assistance = self._polish_assistance(
                analysis.assistance,
                researcher_question,
                expert_answer,
                context_summary,
            )
        if analysis.assistance and (
            not analysis.assistance.relation_type or not analysis.assistance.payload
        ):
            analysis.assistance = None

        if not analysis.assistance:
            analysis.confidence_warning = (
                "No actionable mismatch assistance was produced for the matched leaf pair; "
                "showing the no-mismatch fallback."
            )
            analysis.uncertain_interpretation = self._build_no_mismatch_interpretation()
            return analysis

        return analysis


# ============== Convenience Function ==============

def analyze_turn(
    graph: DSAGGraph,
    researcher_question: str,
    expert_answer: str,
    context_summary: str = "",
) -> Dict[str, Any]:
    """
    Convenience function to analyze a turn.

    Returns the analysis as a dictionary.
    """
    engine = RuntimeEngine(graph)
    analysis = engine.analyze_turn(
        researcher_question,
        expert_answer,
        context_summary=context_summary,
    )
    return analysis.to_dict()
