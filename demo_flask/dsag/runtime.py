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
from .embedding_index import EmbeddingIndex, SearchResult


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
        model = os.getenv("OPENAI_MODEL_RUNTIME", "qwen3-max")
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
    - ConceptualGap: {"analogy": {"source_concept", "structural_mapping": {inputs, logic, outputs},
                      "explanation"}, "scenario": {"inputs", "outputs", "edge_cases"}}
    - TacitGap:      {"attributes": [...], "probes": [{attribute, question, choices}],
                      "hypothetical_scenarios": ["...", "..."]}
    - ScopeGap:      {"validate_focus": "...", "pivot": {limitation, research_goal,
                      compelling_reason, coarse_scenario}}
    - ProcessGap:    {"timeline": [...], "drift_alerts": [...], "current_topic": "..."}
    """
    relation_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    followup_questions: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "payload": self.payload,
            "followup_questions": self.followup_questions,
        }


@dataclass
class RuntimeAnalysis:
    """Complete analysis result for a turn."""
    located: LocatedPosition = field(default_factory=LocatedPosition)
    divergence: Optional[DivergenceInfo] = None
    assistance: Optional[Assistance] = None
    selected_link: Optional[GapLink] = None
    confidence_warning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "located": self.located.to_dict(),
            "confidence_warning": self.confidence_warning,
        }
        if self.divergence:
            result["divergence"] = self.divergence.to_dict()
        if self.assistance:
            result["assistance"] = self.assistance.to_dict()
        if self.selected_link:
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

    def __init__(self, graph: DSAGGraph, embedding_index: EmbeddingIndex):
        self.graph = graph
        self.index = embedding_index
        self._llm = None

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
FROZEN fields (copy verbatim — do NOT alter):
  - term_mapping.expert_term
  - term_mapping.researcher_term

POLISHABLE fields:
  - term_mapping.explanation — make it a crisp, natural-sounding sentence
    that a researcher could say aloud. Weave in the expert's phrasing when
    possible (e.g. "When the expert says '…', it corresponds to '…'").

No follow-up questions expected for this type.""",

        RelationType.CONCEPTUAL_GAP.value: """## ConceptualGap polishing rules
FROZEN fields (copy verbatim — do NOT alter):
  - analogy.source_concept (keep the concept name exactly)
  - analogy.structural_mapping.inputs
  - analogy.structural_mapping.logic
  - analogy.structural_mapping.outputs
  - scenario.inputs
  - scenario.outputs
  - scenario.edge_cases

POLISHABLE fields:
  - analogy.explanation — make it a fluent sentence the researcher could speak
    naturally. Reference the expert's actual vocabulary from their answer.

The analogy and scenario are PARALLEL strategies (no required order).
Do not merge them or remove either one.""",

        RelationType.TACIT_GAP.value: """## TacitGap polishing rules
FROZEN fields (copy verbatim — do NOT alter):
  - attributes (the list of attribute names)
  - probes[].attribute
  - probes[].choices (the option labels must stay exactly as-is)
  - hypothetical_scenarios (keep the variable-change structure intact)

POLISHABLE fields:
  - probes[].question — rephrase for naturalness, but it MUST remain a
    multiple-choice question (NEVER convert to open-ended "Why …?").
    Reference the expert's recent answer where possible.

The 3-step order (attributes → probes → hypothetical checkout) is strict.
Do not reorder or collapse steps.""",

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

The 2-step order (validate THEN pivot) is strict. Do not merge them.""",

        RelationType.PROCESS_GAP.value: """## ProcessGap polishing rules
FROZEN fields (copy verbatim — do NOT alter):
  - timeline (the entire list — structural tracking data)
  - current_topic (label string)
  - expected_steps[].order (numeric ordering)
  - expected_steps[].label (step names)

POLISHABLE fields:
  - expected_steps[].description — make each step description read naturally,
    referencing the expert's terminology and recent answer.
  - tunnel_vision_risks[] — rephrase each risk alert so it sounds like
    a gentle, constructive interviewer note (not a harsh warning).
  - drift_alerts[] — rephrase each drift alert so it reads as a helpful
    suggestion, referencing the expert's own topic labels.

Do NOT alter the ORDER of expected_steps.
Do NOT remove any drift_alerts or tunnel_vision_risks entries.""",
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

        llm = _build_polish_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("user", """You are polishing interview assistance so it sounds natural and directly speakable by the researcher during a live interview.

## Mismatch type: {relation_type}

## Conversation context
- Expert's latest answer: "{expert_answer}"
- Researcher's latest question: "{researcher_question}"
- Recent conversation summary: "{context_summary}"

## Assistance to polish
{assistance_json}

## General rules (apply to ALL types)
1. Keep the EXACT same JSON keys and nesting structure.
2. Make polishable text natural, concise, and directly speakable.
3. Reference the expert's OWN vocabulary and phrasing where possible.
4. Do NOT invent new facts or concepts.
5. Questions MUST remain questions.
6. Output ONLY valid JSON with top-level key "payload" (same structure as the input).

{type_specific_rules}
""")
        ])
        chain = prompt | llm
        response = chain.invoke({
            "relation_type": relation,
            "expert_answer": expert_answer[:500],
            "researcher_question": researcher_question[:300],
            "context_summary": context_summary[:600],
            "assistance_json": json.dumps(
                {"payload": assistance.payload},
                ensure_ascii=False,
            ),
            "type_specific_rules": type_rules,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json_from_text(content)
        if not parsed:
            return assistance

        polished = Assistance(relation_type=assistance.relation_type)
        polished.payload = parsed.get("payload", assistance.payload)
        return polished

    def locate_positions(
        self,
        researcher_question: str,
        expert_answer: str,
    ) -> LocatedPosition:
        """Locate positions in both trees based on the current turn."""
        result = LocatedPosition()

        # Search expert tree with expert's answer
        if expert_answer.strip():
            expert_results = self.index.search_expert_leaves(expert_answer, top_k=3)
            result.expert_results = expert_results
            if expert_results:
                result.best_expert_leaf_id = expert_results[0].node_id
                result.expert_confidence = expert_results[0].score

        # Search researcher tree with researcher's question
        if researcher_question.strip():
            researcher_results = self.index.search_researcher_leaves(researcher_question, top_k=3)
            result.researcher_results = researcher_results
            if researcher_results:
                result.best_researcher_leaf_id = researcher_results[0].node_id
                result.researcher_confidence = researcher_results[0].score

        return result

    def find_best_link(
        self,
        expert_leaf_id: Optional[str],
        researcher_leaf_id: Optional[str],
    ) -> Optional[GapLink]:
        """
        Find the best link connecting the located positions.
        Falls back to finding any link from the expert leaf.
        """
        if not expert_leaf_id:
            return None

        # First try exact match
        if researcher_leaf_id:
            link = self.graph.get_link(expert_leaf_id, researcher_leaf_id)
            if link:
                return link

        # Fallback: find any link from expert leaf
        links = self.graph.get_links_by_expert_leaf(expert_leaf_id)
        if links:
            links.sort(key=lambda l: l.weight, reverse=True)
            return links[0]

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

    def generate_assistance(
        self,
        expert_leaf_id: Optional[str],
        selected_link: Optional[GapLink],
        interview_timeline: Optional[List[Dict[str, Any]]] = None,
    ) -> Assistance:
        """
        Generate type-specific assistance based on the selected link's relation_type.
        """
        if not selected_link:
            return Assistance()

        relation = selected_link.relation_type
        payload = dict(selected_link.assistance_payload)  # copy offline payload
        assistance = Assistance(relation_type=relation, payload=payload)

        expert_node = self.graph.expert_tree.get_node(
            selected_link.expert_leaf_id
        )

        # ---- LexicalGap: 1-to-1 term mapping. No follow-ups. ----
        if relation == RelationType.LEXICAL_GAP.value:
            pass

        # ---- ConceptualGap: analogy + scenario (no order). ----
        elif relation == RelationType.CONCEPTUAL_GAP.value:
            pass

        # ---- TacitGap: strict 3-step pipeline. ----
        elif relation == RelationType.TACIT_GAP.value:
            # Merge live attributes from node into payload
            if expert_node and expert_node.attributes:
                payload["attributes"] = expert_node.attributes
            assistance.payload = payload

        # ---- ScopeGap: strict 2-step (validate then pivot). No follow-ups. ----
        elif relation == RelationType.SCOPE_GAP.value:
            pass

        # ---- ProcessGap: accumulated timeline + drift detection. ----
        elif relation == RelationType.PROCESS_GAP.value:
            timeline = interview_timeline or []
            payload["timeline"] = timeline

            drift_alerts = []
            if expert_node:
                current_label = expert_node.label
                covered_labels = [entry.get("topic_label", "") for entry in timeline]

                # Alert if topic was already discussed
                if current_label in covered_labels:
                    prev_turn = covered_labels.index(current_label) + 1
                    drift_alerts.append(
                        f"Topic '{current_label}' has been discussed before "
                        f"(turn {prev_turn}). Consider probing a different aspect."
                    )

                # Detect skipped steps using expected_steps from factory
                expected_steps = payload.get("expected_steps", [])
                if expected_steps and covered_labels:
                    step_labels = [s.get("label", "") for s in expected_steps]
                    # Find which expected steps have been covered
                    covered_step_orders = []
                    for step in expected_steps:
                        if step.get("label", "") in covered_labels:
                            covered_step_orders.append(step.get("order", 0))
                    # If current topic matches a later step, flag skipped earlier ones
                    current_step_order = None
                    for step in expected_steps:
                        if step.get("label", "") == current_label:
                            current_step_order = step.get("order", 0)
                            break
                    if current_step_order is not None:
                        skipped = [
                            s for s in expected_steps
                            if s.get("order", 0) < current_step_order
                            and s.get("label", "") not in covered_labels
                        ]
                        if skipped:
                            skipped_names = ", ".join(
                                s.get("label", "") for s in skipped
                            )
                            drift_alerts.append(
                                f"Skipped steps: {skipped_names}. "
                                "Consider circling back to these."
                            )

                # Tunnel vision detection
                tunnel_risks = payload.get("tunnel_vision_risks", [])
                if tunnel_risks:
                    turn_counts = {}
                    for entry in timeline:
                        lbl = entry.get("topic_label", "")
                        turn_counts[lbl] = turn_counts.get(lbl, 0) + 1
                    for risk in tunnel_risks:
                        for lbl, count in turn_counts.items():
                            if lbl in risk and count >= 2:
                                drift_alerts.append(
                                    f"Tunnel vision warning: {risk}"
                                )
                                break

                # Suggest uncovered sibling topics
                siblings = self.graph.expert_tree.get_siblings(
                    selected_link.expert_leaf_id
                )
                covered_ids = {entry.get("expert_leaf_id", "") for entry in timeline}
                uncovered = [s for s in siblings if s.id not in covered_ids]
                if uncovered:
                    labels = ", ".join(s.label for s in uncovered[:3])
                    drift_alerts.append(
                        f"Related uncovered topics: {labels}"
                    )

            payload["drift_alerts"] = drift_alerts
            payload["current_topic"] = expert_node.label if expert_node else ""
            assistance.payload = payload

        return assistance

    def analyze_turn(
        self,
        researcher_question: str,
        expert_answer: str,
        context_summary: str = "",
        interview_timeline: Optional[List[Dict[str, Any]]] = None,
    ) -> RuntimeAnalysis:
        """
        Analyze a conversation turn and provide type-specific navigation guidance.

        Args:
            researcher_question: The researcher's question
            expert_answer: The expert's response
            context_summary: Summary of recent conversation turns
            interview_timeline: Accumulated timeline entries for Process Gap

        Returns:
            RuntimeAnalysis with type-specific assistance
        """
        analysis = RuntimeAnalysis()

        # 1. Locate positions
        analysis.located = self.locate_positions(researcher_question, expert_answer)

        # Check confidence
        if analysis.located.expert_confidence < 0.45:
            analysis.confidence_warning = (
                f"Low confidence matching expert's statement "
                f"(score={analysis.located.expert_confidence:.2f}). "
                "Suggestions may not be accurate."
            )

        # 2. Find best link
        analysis.selected_link = self.find_best_link(
            analysis.located.best_expert_leaf_id,
            analysis.located.best_researcher_leaf_id,
        )

        if analysis.selected_link:
            # 3. Compute divergence
            analysis.divergence = self.compute_divergence(analysis.selected_link)

        # 4. Generate type-specific assistance
        analysis.assistance = self.generate_assistance(
            analysis.located.best_expert_leaf_id,
            analysis.selected_link,
            interview_timeline=interview_timeline,
        )

        if analysis.assistance and analysis.assistance.payload:
            analysis.assistance = self._polish_assistance(
                analysis.assistance,
                researcher_question,
                expert_answer,
                context_summary,
            )

        return analysis


# ============== Convenience Function ==============

def analyze_turn(
    graph: DSAGGraph,
    embedding_index: EmbeddingIndex,
    researcher_question: str,
    expert_answer: str,
    context_summary: str = "",
    interview_timeline: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to analyze a turn.

    Returns the analysis as a dictionary.
    """
    engine = RuntimeEngine(graph, embedding_index)
    analysis = engine.analyze_turn(
        researcher_question,
        expert_answer,
        context_summary=context_summary,
        interview_timeline=interview_timeline,
    )
    return analysis.to_dict()
