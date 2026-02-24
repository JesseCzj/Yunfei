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
    - ConceptualGap: {"analogy": {"source_concept", "structural_mapping": {inputs, logic, outputs}},
                      "scenario": {"inputs", "outputs", "edge_cases"}}
    - TacitGap:      {"attributes": [...], "probes": [{attribute, question, choices}],
                      "hypothetical_scenario": "..."}
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

    def _polish_assistance(
        self,
        assistance: Assistance,
        researcher_question: str,
        expert_answer: str,
        context_summary: str,
    ) -> Assistance:
        """Polish assistance text with a low-latency LLM."""
        if not assistance or not assistance.payload:
            return assistance

        # Skip polishing for ProcessGap (timeline data, not prose)
        if assistance.relation_type == RelationType.PROCESS_GAP.value:
            return assistance

        llm = _build_polish_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("user", """You are polishing interview assistance for a researcher.

Type of mismatch: {relation_type}

Inputs:
- Expert answer: "{expert_answer}"
- Researcher question: "{researcher_question}"
- Recent context summary: "{context_summary}"

Assistance to polish (keep structure, improve naturalness, reference expert's wording):
{assistance_json}

Rules:
1) Keep the EXACT same JSON keys and structure.
2) Make wording natural, concise, and directly speakable by the researcher.
3) Do NOT add new facts.
4) Questions must remain questions.
5) For LexicalGap: keep term_mapping exact; only polish the explanation.
6) For TacitGap: probe choices must stay as-is; only polish question phrasing.
7) Output ONLY valid JSON.
""")
        ])
        chain = prompt | llm
        response = chain.invoke({
            "relation_type": assistance.relation_type,
            "expert_answer": expert_answer[:500],
            "researcher_question": researcher_question[:300],
            "context_summary": context_summary[:600],
            "assistance_json": json.dumps(
                {"payload": assistance.payload, "followup_questions": assistance.followup_questions},
                ensure_ascii=False,
            ),
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json_from_text(content)
        if not parsed:
            return assistance

        polished = Assistance(relation_type=assistance.relation_type)
        polished.payload = parsed.get("payload", assistance.payload)
        polished.followup_questions = parsed.get("followup_questions", assistance.followup_questions)
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

        # ---- ConceptualGap: analogy + scenario (no order). ExpandScope follow-ups. ----
        elif relation == RelationType.CONCEPTUAL_GAP.value:
            if expert_leaf_id and expert_node:
                siblings = self.graph.expert_tree.get_siblings(expert_leaf_id)
                for sibling in siblings[:3]:
                    assistance.followup_questions.append({
                        "type": "ExpandScope",
                        "question": (
                            f"After '{expert_node.label}', do you typically "
                            f"also deal with '{sibling.label}'?"
                        ),
                    })

        # ---- TacitGap: strict 3-step pipeline. DeepDive follow-ups. ----
        elif relation == RelationType.TACIT_GAP.value:
            # Merge live attributes from node into payload
            if expert_node and expert_node.attributes:
                payload["attributes"] = expert_node.attributes
                # DeepDive follow-ups using attributes
                for attr in expert_node.attributes[:3]:
                    assistance.followup_questions.append({
                        "type": "DeepDive",
                        "question": (
                            f"When you think about '{expert_node.label}', "
                            f"is '{attr}' something that factors into your judgment?"
                        ),
                    })
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
