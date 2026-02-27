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


# ============== Process Gap Runtime Prompt ==============

PROCESS_GAP_REDIRECT_PROMPT = """You are helping a researcher redirect an interview conversation that has drifted.

**Drift type:** {drift_type}
**Drift detail:** {drift_detail}

**Current topic the expert is discussing:**
- Label: {current_topic_label}
- Description: {current_topic_description}

**Expert's latest answer (verbatim excerpt):**
"{expert_answer}"

**Unvisited related topics the researcher has NOT yet explored:**
{unvisited_siblings_text}

**Misalignment context:** {misalignment_reason}

**Recent conversation flow (last few turns):**
{timeline_summary}

Your task: Generate ONE natural redirect sentence that the researcher can speak directly to steer the conversation. Rules:
1. Reference the expert's actual words or phrasing from their latest answer — show you were listening.
2. Validate what the expert just said before transitioning.
3. Use an exploratory question to introduce the unvisited topic — never assert or correct.
4. Keep it to 1-2 sentences, natural and conversational.
5. Do NOT use jargon the expert hasn't used.

Return ONLY valid JSON:
{{
  "redirect": "Your single redirect sentence here"
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
    - ConceptualGap: {"analogy": {"source_concept", "structural_mapping": {inputs, logic, outputs},
                      "explanation"}, "scenario": {"inputs", "outputs", "edge_cases"}}
    - TacitGap:      {"attributes": [...], "probes": [{attribute, question, choices}],
                      "hypothetical_scenarios": ["...", "..."]}
    - ScopeGap:      {"validate_focus": "...", "pivot": {limitation, research_goal,
                      compelling_reason, coarse_scenario}}
    - ProcessGap:    {"coverage": {"visited": [...], "unvisited_siblings": [...],
                      "coverage_ratio": "2/5"}, "drift_detected": bool,
                      "drift_type": str|null, "drift_detail": str|null,
                      "redirect": str|null}
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

        # ProcessGap is runtime-driven. Its redirect is already LLM-generated
        # with full context, so it skips the polish step entirely.
        # No entry needed here.
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

    def _generate_process_redirect(
        self,
        drift_type: str,
        drift_detail: str,
        expert_node: DSAGNode,
        expert_answer: str,
        unvisited: List[DSAGNode],
        selected_link: Optional[GapLink],
        timeline: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Generate a context-aware redirect sentence via LLM when drift is detected.
        Returns the redirect string, or None if the LLM call fails.
        """
        # Build unvisited siblings text
        if unvisited:
            unvisited_lines = [f"- {s.label}: {s.description}" for s in unvisited[:4]]
            unvisited_text = "\n".join(unvisited_lines)
        else:
            unvisited_text = "(no specific unvisited siblings)"

        # Build a concise timeline summary (last 4 entries)
        recent = timeline[-4:] if len(timeline) > 4 else timeline
        tl_lines = []
        for i, entry in enumerate(recent, 1):
            tl_lines.append(
                f"  Turn {entry.get('turn_index', i)}: "
                f"[{entry.get('topic_label', '?')}] "
                f"{entry.get('summary', '')[:80]}"
            )
        timeline_summary = "\n".join(tl_lines) if tl_lines else "(first turn)"

        # Misalignment reason from the link
        misalignment_reason = ""
        if selected_link and selected_link.assistance_payload:
            misalignment_reason = selected_link.assistance_payload.get(
                "misalignment_reason", ""
            )

        variables = {
            "drift_type": drift_type,
            "drift_detail": drift_detail,
            "current_topic_label": expert_node.label,
            "current_topic_description": expert_node.description,
            "expert_answer": expert_answer[:500],
            "unvisited_siblings_text": unvisited_text,
            "misalignment_reason": misalignment_reason,
            "timeline_summary": timeline_summary,
        }

        try:
            llm = self._get_llm()
            prompt = ChatPromptTemplate.from_messages([
                ("user", PROCESS_GAP_REDIRECT_PROMPT)
            ])
            chain = prompt | llm
            response = chain.invoke(variables)
            content = getattr(response, "content", str(response))
            parsed = _parse_json_from_text(content)
            return parsed.get("redirect") if parsed else None
        except Exception as e:
            print(f"[RuntimeEngine] ProcessGap redirect generation failed: {e}")
            return None

    def generate_assistance(
        self,
        expert_leaf_id: Optional[str],
        selected_link: Optional[GapLink],
        interview_timeline: Optional[List[Dict[str, Any]]] = None,
        expert_answer: str = "",
        researcher_question: str = "",
    ) -> Assistance:
        """
        Generate type-specific assistance based on the selected link's relation_type.
        For ProcessGap, this is entirely runtime-driven (no offline payload).
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

        # ---- ProcessGap: fully runtime-driven. ----
        elif relation == RelationType.PROCESS_GAP.value:
            timeline = interview_timeline or []

            # --- 1. Coverage Analysis ---
            current_label = expert_node.label if expert_node else ""
            covered_ids = {entry.get("expert_leaf_id", "") for entry in timeline}
            visited_labels = [entry.get("topic_label", "") for entry in timeline]

            siblings = self.graph.expert_tree.get_siblings(
                selected_link.expert_leaf_id
            ) if selected_link else []
            unvisited = [s for s in siblings if s.id not in covered_ids]

            total_siblings = len(siblings) + 1  # include current node
            visited_count = total_siblings - len(unvisited)

            coverage_info = {
                "visited": list(dict.fromkeys(visited_labels)),  # deduplicated, ordered
                "unvisited_siblings": [s.label for s in unvisited[:5]],
                "coverage_ratio": f"{visited_count}/{total_siblings}",
            }

            # --- 2. Drift Detection (3 types) ---
            drift_type = None
            drift_detail = None

            # (a) Repeated Topic — same topic appears ≥2 times
            if current_label:
                topic_count = sum(1 for v in visited_labels if v == current_label)
                if topic_count >= 2:
                    drift_type = "repeated_topic"
                    drift_detail = (
                        f"Topic '{current_label}' has been discussed {topic_count} times. "
                        "The conversation may be circling."
                    )

            # (b) Tunnel Vision — expert stays in one sub-branch too long
            #     AND there are unvisited siblings (if all siblings covered, depth is fine)
            if not drift_type and len(timeline) >= 4 and unvisited:
                recent_ids = [entry.get("expert_leaf_id", "") for entry in timeline[-4:]]
                if len(set(recent_ids)) == 1 and recent_ids[0]:
                    drift_type = "tunnel_vision"
                    drift_detail = (
                        f"The last {len(recent_ids)} turns all discuss the same concept. "
                        "Consider broadening the scope."
                    )

            # --- 3. Runtime Redirect (LLM call only when drift detected) ---
            redirect_text = None
            if drift_type and expert_node:
                redirect_text = self._generate_process_redirect(
                    drift_type=drift_type,
                    drift_detail=drift_detail or "",
                    expert_node=expert_node,
                    expert_answer=expert_answer,
                    unvisited=unvisited,
                    selected_link=selected_link,
                    timeline=timeline,
                )

            payload = {
                "coverage": coverage_info,
                "drift_detected": drift_type is not None,
                "drift_type": drift_type,
                "drift_detail": drift_detail,
                "redirect": redirect_text,
            }
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
            expert_answer=expert_answer,
            researcher_question=researcher_question,
        )

        if analysis.assistance and analysis.assistance.payload:
            # ProcessGap redirect is already LLM-generated with full context — skip polish
            if analysis.assistance.relation_type != RelationType.PROCESS_GAP.value:
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
