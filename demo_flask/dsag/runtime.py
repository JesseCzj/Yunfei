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
    - ConceptualGap: {"analogy": {"source_concept", "structural_mapping": {inputs, logic, outputs},
                      "explanation"}, "scenario": {"inputs", "outputs", "edge_cases"}}
    - TacitGap:      {"attributes": [...], "probes": [{attribute, question, choices}],
                      "hypothetical_scenarios": ["...", "..."],
                      "extracted_attributes": [...], "mentioned_attributes": [...]}  // extracted/mentioned attrs are displayed for transparency
    - ScopeGap:      {"validate_focus": "...", "pivot": {limitation, research_goal,
                      compelling_reason, coarse_scenario}}
    - ProcessGap:    Offline payload only ({"misalignment_reason": "..."}).
                      Drift detection is handled separately via DriftSignal.
    """
    relation_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "payload": self.payload,
        }


@dataclass
class DriftSignal:
    """Session-level drift detection result, independent of gap type.

    Computed every turn from interview_timeline + expert tree structure.
    """
    coverage: Dict[str, Any] = field(default_factory=dict)
    drift_detected: bool = False
    drift_type: Optional[str] = None
    drift_detail: Optional[str] = None
    redirect: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage": self.coverage,
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type,
            "drift_detail": self.drift_detail,
            "redirect": self.redirect,
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
    drift_signal: Optional[DriftSignal] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "located": self.located.to_dict(),
            "confidence_warning": self.confidence_warning,
            "uncertain_interpretation": self.uncertain_interpretation,
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
        if self.drift_signal:
            result["drift_signal"] = self.drift_signal.to_dict()
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
The offline payload contains an EXHAUSTIVE arsenal of attributes, probes, and scenarios.
Your primary job is INTELLIGENT FILTERING, then polishing.

### Step A — Extract already-mentioned attributes
Read the expert's latest answer and the conversation context carefully.
Identify which attributes from the "attributes" list the expert has ALREADY
articulated, explained, or demonstrated knowledge of (even partially or
using different words). List them internally.

### Step B — Filter
REMOVE from the output:
  - Any attribute that the expert has already mentioned/articulated.
  - The corresponding probe(s) for those removed attributes.
  - Any hypothetical_scenario whose altered variable maps to an already-mentioned attribute.
Keep ONLY the attributes (and their probes/scenarios) that the expert
has NOT yet surfaced.

### Step C — Polish the remaining items
FROZEN fields (copy verbatim — do NOT alter):
  - The remaining attribute names
  - probes[].attribute
  - probes[].choices (the option labels must stay exactly as-is)

POLISHABLE fields:
  - probes[].question — rephrase for naturalness, but it MUST remain a
    multiple-choice question (NEVER convert to open-ended "Why …?").
    Reference the expert's recent answer where possible.

### Output requirements
- The 3-step structure (attributes → probes → hypothetical_scenarios) is strict.
- If ALL attributes have been mentioned, return empty lists for all three fields.
- Add a top-level field "extracted_attributes": [...] listing the attributes
  you identified as already articulated by the expert (for transparency).
- Also include "mentioned_attributes": [...] with the same content for backward compatibility.
- The extracted/mentioned attributes are DISPLAY-ONLY. Do NOT generate probes or
  hypothetical_scenarios for them.""",

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
            if assistance.relation_type == RelationType.TACIT_GAP.value:
                fallback = Assistance(relation_type=assistance.relation_type)
                fallback_payload = dict(assistance.payload)
                fallback_payload["_expert_answer_hint"] = expert_answer
                fallback.payload = self._postprocess_tacit_payload(fallback_payload)
                return fallback
            return assistance

        polished = Assistance(relation_type=assistance.relation_type)
        polished.payload = parsed.get("payload", assistance.payload)
        if polished.relation_type == RelationType.TACIT_GAP.value:
            polished.payload["_expert_answer_hint"] = expert_answer
            polished.payload = self._postprocess_tacit_payload(polished.payload)
        return polished

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

        attributes = _norm_list(payload.get("attributes", []))
        if isinstance(payload.get("probes", []), list):
            for probe in payload.get("probes", []):
                if isinstance(probe, dict):
                    probe_attr = str(probe.get("attribute", "")).strip()
                    if probe_attr and probe_attr not in attributes:
                        attributes.append(probe_attr)

        extracted = _norm_list(
            payload.get("extracted_attributes", payload.get("mentioned_attributes", []))
        )
        expert_answer = str(payload.get("_expert_answer_hint", "")).strip().lower()
        if not extracted and expert_answer and attributes:
            for attr in attributes:
                attr_lower = attr.lower().strip()
                if not attr_lower:
                    continue
                if attr_lower in expert_answer:
                    extracted.append(attr)
                    continue
                tokens = [t for t in re.findall(r"[a-z0-9]+", attr_lower) if len(t) >= 4]
                if not tokens:
                    continue
                hit_count = sum(1 for t in tokens if t in expert_answer)
                if (len(tokens) == 1 and hit_count == 1) or (
                    len(tokens) > 1 and hit_count >= max(1, len(tokens) - 1)
                ):
                    extracted.append(attr)
        extracted_lower = {a.lower() for a in extracted}

        # Keep only not-yet-mentioned attributes in the actionable list.
        remaining_attrs = [a for a in attributes if a.lower() not in extracted_lower]

        # Remove probes targeting extracted attributes.
        probes_in = payload.get("probes", [])
        probes_out: List[Dict[str, Any]] = []
        if isinstance(probes_in, list):
            for probe in probes_in:
                if not isinstance(probe, dict):
                    continue
                attr = str(probe.get("attribute", "")).strip()
                if attr and attr.lower() in extracted_lower:
                    continue
                probes_out.append(probe)

        # Remove scenarios that mention extracted attributes (best-effort text filter).
        scenarios_in = _norm_list(payload.get("hypothetical_scenarios", []))
        scenarios_out: List[str] = []
        for scenario in scenarios_in:
            lower_scenario = scenario.lower()
            if any(attr in lower_scenario for attr in extracted_lower):
                continue
            scenarios_out.append(scenario)

        payload["attributes"] = remaining_attrs
        payload["probes"] = probes_out
        payload["hypothetical_scenarios"] = scenarios_out
        payload["extracted_attributes"] = extracted
        # Keep old field for compatibility with existing clients/docs.
        payload["mentioned_attributes"] = extracted
        payload.pop("_expert_answer_hint", None)
        return payload

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
            "note": "No mismatch type is shown until the concept is clarified.",
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

    def _detect_drift(
        self,
        expert_leaf_id: Optional[str],
        interview_timeline: List[Dict[str, Any]],
        expert_answer: str = "",
        researcher_question: str = "",
        selected_link: Optional[GapLink] = None,
    ) -> DriftSignal:
        """Detect conversation drift from timeline patterns.

        Runs every turn, independent of gap type.  Uses the expert tree's
        sibling structure to compute coverage and detect narrow-focus patterns.
        """
        signal = DriftSignal()
        timeline = interview_timeline or []

        if not expert_leaf_id or not timeline:
            return signal

        expert_node = self.graph.expert_tree.get_node(expert_leaf_id)
        if not expert_node:
            return signal

        # --- Coverage Analysis ---
        current_label = expert_node.label
        covered_ids = {entry.get("expert_leaf_id", "") for entry in timeline}
        visited_labels = [entry.get("topic_label", "") for entry in timeline]

        siblings = self.graph.expert_tree.get_siblings(expert_leaf_id)
        unvisited = [s for s in siblings if s.id not in covered_ids]

        total_siblings = len(siblings) + 1
        visited_count = total_siblings - len(unvisited)

        signal.coverage = {
            "visited": list(dict.fromkeys(visited_labels)),
            "unvisited_siblings": [s.label for s in unvisited[:5]],
            "coverage_ratio": f"{visited_count}/{total_siblings}",
        }

        # --- Drift Detection ---

        # (a) Repeated Topic — same topic appears ≥2 times across full history
        if current_label:
            topic_count = sum(1 for v in visited_labels if v == current_label)
            if topic_count >= 2:
                signal.drift_detected = True
                signal.drift_type = "repeated_topic"
                signal.drift_detail = (
                    f"Topic '{current_label}' has been discussed {topic_count} times. "
                    "The conversation may be circling."
                )

        # (b) Tunnel Vision / Topic Oscillation — sliding window
        if not signal.drift_detected and len(timeline) >= 3 and unvisited:
            window_size = min(len(timeline), 6)
            recent_ids = [
                entry.get("expert_leaf_id", "")
                for entry in timeline[-window_size:]
                if entry.get("expert_leaf_id")
            ]
            distinct = set(recent_ids)

            if len(recent_ids) >= 3 and len(distinct) == 1 and next(iter(distinct)):
                signal.drift_detected = True
                signal.drift_type = "tunnel_vision"
                signal.drift_detail = (
                    f"The last {len(recent_ids)} turns all focus on the same concept. "
                    f"{len(unvisited)} sibling topics remain unexplored. "
                    "Consider broadening the scope."
                )
            elif len(recent_ids) >= 4 and len(distinct) == 2:
                labels = []
                for eid in distinct:
                    node = self.graph.expert_tree.get_node(eid)
                    if node:
                        labels.append(f"'{node.label}'")
                signal.drift_detected = True
                signal.drift_type = "topic_oscillation"
                signal.drift_detail = (
                    f"The last {len(recent_ids)} turns alternate between "
                    f"{' and '.join(labels)}. "
                    f"{len(unvisited)} related topics remain unexplored."
                )

        # --- Redirect Generation (LLM call only when drift detected) ---
        if signal.drift_detected and expert_node:
            signal.redirect = self._generate_process_redirect(
                drift_type=signal.drift_type or "",
                drift_detail=signal.drift_detail or "",
                expert_node=expert_node,
                expert_answer=expert_answer,
                unvisited=unvisited,
                selected_link=selected_link,
                timeline=timeline,
            )

        return signal

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

        relation = selected_link.relation_type
        payload = dict(selected_link.assistance_payload)
        assistance = Assistance(relation_type=relation, payload=payload)

        # ---- LexicalGap ----
        if relation == RelationType.LEXICAL_GAP.value:
            pass

        # ---- ConceptualGap ----
        elif relation == RelationType.CONCEPTUAL_GAP.value:
            pass

        # ---- TacitGap ----
        elif relation == RelationType.TACIT_GAP.value:
            assistance.payload = payload

        # ---- ScopeGap ----
        elif relation == RelationType.SCOPE_GAP.value:
            pass

        # ---- ProcessGap: offline payload only (to be redesigned) ----
        elif relation == RelationType.PROCESS_GAP.value:
            pass

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

        Drift detection runs every turn regardless of confidence or gap type.
        """
        analysis = RuntimeAnalysis()

        # 1. Locate positions
        analysis.located = self.locate_positions(researcher_question, expert_answer)

        # Confidence gate
        try:
            confidence_threshold = float(os.getenv("DSAG_MATCH_CONFIDENCE_THRESHOLD", "0.45"))
        except Exception:
            confidence_threshold = 0.45

        if analysis.located.expert_confidence < confidence_threshold:
            analysis.confidence_warning = (
                "Low-confidence expert match; showing the top possible expert concepts instead."
            )
            analysis.uncertain_interpretation = self._build_uncertain_interpretation(
                analysis.located
            )
            analysis.uncertain_interpretation = self._polish_uncertain_interpretation(
                analysis.uncertain_interpretation,
                researcher_question,
                expert_answer,
                context_summary,
            )
        else:
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

        # 5. Drift detection — runs every turn, independent of gap type.
        #    Build an extended timeline that includes the current turn so
        #    coverage and pattern detection account for the latest data.
        timeline = list(interview_timeline or [])
        expert_leaf_id = analysis.located.best_expert_leaf_id
        if expert_leaf_id:
            current_node = self.graph.expert_tree.get_node(expert_leaf_id)
            timeline_with_current = timeline + [{
                "turn_index": len(timeline) + 1,
                "topic_label": current_node.label if current_node else "",
                "expert_leaf_id": expert_leaf_id,
            }]
        else:
            timeline_with_current = timeline

        analysis.drift_signal = self._detect_drift(
            expert_leaf_id=expert_leaf_id,
            interview_timeline=timeline_with_current,
            expert_answer=expert_answer,
            researcher_question=researcher_question,
            selected_link=analysis.selected_link,
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
