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
        model = os.getenv("OPENAI_MODEL_RUNTIME", "qwen3-max-2026-01-23")
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
                      "explanation"}, "scenario": "single prose paragraph (situation → researcher lens → tension point, max 60 words)"}
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

    @staticmethod
    def _has_non_empty_payload(payload: Optional[Dict[str, Any]]) -> bool:
        """Treat payload as usable only when it has at least one non-empty value."""
        if not isinstance(payload, dict) or not payload:
            return False
        return any(value not in (None, "", [], {}) for value in payload.values())

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
  - analogy.source_concept — adapt to the terminology the expert actually used
  - scenario — adapt the third sentence (tension beat) to reference the specific situation
    discussed. Keep the three-sentence structure (situation → researcher lens → tension)
    and stay within 60 words total.

STRUCTURAL fields (keep the structure, but you may lightly rephrase for clarity):
  - analogy.structural_mapping.inputs
  - analogy.structural_mapping.logic
  - analogy.structural_mapping.outputs

POLISHABLE fields:
  - analogy.explanation — make it a fluent sentence the researcher could speak
    naturally. Reference the expert's actual vocabulary from their answer.
    Connect it to what was just discussed in the Q&A exchange.

IMPORTANT: Adapt the entire assistance so it feels written specifically for this
conversation moment, not as a generic pre-written template. Use the expert's own
words and examples wherever possible.

The analogy and scenario are PARALLEL strategies (no required order).
Do not merge them or remove either one.""",

        RelationType.TACIT_GAP.value: """## TacitGap polishing rules
The offline payload contains an EXHAUSTIVE arsenal of attributes, probes, and scenarios.
Your primary job is INTELLIGENT FILTERING, then polishing to match the live conversation.

### Step A — Extract already-mentioned attributes
Read the expert's latest answer and the conversation context carefully.
Identify which attributes from the "attributes" list the expert has ALREADY
articulated, explained, or demonstrated knowledge of (even partially or
using different words). List them internally.

### Step B — Filter
REMOVE from the output:
  - Any attribute that the expert has already mentioned/articulated.
  - The corresponding probe(s) for those removed attributes.
  - Any hypothetical_scenario whose targeted attribute maps to an already-mentioned attribute.
Keep ONLY the attributes (and their probes/scenarios) that the expert
has NOT yet surfaced.

NOTE: hypothetical_scenarios are objects with "scenario" (text) and "attribute" (targeted attribute name) fields.

### Step C — Polish the remaining items to match the conversation
FROZEN fields (copy verbatim — do NOT alter):
  - probes[].attribute
  - probes[].choices (the option labels must stay exactly as-is)

CONTEXTUALIZABLE fields (rephrase to match the expert's actual wording):
  - The remaining attribute names — you may lightly rephrase to use the
    expert's own terminology while preserving the core meaning
  - hypothetical_scenarios[].scenario — adapt the scenario text so it
    references the expert's actual situation, examples, or domain context
    from the conversation. The scenario should feel like it was written
    for THIS specific expert, not as a generic template.

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
  hypothetical_scenarios for them.
- IMPORTANT: The entire output should feel grounded in the actual interview
  conversation, not like a generic pre-written template.""",

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
    the expert's actual words from the conversation.""",

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

## General rules (apply to ALL types)
1. Keep the EXACT same JSON keys and nesting structure.
2. Make polishable text natural, concise, and directly speakable.
3. Reference the expert's OWN vocabulary and phrasing where possible.
4. Do NOT invent new facts or concepts.
5. Questions MUST remain questions.
6. Output ONLY valid JSON with top-level key "payload" (same structure as the input).
7. CRITICAL: Adapt the assistance content to directly reference and connect
   with the actual Q&A exchange above. The assistance should feel like it was
   written specifically for THIS conversation moment — not a generic template.
   Use the expert's own words, examples, and domain language wherever possible.

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
        # Scenarios may be plain strings (legacy) or dicts with {scenario, attribute}.
        raw_scenarios = payload.get("hypothetical_scenarios", [])
        if not isinstance(raw_scenarios, list):
            raw_scenarios = []
        scenarios_out: List[Dict[str, str]] = []
        for sc in raw_scenarios:
            if isinstance(sc, str):
                # Legacy plain-string format: wrap into dict
                sc_obj: Dict[str, str] = {"scenario": sc, "attribute": ""}
            elif isinstance(sc, dict):
                sc_obj = {
                    "scenario": str(sc.get("scenario", "")),
                    "attribute": str(sc.get("attribute", "")),
                }
            else:
                continue
            lower_scenario = sc_obj["scenario"].lower()
            sc_attr_lower = sc_obj["attribute"].lower().strip()
            # Filter out scenarios whose targeted attribute was already extracted
            if sc_attr_lower and sc_attr_lower in extracted_lower:
                continue
            if any(attr in lower_scenario for attr in extracted_lower):
                continue
            # If attribute is missing, try to infer from remaining attributes
            if not sc_obj["attribute"].strip() and remaining_attrs:
                best_match = ""
                best_hits = 0
                for attr in remaining_attrs:
                    tokens = [t for t in re.findall(r"[a-z0-9]+", attr.lower()) if len(t) >= 3]
                    if not tokens:
                        continue
                    hits = sum(1 for t in tokens if t in lower_scenario)
                    if hits > best_hits:
                        best_hits = hits
                        best_match = attr
                if best_match and best_hits >= 1:
                    sc_obj["attribute"] = best_match
            scenarios_out.append(sc_obj)

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
        Find the best link connecting the located positions.
        Falls back to finding any link from the expert leaf.
        """
        expert_leaf_id = located.best_expert_leaf_id
        researcher_leaf_id = located.best_researcher_leaf_id
        if not expert_leaf_id:
            return None

        # First try exact match
        if researcher_leaf_id:
            link = self.graph.get_link(expert_leaf_id, researcher_leaf_id)
            if link:
                return self._ensure_actionable_link(link)

        # Fallback: find any link from expert leaf (prefer links tied to top researcher candidates)
        links = self.graph.get_links_by_expert_leaf(expert_leaf_id)
        if links:
            top_researcher_ids = [r.node_id for r in located.researcher_results[:3]]
            preferred = [l for l in links if l.researcher_leaf_id in top_researcher_ids]
            pool = preferred if preferred else links
            pool.sort(key=lambda l: l.weight, reverse=True)
            return self._ensure_actionable_link(pool[0])

        # Runtime hard guarantee requested by product logic:
        # if expert leaf is matched, force-create a minimal actionable link.
        return self._build_runtime_fallback_link(located)

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

        # ---- ProcessGap: offline prevention payload (factual_risk / methodology_conflict) ----
        elif relation == RelationType.PROCESS_GAP.value:
            assistance.payload = payload

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

        # 1. Locate positions
        analysis.located = self.locate_positions(researcher_question, expert_answer)

        # Confidence gate
        try:
            confidence_threshold = float(os.getenv("DSAG_MATCH_CONFIDENCE_THRESHOLD", "0.45"))
        except Exception:
            confidence_threshold = 0.45

        # Product rule: "No mismatch" is shown when expert leaf cannot be matched.
        # A matched expert leaf must always map to an actionable link.
        if not analysis.located.best_expert_leaf_id:
            analysis.confidence_warning = (
                "No expert concept was matched for this turn; showing the no-mismatch fallback."
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
            if analysis.located.expert_confidence < confidence_threshold:
                analysis.confidence_warning = (
                    "Low-confidence expert match; proceeding with conservative link selection."
                )
            # 2. Find best link
            analysis.selected_link = self.find_best_link(
                analysis.located,
            )

            if analysis.selected_link:
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

        return analysis


# ============== Convenience Function ==============

def analyze_turn(
    graph: DSAGGraph,
    embedding_index: EmbeddingIndex,
    researcher_question: str,
    expert_answer: str,
    context_summary: str = "",
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
    )
    return analysis.to_dict()
