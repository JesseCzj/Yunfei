# DSAG — Change Summary

**Scope:** All backend changes from 2026-02-22 to 2026-02-24

---

## 2026-02-22 — Type-Specific Assistance Overhaul

### Problem

The original system generated **identical** assistance structure for all 5 mismatch types:

```
bridge_templates: { coarse: "...", balanced: "...", fine: "..." }
checkout_template: "..."
followup_questions: [...]
```

This was a placeholder design. The user's research requires each mismatch type to produce **fundamentally different** assistance, following type-specific rules defined in `mismatch_types.md`.

### Solution

- Removed `BridgeTemplates` class and the coarse/balanced/fine granularity entirely
- Replaced with a flexible `assistance_payload: Dict[str, Any]` on `GapLink` (offline, built by LLM)
- New `Assistance` dataclass in runtime with `relation_type` + `payload` + `followup_questions`
- 5 type-specific LLM prompts in factory (offline graph build)
- 5 type-specific runtime branches (online per-turn analysis)

### Files Modified

| File | Change |
|------|--------|
| `dsag/schema.py` | Removed `BridgeTemplates` class. `GapLink.bridge_templates` + `checkout_template` replaced by `assistance_payload: Dict[str, Any]`. Added `interview_timeline: List[Dict]` to `DSAGState`. |
| `dsag/factory.py` | Replaced single `BRIDGE_TEMPLATE_PROMPT` with 5 type-specific prompts. `generate_bridge_templates_for_link()` renamed to `generate_assistance_payload_for_link()` with per-type dispatch. `generate_all_bridge_templates()` renamed to `generate_all_assistance_payloads()`. |
| `dsag/runtime.py` | Full rewrite. Removed `ExpertConfusionHelp`, `ResearcherConfusionHelp` and their generators. New `Assistance` dataclass. `generate_assistance()` branches on `relation_type`. `analyze_turn()` now accepts `interview_timeline` parameter. |
| `dsag/__init__.py` | Updated exports: removed `BridgeTemplates`, added `Assistance`. |
| `app.py` | Added `import os`. `/api/dsag/analyze_turn` now passes `interview_timeline` and accumulates timeline entries per turn. |
| `visualize_dsag.py` | Updated assistance display to show `relation_type` + `payload` keys instead of old bridge templates. Added `interview_timeline=[]` to `analyze_turn()` call. |
| `test_dsag_v2.py` | **New file.** Standalone test suite (no dependency on `llm_backend.py` or `app.py`). |

### Type-Specific Assistance Design

**1. LexicalGap** — Offline: `{term_mapping: {expert_term, researcher_term, explanation}}`. Runtime: copy directly, no follow-ups.

**2. ConceptualGap** — Offline: `{analogy: {source_concept, structural_mapping, explanation}, scenario: {inputs, outputs, edge_cases}}`. Runtime: copy + ExpandScope follow-ups from sibling nodes.

**3. TacitGap** — Offline: `{probes: [{attribute, question, choices}], hypothetical_scenarios: [...]}`. Runtime: merge probes with live `node.attributes`, DeepDive follow-ups.

**4. ScopeGap** — Offline: `{validate_focus, pivot: {limitation, research_goal, compelling_reason, coarse_scenario}}`. Runtime: copy directly, no follow-ups.

**5. ProcessGap** — Offline: `{initial_topics: [...]}`. Runtime: inject interview timeline, drift detection, `{timeline, drift_alerts, current_topic}`.

### Interview Timeline

`DSAGState.interview_timeline` accumulates one entry per analyzed turn:
```json
{"turn_index": 0, "topic_label": "...", "expert_leaf_id": "...", "researcher_leaf_id": "...", "relation_type": "TacitGap", "summary": "..."}
```
Used by ProcessGap's drift detection to identify repeated topics or missed areas.

### Untouched

- Tree builders (Agent A / Agent B prompts)
- Alignment judge (Agent C prompt)
- `_infer_relation_type()` logic
- `EmbeddingIndex` / `embedding_index.py`
- Frontend (`templates/`, `static/`)
- `llm_backend.py` (legacy, unused)

---

## 2026-02-24 — Polish Agent Upgrade

### Problem

The original `_polish_assistance()` in `runtime.py` had 3 issues:

1. **ProcessGap was blanket-skipped** — `expected_steps[].description`, `tunnel_vision_risks[]`, `drift_alerts[]` are all prose that should be polished, but were returned raw.
2. **ConceptualGap and ScopeGap had no polishing rules** — only LexicalGap and TacitGap had type-specific instructions. The LLM could corrupt structural fields.
3. **No FROZEN/POLISHABLE field distinction** — the LLM had to guess which fields are structural data vs polishable prose, risking key corruption or fact alteration.

### Solution

Rewrote `_polish_assistance()` with a class-level `_POLISH_RULES_BY_TYPE` dict. Every field in every type's payload is explicitly classified as either **FROZEN** (copy verbatim) or **POLISHABLE** (make natural, weave in expert's wording).

| Type | FROZEN fields | POLISHABLE fields |
|------|--------------|-------------------|
| **LexicalGap** | `term_mapping.expert_term`, `term_mapping.researcher_term` | `term_mapping.explanation` |
| **ConceptualGap** | `analogy.source_concept`, `analogy.structural_mapping.*`, `scenario.*` | `analogy.explanation`, `followup_questions[].question` |
| **TacitGap** | `attributes`, `probes[].attribute`, `probes[].choices`, `hypothetical_scenarios` | `probes[].question`, `followup_questions[].question` |
| **ScopeGap** | `pivot.limitation`, `pivot.research_goal` | `validate_focus`, `pivot.compelling_reason`, `pivot.coarse_scenario` |
| **ProcessGap** | `timeline`, `current_topic`, `expected_steps[].order`, `expected_steps[].label` | `expected_steps[].description`, `tunnel_vision_risks[]`, `drift_alerts[]` |

### Files Modified

| File | Change |
|------|--------|
| `dsag/runtime.py` | Removed ProcessGap blanket skip. Added `_POLISH_RULES_BY_TYPE` class dict with 5 type-specific rule blocks. Rewrote polish prompt to include general rules (6 universal) + type-specific rules injected via `{type_specific_rules}` variable. |

---

## 2026-02-24 — Edge Type Classification: Keyword Matching → Agent C

### Problem

Keyword matching (`_infer_relation_type`) suffered from:
1. **Order bias** — first-match wins regardless of which type is more prominent.
2. **Multi-keyword sentences** — a reason containing both lexical and conceptual cues is always mis-typed as Lexical.

### Solution

Agent C now emits `relation_type` alongside `reason` in the same JSON response for each misaligned pair. Agent C has full semantic context (both trees + the reason it just wrote), making it far better suited to pick the correct gap type. The keyword function is kept as a fallback.

### Files Modified

| File | Change |
|------|--------|
| `schema.py` | Added `relation_type: Optional[str] = None` to `NodeAlignment` |
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` — added `relation_type` field + 5-type definitions to JSON schema |
| `factory.py` | `_parse_alignments` — reads and validates `relation_type` from Agent C response |
| `factory.py` | GapLink builder — uses `leaf_align.relation_type or _infer_relation_type(reason)` |

---

## 2026-02-24 — Keyword Fallback Classifier: Counting + Two-Tier Redesign

### Problem

The previous keyword fallback classifier used **first-match-wins** with fixed order (Lexical, Conceptual, Tacit, Process, Scope). The keyword lists also mixed conversational phrases (which Agent C never generates) with over-generic single words.

### Solution: Counting + Weighted Scoring

- Multi-word phrases = 2 points (Tier 2, high precision), single words = 1 point (Tier 1, broad recall).
- All 5 categories scored in parallel — no ordering bias.
- Highest total score wins. Tie defaults to ConceptualGap.
- All keywords redesigned to match Agent C analytical register. Conversational phrases removed.

| Category | Tier 1 | Tier 2 | Total |
|----------|--------|--------|-------|
| Lexical | 14 | 10 | 24 |
| Conceptual | 11 | 11 | 22 |
| Tacit | 15 | 13 | 28 |
| Scope | 16 | 17 | 33 |
| Process | 10 | 11 | 21 |

### Bug Fixes in Same Pass

- **Fallback pairing `relation_type` not set**: `NodeAlignment` constructor now includes `relation_type=gap_type`.
- **Agent C `relation_type` emission**: Prompt changed from "omit or null if aligned" to "you MUST set" + "REQUIRED when is_aligned=false".

### Files Modified

| File | Change |
|------|--------|
| `factory.py` | `_infer_relation_type()` — counting mechanism + two-tier Agent-C keyword redesign |
| `factory.py` | Fallback pairing `NodeAlignment` — added `relation_type=gap_type` |
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` — strengthened `relation_type` requirement |

---

## 2026-02-24 — Prompt Bug Fixes (5 fixes)

Cross-referenced `mismatch_types.md` and `improvement_goal.md` against the actual prompts in `factory.py`. Found 5 bugs; all fixed.

### Fix 1: ScopeGap definition in ALIGNMENT_JUDGE_PROMPT

**Bug:** Definition said "differ in focus boundary — one is too detailed, too high-level, or out of scope." This describes a *granularity* mismatch.

**Spec says:** The expert focuses on practical utility while the researcher focuses on research value — a *purpose/expectation* mismatch about "what to do."

**Fix:** Rewrote to: "The two sides differ in purpose or expectations — the expert focuses on practical utility while the researcher focuses on research value, leading to inconsistent goals about 'what to do.'"

### Fix 2: ProcessGap definition in ALIGNMENT_JUDGE_PROMPT

**Bug:** Definition only mentioned "workflow, sequence, or procedure — how steps are ordered or executed."

**Spec says:** ProcessGap is also triggered by **factual errors** disrupting the discussion and **narrow expert narratives (tunnel vision)**.

**Fix:** Rewrote to: "Factual errors disrupt the discussion, or the expert lacks standardized procedures or falls into narrow narratives (tunnel vision), causing workflow/sequence misalignment."

### Fix 3: TacitGap classification hint for `attributes`

**Bug:** Agent A marks intuition-based expert leaves with `attributes`. The ALIGNMENT_JUDGE_PROMPT never told Agent C what `attributes` means or how to use it.

**Fix:** Added classification hint: "Expert leaf nodes that contain a non-empty 'attributes' list indicate tacit, intuition-based knowledge. When such a leaf is part of a misaligned pair, strongly prefer TacitGap."

### Fix 4: TacitGap bridge prompt — singular → plural

**Bug:** Prompt generated ONE `hypothetical_scenario` (string). Spec says "Come up with **some** hypothetical scenarios."

**Fix:** Changed to 2-3 `hypothetical_scenarios` (array). Updated `runtime.py` Assistance docstring.

### Fix 5: ConceptualGap bridge prompt — missing `explanation` field

**Bug:** Prompt says "Explain how the researcher's concept behaves similarly" but JSON schema had no field for it.

**Fix:** Added `"explanation"` field to analogy object. Updated `runtime.py` Assistance docstring.

### Files Modified

| File | Change |
|------|--------|
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` — rewrote ScopeGap definition (Fix 1), ProcessGap definition (Fix 2), added TacitGap classification hint (Fix 3) |
| `factory.py` | `TACIT_GAP_PROMPT` — `hypothetical_scenario` string → `hypothetical_scenarios` array (Fix 4) |
| `factory.py` | `CONCEPTUAL_GAP_PROMPT` — added `explanation` field to analogy output (Fix 5) |
| `runtime.py` | `Assistance` docstring — updated payload shapes for ConceptualGap and TacitGap (Fix 4 & 5) |

---

## 2026-02-24 — ConceptualGap: Expert Sibling Injection

### Problem

`CONCEPTUAL_GAP_PROMPT` Strategy 1 (Analogy Construction) tells the LLM "Identify a Sibling Concept the expert already knows" but provides no candidates. The LLM must hallucinate what the expert might know.

### Solution

Added `{expert_siblings}` field to `CONCEPTUAL_GAP_PROMPT` — up to 5 sibling leaves from the expert tree under the same L2 parent. Framed as **candidates, not constraints** (LLM can still pick a better concept from outside the tree).

### Files Modified

| File | Change |
|------|--------|
| `factory.py` | `CONCEPTUAL_GAP_PROMPT` — added `{expert_siblings}` field + rewrote Strategy 1 instruction |
| `factory.py` | `generate_assistance_payload_for_link()` — collect siblings, build text, inject into variables |
