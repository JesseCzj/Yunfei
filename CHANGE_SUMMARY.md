# DSAG Type-Specific Assistance — Change Summary

**Date:** 2026-02-22
**Scope:** Replace uniform `BridgeTemplates(coarse, balanced, fine)` with 5 mismatch-type-specific assistance payloads

---

## What Changed

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

---

## Files Modified

| File | Change |
|------|--------|
| `dsag/schema.py` | Removed `BridgeTemplates` class. `GapLink.bridge_templates` + `checkout_template` replaced by `assistance_payload: Dict[str, Any]`. Added `interview_timeline: List[Dict]` to `DSAGState`. |
| `dsag/factory.py` | Replaced single `BRIDGE_TEMPLATE_PROMPT` with 5 type-specific prompts. `generate_bridge_templates_for_link()` renamed to `generate_assistance_payload_for_link()` with per-type dispatch. `generate_all_bridge_templates()` renamed to `generate_all_assistance_payloads()`. |
| `dsag/runtime.py` | Full rewrite. Removed `ExpertConfusionHelp`, `ResearcherConfusionHelp` and their generators. New `Assistance` dataclass. `generate_assistance()` branches on `relation_type`. `analyze_turn()` now accepts `interview_timeline` parameter. |
| `dsag/__init__.py` | Updated exports: removed `BridgeTemplates`, added `Assistance`. |
| `app.py` | Added `import os`. `/api/dsag/analyze_turn` now passes `interview_timeline` and accumulates timeline entries per turn. |
| `visualize_dsag.py` | Updated assistance display to show `relation_type` + `payload` keys instead of old bridge templates. Added `interview_timeline=[]` to `analyze_turn()` call. |
| `test_dsag_v2.py` | **New file.** Standalone test suite (no dependency on `llm_backend.py` or `app.py`). |

### Untouched

- Tree builders (Agent A / Agent B prompts)
- Alignment judge (Agent C prompt)
- `_infer_relation_type()` logic
- `EmbeddingIndex` / `embedding_index.py`
- Frontend (`templates/`, `static/`)
- `llm_backend.py` (legacy, unused)

---

## Type-Specific Assistance Design

### 1. LexicalGap

**Offline payload** (factory prompt → `assistance_payload`):
```json
{
  "term_mapping": {
    "expert_term": "...",
    "researcher_term": "...",
    "explanation": "..."
  }
}
```

**Runtime behavior:**
- Copies offline `term_mapping` directly into `Assistance.payload`
- No follow-up questions (terminology mapping is self-contained)

---

### 2. ConceptualGap

**Offline payload** (factory prompt → `assistance_payload`):
```json
{
  "analogy": {
    "source_concept": "...",
    "structural_mapping": {
      "inputs": "...",
      "logic": "...",
      "outputs": "..."
    }
  },
  "scenario": {
    "inputs": "...",
    "outputs": "...",
    "edge_cases": "..."
  }
}
```

**Runtime behavior:**
- Copies offline `analogy` and `scenario` into payload
- Generates **ExpandScope** follow-up questions from sibling nodes in the researcher tree

---

### 3. TacitGap

**Offline payload** (factory prompt → `assistance_payload`):
```json
{
  "probes": [
    {
      "attribute": "...",
      "question": "...",
      "choices": "..."
    }
  ],
  "hypothetical_scenario": "..."
}
```

**Runtime behavior (3-step sequential):**
1. Merge offline probes with live `node.attributes` from the expert leaf
2. Include hypothetical scenario for step 3 checkout
3. Generate **DeepDive** follow-up questions based on attributes

---

### 4. ScopeGap

**Offline payload** (factory prompt → `assistance_payload`):
```json
{
  "validate_focus": "...",
  "pivot": {
    "limitation": "...",
    "research_goal": "...",
    "compelling_reason": "...",
    "coarse_scenario": "..."
  }
}
```

**Runtime behavior:**
- Copies offline `validate_focus` and `pivot` into payload
- No follow-up questions (scope redirection is a one-shot intervention)

---

### 5. ProcessGap

**Offline payload** (factory prompt → `assistance_payload`):
```json
{
  "initial_topics": ["..."]
}
```

**Runtime behavior:**
- Injects `interview_timeline` from `DSAGState` (accumulated across turns)
- Runs **drift detection**: identifies repeated topics and uncovered sibling nodes
- Payload includes: `timeline`, `drift_alerts`, `current_topic`
- No polish step (timeline data, not prose)

---

## Interview Timeline (New)

`DSAGState.interview_timeline` accumulates one entry per analyzed turn:

```json
{
  "turn_index": 0,
  "topic_label": "...",
  "expert_leaf_id": "exp_leaf_00_01_02",
  "researcher_leaf_id": "res_leaf_00_00_01",
  "relation_type": "TacitGap",
  "summary": "..."
}
```

This is used by ProcessGap's drift detection to identify when the interview revisits the same topics or misses important areas.

---

## Test Results

Ran `test_dsag_v2.py` (3 offline tests, no API calls):

```
=== Testing Schema (v2) ===
  [OK] Node creation with attributes
  [OK] LCA computation
  [OK] GapLink with assistance_payload (LexicalGap)
  [OK] GapLink serialization round-trip
  [OK] GapLink with assistance_payload (TacitGap)
  [OK] GapLink with assistance_payload (ScopeGap)
  [OK] DSAGState with interview_timeline
  Schema v2 tests passed!

=== Testing GraphFactory (v2 structure) ===
  [OK] Expert tree: 7 nodes, 3 leaves
  [OK] Researcher tree: 5 nodes, 2 leaves
  [OK] Intuition node has attributes: ['confidence level', 'patient history completeness']
  [OK] Relation types correctly inferred from reason text
  GraphFactory v2 tests passed!

=== Testing Runtime Type Branching ===
  [OK] LexicalGap: term_mapping present, no follow-ups
  [OK] TacitGap: 1 probes, 2 DeepDive follow-ups
  [OK] ConceptualGap: analogy + scenario present, 2 ExpandScope follow-ups
  [OK] ProcessGap: timeline=1 entries, drift_alerts=2
  [OK] Assistance.to_dict() has correct shape (no old fields)
  Runtime type branching tests passed!

Total: 3/3 offline tests passed
```

Test 4 (full generation with API calls) is available but requires 30-60s for LLM calls. Run with: `python test_dsag_v2.py`

---

## Cache Invalidation

Old cached DSAG graphs (built with `BridgeTemplates`) are **incompatible** with the new structure. The first `/api/dsag/init` call after this change will build a new graph. Old `dsag_output.json` files will fail on `GapLink.from_dict()` if they contain `bridge_templates` keys instead of `assistance_payload`.

---

## Polish Agent Upgrade (2026-02-24)

### Problem

The original `_polish_assistance()` in `runtime.py` had 3 issues:

1. **ProcessGap was blanket-skipped** — `expected_steps[].description`, `tunnel_vision_risks[]`, `drift_alerts[]` are all prose that should be polished, but were returned raw.
2. **ConceptualGap and ScopeGap had no polishing rules** — only LexicalGap (rule #5) and TacitGap (rule #6) had type-specific instructions. The LLM could corrupt `source_concept`, `structural_mapping`, `limitation`, or `research_goal`.
3. **No FROZEN/POLISHABLE field distinction** — the LLM had to guess which fields are structural data vs polishable prose, risking key corruption or fact alteration.

### Solution

Rewrote `_polish_assistance()` with a class-level `_POLISH_RULES_BY_TYPE` dict that provides per-type polishing specifications.

**Design principle**: every field in every type's payload is explicitly classified as either **FROZEN** (copy verbatim) or **POLISHABLE** (make natural, weave in expert's wording).

#### Per-type FROZEN / POLISHABLE designation:

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

### What's NOT changed

- Factory prompts (all 5 type-specific prompts in `factory.py`) — reviewed, confirmed correct against `mismatch_types.md`
- Alignment judge prompt (`ALIGNMENT_JUDGE_PROMPT`) — reviewed, confirmed correct
- `generate_assistance()` runtime branches — no changes
- Schema, embedding index, frontend — no changes

---

## What's Next

- **Frontend**: Update `static/app.js` to render type-specific assistance cards (not yet started)
- **End-to-end test**: Run `python visualize_dsag.py` or the Flask app with real API calls to verify full pipeline
- **Tuning**: Adjust the 5 factory prompts based on real interview data quality