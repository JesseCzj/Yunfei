# Change Summary

## Scope
This round focused on introducing a dedicated `Process Guidance` panel for `ProcessGap`-style session monitoring, while keeping the existing message-level DSAG assistance cards intact.

## Code Changes

### `demo_flask/app.py`
- Added session-scoped interview timeline helpers:
  - `get_interview_timeline()`
  - `set_interview_timeline()`
- Added timeline construction and reuse helpers:
  - `build_timeline_entry()`
  - `analyze_turn_with_dsag()`
  - `get_latest_process_payload()`
  - `build_process_panel_state()`
- Moved ProcessGap tracking state away from the shared cached `DSAGState.interview_timeline` path and into the browser session, so repeated transcript testing is isolated per session.
- Updated turn analysis flow so the current turn is appended to timeline state before the ProcessGap payload is regenerated for that turn.
- Reused the same analysis helper in both:
  - `/api/dsag/analyze_turn`
  - server-rendered chat submission path in `index()`
- Passed a new `process_panel` view model into `render_template(...)`.
- Slightly adjusted cache reuse flow during DSAG init so cached state lookup and session mapping are handled more safely.

### `demo_flask/templates/index.html`
- Added a new right-side `Process Guidance` panel.
- The panel has four display blocks:
  - `Current Focus`
  - `Coverage View`
  - `Recent Topic Trail`
  - `Latest Process Signal`
- Added empty states for:
  - DSAG not initialized
  - DSAG ready but no interview turns yet
- Reused existing clickable follow-up behavior so redirect text in the process panel can still feed the researcher input.
- Kept the existing in-message DSAG assistance cards for cross-checking during testing.

### `demo_flask/static/style.css`
- Expanded the layout from a two-column shell to a three-column shell:
  - left script / setup panel
  - center interview panel
  - right process panel
- Added process-panel-specific styles for:
  - cards
  - pills
  - progress bar
  - topic chips
  - recent topic trail
  - process alert state
- Added responsive handling so the process panel falls below the main content on narrower screens.

### `simu_transcript_process_panel.txt`
- Added a dedicated simulation transcript for manual UI and runtime testing.
- The transcript emphasizes:
  - a clean warm-up phase
  - a repeated-topic phase meant to stress `ProcessGap`
  - suggested fallback prompts if repeated-topic detection is weak

## What The New Panel Currently Does Well
- Tracks the current matched expert topic.
- Shows local branch coverage around that topic.
- Accumulates recent topic history across turns.
- Preserves a clear separation between message-level semantic gaps and session-level process visualization.

## Current Known Limitations

### 1. `Latest Process Signal` is still coupled to `ProcessGap` classification
The right-side panel already shows repeated topic history and branch-local coverage, but its alert block still depends on the most recent message whose `assistance.relation_type == "ProcessGap"`.

Implication:
- You can observe repeated conceptual looping in the trail,
- while `Latest Process Signal` stays empty,
- because the selected link for those turns remains `ConceptualGap`.

### 2. Low-confidence turns are not visible enough in the UI
When expert matching confidence is below `DSAG_MATCH_CONFIDENCE_THRESHOLD`, the backend returns a `confidence_warning` without a normal assistance payload.

Current frontend behavior:
- the DSAG card is rendered only when a non-empty assistance payload exists,
- so low-confidence warnings can become invisible,
- making those turns look like “no analysis happened”.

### 3. Runtime topic grounding can drift toward narrow technical leaves
In the calibration stress test, repeated turns were consistently mapped to:
- `Slice thickness and spacing sensitivity`

This is useful for observing stable topic tracking, but it also shows that runtime matching currently tends to collapse multiple calibration-related utterances into a very specific technical leaf.

## Practical Outcome Of This Round
- The new `Process Guidance` panel is implemented and working.
- Session-level process tracking now updates across turns.
- Manual testing confirmed that topic trail and local coverage are useful.
- Manual testing also confirmed that the current `ProcessGap` alert logic should be decoupled from the main selected gap type in a future revision.
# DSAG — Change Summary

**Scope:** All backend changes from 2026-02-22 to 2026-02-26

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

---

## 2026-02-26 — Polish Process Cleanup: Remove followup_questions

### Problem

The polish LLM was handling `followup_questions` alongside `payload`, which went beyond its intended role of pure text polishing. Additionally, `generate_assistance()` was generating ExpandScope (ConceptualGap) and DeepDive (TacitGap) follow-up questions — artifacts of an earlier design that are not defined in `mismatch_types.md`. The system should follow the spec table exactly; no mismatch type requires follow-up question generation.

### Solution

1. Removed ExpandScope follow-up generation from ConceptualGap branch in `generate_assistance()`
2. Removed DeepDive follow-up generation from TacitGap branch in `generate_assistance()` (attribute merge into payload retained)
3. Removed `followup_questions` references from `_POLISH_RULES_BY_TYPE` for ConceptualGap and TacitGap
4. Polish prompt now only sends and expects `{"payload": ...}` — no longer includes or parses `followup_questions`
5. **Fully removed** `followup_questions` field from `Assistance` dataclass and `to_dict()` — no longer part of the data model or API response
6. Cleaned up all test/visualization references to the removed field

### Files Modified

| File | Change |
|------|--------|
| `dsag/runtime.py` | `generate_assistance()` — removed ExpandScope (ConceptualGap) and DeepDive (TacitGap) followup generation. `_POLISH_RULES_BY_TYPE` — removed `followup_questions[].question` from ConceptualGap and TacitGap POLISHABLE fields. `_polish_assistance()` — prompt output instruction changed from `"payload", "followup_questions"` to `"payload"` only; input JSON no longer includes `followup_questions`; output parsing no longer reads `followup_questions`. `Assistance` dataclass — removed `followup_questions` field and its `to_dict()` serialization. |
| `test_dsag_v2.py` | Removed 6 `followup_questions` assertions (LexicalGap empty check, TacitGap DeepDive check, ConceptualGap ExpandScope check, serialization check). Changed serialization assertion to `"followup_questions" not in assist_dict`. |
| `visualize_dsag.py` | Removed `followup_questions` display block. |
| `test_dsag.py` | Replaced `followup_questions` print with `payload.keys()` print. |
| `app.py` | Updated `/api/dsag/analyze_turn` docstring — removed `followup_questions` from response schema. |

---

## 2026-02-27 — LexicalGap Definition: "same or related" → "same"

### Problem

The LexicalGap definition in `ALIGNMENT_JUDGE_PROMPT` read: *"The two sides use different terms, jargon, or labels for the same **or related** concept."* The phrase "or related concept" is too permissive — pairs where the underlying constructs differ (i.e., ConceptualGap) could be misclassified as purely terminological. `mismatch_types.md` defines LexicalGap strictly as "different words for the **same** concept."

### Fix

Removed "or related" from the definition. New definition: *"The two sides use different terms, jargon, or labels for the same concept."* This aligns exactly with `mismatch_types.md` and keeps the Lexical/Conceptual boundary sharp: if a shared label would not fully close the gap, the pair must be evaluated for ConceptualGap or another type.

### Files Modified

| File | Change |
|------|--------|
| `dsag/factory.py` | `ALIGNMENT_JUDGE_PROMPT` — LexicalGap definition: removed "or related" |

---

## 2026-02-26 -- ALIGNMENT_JUDGE_PROMPT: Examples, Edge Cases & LexicalGap Definition Fix

### Problem

`ALIGNMENT_JUDGE_PROMPT` provided only one-line definitions for each gap type. Agent C had no grounded reference for boundary cases where two types share surface features, leading to misclassification at the four most confusable type boundaries: Lexical/Conceptual, Conceptual/Tacit, Conceptual/Scope, and Tacit/Scope. Additionally, the LexicalGap definition included "or related concept," which was too permissive and invited over-classification of ConceptualGap pairs as LexicalGap.

### Changes

1. **LexicalGap definition tightened** -- Removed "or related concept." New definition: "The two sides use different terms, jargon, or labels for the **same** concept." This prevents pairs where the underlying constructs differ from being misclassified as purely terminological.

2. **One canonical example per gap type added** (ProcessGap excluded -- it is runtime-driven with no offline example value):
   - Each example is written as expert-leaf / researcher-leaf node pairs (label + description), matching the exact input format Agent C receives
   - Each example ends with a one-sentence explanation naming the discriminating condition

3. **One boundary edge case per type-pair added** (6 pairs total: Lexical/Conceptual, Lexical/Tacit, Lexical/Scope, Conceptual/Tacit, Conceptual/Scope, Tacit/Scope):
   - Each edge case shows a node pair that superficially resembles one type but correctly classifies as another
   - Each ends with "X (NOT Y): [reason why Y does not apply]" -- the contrastive phrasing forces Agent C to evaluate the discriminating condition rather than pattern-match to a prototype

### Design Decisions

- Examples and edge cases use the medical/clinical domain (consistent with the system's primary use case and existing prompt examples in `process_gap.md`)
- ProcessGap excluded from examples: its classification depends on dynamic runtime signals (tunnel vision, procedural drift) that cannot be demonstrated with static tree node pairs
- New content is inserted between the existing `Classification hint` and `Return ONLY valid JSON` -- the optimal position for few-shot examples to influence output classification

### Files Modified

| File | Change |
|------|--------|
| `dsag/factory.py` | `ALIGNMENT_JUDGE_PROMPT` -- LexicalGap definition ("same or related" to "same"); added 4 canonical examples + 6 boundary edge cases between classification hint and JSON schema |

---

## 2026-02-27 — ProcessGap: Runtime-Driven Redesign

### Problem

The original ProcessGap implementation had three fundamental issues:

1. **Scaffold was LLM hallucination**: An offline prompt (`PROCESS_GAP_PROMPT`) asked the LLM to *guess* the expert's expected workflow steps (`expected_steps`) and tunnel vision risks (`tunnel_vision_risks`) from node labels alone. If the guess was wrong, all downstream drift detection produced noise.
2. **Alerts without actionable speech**: Drift detection generated plain-text warnings (e.g. "Skipped steps: X. Consider circling back.") but no conversational strategy the researcher could actually speak.
3. **Fake vs real mismatch**: Offline-guessed scaffolds were compared against real conversation data — a fundamentally unreliable basis for drift detection.

### Solution

Replaced the entire ProcessGap pipeline with a **fully runtime-driven** approach. No offline LLM call, no scaffold, no `expected_steps`. All drift detection uses real data (`interview_timeline` + expert tree structure). When drift is detected, a runtime LLM call generates a context-aware redirect sentence referencing the expert's actual words.

#### Offline Changes (`factory.py`)

| Change | Detail |
|--------|--------|
| Removed `PROCESS_GAP_PROMPT` | 37-line prompt replaced with 2-line comment |
| ProcessGap early return | `generate_assistance_payload_for_link()` returns `{"misalignment_reason": reason}` for ProcessGap (no LLM call). Stores Agent C's alignment reason for runtime consumption |
| Removed from `prompt_map` | ProcessGap no longer in the dispatch dict |

#### Runtime Changes (`runtime.py`)

| Change | Detail |
|--------|--------|
| New `PROCESS_GAP_REDIRECT_PROMPT` | Runtime prompt that takes drift context + expert's actual words + unvisited siblings → generates one natural redirect sentence |
| New `_generate_process_redirect()` | Method on `RuntimeEngine`; builds context variables, calls LLM, parses redirect. Only invoked when drift is detected |
| Rewritten ProcessGap branch | `generate_assistance()` ProcessGap branch now: (1) coverage analysis, (2) rule-based drift detection, (3) conditional LLM redirect |
| Updated `generate_assistance()` signature | Added `expert_answer` and `researcher_question` parameters |
| Updated `analyze_turn()` | Passes `expert_answer` and `researcher_question` to `generate_assistance()`. ProcessGap skips `_polish_assistance()` |
| Updated `Assistance` docstring | ProcessGap payload shape: `{coverage, drift_detected, drift_type, drift_detail, redirect}` |
| Updated `_POLISH_RULES_BY_TYPE` | ProcessGap entry replaced with comment (no polish needed) |

#### Drift Detection Rules (2 alert types + 1 info type)

| Type | Trigger | LLM Redirect? |
|------|---------|---------------|
| **Repeated Topic** | `topic_count >= 2` in history (3rd+ discussion) | Yes |
| **Tunnel Vision** | Last 4 turns same `expert_leaf_id` AND unvisited siblings exist | Yes |
| **Coverage Gap** | Always computed; `{visited, unvisited_siblings, coverage_ratio}` | No — info only |

Priority: Repeated Topic > Tunnel Vision. Only one drift type per turn (mutually exclusive). Coverage info always returned regardless of drift.

#### Design Decisions

- **Coverage Gap is info, not alert**: It tells the researcher what's unexplored, not that something is wrong. No LLM redirect to avoid paternalistic over-prompting.
- **Tunnel Vision requires unvisited siblings**: If all siblings are already covered, deep-diving one topic is reasonable (breadth done, doing depth). Only "insufficient breadth + excessive depth" qualifies.
- **`misalignment_reason` via payload**: Other 4 types consume `misalignment_reason` through their offline LLM prompts. ProcessGap stores it in `assistance_payload` for runtime consumption — same data flow pattern, no schema change needed.
- **Skip polish**: The redirect is already generated by LLM with full conversation context. A second LLM call to "polish" it would add latency without value.

### New ProcessGap Payload Structure

```json
{
  "relation_type": "ProcessGap",
  "payload": {
    "coverage": {
      "visited": ["topic_A", "topic_B"],
      "unvisited_siblings": ["topic_C", "topic_D"],
      "coverage_ratio": "2/4"
    },
    "drift_detected": true,
    "drift_type": "repeated_topic",
    "drift_detail": "Topic 'topic_A' has been discussed 2 times. ...",
    "redirect": "You mentioned [expert's words]... I'm curious about [unvisited topic]..."
  }
}
```

### Files Modified

| File | Change |
|------|--------|
| `dsag/factory.py` | Removed `PROCESS_GAP_PROMPT`. ProcessGap returns `{"misalignment_reason": reason}` (no LLM). Removed from `prompt_map`. |
| `dsag/runtime.py` | New `PROCESS_GAP_REDIRECT_PROMPT`. New `_generate_process_redirect()`. Rewritten ProcessGap branch. Updated signature. ProcessGap skips polish. Updated `Assistance` docstring + `_POLISH_RULES_BY_TYPE`. |
| `CLAUDE.md` | Updated runtime description for ProcessGap. |
| `codebase_overview.md` | Updated ProcessGap offline payload shape and runtime description. |

### Untouched

- `app.py` — timeline accumulation logic unchanged
- `schema.py` — `GapLink.assistance_payload` is `Dict[str, Any]`, accepts any shape
- All other gap types — no changes to Lexical, Conceptual, Scope pipelines

---

## 2026-02-27 — TacitGap Two-Phase Design: Exhaustive Offline → Intelligent Runtime Filtering

### Problem

The original TacitGap pipeline had two weaknesses:

1. **Attribute list too narrow**: Agent A generates only 2-4 seed attributes per intuition-based leaf. The `TACIT_GAP_PROMPT` simply used these as-is, producing a small set of probes/scenarios. Many relevant tacit dimensions went unprobed.
2. **Polish was dumb rephrasing**: The runtime polish step only rephrased probe questions for naturalness. It had no awareness of what the expert had already articulated during the live interview, so it would present probes for attributes the expert had already explained — redundant and potentially annoying.

### Solution: Two-Phase Design

| Phase | Before | After |
|-------|--------|-------|
| **Offline (payload generation)** | Use Agent A's 2-4 seed attributes directly | **Expand**: use seed attributes as starting points, generate 6-10 total attributes covering decision heuristics, contextual cues, experience thresholds, subconscious variables. Generate probes + scenarios for ALL of them. |
| **Runtime (polish)** | Simple rephrasing of probe questions | **Intelligent filtering**: extract which attributes the expert has already mentioned → remove those attributes + their probes/scenarios → polish only the remaining unmentioned items |

### Design Rationale

- **Offline = exhaustive arsenal**: One-time LLM call with full context produces higher quality. Generate everything possible upfront.
- **Runtime = smart filtering**: Lightweight step that uses real conversation data to select what's actually needed. No wasted probes on already-surfaced knowledge.
- The polish LLM now outputs a `"mentioned_attributes"` field for transparency — the researcher can see which attributes the system detected as already discussed.

### Touch Points

| File | Change |
|------|--------|
| `dsag/factory.py` | `TACIT_GAP_PROMPT` — Step 1 rewritten from "extract" to "expand" (seed → 6-10 attributes). Steps 2-3 updated to cover all attributes. |
| `dsag/runtime.py` | `_POLISH_RULES_BY_TYPE[TacitGap]` — rewritten from simple rephrasing to 3-phase filtering (extract mentioned → filter → polish remainder). Added `mentioned_attributes` output field. |
| `dsag/runtime.py` | `generate_assistance()` TacitGap branch — removed old attribute-override logic (no longer needed; expanded attributes live in offline payload). |
| `dsag/runtime.py` | `Assistance` docstring — updated TacitGap payload shape to include `mentioned_attributes`. |
| `mismatch_types.md` | Added "Two-Phase Design" section under Tacit Gap. |
| `improvement_goal.md` | Added "Two-Phase Design" section under Tacit Gap. |
