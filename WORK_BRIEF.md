# Work Brief

## What I changed
- Added a dedicated right-side `Process Guidance` panel to the UI.
- Added a session-scoped process view model in `demo_flask/app.py` to drive:
  - `Current Focus`
  - `Coverage View`
  - `Recent Topic Trail`
  - `Latest Process Signal`
- Moved interview timeline tracking into browser session storage instead of relying on shared cached DSAG state.
- Updated turn analysis flow so the current turn can contribute to process tracking immediately.
- Added `simu_transcript_process_panel.txt` for repeatable manual testing.

## What we verified
- DSAG init and the new panel render correctly.
- Topic trail and local branch coverage update across turns.
- Repeated-topic behavior is visible in the panel during calibration-focused questioning.

## Main issue found
- `Latest Process Signal` still depends on the latest analyzed message being explicitly typed as `ProcessGap`.
- As a result, repeated topic fixation can be visible in the panel while no process alert appears.

## Most likely next step
- Decouple process drift detection in the right-side panel from the main selected `gap type`.
- Surface low-confidence warnings more clearly when a turn is gated and no assistance card is shown.

## Historical Notes

# DSAG — Work Brief

## 2026-02-22

1. **Type-Specific Assistance Overhaul** — Replaced uniform `BridgeTemplates(coarse/balanced/fine)` with 5 mismatch-type-specific assistance pipelines. Each type (Lexical, Conceptual, Tacit, Scope, Process) now has its own factory prompt, runtime branch, and output shape. Added `interview_timeline` for cross-turn ProcessGap tracking. Rewrote `runtime.py`; touched `schema.py`, `factory.py`, `__init__.py`, `app.py`, `visualize_dsag.py`. New `test_dsag_v2.py`.

## 2026-02-24

2. **Polish Agent Upgrade** — Rewrote `_polish_assistance()` with per-type FROZEN/POLISHABLE field rules so the polish LLM knows exactly which fields to leave untouched vs. make natural. Covers all 5 types. (`runtime.py`)

3. **Edge Type Classification → Agent C** — Moved `relation_type` classification from keyword-matching to Agent C (AlignmentJudge), which has full semantic context. Keyword function kept as fallback. (`schema.py`, `factory.py`)

4. **Keyword Fallback Redesign** — Replaced first-match-wins with counting + weighted scoring (Tier 1 = 1pt, Tier 2 = 2pt). Removed dead conversational keywords; added Agent-C analytical-register keywords. Fixed fallback pairing bug and Agent C emission prompt. (`factory.py`)

5. **5 Prompt Bug Fixes** — Cross-referenced `mismatch_types.md` against prompts. Fixed: ScopeGap definition (granularity → purpose), ProcessGap definition (added factual-error + tunnel-vision triggers), TacitGap `attributes` classification hint, `hypothetical_scenario` → plural array, ConceptualGap missing `explanation` field. (`factory.py`, `runtime.py`)

6. **ConceptualGap Sibling Injection** — Analogy prompt now receives up to 5 expert sibling leaves as grounded source candidates instead of hallucinating. (`factory.py`)

## 2026-02-26

7. **Polish Process Cleanup** — Removed ExpandScope and DeepDive followup_questions from `generate_assistance()` (not in spec). Polish LLM now only polishes `payload`, no longer handles `followup_questions`. Fully removed `followup_questions` field from `Assistance` dataclass, API response, and all test/visualization references. (`runtime.py`, `test_dsag_v2.py`, `test_dsag.py`, `visualize_dsag.py`, `app.py`)

8. **LexicalGap Definition Tightened** — Removed "or related concept" from the LexicalGap definition in `ALIGNMENT_JUDGE_PROMPT`. Now reads "same concept" only, matching `mismatch_types.md` strictly. Prevents ConceptualGap pairs from being misclassified as purely terminological. (`factory.py`)

9. **ALIGNMENT_JUDGE_PROMPT — Examples & Edge Cases** — Added one canonical example per gap type (4 types, ProcessGap excluded) and one boundary edge case per type-pair (6 pairs) to the alignment prompt. Each edge case explicitly names the correct type and states why the alternative does not apply, targeting the four most common misclassification boundaries (Lexical/Conceptual, Conceptual/Tacit, Conceptual/Scope, Tacit/Scope). Also tightened LexicalGap definition: changed "same or related concept" to "same concept" to prevent over-classification of ConceptualGap pairs as Lexical. (`factory.py`)

## 2026-02-27

10. **ProcessGap Runtime-Driven Redesign** — Replaced offline scaffold-based ProcessGap with a fully runtime-driven implementation. Removed `PROCESS_GAP_PROMPT` and offline `expected_steps`/`tunnel_vision_risks` generation. Drift detection now uses real `interview_timeline` data: Repeated Topic (≥2 history occurrences) and Tunnel Vision (4 consecutive same-leaf turns + unvisited siblings). Coverage info is always returned as informational data (no LLM call). On-demand `PROCESS_GAP_REDIRECT_PROMPT` generates a context-aware redirect sentence only when drift is detected. `misalignment_reason` stored in ProcessGap payload for runtime consumption. ProcessGap skips polish step. (`factory.py`, `runtime.py`)

11. **TacitGap Two-Phase Design** — Offline phase now expands Agent A's 2-4 seed attributes to 6-10 by generating additional potential tacit dimensions (decision heuristics, contextual cues, experience thresholds, subconscious variables), then builds probes + scenarios for all of them as an exhaustive arsenal. Runtime polish phase upgraded from simple rephrasing to intelligent filtering: extracts which attributes the expert has already articulated, removes those and their corresponding probes/scenarios, presents only unmentioned remainder. New `mentioned_attributes` output field for transparency. (`factory.py`, `runtime.py`)