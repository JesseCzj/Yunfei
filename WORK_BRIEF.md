# DSAG — Work Brief

## Historical Notes

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

## 2026-03-07

12. **Process Guidance Panel + Session-Scoped Timeline Tracking** — Added a dedicated right-side `Process Guidance` panel to surface `Current Focus`, `Coverage View`, `Recent Topic Trail`, and `Latest Process Signal`. Introduced a session-scoped process view model in `demo_flask/app.py`, moved interview timeline tracking into browser session storage instead of shared cached DSAG state, updated turn analysis so the current turn contributes to process tracking immediately, and added `simu_transcript_process_panel.txt` for repeatable manual testing. Verified DSAG init + panel rendering, topic trail/local branch coverage updates, and repeated-topic visibility during calibration-focused questioning.

13. **Known Limitation: Process Signal Still Gap-Type-Gated** — `Latest Process Signal` still depends on the latest analyzed message being explicitly typed as `ProcessGap`, so repeated-topic fixation can already be visible in the panel while no process alert appears. Most likely next step: decouple process drift detection in the right-side panel from the main selected `gap type`, and surface low-confidence warnings more clearly when a turn is gated and no assistance card is shown.

14. **Low-Confidence Uncertainty Card (Phase 1)** — Kept the existing `DSAG_MATCH_CONFIDENCE_THRESHOLD` gate unchanged, but replaced the effectively invisible low-confidence warning path with a dedicated `Uncertain Interpretation` UI state. `demo_flask/dsag/runtime.py` now returns a structured `uncertain_interpretation` payload on low-confidence expert matches, including the top-2 expert candidates (`label`, `description`, `score`), a short status line, a note that formal mismatch output is withheld until clarification, and a template-generated interview-style `Suggested Follow-up` based on candidate labels only. `demo_flask/templates/index.html` now renders this as a standalone DSAG card instead of requiring a normal assistance payload, and `demo_flask/static/style.css` adds a side-by-side two-column candidate layout with score chips. Verified Python syntax and Jinja template parsing. Future plan: keep the same card structure, but optionally upgrade the follow-up sentence from template-based generation to a light LLM-polished variant after UI behavior is validated.

## 2026-03-08

15. **Low-Confidence Uncertainty Card (Phase 2): Prompt, Fallback & Polish Fixes** — Rewrote `UNCERTAIN_INTERPRETATION_FOLLOWUPS_PROMPT` from independent per-candidate question generation to contrastive disambiguation: the LLM now first identifies the core semantic distinction between candidates, then generates questions targeting that distinction (with "or neither" to break the closed-set assumption). Stripped `score` from candidates sent to the polish LLM to prevent implicit bias. Rewrote `_fallback_uncertainty_followups()` from yes/no label confirmation ("Are you pointing more to {label}?") to open-ended description probes ("Could you tell me more about how {description} plays into what you're describing?"), unifying fallback behavior with the prompt's own rules. Gate logic unchanged: pure absolute threshold (< 0.45 → uncertainty card, ≥ 0.45 → normal pipeline). (`runtime.py`)

## 2026-03-09

16. **Low-Confidence Timeline Gating** — Low-confidence turns (expert match below `DSAG_MATCH_CONFIDENCE_THRESHOLD`) are now excluded from both the persistent interview timeline (`app.py: build_timeline_entry`) and the synthetic within-turn timeline entry (`runtime.py: analyze_turn`). Prevents unreliable topic mappings from polluting drift detection with phantom topic patterns. Both sites use the same threshold (default 0.45). (`app.py`, `runtime.py`)

17. **TacitGap vs ProcessGap Boundary Strengthening** — Expanded TacitGap definition to explicitly cover methodology-level tacit knowledge (expert's approach is an intuitive blend of multiple frameworks, not a nameable paradigm). Added KEY BOUNDARY to ProcessGap's methodology_conflict: only applies when the expert CAN name their approach; otherwise classify as TacitGap. Added new [TacitGap vs ProcessGap] boundary edge case (therapist blending CBT/psychodynamic/mindfulness → TacitGap, not ProcessGap). Three-point reinforcement: definition → boundary condition → concrete example. (`factory.py`)

18. **Drift Detection Simplified to narrow_focus** — Merged three drift types (repeated_topic, tunnel_vision, topic_oscillation) into a single `narrow_focus` signal. Trigger: recent window (4-6 turns) has low topic diversity (distinct topics < half window size) AND unvisited siblings exist. Removed `drift_type` field from `DriftSignal`, removed type parameter from redirect generation, simplified redirect prompt, removed type badge from UI, updated design doc. The three former types were different names for the same phenomenon with the same researcher action (broaden scope); separating them added complexity without decision value. (`runtime.py`, `app.py`, `index.html`, `style.css`, `mismatch_types.md`)