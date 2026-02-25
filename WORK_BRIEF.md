# DSAG — Work Brief

## 2026-02-22

1. **Type-Specific Assistance Overhaul** — Replaced uniform `BridgeTemplates(coarse/balanced/fine)` with 5 mismatch-type-specific assistance pipelines. Each type (Lexical, Conceptual, Tacit, Scope, Process) now has its own factory prompt, runtime branch, and output shape. Added `interview_timeline` for cross-turn ProcessGap tracking. Rewrote `runtime.py`; touched `schema.py`, `factory.py`, `__init__.py`, `app.py`, `visualize_dsag.py`. New `test_dsag_v2.py`.

## 2026-02-24

2. **Polish Agent Upgrade** — Rewrote `_polish_assistance()` with per-type FROZEN/POLISHABLE field rules so the polish LLM knows exactly which fields to leave untouched vs. make natural. Covers all 5 types. (`runtime.py`)

3. **Edge Type Classification → Agent C** — Moved `relation_type` classification from keyword-matching to Agent C (AlignmentJudge), which has full semantic context. Keyword function kept as fallback. (`schema.py`, `factory.py`)

4. **Keyword Fallback Redesign** — Replaced first-match-wins with counting + weighted scoring (Tier 1 = 1pt, Tier 2 = 2pt). Removed dead conversational keywords; added Agent-C analytical-register keywords. Fixed fallback pairing bug and Agent C emission prompt. (`factory.py`)

5. **5 Prompt Bug Fixes** — Cross-referenced `mismatch_types.md` against prompts. Fixed: ScopeGap definition (granularity → purpose), ProcessGap definition (added factual-error + tunnel-vision triggers), TacitGap `attributes` classification hint, `hypothetical_scenario` → plural array, ConceptualGap missing `explanation` field. (`factory.py`, `runtime.py`)

6. **ConceptualGap Sibling Injection** — Analogy prompt now receives up to 5 expert sibling leaves as grounded source candidates instead of hallucinating. (`factory.py`)
