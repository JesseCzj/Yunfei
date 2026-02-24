# Edge Type Classification: Keyword Matching → Agent C

## What changed

GapLink `relation_type` is now classified by **Agent C** (AlignmentJudge) instead of the keyword-matching function `_infer_relation_type`.

Agent C emits `relation_type` alongside `reason` in the same JSON response for each misaligned pair. The keyword function is kept as a fallback in case Agent C omits the field.

## Why

Keyword matching suffers from two problems:
1. **Order bias** — first-match wins regardless of which type is more prominent.
2. **Multi-keyword sentences** — a reason containing both lexical and conceptual cues is always mis-typed as Lexical.

Agent C has full semantic context (both trees + the reason it just wrote), making it far better suited to pick the correct gap type.

## What did NOT change

The GapLink construction algorithm (tree traversal, LCA computation, divergence depth, bridge templates) is entirely unchanged.

## Touch points

| File | Change |
|------|--------|
| `schema.py` | Added `relation_type: Optional[str] = None` to `NodeAlignment` |
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` — added `relation_type` field + 5-type definitions to JSON schema |
| `factory.py` | `_parse_alignments` — reads and validates `relation_type` from Agent C's response |
| `factory.py` | GapLink builder — uses `leaf_align.relation_type or _infer_relation_type(reason)` |

---

# Keyword Fallback Classifier: Counting Mechanism Overhaul

**Date:** 2026-02-24
**Scope:** `_infer_relation_type()` in `factory.py`

## Problem

The previous keyword fallback classifier used **first-match-wins**: it checked categories in a fixed order (Lexical → Conceptual → Tacit → Process → Scope) and returned the first category that had any keyword hit. This caused two problems:

1. **Order bias** — If a reason contained keywords from multiple categories, the earlier category always won. ScopeGap was checked last and thus had the lowest priority.
2. **Keyword overlap** — Some keywords sat in the wrong category (e.g., `"depends"` and `"varies"` were in Process but are actually Tacit signals), while others like `"detail"` were too generic to discriminate.

Note: since Agent C is the primary classifier and this function is only a fallback, the impact was limited — but when Agent C omits `relation_type`, the fallback must still be reasonably accurate.

## Solution: Counting + Weighted Scoring

Replaced first-match-wins with a **counting mechanism**:

- Each keyword match adds a score: **multi-word phrases = 2**, **single words = 1**.
- Multi-word phrases (e.g., `"hard to explain"`, `"refers to the same"`) are more discriminative, hence the higher weight.
- All 5 categories are scored in parallel — no ordering bias.
- The category with the highest total score wins.
- On tie (including all-zero) → default to ConceptualGap (the most general type).

### Why substring matching is kept

We deliberately kept substring matching (`kw in reason_lower`) instead of word-boundary regex. Reasons:

1. **Tense forms are valid matches** — `"stepped"` matching `"step"`, `"labeled"` matching `"label"` are legitimate hits, not false positives.
2. **The counting mechanism mitigates false substring matches** — e.g., `"term"` inside `"determine"` adds only +1 to Lexical, which gets outvoted by multiple genuine matches in the correct category.
3. **Simplicity** — regex word boundaries add complexity (escaping, Unicode edge cases) for minimal gain given that Agent C is the primary classifier.

## Keyword Changes

### Removed (false-match or too generic)

| Keyword | Was in | Reason for removal |
|---------|--------|-------------------|
| `"aim"` | Scope | False-matches `"claim"`, `"main"` (common in Agent C's analytical text). Covered by `"objective"` and `"goal"`. |
| `"phase"` | Process | False-matches `"emphasis"` (very common in Agent C output). Covered by `"stage"`. |
| `"detail"` | Scope | Too generic — matches almost any analytical sentence. Not discriminative. |

### Moved between categories

| Keyword | From | To | Reason |
|---------|------|------|--------|
| `"depends"` | Process | Tacit | "It depends" is a classic tacit knowledge signal (context-dependent judgment), not a workflow issue. |
| `"varies"` | Process | Tacit | Same reasoning — indicates implicit, situation-dependent decision-making. |

### Added (Agent-C-style analytical language)

| Keyword | Category | Rationale |
|---------|----------|-----------|
| `"nomenclature"` | Lexical | Formal synonym for "terminology"; Agent C uses this in analytical text. |
| `"refers to the same"` | Lexical | Agent C often writes "both refer to the same concept". Multi-word phrase, scores 2. |
| `"equivalent term"` | Lexical | Directly describes a 1-to-1 term mapping. |
| `"structural difference"` | Conceptual | Agent C uses this when describing mental model divergence. |
| `"mental model"` | Conceptual | Core concept in the mismatch definition; Agent C frequently uses this exact phrase. |
| `"difficult to articulate"` | Tacit | Formal version of "hard to explain"; matches Agent C's register. |
| `"non-quantifiable"` | Tacit | Agent C uses this when the expert's knowledge resists measurement. |
| `"implicit judgment"` | Tacit | Describes exactly what TacitGap is about. |
| `"misaligned"` | Scope | Agent C writes "misaligned focus/priorities" for scope divergence. |
| `"divergent"` | Scope | Agent C writes "divergent priorities/expectations" for scope issues. |

## What did NOT change

- Substring matching logic (kept as-is).
- The 5 keyword category lists remain the primary data structure.
- Agent C's `relation_type` is still preferred; this function is only the fallback.
- No changes to `schema.py`, `runtime.py`, or any other file.

## Touch points

| File | Change |
|------|--------|
| `factory.py` | `_infer_relation_type()` — replaced first-match-wins with counting/scoring mechanism; updated keyword lists |
