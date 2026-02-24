# Edge Type Classification: Keyword Matching to Agent C

## What changed

GapLink `relation_type` is now classified by **Agent C** (AlignmentJudge) instead of the keyword-matching function `_infer_relation_type`.

Agent C emits `relation_type` alongside `reason` in the same JSON response for each misaligned pair. The keyword function is kept as a fallback in case Agent C omits the field.

## Why

Keyword matching suffers from two problems:
1. **Order bias** -- first-match wins regardless of which type is more prominent.
2. **Multi-keyword sentences** -- a reason containing both lexical and conceptual cues is always mis-typed as Lexical.

Agent C has full semantic context (both trees + the reason it just wrote), making it far better suited to pick the correct gap type.

## Touch points

| File | Change |
|------|--------|
| `schema.py` | Added `relation_type: Optional[str] = None` to `NodeAlignment` |
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` -- added `relation_type` field + 5-type definitions to JSON schema |
| `factory.py` | `_parse_alignments` -- reads and validates `relation_type` from Agent C response |
| `factory.py` | GapLink builder -- uses `leaf_align.relation_type or _infer_relation_type(reason)` |

---

# Keyword Fallback Classifier: Counting Mechanism Overhaul

**Date:** 2026-02-24

## Problem

The previous keyword fallback classifier used **first-match-wins** with fixed order (Lexical, Conceptual, Tacit, Process, Scope). This caused order bias and keyword overlap issues.

## Solution: Counting + Weighted Scoring

- Multi-word phrases = 2 points, single words = 1 point.
- All 5 categories scored in parallel -- no ordering bias.
- Highest total score wins. Tie defaults to ConceptualGap.
- Substring matching kept (tense forms are valid, counting mitigates false hits).

---

## Bug Fix: Fallback Pairing `relation_type` Not Set

The fallback pairing code created `NodeAlignment` without setting `relation_type`. TacitGap and LexicalGap fallback pairs were silently reclassified as ConceptualGap.

**Fix:** Added `relation_type=gap_type` to the `NodeAlignment` constructor.

---

## Prompt Fix: Agent C `relation_type` Emission

Agent C never emitted `relation_type` because the prompt said "(misaligned only, omit or null if aligned)" which the LLM interpreted as omission is always acceptable.

**Fix:** Changed to "you MUST set" + "REQUIRED when is_aligned=false".

---

# Keyword Redesign: Two-Tier Agent-C-Style Architecture

**Date:** 2026-02-24

## Problem

The keyword lists mixed two incompatible registers:
1. **Conversational phrases** -- Agent C never generates these. Dead keywords.
2. **Over-generic single words** -- removed previously to avoid overlap, losing recall.

## Solution: Two-Tier Keyword Architecture

| Tier | Score | Role | Example |
|------|-------|------|---------|
| Tier 1 (single words) | 1pt | Broad recall | `focus`, `intuitive`, `workflow` |
| Tier 2 (multi-word) | 2pt | High precision | `different focus`, `contextual cue` |

### Key design decisions

1. All keywords match Agent C analytical register. Conversational phrases removed.
2. Generic single words restored (`focus`, `goal`, `value`) as 1pt weak votes.
3. Process keywords completely overhauled with Agent-C-style phrases.

## Removed (conversational / dead)

Lexical: `word`, `phrase`, `lingo`, `buzzword`
Conceptual: `picture`, `idea`, `wrap my head around`
Tacit: `gut`, `feeling`, `sense`, `feel like`, `ballpark`, `roughly`, `just know`, `over the years`, `you'd know it when you see it`
Scope: `user`
Process: `step`, `in our system`, `the way we do it`, `actually that's not right`, `that's not how it works`

## Added (Agent-C analytical style)

Lexical: `different terminology`, `different label`
Conceptual: `different interpretation`, `conceptual mismatch`, `different framework`
Tacit: `experience-based`, `implicit knowledge`, `tacit knowledge`
Scope: `different scope`, `different priority`, `different level`, `different levels of analysis`
Process: `different workflow`, `different ordering`, `different procedure`, `procedural difference`, `procedural mismatch`, `operational mismatch`, `methodological difference`

## Restored (previously over-removed)

`focus`, `goal`, `value` restored to Scope as 1pt weak votes, disambiguated by Tier 2 anchors.

## Final keyword counts

| Category | Tier 1 | Tier 2 | Total |
|----------|--------|--------|-------|
| Lexical | 14 | 10 | 24 |
| Conceptual | 11 | 11 | 22 |
| Tacit | 15 | 13 | 28 |
| Scope | 16 | 17 | 33 |
| Process | 10 | 11 | 21 |

---

## What did NOT change

- Substring matching logic (kept as-is).
- Counting/scoring mechanism (Tier 2 = 2pt, Tier 1 = 1pt, highest wins, tie = ConceptualGap).
- Agent C `relation_type` is still preferred; this function is only the fallback.

## All touch points

| File | Change |
|------|--------|
| `factory.py` | `_infer_relation_type()` -- counting mechanism + two-tier Agent-C keyword redesign |
| `factory.py` | Fallback pairing `NodeAlignment` -- added `relation_type=gap_type` |
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` -- strengthened `relation_type` requirement |

---

# Prompt Bug Fixes: Alignment & Bridge Prompts

**Date:** 2026-02-24

Cross-referenced `mismatch_types.md` and `improvement_goal.md` against the actual prompts in `factory.py`. Found 5 bugs; all fixed.

## Fix 1: ScopeGap definition in ALIGNMENT_JUDGE_PROMPT

**Bug:** Definition said "differ in focus boundary — one is too detailed, too high-level, or out of scope." This describes a *granularity* mismatch.

**Spec says:** The expert focuses on practical utility while the researcher focuses on research value — a *purpose/expectation* mismatch about "what to do."

**Risk:** Agent C would misclassify detail-level mismatches as ScopeGap, while missing the true ScopeGap pattern (practical vs. research value). Wrong classification → wrong bridge prompt (Validate & Pivot) fires on the wrong pairs.

**Fix:** Rewrote to: "The two sides differ in purpose or expectations — the expert focuses on practical utility while the researcher focuses on research value, leading to inconsistent goals about 'what to do.'"

## Fix 2: ProcessGap definition in ALIGNMENT_JUDGE_PROMPT

**Bug:** Definition only mentioned "workflow, sequence, or procedure — how steps are ordered or executed."

**Spec says:** ProcessGap is also triggered by **factual errors** disrupting the discussion and **narrow expert narratives (tunnel vision)**.

**Risk:** Agent C would only flag workflow ordering differences, missing the two other core triggers. Factual-error and tunnel-vision pairs would be classified as ConceptualGap or ScopeGap, causing the Process timeline/drift-alert pipeline in `runtime.py` to never fire for those pairs.

**Fix:** Rewrote to: "Factual errors disrupt the discussion, or the expert lacks standardized procedures or falls into narrow narratives (tunnel vision), causing workflow/sequence misalignment."

## Fix 3: TacitGap classification hint for `attributes`

**Bug:** Agent A marks intuition-based expert leaves with `attributes` (Tacit Knowledge Facets). `_tree_to_summary()` includes these in the JSON sent to Agent C. But the ALIGNMENT_JUDGE_PROMPT never told Agent C what `attributes` means or how to use it for classification.

**Risk:** Broken data chain: Agent A marks tacit leaves → Agent C ignores the signal → misclassifies as ConceptualGap → TacitGap bridge prompt never fires → the 3-step pipeline (extract → probe → checkout) is skipped, and the attributes are wasted.

**Fix:** Added classification hint after gap type definitions: "Expert leaf nodes that contain a non-empty 'attributes' list indicate tacit, intuition-based knowledge. When such a leaf is part of a misaligned pair, strongly prefer TacitGap."

## Fix 4: TacitGap bridge prompt — singular scenario → plural array

**Bug:** Prompt said "Generate ONE hypothetical scenario" with output `"hypothetical_scenario": "..."` (a single string).

**Spec says:** "Come up with **some** hypothetical scenarios for checkout." Each scenario alters only one variable at a time.

**Risk:** Only one checkout scenario generated, providing insufficient coverage to test the boundaries of the expert's intuition. The spec intends multiple scenarios, each isolating a different variable.

**Fix:** Changed to "Generate 2-3 hypothetical scenarios." Output field changed from `"hypothetical_scenario"` (string) to `"hypothetical_scenarios"` (array). Updated `runtime.py` Assistance docstring to match.

## Fix 5: ConceptualGap bridge prompt — missing `explanation` field

**Bug:** The prompt instructs the LLM to "Explain how the researcher's concept behaves similarly to the expert's familiar concept," but the JSON output schema had no field for this explanation — only `source_concept` and `structural_mapping`.

**Risk:** The LLM is told to explain but has nowhere to put the explanation in the structured output. The explanation either gets silently dropped or crammed into `source_concept`, making the analogy output incomplete and unreliable for the frontend.

**Fix:** Added `"explanation": "How the researcher's concept behaves similarly to the source concept"` to the analogy object in the JSON schema. Updated `runtime.py` Assistance docstring to match.

## Touch points

| File | Change |
|------|--------|
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` — rewrote ScopeGap definition (Fix 1) |
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` — rewrote ProcessGap definition (Fix 2) |
| `factory.py` | `ALIGNMENT_JUDGE_PROMPT` — added TacitGap classification hint for `attributes` (Fix 3) |
| `factory.py` | `TACIT_GAP_PROMPT` — `hypothetical_scenario` string → `hypothetical_scenarios` array (Fix 4) |
| `factory.py` | `CONCEPTUAL_GAP_PROMPT` — added `explanation` field to analogy output (Fix 5) |
| `runtime.py` | `Assistance` docstring — updated payload shapes for ConceptualGap and TacitGap (Fix 4 & 5) |

---

# ConceptualGap Analogy: Expert Sibling Injection

**Date:** 2026-02-24

## Problem

`CONCEPTUAL_GAP_PROMPT` Strategy 1 (Analogy Construction) tells the LLM "Identify a Sibling Concept the expert already knows" but provides no candidates. The LLM must hallucinate what the expert might know — ungrounded and unreliable.

Meanwhile, the expert taxonomy tree already contains sibling leaves under the same L2 parent. These are concepts the expert demonstrably articulated, at the same abstraction level, in the same category — ideal analogy source candidates.

## Risk (before fix)

The LLM invents a "Sibling Concept" from imagination rather than from the expert's actual knowledge. This leads to analogies that reference concepts the expert may not recognize, defeating the purpose of making the researcher's concept relatable through familiar terms.

## Fix

1. **Prompt**: Added `{expert_siblings}` field to `CONCEPTUAL_GAP_PROMPT`, placed before the divergence info. The analogy instruction now says "Consider the expert's related concepts listed above as potential analogy sources" with an escape hatch: "You may also use a different concept if it provides a stronger structural mapping."

2. **Code**: In `generate_assistance_payload_for_link()`, when `relation_type == ConceptualGap`, collect up to 5 siblings via `expert_tree.get_siblings(link.expert_leaf_id)` and format them as `"label": description` lines. Injected into variables as `expert_siblings`.

## Design decisions

- Siblings are framed as **candidates, not constraints** — LLM can still pick a better concept from outside the tree.
- Capped at 5 siblings to limit token cost.
- Only collected for ConceptualGap (other types don't need analogy candidates).

## Touch points

| File | Change |
|------|--------|
| `factory.py` | `CONCEPTUAL_GAP_PROMPT` — added `{expert_siblings}` field + rewrote Strategy 1 instruction |
| `factory.py` | `generate_assistance_payload_for_link()` — collect siblings, build text, inject into variables |