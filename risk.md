# Potential Risks & Known Limitations

## 1. Fallback Pairing Gap Type Assignment (factory.py, lines ~1357-1365)

**Status:** Under discussion — pending partner review

**Location:** `GraphFactory.generate_links()` → fallback pairing block

**Description:**
When Agent C's alignment coverage is below the threshold (default 80%), a fallback mechanism force-pairs uncovered expert leaves with researcher leaves using embedding similarity. The gap type for these fallback pairs is assigned by a **pure structural heuristic** (tree depth + node attributes), with no semantic analysis:

```
gap_type = ConceptualGap (default)
if expert leaf has attributes     → TacitGap
elif path_mismatch                → ProcessGap
elif LCA at Root                  → ConceptualGap
elif LCA at L1                    → ScopeGap
```

**Risks:**

1. **LexicalGap is never assigned** — there is no condition that produces LexicalGap for fallback pairs. A genuine terminology mismatch missed by Agent C will be misclassified.
2. **`path_mismatch → ProcessGap` is a weak heuristic** — L1 aligned but L2 divergent could also be ConceptualGap or ScopeGap, not necessarily a workflow issue.
3. **`LCA at L1 → ScopeGap` is rough** — L1-level divergence could be ConceptualGap (different mental models) rather than ScopeGap (different focus boundary).
4. **Wrong gap type → wrong assistance prompt** — since each gap type dispatches to a different type-specific prompt (LexicalGap prompt, TacitGap prompt, etc.), a misclassified pair will generate the wrong kind of assistance.

**Impact:**
Limited when Agent C achieves good coverage (>80%). Significant when coverage is low and many pairs go through the fallback path.

**Possible improvements (to discuss):**
- Call `_infer_relation_type()` on a synthetic reason string built from node labels/descriptions
- Add a LexicalGap condition (e.g., check if expert leaf has `aliases` or if label similarity is very high)
- Use a lightweight LLM call for fallback pairs

---

## 2. Agent C Alignment Prompt Improvements (factory.py, ALIGNMENT_JUDGE_PROMPT)

**Status:** Not implemented — planned for next iteration

**Description:**
The current alignment prompt gives Agent C abstract one-line definitions for each gap type, with no examples, no disambiguation guidance, and no requirement to justify the chosen type. This leads to inconsistent or biased classification.

### 2a. Add Concrete Examples for Each Gap Type

**Problem:** Abstract definitions alone are insufficient for LLMs. Without few-shot examples, Agent C defaults to the most generic-sounding type (ConceptualGap).

**Recommendation:** Add 1 generic/domain-agnostic example per gap type showing a misaligned pair and explaining why it belongs to that type. Use generic examples (not domain-specific) so they transfer across interview topics.

**Risk:** Adds ~500 tokens to the prompt. Must verify this doesn't exceed the context window when combined with large taxonomy trees.

### 2b. Add Disambiguation Rules for Confusable Type Pairs

**Problem:** The most common misclassifications are between:
- TacitGap vs. ConceptualGap — "experience-based intuition" could be either
- ScopeGap vs. ConceptualGap — "different levels of analysis" could be either
- ProcessGap vs. ScopeGap — "different workflow with different priorities" could be either

**Recommendation:** Add explicit decision boundaries using the "what's the fix?" test:
- If the fix is **translating terminology** -> LexicalGap
- If the fix is **building a new mental model** (analogy, scenario) -> ConceptualGap
- If the fix is **making implicit knowledge explicit** (probing, quantifying) -> TacitGap
- If the fix is **realigning priorities/focus** (validate & pivot) -> ScopeGap
- If the fix is **correcting workflow/sequence** (timeline, flowchart) -> ProcessGap

This is the highest-value improvement of the three.

### 2c. Require Reason to Include Gap Type Justification

**Problem:** Agent C currently writes the reason as pure semantic analysis without explaining why a particular `relation_type` was chosen. This means no chain-of-thought reasoning before classification, reducing accuracy.

**Recommendation:** Instruct Agent C to append a brief classification note at the END of the reason field, after the semantic analysis. Example format:

> "The expert highlights spatial compartmentalization as a contextual cue affecting trust, while the researcher focuses on output format mismatches. [TacitGap: expert relies on domain-specific experiential knowledge not transferable to the researcher's cognitive frame.]"

**Risk:** The `reason` field is used downstream by bridge prompts and keyword fallback. Embedding classification reasoning into it could pollute those consumers. Appending at the end (rather than interleaving) mitigates this — bridge prompts will still see the semantic content first.
