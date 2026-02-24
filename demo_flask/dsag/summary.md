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
