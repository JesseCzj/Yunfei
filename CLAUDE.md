# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask development server (http://localhost:5000)
python app.py

# Run DSAG smoke tests (schema, factory, embeddings, runtime)
python test_dsag.py

# Build DSAG offline and generate visualization
python visualize_dsag.py
# Outputs: dsag_output.json, dsag_alignments.json, dsag_visualization.html
```

## Environment Configuration

The `.env` file configures separate providers for text generation and embeddings:
- **Text generation:** Qwen via DashScope (OpenAI-compatible API). Multiple model slots: `OPENAI_MODEL` (general), `OPENAI_MODEL_GRAPH` (graph building), `OPENAI_MODEL_RUNTIME` (runtime analysis), `OPENAI_MODEL_POLISH` (fast/cheap refinement).
- **Embeddings:** `text-embedding-3-small` via a separate API endpoint (`OPENAI_EBD_API_KEY` / `OPENAI_EBD_BASE_URL`).

## Architecture

This is an **HCI interview assistant** that helps researchers interview domain experts by detecting communication misalignments in real time. The core backend is **DSAG (Dual-Tower Semantic Alignment Graph)**.

### DSAG Pipeline (dsag/)

The system builds a knowledge graph before the interview starts, then uses it for real-time assistance during each Q&A turn:

1. **Schema** (`dsag/schema.py`) — Core data structures: `DSAGNode` (4-layer hierarchy: root/L1/L2/leaf), `TaxonomyTree`, `GapLink` (misalignment edges with bridge templates), `RelationType` (5 gap types: Lexical, Conceptual, Tacit, Scope, Process), `DSAGGraph`, `DSAGState`.

2. **Factory** (`dsag/factory.py`) — Multi-agent graph construction (~30-60s):
   - **Agent A (ExpertPersona):** Generates expert taxonomy tree from topic + expert background. Leaves marked `is_intuition=true` get `attributes` (tacit knowledge facets).
   - **Agent B (ResearcherPersona):** Generates researcher taxonomy tree from topic + researcher background + `questionnaire.docx`. The questionnaire is **only visible to Agent B**, not Agent A.
   - **Agent C (AlignmentJudge):** Judges semantic alignment between leaf/concept pairs across both trees.
   - **GapLink builder:** Creates edges only for misaligned pairs, with LCA computation, relation typing, bridge templates (3 granularities: coarse/balanced/fine), and checkout questions. Fallback pairing uses embeddings when coverage < `DSAG_GAPLINK_MIN_COVERAGE` (default 0.8).

3. **Embedding Index** (`dsag/embedding_index.py`) — Indexes only leaf nodes (text = label + description + aliases). Separate indices for expert and researcher leaves. Used for runtime position-matching and fallback pairing.

4. **Runtime** (`dsag/runtime.py`) — Per-turn analysis:
   - Locates positions by embedding the Q&A into both trees.
   - Queries relevant GapLinks between matched leaves.
   - Generates type-specific `Assistance` payloads (structure varies by `relation_type`).
   - **ProcessGap is entirely runtime-driven** — no offline payload. Uses `interview_timeline` for drift detection (repeated topic / tunnel vision / coverage gap) and an on-demand LLM call to generate a context-aware redirect sentence when drift is detected.
   - **Polish step:** Uses a faster/cheaper model (`OPENAI_MODEL_POLISH`) to refine assistance text using recent conversation context. ProcessGap skips this step (its redirect is already LLM-generated with full context).

### Flask Application (app.py)

- Server-side DSAG state cached in `DSAG_CACHE` dict, keyed by SHA256 hash of inputs.
- `SESSION_TO_DSAG` maps browser sessions to cached graphs.
- Thread safety via `DSAG_LOCK`.
- Context summary for polish is built from the last 3 Q&A turns in the session.

### Key API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/dsag/init` | POST | Start DSAG graph build (async) |
| `/api/dsag/status` | GET | Poll build status |
| `/api/dsag/analyze_turn` | POST | Analyze one Q&A turn |
| `/api/dsag/graph` | GET | Fetch full graph JSON |

Additional helper endpoints: `/api/get_refinement`, `/api/get_examples`, `/api/explain`, `/api/extract_keypoints`, `/api/suggest_followups`, `/api/hci_mapping`.

### Frontend (templates/ + static/)

Vanilla HTML/JS/CSS — no frontend framework. `templates/index.html` is the single-page layout, `static/app.js` handles API calls and UI interactions, `static/style.css` provides styling. Features: jargon highlighting with tooltips, mismatch detection badges, collapsible analysis cards, voice input (Web Speech API), file upload for interview guides.

### Node ID Convention

- Expert nodes: `exp_L1_01`, `exp_L2_03`, `exp_leaf_12`
- Researcher nodes: `res_L1_01`, `res_leaf_05`

## Important Notes

- `llm_backend.py` is a legacy demo module — ignore it. All core logic lives in `dsag/`.
- All DSAG data classes use `to_dict()` / `from_dict()` for JSON serialization.
- LLM prompts include JSON schemas inline; responses are parsed with regex fallback (`_parse_json()` in factory.py).
- The project README is written in Chinese.
