# Codebase Overview — HCI Empathy Bridge (DSAG)

> Last updated: 2026-02-24

---

## 1. What the System Does

This is a **real-time HCI interview assistant** ("Empathy Bridge") that helps researchers interview domain experts across disciplinary boundaries. The core problem it solves: experts and researchers frame topics through completely different mental models, causing "talking past each other" communication failures.

The system detects these **semantic misalignments** in real time and gives the researcher **type-specific assistance** — each of the 5 mismatch types (Lexical, Conceptual, Tacit, Scope, Process) produces a fundamentally different assistance structure with its own strategies and follow-up logic.

The theoretical backbone is the **DSAG (Dual-Tower Semantic Alignment Graph)**, a knowledge graph pre-built before each interview session starts.

---

## 2. Repository Layout

```
demo_flask/
├── app.py                   # Flask server — routes, session management, DSAG orchestration
├── requirements.txt
├── questionnaire.docx       # Default researcher interview guide (used by Agent B)
├── test_dsag.py             # Smoke tests (schema, factory, embeddings, runtime)
├── test_dsag_v2.py          # Standalone v2 test suite (type-specific assistance)
├── visualize_dsag.py        # Offline DSAG build + generates dsag_visualization.html
├── dsag_output.json         # Output artifact from visualize_dsag.py
├── dsag_alignments.json     # Alignment data artifact
├── dsag_visualization.html  # Interactive graph visualization artifact
└── dsag/                    # Core DSAG library
    ├── __init__.py          # Public API exports
    ├── schema.py            # All data structures (nodes, trees, links, enums)
    ├── factory.py           # Multi-agent graph construction + type-specific prompts
    ├── embedding_index.py   # Leaf-node embedding index for semantic search
    └── runtime.py           # Per-turn real-time analysis engine (type-branched)
```

---

## 3. Core Data Model (`dsag/schema.py`)

### Enums

| Enum | Values |
|------|--------|
| `Tower` | `expert`, `researcher`, `shared` |
| `Layer` | `root` (L0), `L1` (Perspective), `L2` (Category), `leaf` (L3) |
| `RelationType` | `LexicalGap`, `ConceptualGap`, `TacitGap`, `ScopeGap`, `ProcessGap` |

### Key Classes

**`DSAGNode`** — One node in a taxonomy tree.
- `id`: String like `exp_leaf_02_01_03` or `res_L1_00`
- `tower`: which tower it belongs to
- `layer`: depth in the tree (`root`/`L1`/`L2`/`leaf`)
- `label`, `description`: human-readable text
- `parent_id`, `children_ids`: tree linkage
- `attributes`: **Tacit Knowledge Facets** — only populated on expert leaf nodes where `is_intuition=true` during generation. Examples: "Visual clutter level", "Class imbalance". Used to generate DeepDive follow-up questions and to signal TacitGap to Agent C.
- `aliases`: HCI terms from researcher leaves; also used in embedding text

**`TaxonomyTree`** — Container for one tower's tree.
- Key methods: `get_leaves()`, `get_path_to_root()`, `get_aligned_path()`, `get_siblings()`
- `get_aligned_path(node_id)` → `[root_label, L1_label, L2_label, leaf_label]` — used for LCA computation
- `get_siblings(node_id)` → sibling nodes under same parent — used for ConceptualGap analogy candidates and ExpandScope follow-ups

**`GapLink`** — An edge in the gap graph connecting one expert leaf to one researcher leaf.
- `expert_leaf_id`, `researcher_leaf_id`
- `aligned_path_expert`, `aligned_path_researcher`: label paths from root to each leaf
- `lca_layer`: where the paths first diverge (`root`, `L1`, or `L2`)
- `expert_lca_id`, `researcher_lca_id`: the actual ancestor node IDs at LCA
- `conflict`: `{expert_branch: "...", researcher_branch: "..."}` — divergent branch labels
- `relation_type`: one of the 5 `RelationType` values (set by Agent C, keyword fallback if omitted)
- `assistance_payload`: `Dict[str, Any]` — type-specific assistance data built offline by LLM (structure varies per `relation_type`)
- `weight`: priority score (higher = more important gap; computed as `(1 - semantic_similarity) * 10`)

**`DSAGGraph`** — The complete graph.
- `topic`, `researcher_bg`, `expert_bg`
- `expert_tree`, `researcher_tree` (both `TaxonomyTree`)
- `links` (list of `GapLink`)
- `metadata`: creation time, model info, link count stats
- Cache key: SHA256 of `topic + researcher_bg + expert_bg + questionnaire` (first 16 chars)

**`DSAGState`** — Server-side session state.
- `graph`, `cache_key`
- `expert_leaf_embeddings`, `researcher_leaf_embeddings`: dict mapping `leaf_id → vector`
- `status`: `"building"` | `"ready"` | `"error"`
- `interview_timeline`: accumulated timeline entries for ProcessGap drift detection (one entry per analyzed turn)

**`NodeAlignment`** / **`TreeAlignments`** — Agent C output.
- `leaf_alignments`: pairings between expert and researcher leaves with `is_aligned` flag, `reason`, and `relation_type`
- `concept_alignments`: L1/L2 level pairings (used for LCA lookup)

### Validation & LCA Utilities
- `validate_node/tree/graph()`: structural integrity checks
- `compute_lca_layer(path_expert, path_researcher)`: compares two label paths, returns divergence layer

---

## 4. Graph Construction Pipeline (`dsag/factory.py`)

**Entry point:** `GraphFactory.generate_graph()` or convenience `create_dsag_graph()`

### Agents

**Agent A — ExpertPersona** (`generate_expert_tree`)
- Prompt: given `topic` + `expert_bg`, generate 3–5 perspectives → 2–4 categories → 3–5 pain points each
- Pain points flagged `is_intuition=true` get `attributes[]` (tacit knowledge facets)
- Tree builder: `_build_expert_tree()` → node IDs `exp_L1_00`, `exp_L2_00_01`, `exp_leaf_00_01_02`

**Agent B — ResearcherPersona** (`generate_researcher_tree`)
- Prompt: given `topic` + `researcher_bg` + **questionnaire text** (questionnaire.docx)
- Goals extracted from the questionnaire; HCI terms stored as `aliases` on leaf nodes
- Tree builder: `_build_researcher_tree()` → node IDs `res_L1_00`, `res_leaf_00_01_02`

**Agent C — AlignmentJudge** (`generate_alignments`)
- Prompt: both trees in JSON summary form → judge semantic alignment of all leaf pairs and concept (L1/L2) pairs
- Returns `TreeAlignments` with `is_aligned` bool, `reason` text, and `relation_type` per pair
- `relation_type` is classified by Agent C directly (prompt includes 5-type definitions + TacitGap classification hint for `attributes`)

### Math-Based Link Construction (`build_links_from_alignments`)

For each **misaligned** leaf pair from Agent C:
1. Build concept alignment map: `expert_node_id → researcher_node_id` (for aligned L1/L2 pairs)
2. Traverse `expert_leaf → root` path to find the first ancestor that appears in the concept alignment map → this is the **LCA**
3. Validate: check the mapped researcher LCA is actually an ancestor of the researcher leaf (walk up researcher path if not)
4. Record divergence: labels of the first node below LCA on each side → stored as `conflict`
5. Set `relation_type` from Agent C's classification; fall back to `_infer_relation_type()` keyword scoring if Agent C omits it
6. Compute `weight` = `(1 - semantic_similarity) * 10`
7. Build `GapLink`

### Keyword Fallback Classifier (`_infer_relation_type`)

Two-tier counting + weighted scoring (fallback when Agent C omits `relation_type`):
- Tier 1 (single words) = 1pt broad recall
- Tier 2 (multi-word phrases) = 2pt high precision
- All 5 categories scored in parallel; highest total wins; tie defaults to ConceptualGap

### Fallback Pairing

If gap coverage (fraction of expert leaves with at least one misaligned link) is below `DSAG_GAPLINK_MIN_COVERAGE` (default 0.8), a fallback uses the embedding index to find candidate researcher leaves for uncovered expert leaves. Filtering criteria:
- Cosine similarity in range `[sim_min, sim_max]` (defaults 0.25–0.75)
- Either high-level divergence (LCA at root or L1) OR structural path mismatch (same L1 but different L2)
- Gap score = `alpha * (1 - sim) + beta * divergence_level` (defaults: alpha=0.6, beta=0.4)

### Assistance Payload Generation (`generate_all_assistance_payloads`)

After links are built, one LLM call per link generates the type-specific `assistance_payload`. Uses creativity-oriented temperature (0.7). Each type has its own prompt:

| Type | Offline Payload Shape |
|------|----------------------|
| **LexicalGap** | `{term_mapping: {expert_term, researcher_term, explanation}}` |
| **ConceptualGap** | `{analogy: {source_concept, structural_mapping, explanation}, scenario: {inputs, outputs, edge_cases}}` — receives `{expert_siblings}` as grounded analogy candidates |
| **TacitGap** | `{probes: [{attribute, question, choices}], hypothetical_scenarios: [...]}` |
| **ScopeGap** | `{validate_focus, pivot: {limitation, research_goal, compelling_reason, coarse_scenario}}` |
| **ProcessGap** | `{initial_topics: [...]}` |

### LLM Configuration in factory.py

Uses `_build_llm()` which reads:
- `LLM_PROVIDER` → `"openai"` or `"deepseek"`
- For OpenAI: `OPENAI_MODEL_GRAPH` (falls back to `OPENAI_MODEL`, default `"qwen3-max"`)
- Trees generated at temperature 0.3; assistance payloads at 0.7

---

## 5. Embedding Index (`dsag/embedding_index.py`)

**What is indexed:** Only leaf nodes. Text = `label | description | alias1 | alias2 | ...`

**`EmbeddingIndex`** class:
- `build_index()`: batch-embeds all expert + researcher leaves in one API call
- `search_expert_leaves(query, top_k=3)`: embed query → cosine similarity search over expert embeddings → returns `SearchResult[]`
- `search_researcher_leaves(query, top_k=3)`: same for researcher side
- `get_embeddings_data()` / `load_embeddings_data()`: serialization for storing in `DSAGState`

**Thresholds:**
- `HIGH_CONFIDENCE_THRESHOLD = 0.75` — below this, result is marked `snapped=True`
- `SNAP_THRESHOLD = 0.45` — below this, "very low confidence" warning

**Embedding model config:**
- `EMBEDDING_PROVIDER` (or falls back to `LLM_PROVIDER`)
- `OPENAI_EBD_API_KEY`, `OPENAI_EBD_BASE_URL`, `OPENAI_EMBEDDING_MODEL` (default `"text-embedding-3-small"`)

---

## 6. Runtime Engine (`dsag/runtime.py`)

**Entry point:** `RuntimeEngine.analyze_turn(researcher_question, expert_answer, context_summary, interview_timeline)`

### Per-turn Flow

1. **Locate positions** (`locate_positions`):
   - Embed `expert_answer` → search expert leaves → best match = `best_expert_leaf_id`
   - Embed `researcher_question` → search researcher leaves → best match = `best_researcher_leaf_id`
   - Confidence warning if expert confidence < 0.45

2. **Find best link** (`find_best_link`):
   - Try exact match: `graph.get_link(expert_leaf_id, researcher_leaf_id)`
   - Fallback: any link from expert leaf, sorted by weight descending

3. **Compute divergence** (`compute_divergence`):
   - Reads `lca_layer` and `conflict` from selected link
   - Generates English explanation of the gap severity (root > L1 > L2)

4. **Generate assistance** (`generate_assistance`) — **branches on `relation_type`**:
   - **LexicalGap**: copies offline `term_mapping` directly; no follow-ups
   - **ConceptualGap**: copies offline `analogy` + `scenario`; generates **ExpandScope** follow-ups from researcher tree siblings
   - **TacitGap**: merges offline probes with live `node.attributes`; includes `hypothetical_scenarios`; generates **DeepDive** follow-ups
   - **ScopeGap**: copies offline `validate_focus` + `pivot`; no follow-ups
   - **ProcessGap**: injects `interview_timeline`, runs drift detection (repeated topics / uncovered siblings); payload includes `timeline`, `drift_alerts`, `current_topic`

5. **Polish step** (`_polish_assistance`):
   - Uses cheap/fast model (`OPENAI_MODEL_POLISH`, default `"qwen-turbo"`)
   - Per-type `_POLISH_RULES_BY_TYPE` dict classifies every field as **FROZEN** (copy verbatim) or **POLISHABLE** (make natural, weave in expert's wording)
   - 6 universal rules + type-specific rules injected into prompt

### Output Structure (`Assistance`)

```python
{
  "relation_type": "TacitGap",
  "payload": { ... },           # structure varies per relation_type
  "followup_questions": [
    {"type": "DeepDive" | "ExpandScope", "question": "..."},
    ...
  ]
}
```

### LLM Configuration in runtime.py

- `_build_light_llm()`: `OPENAI_MODEL_RUNTIME` (default `"qwen3-max"`) — used for template filling
- `_build_polish_llm()`: `OPENAI_MODEL_POLISH` (default `"qwen-turbo"`) — used for polish step

---

## 7. Flask Application (`app.py`)

### Session & Caching Architecture

```python
DSAG_CACHE: Dict[str, DSAGState]       # cache_key → DSAGState (graph + embeddings)
SESSION_TO_DSAG: Dict[str, str]        # session_id → cache_key
DSAG_LOCK: threading.Lock              # protects both dicts
```

- Cache key is deterministic (SHA256 hash of inputs), so identical inputs reuse the graph without rebuilding
- Graph builds are **synchronous** inside the request handler (30–60s blocking call)

### Context Summary

`build_context_summary(messages, max_turns=3)`: collects the last 3 Q&A pairs from session, truncates to 160/220 chars, formats as `"Turn N | R: ... | E: ..."`. Passed to `analyze_turn()` for the polish step.

### Interview Timeline

Each `analyze_turn` call appends a timeline entry to `DSAGState.interview_timeline`:
```json
{"turn_index": 0, "topic_label": "...", "expert_leaf_id": "...", "researcher_leaf_id": "...", "relation_type": "...", "summary": "..."}
```
This is passed to the runtime engine for ProcessGap drift detection.

### DSAG API Endpoints

| Endpoint | Method | Behavior |
|----------|--------|----------|
| `/api/dsag/init` | POST | Checks cache → builds graph if miss → builds embedding index → stores in `DSAGState` |
| `/api/dsag/status` | GET | Returns `{ready, status, error, metadata}` |
| `/api/dsag/analyze_turn` | POST | Reconstructs `EmbeddingIndex` from stored vectors → runs `RuntimeEngine.analyze_turn()` with `interview_timeline` |
| `/api/dsag/graph` | GET | Returns full `DSAGGraph.to_dict()` |

### File Upload

`extract_text_from_upload()` handles `.txt`, `.docx` (via python-docx), `.pdf` (via pdfplumber). Used for the Interview Script panel.

`load_questionnaire_text()`: auto-loads `questionnaire.docx` from the app root at DSAG init time. This text is passed exclusively to Agent B.

---

## 8. Node ID Convention

| Level | Expert | Researcher |
|-------|--------|------------|
| Root | `shared_root` | `shared_root` (same node) |
| L1 | `exp_L1_00`, `exp_L1_01` | `res_L1_00`, `res_L1_01` |
| L2 | `exp_L2_00_01` (L1_idx + L2_idx) | `res_L2_00_01` |
| Leaf | `exp_leaf_00_01_02` (L1+L2+leaf indices) | `res_leaf_00_01_02` |

---

## 9. Data Flow Summary

```
[Input: topic + researcher_bg + expert_bg + questionnaire.docx]
        ↓
POST /api/dsag/init
        ↓
GraphFactory.generate_graph()
    ├─ Agent A → expert_tree (LLM: OPENAI_MODEL_GRAPH)
    ├─ Agent B → researcher_tree (LLM: OPENAI_MODEL_GRAPH) + questionnaire
    ├─ Agent C → TreeAlignments with relation_type (LLM: OPENAI_MODEL_GRAPH)
    ├─ build_links_from_alignments() → GapLink[] (math + Agent C relation_type)
    └─ generate_all_assistance_payloads() → fills type-specific payloads (LLM: OPENAI_MODEL_GRAPH)
        ↓
build_embedding_index() → embed all leaves (OPENAI_EBD_API_KEY)
        ↓
DSAGState stored in DSAG_CACHE[cache_key]
        ↓
[Each Q&A turn → POST /api/dsag/analyze_turn]
        ↓
RuntimeEngine.analyze_turn()
    ├─ locate_positions() → embed Q&A → find expert_leaf + researcher_leaf
    ├─ find_best_link() → select GapLink
    ├─ compute_divergence() → DivergenceInfo
    ├─ generate_assistance() → Assistance (type-branched: payload + follow-ups)
    └─ _polish_assistance() → refine text with FROZEN/POLISHABLE rules (LLM: OPENAI_MODEL_POLISH)
        ↓
→ Assistance JSON returned to caller
→ Timeline entry appended to DSAGState.interview_timeline
```

---

## 10. Known Design Decisions & Gotchas

- **DSAG init is synchronous**: The `/api/dsag/init` endpoint blocks for 30–60 seconds. There is a `/api/dsag/status` polling endpoint but no background thread — the build happens inside the init request itself.
- **Embeddings are reconstructed per turn**: `EmbeddingIndex` is instantiated fresh in each `analyze_turn` call, then `load_embeddings_data()` fills it from the stored vectors in `DSAGState`. No re-embedding happens at runtime.
- **The questionnaire is Agent B only**: `questionnaire.docx` is loaded server-side via `load_questionnaire_text()` and only passed to the researcher tree prompt. Agent A (expert) never sees it.
- **Agent C classifies relation_type**: `relation_type` is set by Agent C in its JSON response. `_infer_relation_type()` keyword fallback only fires if Agent C omits the field.
- **Thread safety**: Only `DSAG_CACHE` and `SESSION_TO_DSAG` are protected by `DSAG_LOCK`. The Flask session (`session["messages"]`) uses Flask's built-in session cookie mechanism.
- **No persistence**: All state is in-memory. Server restart loses all session data and cached graphs.
- **Cache incompatibility**: Old cached graphs built with the `BridgeTemplates` structure will fail on `GapLink.from_dict()`. Delete old `dsag_output.json` files after upgrading.
