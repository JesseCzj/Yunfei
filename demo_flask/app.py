from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_session import Session
import io
import os
import uuid
import threading
from typing import Any, Dict, Optional

# Import DSAG modules
from dsag import (
    DSAGGraph,
    DSAGState,
    GraphFactory,
    EmbeddingIndex,
    RuntimeEngine,
    build_embedding_index,
)


app = Flask(__name__)
app.secret_key = "demo_secret_key"

# Use server-side session storage to avoid oversized cookie payloads.
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(app.root_path, ".flask_session")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)


# ============== Server-Side DSAG State ==============
# Store DSAG graphs in memory, keyed by cache_key
# Format: {cache_key: DSAGState}
DSAG_CACHE: Dict[str, DSAGState] = {}

# Map session IDs to their current cache_key
# Format: {session_id: cache_key}
SESSION_TO_DSAG: Dict[str, str] = {}

# Lock for thread-safe DSAG state access
DSAG_LOCK = threading.Lock()


def get_session_id() -> str:
    """Get or create a session ID."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def get_dsag_state() -> Optional[DSAGState]:
    """Get the DSAG state for the current session."""
    sid = get_session_id()
    with DSAG_LOCK:
        cache_key = SESSION_TO_DSAG.get(sid)
        if cache_key:
            return DSAG_CACHE.get(cache_key)
    return None


def set_dsag_state(state: DSAGState, cache_key: str) -> None:
    """Set the DSAG state for the current session."""
    sid = get_session_id()
    with DSAG_LOCK:
        DSAG_CACHE[cache_key] = state
        SESSION_TO_DSAG[sid] = cache_key


def clear_dsag_state() -> None:
    """Clear DSAG state for current session (but keep cache for reuse)."""
    sid = get_session_id()
    with DSAG_LOCK:
        SESSION_TO_DSAG.pop(sid, None)


DEFAULT_GUIDE_TEXT = """
Please upload your interview script
"""


def get_messages():
    if "messages" not in session:
        session["messages"] = []
    return session["messages"]


def build_context_summary(messages, max_turns: int = 3) -> str:
    """Build a lightweight summary of the most recent turns."""
    turns = []
    i = len(messages) - 1
    while i >= 0 and len(turns) < max_turns:
        msg = messages[i]
        if msg.get("role") == "expert":
            answer = msg.get("content", "")
            # Find the closest prior researcher question
            j = i - 1
            while j >= 0 and messages[j].get("role") != "researcher":
                j -= 1
            question = messages[j].get("content", "") if j >= 0 else ""
            turns.append((question, answer))
            i = j - 1
        else:
            i -= 1
    turns.reverse()

    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "…"

    lines = []
    for idx, (q, a) in enumerate(turns, 1):
        q = _truncate((q or "").strip(), 160)
        a = _truncate((a or "").strip(), 220)
        if q or a:
            lines.append(f"Turn {idx} | R: {q} | E: {a}")
    return "\n".join(lines)


def get_guide_text():
    return session.get("guide_text", DEFAULT_GUIDE_TEXT)


def set_guide_text(text: str):
    session["guide_text"] = text


def set_guide_error(message: str | None):
    if message:
        session["guide_error"] = message
    else:
        session.pop("guide_error", None)


def extract_text_from_upload(upload):
    filename = upload.filename or ""
    lower = filename.lower()
    data = upload.read()
    if lower.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if lower.endswith(".docx"):
        try:
            import docx
        except Exception as exc:
            raise RuntimeError("Missing dependency: python-docx") from exc
        document = docx.Document(io.BytesIO(data))
        return "\n".join([p.text for p in document.paragraphs if p.text])
    if lower.endswith(".pdf"):
        try:
            import pdfplumber
        except Exception as exc:
            raise RuntimeError("Missing dependency: pdfplumber") from exc
        text_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                if extracted.strip():
                    text_parts.append(extracted)
        return "\n".join(text_parts)
    raise RuntimeError("Unsupported file type. Use .txt, .docx, or .pdf")


def load_questionnaire_text() -> str:
    """Load questionnaire.docx from workspace root if present."""
    questionnaire_path = os.path.join(app.root_path, "questionnaire.docx")
    if not os.path.exists(questionnaire_path):
        return ""
    try:
        with open(questionnaire_path, "rb") as f:
            data = f.read()
        try:
            import docx
        except Exception:
            return ""
        document = docx.Document(io.BytesIO(data))
        return "\n".join([p.text for p in document.paragraphs if p.text])
    except Exception:
        return ""


def ensure_manual_mismap(message):
    """Mark a message as manually tagged mis-map."""
    message["manual_mismap"] = True
    if "mismap" not in message:
        message["mismap"] = {
            "detected": True,
            "type": "Manual",
            "reason": "Manually tagged by researcher.",
        }
    else:
        message["mismap"]["detected"] = True
        if not message["mismap"].get("reason"):
            message["mismap"]["reason"] = "Manually tagged by researcher."


def clear_manual_mismap(message):
    """Clear manual mis-map tag from a message."""
    message.pop("manual_mismap", None)
    mismap = message.get("mismap")
    if not mismap:
        return
    # Only remove mismap if it was purely manual (no LLM-detected reason)
    if mismap.get("reason") == "Manually tagged by researcher.":
        # Check if there's any generated content we should keep
        has_refinement = bool(message.get("refinement"))
        has_contexts = bool(message.get("contexts"))
        has_explanation = bool(message.get("explanation"))
        if not has_refinement and not has_contexts and not has_explanation:
            message.pop("mismap", None)


def find_last_question(messages, expert_index):
    """Find the last researcher question before the expert message at given index."""
    for i in range(expert_index - 1, -1, -1):
        if messages[i].get("role") == "researcher":
            return messages[i].get("content", "")
    return ""


@app.route("/clear", methods=["GET"])
def clear_session():
    """Clear all session data and redirect to index."""
    clear_dsag_state()
    session.clear()
    return redirect(url_for("index"))


# ============== DSAG API Routes ==============

@app.route("/api/dsag/init", methods=["POST"])
def api_dsag_init():
    """
    Initialize a DSAG graph for the interview session.
    
    Request body:
    {
        "topic": "Interview topic (becomes shared root)",
        "researcher_bg": "Researcher background description",
        "expert_bg": "Expert background description"
    }
    
    Response:
    {
        "success": true,
        "cached": true/false,
        "cache_key": "...",
        "metadata": {...}
    }
    """
    try:
        data = request.get_json()
        topic = data.get("topic", "").strip()
        researcher_bg = data.get("researcher_bg", "").strip()
        expert_bg = data.get("expert_bg", "").strip()
        
        if not topic:
            return jsonify({"success": False, "error": "Topic is required"})
        if not researcher_bg:
            return jsonify({"success": False, "error": "Researcher background is required"})
        if not expert_bg:
            return jsonify({"success": False, "error": "Expert background is required"})
        
        # Compute cache key
        questionnaire = load_questionnaire_text()
        cache_key = DSAGGraph.compute_cache_key(topic, researcher_bg, expert_bg, questionnaire)
        
        # Check if already cached
        with DSAG_LOCK:
            existing_state = DSAG_CACHE.get(cache_key)
            if existing_state and existing_state.is_ready():
                # Reuse cached graph
                set_dsag_state(existing_state, cache_key)
                return jsonify({
                    "success": True,
                    "cached": True,
                    "cache_key": cache_key,
                    "metadata": existing_state.graph.metadata if existing_state.graph else {},
                })
        
        # Create new state (building)
        new_state = DSAGState(
            cache_key=cache_key,
            status="building",
        )
        set_dsag_state(new_state, cache_key)
        
        # Build the graph (this may take 30-60 seconds)
        try:
            factory = GraphFactory()
            graph = factory.generate_graph(topic, researcher_bg, expert_bg, questionnaire=questionnaire)
            
            # Build embeddings index
            embedding_index = build_embedding_index(graph)
            
            # Update state
            with DSAG_LOCK:
                new_state.graph = graph
                new_state.expert_leaf_embeddings = embedding_index.expert_leaf_embeddings
                new_state.researcher_leaf_embeddings = embedding_index.researcher_leaf_embeddings
                new_state.status = "ready"
                new_state.error = ""
            
            return jsonify({
                "success": True,
                "cached": False,
                "cache_key": cache_key,
                "metadata": graph.metadata,
            })
        
        except Exception as e:
            with DSAG_LOCK:
                new_state.status = "error"
                new_state.error = str(e)
            return jsonify({
                "success": False,
                "error": f"Failed to build DSAG graph: {str(e)}",
            })
    
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/dsag/status", methods=["GET"])
def api_dsag_status():
    """
    Get the status of the current DSAG session.
    
    Response:
    {
        "ready": true/false,
        "status": "building|ready|error",
        "error": "...",
        "metadata": {...}
    }
    """
    try:
        state = get_dsag_state()
        if state is None:
            return jsonify({
                "ready": False,
                "status": "not_initialized",
                "error": "DSAG not initialized. Call /api/dsag/init first.",
            })
        
        return jsonify({
            "ready": state.is_ready(),
            "status": state.status,
            "error": state.error,
            "metadata": state.graph.metadata if state.graph else {},
        })
    
    except Exception as exc:
        return jsonify({"ready": False, "status": "error", "error": str(exc)})


@app.route("/api/dsag/analyze_turn", methods=["POST"])
def api_dsag_analyze_turn():
    """
    Analyze a conversation turn using the DSAG graph.

    Request body:
    {
        "researcher_question": "The researcher's question",
        "expert_answer": "The expert's response"
    }

    Response:
    {
        "success": true,
        "analysis": {
            "located": {...},
            "divergence": {...},
            "assistance": {
                "relation_type": "LexicalGap|ConceptualGap|TacitGap|ScopeGap|ProcessGap",
                "payload": {...}
            },
            ...
        }
    }
    """
    try:
        state = get_dsag_state()
        if state is None or not state.is_ready():
            return jsonify({
                "success": False,
                "error": "DSAG not ready. Call /api/dsag/init first.",
            })

        data = request.get_json()
        researcher_question = data.get("researcher_question", "").strip()
        expert_answer = data.get("expert_answer", "").strip()

        if not expert_answer:
            return jsonify({"success": False, "error": "Expert answer is required"})

        # Create embedding index from stored embeddings
        embedding_index = EmbeddingIndex(state.graph)
        embedding_index.load_embeddings_data({
            "expert": state.expert_leaf_embeddings,
            "researcher": state.researcher_leaf_embeddings,
        })

        # Create runtime engine and analyze with timeline
        context_summary = build_context_summary(get_messages())
        engine = RuntimeEngine(state.graph, embedding_index)
        analysis = engine.analyze_turn(
            researcher_question,
            expert_answer,
            context_summary=context_summary,
            interview_timeline=state.interview_timeline,
        )

        # Accumulate timeline entry for Process Gap tracking
        if analysis.located.best_expert_leaf_id:
            expert_node = state.graph.expert_tree.get_node(
                analysis.located.best_expert_leaf_id
            )
            timeline_entry = {
                "turn_index": len(state.interview_timeline) + 1,
                "topic_label": expert_node.label if expert_node else "",
                "expert_leaf_id": analysis.located.best_expert_leaf_id or "",
                "researcher_leaf_id": analysis.located.best_researcher_leaf_id or "",
                "relation_type": (
                    analysis.selected_link.relation_type
                    if analysis.selected_link else ""
                ),
                "summary": f"Q: {researcher_question[:80]} | A: {expert_answer[:80]}",
            }
            with DSAG_LOCK:
                state.interview_timeline.append(timeline_entry)

        return jsonify({
            "success": True,
            "analysis": analysis.to_dict(),
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/dsag/graph", methods=["GET"])
def api_dsag_graph():
    """
    Get the full DSAG graph structure (for debugging/visualization).
    
    Response:
    {
        "success": true,
        "graph": {...}
    }
    """
    try:
        state = get_dsag_state()
        if state is None or not state.is_ready():
            return jsonify({
                "success": False,
                "error": "DSAG not ready. Call /api/dsag/init first.",
            })
        
        return jsonify({
            "success": True,
            "graph": state.graph.to_dict(),
        })
    
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


# ============== Main Routes ==============

@app.route("/", methods=["GET", "POST"])
def index():
    # Clear session if requested via query param
    if request.args.get("reset") == "1":
        clear_dsag_state()
        session.clear()
        return redirect(url_for("index"))

    messages = get_messages()
    if request.method == "POST":
        action = request.form.get("action", "chat")

        if action == "save_guide":
            guide_text = request.form.get("guide_text", "")
            set_guide_text(guide_text)
            set_guide_error(None)
            return redirect(url_for("index"))

        if action == "upload_guide":
            uploaded = request.files.get("guide_file")
            if not uploaded or not uploaded.filename:
                set_guide_error("Please choose a file to upload.")
            else:
                try:
                    extracted = extract_text_from_upload(uploaded)
                    set_guide_text(extracted)
                    set_guide_error(None)
                except Exception as exc:
                    set_guide_error(str(exc))
            return redirect(url_for("index"))

        if action == "toggle_mismap":
            try:
                msg_index = int(request.form.get("msg_index", "-1"))
            except ValueError:
                msg_index = -1
            if 0 <= msg_index < len(messages):
                target = messages[msg_index]
                if target.get("role") == "expert":
                    if target.get("manual_mismap"):
                        clear_manual_mismap(target)
                    else:
                        ensure_manual_mismap(target)
                session["messages"] = messages
            return redirect(url_for("index"))

        # Handle chat messages
        researcher_text = request.form.get("researcher_input", "").strip()
        expert_text = request.form.get("expert_input", "").strip()
        researcher_source = request.form.get("researcher_source", "text")
        expert_source = request.form.get("expert_source", "text")

        if researcher_text:
            researcher_msg = {
                "role": "researcher",
                "content": researcher_text,
                "source": researcher_source,
            }
            messages.append(researcher_msg)

        if expert_text:
            msg = {"role": "expert", "content": expert_text, "source": expert_source}
            last_question = ""
            for m in reversed(messages):
                if m["role"] == "researcher":
                    last_question = m["content"]
                    break

            # ---- DSAG auto-analysis ----
            dsag_state = get_dsag_state()
            if dsag_state and dsag_state.is_ready() and last_question:
                try:
                    embedding_index = EmbeddingIndex(dsag_state.graph)
                    embedding_index.load_embeddings_data({
                        "expert": dsag_state.expert_leaf_embeddings,
                        "researcher": dsag_state.researcher_leaf_embeddings,
                    })
                    context_summary = build_context_summary(messages)
                    engine = RuntimeEngine(dsag_state.graph, embedding_index)
                    dsag_analysis = engine.analyze_turn(
                        last_question,
                        expert_text,
                        context_summary=context_summary,
                        interview_timeline=dsag_state.interview_timeline,
                    )

                    # Accumulate timeline entry
                    if dsag_analysis.located.best_expert_leaf_id:
                        expert_node = dsag_state.graph.expert_tree.get_node(
                            dsag_analysis.located.best_expert_leaf_id
                        )
                        timeline_entry = {
                            "turn_index": len(dsag_state.interview_timeline) + 1,
                            "topic_label": expert_node.label if expert_node else "",
                            "expert_leaf_id": dsag_analysis.located.best_expert_leaf_id or "",
                            "researcher_leaf_id": dsag_analysis.located.best_researcher_leaf_id or "",
                            "relation_type": (
                                dsag_analysis.selected_link.relation_type
                                if dsag_analysis.selected_link else ""
                            ),
                            "summary": f"Q: {last_question[:80]} | A: {expert_text[:80]}",
                        }
                        with DSAG_LOCK:
                            dsag_state.interview_timeline.append(timeline_entry)

                    # Store analysis result in message for template rendering
                    msg["dsag_analysis"] = dsag_analysis.to_dict()
                except Exception as dsag_exc:
                    print(f"[DSAG auto-analysis] Error: {dsag_exc}")

            messages.append(msg)

        session["messages"] = messages
        return redirect(url_for("index"))

    # Check if DSAG is ready for template rendering
    dsag_state = get_dsag_state()
    dsag_ready = dsag_state is not None and dsag_state.is_ready()

    return render_template(
        "index.html",
        messages=messages,
        guide_text=get_guide_text(),
        guide_error=session.get("guide_error"),
        dsag_ready=dsag_ready,
    )


if __name__ == "__main__":
    app.run(debug=True)
