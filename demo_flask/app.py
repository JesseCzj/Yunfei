from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import io
import os
import uuid
import threading
from typing import Any, Dict, Optional

from llm_backend import (
    analyze_exchange,
    generate_refinement,
    generate_context_examples,
    explain_expert_response,
    extract_key_points,
    suggest_followup_questions,
    detect_hci_terms,
    map_to_hci,
)

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


# ============== API Routes for On-Demand Assistance ==============

@app.route("/api/get_refinement", methods=["POST"])
def api_get_refinement():
    """Generate and store question refinement for a specific message."""
    try:
        data = request.get_json()
        msg_index = int(data.get("msg_index", -1))
        messages = get_messages()
        
        if not (0 <= msg_index < len(messages)):
            return jsonify({"success": False, "error": "Invalid message index"})
        
        target = messages[msg_index]
        if target.get("role") != "expert":
            return jsonify({"success": False, "error": "Not an expert message"})
        
        # Find the question that preceded this answer
        question = find_last_question(messages, msg_index)
        if not question:
            return jsonify({"success": False, "error": "No prior question found"})
        
        answer = target.get("content", "")
        mismap = target.get("mismap", {})
        reason = mismap.get("reason", "")
        
        # Generate refinement
        refinement = generate_refinement(question, answer, reason)
        if not refinement:
            return jsonify({"success": False, "error": "Failed to generate refinement"})
        
        # Store in message and save
        target["refinement"] = refinement
        session["messages"] = messages
        
        return jsonify({"success": True, "data": refinement})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/get_examples", methods=["POST"])
def api_get_examples():
    """Generate and store context examples for a specific message."""
    try:
        data = request.get_json()
        msg_index = int(data.get("msg_index", -1))
        messages = get_messages()
        
        if not (0 <= msg_index < len(messages)):
            return jsonify({"success": False, "error": "Invalid message index"})
        
        target = messages[msg_index]
        if target.get("role") != "expert":
            return jsonify({"success": False, "error": "Not an expert message"})
        
        # Find the question that preceded this answer
        question = find_last_question(messages, msg_index)
        if not question:
            return jsonify({"success": False, "error": "No prior question found"})
        
        answer = target.get("content", "")
        mismap = target.get("mismap", {})
        reason = mismap.get("reason", "")
        
        # Generate context examples
        contexts = generate_context_examples(question, answer, reason)
        if not any(contexts.values()):
            return jsonify({"success": False, "error": "Failed to generate examples"})
        
        # Store in message and save
        target["contexts"] = contexts
        session["messages"] = messages
        
        return jsonify({"success": True, "data": contexts})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """Generate and store explanation for expert's response."""
    try:
        data = request.get_json()
        msg_index = int(data.get("msg_index", -1))
        messages = get_messages()
        
        if not (0 <= msg_index < len(messages)):
            return jsonify({"success": False, "error": "Invalid message index"})
        
        target = messages[msg_index]
        if target.get("role") != "expert":
            return jsonify({"success": False, "error": "Not an expert message"})
        
        # Find the question that preceded this answer
        question = find_last_question(messages, msg_index)
        if not question:
            return jsonify({"success": False, "error": "No prior question found"})
        
        answer = target.get("content", "")
        
        # Generate explanation
        explanation = explain_expert_response(question, answer)
        if not explanation:
            return jsonify({"success": False, "error": "Failed to generate explanation"})
        
        # Store in message and save
        target["explanation"] = explanation
        session["messages"] = messages
        
        return jsonify({"success": True, "data": explanation})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/extract_keypoints", methods=["POST"])
def api_extract_keypoints():
    """Extract key points from multiple selected expert messages."""
    try:
        data = request.get_json()
        msg_indices = data.get("msg_indices", [])
        messages = get_messages()
        
        if not msg_indices:
            return jsonify({"success": False, "error": "No messages selected"})
        
        # Validate all indices and collect answers
        answers = []
        for idx in msg_indices:
            if not (0 <= idx < len(messages)):
                continue
            target = messages[idx]
            if target.get("role") == "expert":
                answers.append(target.get("content", ""))
        
        if not answers:
            return jsonify({"success": False, "error": "No valid expert messages found"})
        
        # Find the most recent researcher question
        question = ""
        for m in reversed(messages):
            if m.get("role") == "researcher":
                question = m.get("content", "")
                break
        
        # Generate key points
        keypoints = extract_key_points(question, answers)
        if not keypoints:
            return jsonify({"success": False, "error": "Failed to extract key points"})
        
        # Store in session for display (not in individual messages)
        session["keypoints_result"] = {
            "msg_indices": msg_indices,
            "keypoints": keypoints,
        }
        
        return jsonify({"success": True, "data": keypoints})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/suggest_followups", methods=["POST"])
def api_suggest_followups():
    """Generate follow-up questions for a specific expert message."""
    try:
        data = request.get_json()
        msg_index = int(data.get("msg_index", -1))
        messages = get_messages()
        
        if not (0 <= msg_index < len(messages)):
            return jsonify({"success": False, "error": "Invalid message index"})
        
        target = messages[msg_index]
        if target.get("role") != "expert":
            return jsonify({"success": False, "error": "Not an expert message"})
        
        # Find the question that preceded this answer
        question = find_last_question(messages, msg_index)
        if not question:
            return jsonify({"success": False, "error": "No prior question found"})
        
        answer = target.get("content", "")
        
        # Generate follow-up questions
        followups = suggest_followup_questions(question, answer)
        if not followups:
            return jsonify({"success": False, "error": "Failed to generate follow-ups"})
        
        # Store in message and save
        target["followups"] = followups
        session["messages"] = messages
        
        return jsonify({"success": True, "data": followups})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route("/api/hci_mapping", methods=["POST"])
def api_hci_mapping():
    """Map expert concepts to HCI domain."""
    try:
        data = request.get_json()
        msg_index = int(data.get("msg_index", -1))
        messages = get_messages()
        
        if not (0 <= msg_index < len(messages)):
            return jsonify({"success": False, "error": "Invalid message index"})
        
        target = messages[msg_index]
        if target.get("role") != "expert":
            return jsonify({"success": False, "error": "Not an expert message"})
        
        # Find the question that preceded this answer
        question = find_last_question(messages, msg_index)
        if not question:
            return jsonify({"success": False, "error": "No prior question found"})
        
        answer = target.get("content", "")
        
        # Generate HCI domain mappings
        mappings = map_to_hci(question, answer)
        if not mappings:
            return jsonify({"success": False, "error": "No domain concepts found to map"})
        
        # Store in message and save
        target["hci_mapping"] = mappings
        session["messages"] = messages
        
        return jsonify({"success": True, "data": mappings})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


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
                "payload": {...},
                "followup_questions": [...]
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
            # Detect HCI terms in researcher's question
            hci_terms = detect_hci_terms(researcher_text)
            if hci_terms:
                researcher_msg["hci_terms"] = hci_terms
            messages.append(researcher_msg)

        if expert_text:
            msg = {"role": "expert", "content": expert_text, "source": expert_source}
            last_question = ""
            for m in reversed(messages):
                if m["role"] == "researcher":
                    last_question = m["content"]
                    break

            # Lightweight analysis: only detect jargon and mis-map
            if last_question:
                analysis = analyze_exchange(last_question, expert_text)
                if analysis.get("jargon"):
                    msg["jargon"] = analysis["jargon"]

                mismap = analysis.get("mismap", {})
                if mismap.get("detected"):
                    msg["mismap"] = mismap

            messages.append(msg)

        session["messages"] = messages
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        messages=messages,
        guide_text=get_guide_text(),
        guide_error=session.get("guide_error"),
    )


if __name__ == "__main__":
    app.run(debug=True)
