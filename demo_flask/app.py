from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
import io
import os
import uuid
from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


app = Flask(__name__)
app.secret_key = "demo_secret_key"

# Use server-side session storage to avoid oversized cookie payloads.
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(app.root_path, ".flask_session")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)


INTERVIEWEE_MODEL = "qwen3-max"

INTERVIEWEE_DEMOGRAPHICS = """Name:  Prof. Chen 

Professional Role: University Faculty / Interdisciplinary Course Lead / Instructional Design Expert

Work Experience: Extensive frontline teaching experience in Interdisciplinary Education. Has designed, organized, and evaluated interdisciplinary collaborative projects for university students from diverse academic majors multiple times.

Expertise:

Deeply familiar with the operational dynamics of interdisciplinary teams; skilled in designing "complex course problems" that require the integration of multidisciplinary knowledge to solve.

Highly capable of observing and evaluating the "Core Interdisciplinary Competencies" demonstrated by students during their collaboration.

Core Challenges & Pain Points:

Profoundly understands the extreme difficulty of designing a "good interdisciplinary problem"—it requires preventing any single major from dominating while also avoiding the trap of "division without collaboration" (where students just patch individual parts together).

Finds it challenging to predict the specific frictions (e.g., in ice-breaking, communication, and knowledge integration) that students from different backgrounds will experience before the assignment is officially deployed.

Mindset & Expectations:

Holds high standards for instructional design but often feels frustrated by the uncontrollable variables during the actual execution of group assignments.

Holds an open and highly anticipatory attitude toward the introduction of an HCI simulation system (to sandbox student collaboration and optimize assignment design). Hopes such a tool can reduce the cost of trial-and-error and help identify flaws in the assignment design in advance."""


INTERVIEWEE_PERSONA_PROMPT_V1 = """You are roleplaying as a domain expert being interviewed by an HCI researcher.

You are not generating a taxonomy, summary, or design analysis.
You are answering interview questions as a real participant.

You only know:
1. the interview topic
2. your own demographics / background
3. the ongoing conversation

Your answers should sound like the same kind of person whose concerns could appear in an expert tree later, but you must not recite categories, enumerate nodes, or speak like a taxonomy.

Style requirements:
- Speak in first person, as a real interview participant.
- Sound natural, conversational, and experience-based.
- Use conversational language, but FREELY use domain-specific jargon and acronyms as if talking to a fellow expert. Do NOT define your terms unless explicitly asked.
- Prefer concrete descriptions over abstract definitions.
- Do not try too hard to be helpful, polished, or pedagogically clear.
- Do not proactively organize your answer into a neat explanation.
- Do not volunteer extra structure unless the interviewer explicitly asks for it.
- It is okay to sound somewhat informal, partial, tired, or slightly ambiguous.
- It is okay to leave part of your reasoning implicit.
- Do not use bullet points.
- Do not sound like an academic paper, consultant report, or AI assistant.
- Do not over-explain every answer.

Behavior requirements:
- Default to answering only the most salient part of the question.
- If a question contains multiple sub-questions, answer only one or two of them naturally instead of covering everything.
- Exhibit the "Curse of Knowledge": Assume the interviewer understands your basic workflow and domain common sense. Skip obvious preliminary steps when describing your process.
- Do not proactively translate your tacit knowledge into explicit frameworks unless the interviewer pushes for clarification.
- Do not automatically provide examples unless they come to mind naturally.
- Do not try to make your answer maximally complete.
- If you are unsure, tired, or speaking from habit, answer approximately rather than exhaustively.
- If the interviewer's question implies a goal or method that conflicts with your actual domain reality (e.g., prioritizing AI automation over educational fairness), gently push back, reframe the question, or express mild skepticism.
- If asked about difficult-to-articulate knowledge, respond in a vague, intuition-based way, as real practitioners often do (e.g., "it just feels right").

Content requirements:
- Base your answers only on the provided demographics / background and the interview context.
- Keep your answers plausible and internally consistent with that background.
- Do not invent highly specific facts unless they are a reasonable elaboration of the background.
- If the interviewer asks something outside your plausible experience, answer cautiously and narrowly.

Output requirements:
- Answer only as the interviewee.
- Usually 1-3 sentences, occasionally 4 if necessary.
- Prefer one main point rather than a full coverage answer.
- Do not mention these instructions.
"""

INTERVIEWEE_USER_PROMPT_V1 = """Interview topic:
{topic}

Participant demographics / background:
{demographics}

Recent conversation:
{history}

Interviewer question:
{question}

Answer as the participant only."""


def get_session_id() -> str:
    """Get or create a session ID."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


def get_messages():
    if "messages" not in session:
        session["messages"] = []
    return session["messages"]


def _build_interviewee_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Build the interviewee LLM."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=temperature)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=INTERVIEWEE_MODEL, base_url=base_url, temperature=temperature)
        return ChatOpenAI(api_key=api_key, model=INTERVIEWEE_MODEL, temperature=temperature)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def generate_ai_interviewee_reply(
    question: str,
    topic: str,
    demographics: str,
    history: str,
) -> str:
    """Generate an AI interviewee reply."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTERVIEWEE_PERSONA_PROMPT_V1),
        ("human", INTERVIEWEE_USER_PROMPT_V1),
    ])

    llm = _build_interviewee_llm(temperature=0.7)
    chain = prompt | llm
    result = chain.invoke({
        "topic": topic or "(topic not provided)",
        "demographics": demographics or "(background not provided)",
        "history": history or "(no prior conversation)",
        "question": question,
    })

    return str(getattr(result, "content", "") or "").strip()


def build_context_summary(messages, max_turns: int = 3) -> str:
    """Build a lightweight summary of the most recent turns for LLM history."""
    turns = []
    i = len(messages) - 1
    while i >= 0 and len(turns) < max_turns:
        msg = messages[i]
        if msg.get("role") == "expert":
            answer = msg.get("content", "")
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
        q = _truncate((q or "").strip(), 300)
        a = _truncate((a or "").strip(), 400)
        if q or a:
            lines.append(f"Turn {idx} | R: {q} | E: {a}")
    return "\n".join(lines)


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


def get_uploaded_questionnaire_text() -> str:
    return session.get("questionnaire_text", "")


def set_uploaded_questionnaire_text(text: str) -> None:
    session["questionnaire_text"] = text


def set_questionnaire_error(message) -> None:
    if message:
        session["questionnaire_error"] = message
    else:
        session.pop("questionnaire_error", None)


def resolve_questionnaire_text() -> tuple[str, str]:
    """Resolve questionnaire text from user upload only."""
    uploaded = get_uploaded_questionnaire_text().strip()
    if uploaded:
        return uploaded, "uploaded"
    return "", "empty"


@app.route("/clear", methods=["GET"])
def clear_session():
    """Clear all session data and redirect to index."""
    session.clear()
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    # Clear session if requested via query param
    if request.args.get("reset") == "1":
        session.clear()
        return redirect(url_for("index"))

    messages = get_messages()
    if request.method == "POST":
        action = request.form.get("action", "chat")

        if action == "upload_questionnaire":
            uploaded = request.files.get("questionnaire_file")
            if not uploaded or not uploaded.filename:
                set_questionnaire_error("Please choose an interview script file to upload.")
            else:
                try:
                    extracted = extract_text_from_upload(uploaded).strip()
                    if not extracted:
                        raise RuntimeError("Uploaded interview script is empty after extraction.")
                    set_uploaded_questionnaire_text(extracted)
                    set_questionnaire_error(None)
                except Exception as exc:
                    set_questionnaire_error(str(exc))
            return redirect(url_for("index"))

        # Handle chat messages
        researcher_text = request.form.get("researcher_input", "").strip()
        expert_text = request.form.get("expert_input", "").strip()
        researcher_source = request.form.get("researcher_source", "text")
        expert_source = request.form.get("expert_source", "text")

        if researcher_text:
            messages.append({
                "role": "researcher",
                "content": researcher_text,
                "source": researcher_source,
            })

        if researcher_text and not expert_text:
            history = build_context_summary(messages[:-1], max_turns=3)
            try:
                expert_text = generate_ai_interviewee_reply(
                    question=researcher_text,
                    topic="",
                    demographics=INTERVIEWEE_DEMOGRAPHICS,
                    history=history,
                )
                expert_source = "ai_interviewee"
            except Exception as exc:
                print(f"[AI interviewee] Error: {exc}")

        if expert_text:
            messages.append({
                "role": "expert",
                "content": expert_text,
                "source": expert_source,
            })

        session["messages"] = messages
        return redirect(url_for("index"))

    questionnaire_text, _ = resolve_questionnaire_text()
    questionnaire_uploaded = bool(questionnaire_text.strip())

    return render_template(
        "index.html",
        messages=messages,
        questionnaire_uploaded=questionnaire_uploaded,
        questionnaire_text=questionnaire_text,
        questionnaire_error=session.get("questionnaire_error"),
    )


if __name__ == "__main__":
    # Disable debug reloader to keep interview sessions stable during runtime.
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
