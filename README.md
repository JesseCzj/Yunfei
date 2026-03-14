# HCI Interview Baseline

A plain interview system for HCI research. An AI-simulated domain expert answers researcher questions in real time — no gap detection, no assistance, no analysis overlays.

## What it does

- Researcher types (or speaks) a question; the expert LLM responds automatically
- Interview questionnaire can be uploaded (txt / docx / pdf) for reference during the session
- Voice input supported via Web Speech API (Chrome / Edge)

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your API credentials:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_BASE_URL=...   # optional, for compatible endpoints
```

## Run

```bash
python -m dotenv run python app.py
```

Then open `http://localhost:5000`.

## Expert persona

The simulated expert is a senior university faculty member with 15+ years of experience grading project-based assignments. The persona is defined in `app.py` (`INTERVIEWEE_DEMOGRAPHICS` and `INTERVIEWEE_PERSONA_PROMPT_V1`).
