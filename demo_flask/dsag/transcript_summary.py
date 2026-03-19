"""
Transcript Summary Module

Provides two LLM-driven functions:
1. parse_questionnaire() — extracts main bullet points from a questionnaire (build-time)
2. classify_and_update() — classifies a QA turn into a bullet and updates the summary (per-turn)
"""

import json
import os
import re
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .schema import TranscriptSummary, MainBullet, SubBullet


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with fallback regex extraction."""
    try:
        return json.loads(text)
    except Exception:
        pass
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except Exception:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def _build_llm() -> ChatOpenAI:
    """Build LLM for transcript summary operations."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=0.3)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        model = os.getenv("OPENAI_MODEL_POLISH", "qwen-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=0.3)
        return ChatOpenAI(api_key=api_key, model=model, temperature=0.3)

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


PARSE_QUESTIONNAIRE_PROMPT = """You are extracting the main topic blocks from an interview questionnaire/script.

## Questionnaire text
{questionnaire_text}

## Task
Identify the TOP-LEVEL questions or topic blocks and extract each as one bullet.

### How to identify top-level questions
Questionnaires come in many styles. Use these heuristics to find the top-level boundaries:
- **Numbered items** (1. / 2. / Q1 / Question 1) are top-level.
- **Bolded or larger headings** that introduce a new theme are top-level.
- **Sub-items** nested under a top-level question (marked by "o", "-", "•", "a)", indentation, or follow-up probes) are NOT separate bullets — they belong to their parent.
- If the questionnaire has NO explicit numbering or headings, group by thematic shift: each distinct topic area becomes one bullet.
- When in doubt, prefer FEWER, BROADER bullets over many narrow ones.

### Output per bullet
- "label": a short descriptive label (3-10 words) summarizing the entire question block
- "source_question": the FULL original text of this question block INCLUDING all its sub-items, verbatim or lightly cleaned

Return ONLY valid JSON:
{{
  "bullets": [
    {{
      "label": "short label for the topic block",
      "source_question": "full question text with all sub-items"
    }}
  ]
}}
"""


CLASSIFY_AND_UPDATE_PROMPT = """You are maintaining a structured interview transcript summary.

## Current summary structure (main bullets with existing sub-bullets)
{summary_json}

## New Q&A turn (turn {turn_index})
- Researcher question: "{question}"
- Expert answer: "{answer}"

## Task

### Step 1 — Match to a main bullet
- Compare the Q&A content against each main bullet's label AND source_question (which contains the original interview question and its sub-items).
- Pick the BEST semantic match. The Q&A does not need to use the same words — match by underlying topic and intent.
- Conversations often drift or use domain jargon. Focus on what the expert is actually discussing, not surface keywords.
- You MUST always pick one main bullet. Every Q&A belongs somewhere — choose the closest match even if the fit is loose.

### Step 2 — Match or create a sub-bullet
- Look at the existing sub-bullets under the matched main bullet.
- If this Q&A covers the SAME specific sub-topic as an existing sub-bullet, use "merge" and provide that sub-bullet's id. Generate a merged summary that integrates both the old and new information.
- If this Q&A introduces a DIFFERENT angle or sub-topic, use "new".

### Step 3 — Write the summary
The "summary" MUST be a short descriptive phrase (3-10 words). Do NOT write full sentences.
Examples: "False alarms cause alarm fatigue", "Domain knowledge corrects AI errors", "Seurat preferred over Scanpy for clustering".

Return ONLY valid JSON:
{{
  "main_bullet_id": "mb_01",
  "action": "merge" or "new",
  "existing_sub_bullet_id": "sb_01_1 (only if action=merge, else empty string)",
  "summary": "short descriptive phrase (3-10 words)"
}}
"""


def _clean_block_text(text: str) -> str:
    """Normalize whitespace inside a questionnaire block."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _derive_block_label(source_question: str) -> str:
    """Create a short deterministic label from the first question line."""
    text = _clean_block_text(source_question)
    if not text:
        return ""
    first_line = text.splitlines()[0]
    first_line = re.sub(
        r"^\s*(?:question\s*)?(?:q\s*)?\d+\s*[\.\):：-]?\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    ).strip()
    words = first_line.split()
    if not words:
        return ""
    label = " ".join(words[:8]).strip()
    return label[:80]


def _split_numbered_questionnaire_blocks(questionnaire_text: str) -> list[str]:
    """Split questionnaire into top-level numbered blocks when clear numbering exists."""
    text = str(questionnaire_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    header_re = re.compile(
        r"^\s*(?:question\s*)?(?:q\s*)?\d+\s*[\.\):：-]\s+\S+",
        flags=re.IGNORECASE,
    )

    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if header_re.match(line):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)

    normalized_blocks = [_clean_block_text("\n".join(block)) for block in blocks]
    normalized_blocks = [block for block in normalized_blocks if block]
    return normalized_blocks if len(normalized_blocks) >= 2 else []


def parse_questionnaire(questionnaire_text: str) -> TranscriptSummary:
    """Extract main bullet points from questionnaire text.

    Called at build-time after graph construction.
    Returns a TranscriptSummary with empty sub_bullets.
    """
    if not questionnaire_text.strip():
        return TranscriptSummary()

    numbered_blocks = _split_numbered_questionnaire_blocks(questionnaire_text)
    if numbered_blocks:
        main_bullets = []
        for i, block in enumerate(numbered_blocks):
            label = _derive_block_label(block)
            mb = MainBullet(
                id=f"mb_{i+1:02d}",
                label=label or f"Question {i+1}",
                keywords=[],
                source_question=block,
            )
            main_bullets.append(mb)
        return TranscriptSummary(main_bullets=main_bullets, last_updated_turn=0)

    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("user", PARSE_QUESTIONNAIRE_PROMPT)
    ])
    chain = prompt | llm
    response = chain.invoke({"questionnaire_text": questionnaire_text[:4000]})
    content = getattr(response, "content", str(response))
    parsed = _parse_json(content)

    bullets_raw = parsed.get("bullets", [])
    main_bullets = []
    for i, b in enumerate(bullets_raw):
        if not isinstance(b, dict):
            continue
        mb = MainBullet(
            id=f"mb_{i+1:02d}",
            label=str(b.get("label", "")).strip(),
            keywords=[str(k).strip() for k in b.get("keywords", []) if str(k).strip()],
            source_question=str(b.get("source_question", "")).strip(),
        )
        if mb.label:
            main_bullets.append(mb)

    return TranscriptSummary(main_bullets=main_bullets, last_updated_turn=0)


def classify_and_update(
    transcript_summary: TranscriptSummary,
    question: str,
    answer: str,
    turn_index: int,
) -> TranscriptSummary:
    """Classify a QA turn into a main bullet and update the summary.

    Called per-turn during analyze_turn.
    """
    if not transcript_summary or not transcript_summary.main_bullets:
        return transcript_summary

    # Build a compact JSON representation for the LLM
    summary_for_llm = []
    for mb in transcript_summary.main_bullets:
        mb_data = {
            "id": mb.id,
            "label": mb.label,
            "source_question": mb.source_question,
            "sub_bullets": [
                {"id": sb.id, "summary": sb.summary, "turn_indices": sb.turn_indices}
                for sb in mb.sub_bullets
            ],
        }
        summary_for_llm.append(mb_data)

    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("user", CLASSIFY_AND_UPDATE_PROMPT)
    ])
    chain = prompt | llm
    response = chain.invoke({
        "summary_json": json.dumps(summary_for_llm, ensure_ascii=False),
        "turn_index": turn_index,
        "question": question[:500],
        "answer": answer[:800],
    })
    content = getattr(response, "content", str(response))
    parsed = _parse_json(content)

    if not parsed:
        return transcript_summary

    main_bullet_id = str(parsed.get("main_bullet_id", "")).strip()
    action = str(parsed.get("action", "new")).strip()
    existing_sb_id = str(parsed.get("existing_sub_bullet_id", "")).strip()
    summary_text = str(parsed.get("summary", "")).strip()

    if not main_bullet_id or main_bullet_id == "none" or not summary_text:
        return transcript_summary

    # Find the target main bullet
    target_mb = None
    for mb in transcript_summary.main_bullets:
        if mb.id == main_bullet_id:
            target_mb = mb
            break

    if not target_mb:
        return transcript_summary

    if action == "merge" and existing_sb_id:
        # Find and update existing sub-bullet
        for sb in target_mb.sub_bullets:
            if sb.id == existing_sb_id:
                sb.summary = summary_text
                if turn_index not in sb.turn_indices:
                    sb.turn_indices.append(turn_index)
                break
        else:
            # Sub-bullet not found, create new
            new_sb = SubBullet(
                id=f"sb_{main_bullet_id.replace('mb_', '')}_{len(target_mb.sub_bullets)+1}",
                summary=summary_text,
                turn_indices=[turn_index],
            )
            target_mb.sub_bullets.append(new_sb)
    else:
        # Create new sub-bullet
        new_sb = SubBullet(
            id=f"sb_{main_bullet_id.replace('mb_', '')}_{len(target_mb.sub_bullets)+1}",
            summary=summary_text,
            turn_indices=[turn_index],
        )
        target_mb.sub_bullets.append(new_sb)

    transcript_summary.last_updated_turn = turn_index
    return transcript_summary
