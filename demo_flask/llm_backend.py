import json
import os
import re
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# ============== HCI Terms Dictionary ==============
# Predefined dictionary of HCI/AI/UX terms with explanations
HCI_TERMS_DICTIONARY = {
    "HCI": "Human-Computer Interaction, the study of how people interact with computers",
    "UX": "User Experience, the overall experience of using a product",
    "UI": "User Interface, the visual elements users interact with",
    "AI": "Artificial Intelligence, computer systems that can perform human-like tasks",
    "ML": "Machine Learning, AI systems that learn from data",
    "usability": "how easy and efficient a product is to use",
    "persona": "a fictional character representing a user type",
    "wireframe": "a basic visual guide for page structure",
    "prototype": "an early sample or model of a product",
    "user research": "studying users to understand their needs and behaviors",
    "interaction design": "designing how users interact with products",
    "affordance": "visual cues that suggest how an object should be used",
    "heuristic": "a rule of thumb for evaluating usability",
    "cognitive load": "the mental effort required to use a system",
    "accessibility": "making products usable by people with disabilities",
    "user journey": "the path a user takes through a product",
    "stakeholder": "anyone with interest in a project's outcome",
    "iteration": "repeating a process to improve results",
    "feedback loop": "a cycle where outputs influence future inputs",
    "mental model": "a user's understanding of how something works",
}


# Lightweight analysis prompt - detects jargon and mis-map
ANALYZE_PROMPT = """You are an assistant for interdisciplinary HCI interviews.
Given a researcher question and an expert answer, identify:
1) domain jargon terms with simplified explanations
2) whether there is a mis-map (communication mismatch), with the type and reason

Return ONLY valid JSON with this schema:
{{ 
  "jargon": [{{
    "term": "...",
    "desc": "simplified explanation in plain language"
  }}],
  "mismap": {{
    "detected": true/false,
    "type": "...",
    "reason": "..."
  }}
}}

Mis-map types include:
- "Terminology gap": The expert used domain-specific language the researcher may not understand
- "Conceptual mismatch": The expert answered a different aspect than what was asked
- "Assumption gap": The expert assumed background knowledge the researcher lacks
- "Scope mismatch": The expert's answer is too broad/narrow for the question

If no jargon, return empty list.
If no mis-map, set detected=false and use empty strings for type/reason.
"""

# Prompt for generating question refinement
REFINEMENT_PROMPT = """You are an assistant helping HCI researchers communicate with domain experts.

The researcher asked a question and the expert's response indicates a communication mismatch.

Researcher question: {question}
Expert answer: {answer}
Mis-map reason: {reason}

Generate a refined/translated version of the researcher's question that:
1. Uses terminology the expert would better understand
2. Clarifies the intent based on the mis-map reason
3. Is more specific and actionable

Return ONLY the refined question text, no JSON or extra formatting.
"""

# Prompt for generating context examples
EXAMPLES_PROMPT = """You are an assistant helping HCI researchers explain their questions to domain experts.

The researcher asked a question and the expert didn't fully understand. Generate concrete examples
the researcher can use to illustrate their question at three granularity levels.

Researcher question: {question}
Expert answer: {answer}
Mis-map reason: {reason}

Return ONLY valid JSON with this schema:
{{
  "Coarse": "A high-level, simple example (1-2 sentences)",
  "Balanced": "A medium-detail example with some specifics",
  "Fine": "A detailed, concrete scenario with step-by-step context"
}}

The examples should be things the researcher can SAY to the expert to help explain the question,
not explanations of the conversation itself.
"""

# Prompt for explaining expert's response (kept for backward compatibility)
EXPLAIN_PROMPT = """You are an assistant helping HCI researchers understand domain experts' responses.

The researcher doesn't fully understand the expert's answer. Provide a clear explanation
in plain terms that helps the researcher understand what the expert means.

Researcher question: {question}
Expert answer: {answer}

Explain the expert's response in simple terms:
1. What is the expert actually saying?
2. What domain concepts or assumptions are implicit?
3. What is the key takeaway for the researcher?

Return a clear, helpful explanation (2-4 sentences). No JSON, just plain text.
"""

# Prompt for extracting key points from multiple expert responses
KEYPOINTS_PROMPT = """You are an assistant helping HCI researchers extract key information from domain expert interviews.

The researcher has selected several expert responses and wants to extract the key points.

Researcher question: {question}

Expert responses:
{answers}

Extract the most important key points from these responses. Focus on:
1. Core facts and information the researcher should note
2. Key workflow steps or processes mentioned
3. Pain points or challenges identified
4. Important domain concepts explained

Return ONLY valid JSON with this schema:
{{
  "keypoints": [
    "Key point 1...",
    "Key point 2...",
    "Key point 3..."
  ]
}}

Provide 3-6 concise, actionable key points.
"""

# Prompt for suggesting follow-up questions to extract criteria from intuition
FOLLOWUPS_PROMPT = """You are an assistant helping HCI researchers conduct better interviews with domain experts.

The expert gave responses that contain intuition-based or implicit knowledge. Help the researcher
formulate follow-up questions to extract specific, actionable criteria from the expert's intuition.

Researcher question: {question}
Expert answer: {answer}

Generate follow-up questions that will help the researcher:
1. Convert vague intuitions into specific observable criteria
2. Understand the decision-making process step by step
3. Extract concrete examples or scenarios

Return ONLY valid JSON with this schema:
{{
  "followups": [
    "Follow-up question 1?",
    "Follow-up question 2?",
    "Follow-up question 3?"
  ]
}}

Provide 2-4 thoughtful follow-up questions.
"""

USER_PROMPT = """Researcher question:
{question}

Expert answer:
{answer}
"""

# Prompt for mapping expert concepts to HCI domain (combined extract + mapping)
HCI_MAPPING_PROMPT = """You are an assistant helping HCI researchers map domain expert knowledge to HCI/UX research.

The expert shared domain-specific concepts, processes, and experiences. Help the researcher understand 
how these relate to HCI/UX research by providing clear mappings.

Researcher question: {question}
Expert answer: {answer}

For each significant concept, process, or workflow in the expert's answer:
1. Identify the domain-specific element (concept, term, process, criterion, etc.)
2. Provide the HCI/UX research equivalent (could be a methodology, workflow step, design principle, keyword, etc.)
3. Briefly explain how this mapping helps the researcher understand or apply the knowledge

Return ONLY valid JSON with this schema:
{{
  "mappings": [
    {{
      "expert_concept": "The domain-specific concept/process/criterion",
      "hci_equivalent": "The HCI/UX equivalent (e.g., 'user research workflow', 'usability heuristic', 'design pattern')",
      "explanation": "Brief explanation of how this maps to HCI research practice"
    }}
  ]
}}

Examples of good mappings:
- Expert "patient triage" → HCI "user feedback prioritization"
- Expert "clinical workflow" → HCI "user task flow analysis"
- Expert "diagnostic criteria" → HCI "usability evaluation criteria"

Focus on mappings that are practical and actionable for HCI researchers.
"""


def _default_result() -> Dict[str, Any]:
    """Default result for lightweight analysis (no refinement/contexts)."""
    return {
        "jargon": [],
        "mismap": {
            "detected": False,
            "type": "",
            "reason": "",
        },
    }


def _sanitize_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize and normalize the lightweight analysis result."""
    result = _default_result()
    jargon = data.get("jargon", [])
    if isinstance(jargon, list):
        normalized: List[Dict[str, str]] = []
        for item in jargon:
            term = str(item.get("term", "")).strip()
            desc = str(item.get("desc", "")).strip()
            if term and desc:
                normalized.append({
                    "term": term,
                    "desc": desc,
                })
        result["jargon"] = normalized

    mismap = data.get("mismap", {})
    if isinstance(mismap, dict):
        detected = bool(mismap.get("detected", False))
        result["mismap"]["detected"] = detected
        if detected:
            result["mismap"]["type"] = str(mismap.get("type", "")).strip()
            result["mismap"]["reason"] = str(mismap.get("reason", "")).strip()
    return result


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with fallback regex extraction."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def _build_llm() -> ChatOpenAI:
    """Build LLM instance based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=0.2,
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        model = os.getenv("OPENAI_MODEL", "qwen3-max")
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            return ChatOpenAI(api_key=api_key, model=model, base_url=base_url, temperature=0.2)
        return ChatOpenAI(api_key=api_key, model=model, temperature=0.2)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def analyze_exchange(question: str, answer: str) -> Dict[str, Any]:
    """
    Lightweight analysis: detect jargon and mis-map only.
    Does NOT generate refinement or context examples (those are on-demand).
    """
    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("system", ANALYZE_PROMPT), ("user", USER_PROMPT)]
        )
        chain = prompt | llm
        response = chain.invoke({"question": question, "answer": answer})
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        return _sanitize_result(parsed)
    except Exception as exc:
        print(f"[LLM Backend Error] {exc}")
        return _default_result()


def generate_refinement(question: str, answer: str, reason: str) -> str:
    """
    Generate a refined/translated version of the researcher's question.
    Called on-demand when user clicks "Get Refinement" button.
    """
    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("user", REFINEMENT_PROMPT)]
        )
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "reason": reason or "Communication mismatch detected",
        })
        content = getattr(response, "content", str(response))
        return content.strip()
    except Exception as exc:
        print(f"[LLM Backend Error - Refinement] {exc}")
        return ""


def generate_context_examples(question: str, answer: str, reason: str) -> Dict[str, str]:
    """
    Generate context examples at three granularities.
    Called on-demand when user clicks "Get Examples" button.
    """
    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("user", EXAMPLES_PROMPT)]
        )
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "reason": reason or "Communication mismatch detected",
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        # Ensure all three granularities exist
        return {
            "Coarse": str(parsed.get("Coarse", "")).strip(),
            "Balanced": str(parsed.get("Balanced", "")).strip(),
            "Fine": str(parsed.get("Fine", "")).strip(),
        }
    except Exception as exc:
        print(f"[LLM Backend Error - Examples] {exc}")
        return {"Coarse": "", "Balanced": "", "Fine": ""}


def explain_expert_response(question: str, answer: str) -> str:
    """
    Explain the expert's response in plain terms for the researcher.
    Called on-demand when user clicks "I Don't Understand" button.
    """
    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("user", EXPLAIN_PROMPT)]
        )
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "answer": answer,
        })
        content = getattr(response, "content", str(response))
        return content.strip()
    except Exception as exc:
        print(f"[LLM Backend Error - Explain] {exc}")
        return ""


def extract_key_points(question: str, answers: List[str]) -> List[str]:
    """
    Extract key information from multiple expert answers.
    Called when user selects multiple expert messages and clicks "Extract Key Points".
    """
    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("user", KEYPOINTS_PROMPT)]
        )
        chain = prompt | llm
        
        # Combine multiple answers with separators
        combined_answers = "\n---\n".join(
            f"Response {i+1}: {ans}" for i, ans in enumerate(answers)
        )
        
        response = chain.invoke({
            "question": question,
            "answers": combined_answers,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        keypoints = parsed.get("keypoints", [])
        if isinstance(keypoints, list):
            return [str(kp).strip() for kp in keypoints if str(kp).strip()]
        return []
    except Exception as exc:
        print(f"[LLM Backend Error - KeyPoints] {exc}")
        return []


def suggest_followup_questions(question: str, answer: str) -> List[str]:
    """
    Suggest follow-up questions to help extract criteria from expert's intuition.
    Called when user clicks "Suggest Follow-ups" button.
    """
    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("user", FOLLOWUPS_PROMPT)]
        )
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "answer": answer,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        followups = parsed.get("followups", [])
        if isinstance(followups, list):
            return [str(fq).strip() for fq in followups if str(fq).strip()]
        return []
    except Exception as exc:
        print(f"[LLM Backend Error - Followups] {exc}")
        return []


def detect_hci_terms(text: str) -> List[Dict[str, str]]:
    """
    Detect HCI/AI/UX terms in text using predefined dictionary.
    Returns list of detected terms with their explanations.
    """
    detected = []
    text_lower = text.lower()
    
    for term, explanation in HCI_TERMS_DICTIONARY.items():
        # Check for whole word match (case insensitive)
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, text_lower):
            # Find the original case in the text
            match = re.search(pattern, text_lower)
            if match:
                start = match.start()
                original_term = text[start:start + len(term)]
                detected.append({
                    "term": original_term,
                    "explanation": explanation
                })
    
    return detected


def map_to_hci(question: str, answer: str) -> List[Dict[str, str]]:
    """
    Map expert's domain concepts to HCI/UX research equivalents.
    Called when user clicks "Map to HCI" button.
    """
    try:
        llm = _build_llm()
        prompt = ChatPromptTemplate.from_messages(
            [("user", HCI_MAPPING_PROMPT)]
        )
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "answer": answer,
        })
        content = getattr(response, "content", str(response))
        parsed = _parse_json(content)
        
        mappings = parsed.get("mappings", [])
        if isinstance(mappings, list):
            result = []
            for m in mappings:
                if isinstance(m, dict):
                    result.append({
                        "expert_concept": str(m.get("expert_concept", "")).strip(),
                        "hci_equivalent": str(m.get("hci_equivalent", "")).strip(),
                        "explanation": str(m.get("explanation", "")).strip(),
                    })
            return result
        return []
    except Exception as exc:
        print(f"[LLM Backend Error - HCI Mapping] {exc}")
        return []
