"""
Agentic pipeline — 4 separate LLM calls, each with a targeted prompt.
Multi-step reasoning: decisions → action items → conflicts → unresolved questions.
Uses llm_service.call_llm so any provider (Gemini/Groq/Ollama) works.
"""

import logging
from typing import Dict, Any

from app.services.llm_service import call_llm, parse_json_response

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────

_DECISIONS_PROMPT = """You are analyzing a meeting transcript. Extract ONLY the decisions that were clearly made and agreed upon.

TRANSCRIPT:
{transcript}

Return a JSON array of decisions. Each object must have:
- "text": the decision (1-2 sentences, factual)
- "owner": person responsible (or null if not specified)

Rules:
- Only include things that were DECIDED, not ideas/suggestions
- If nothing was decided, return []
- Return ONLY the JSON array, no other text

Example: [{{"text": "We will use PostgreSQL for the database.", "owner": "Alice"}}]
"""

_ACTION_ITEMS_PROMPT = """You are analyzing a meeting transcript. Extract action items — tasks assigned to specific people.

TRANSCRIPT:
{transcript}

Return a JSON array of action items. Each object must have:
- "text": what needs to be done (clear, actionable)
- "owner": who is responsible (or null if unclear)
- "deadline": when it's due (e.g. "Friday", "March 15", "next week", or null)

Rules:
- An action item must be something someone agreed to DO
- If no action items, return []
- Return ONLY the JSON array, no other text

Example: [{{"text": "Set up the CI pipeline", "owner": "Tom", "deadline": "Friday"}}]
"""

_CONFLICTS_PROMPT = """You are analyzing a meeting transcript. Identify conflicts — where two or more people argued different positions and NO clear resolution was reached.

TRANSCRIPT:
{transcript}

Return a JSON array of conflicts. Each object must have:
- "party_a": first person/side (or null if unnamed)
- "party_b": second person/side (or null if unnamed)
- "issue": what they disagreed about (1-2 sentences)
- "resolved": false (since these are unresolved conflicts)

Rules:
- Only include UNRESOLVED disagreements where both sides are mentioned
- If no conflicts exist, return []
- Return ONLY the JSON array, no other text

Example: [{{"party_a": "John", "party_b": "Sarah", "issue": "John prefers PostgreSQL, Sarah prefers MySQL. No decision reached.", "resolved": false}}]
"""

_QUESTIONS_PROMPT = """You are analyzing a meeting transcript. Find questions that were asked but NEVER answered during the meeting.

TRANSCRIPT:
{transcript}

Return a JSON array of unresolved questions. Each object must have:
- "question": the unanswered question (quoted or paraphrased)
- "asker": who asked it (or null if unclear)

Rules:
- Only include questions that received NO answer in the transcript 
- Skip questions that were clearly answered
- If all questions were answered, return []
- Return ONLY the JSON array, no other text

Example: [{{"question": "What is the timeline for the mobile app?", "asker": "Maria"}}]
"""


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(transcript: str) -> Dict[str, Any]:
    """
    Run 4 sequential LLM calls on the transcript.
    Returns dict with decisions, action_items, conflicts, unresolved_questions,
    and the llm_provider that was used.
    """
    results = {}
    provider_used = None

    steps = [
        ("decisions",             _DECISIONS_PROMPT),
        ("action_items",          _ACTION_ITEMS_PROMPT),
        ("conflicts",             _CONFLICTS_PROMPT),
        ("unresolved_questions",  _QUESTIONS_PROMPT),
    ]

    for key, prompt_template in steps:
        prompt = prompt_template.format(transcript=transcript)
        try:
            raw, provider = call_llm(prompt)
            if provider_used is None:
                provider_used = provider  # track first responding provider
            data = parse_json_response(raw)
            if not isinstance(data, list):
                data = []
            results[key] = data
            logger.info(f"[AGENT] Step '{key}' → {len(data)} items via {provider}")
        except Exception as e:
            logger.error(f"[AGENT] Step '{key}' failed: {e}")
            results[key] = []

    results["llm_provider"] = provider_used or "unknown"
    return results
