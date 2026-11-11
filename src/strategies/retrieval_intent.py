"""Shared retrieval-intent helpers for local MAF strategies."""

from typing import Any, Iterable, Mapping


GREETING = "greeting"
NO_RETRIEVAL = "no_retrieval"
QUESTION = "question"

RETRIEVAL_INTENT_SYSTEM_PROMPT = (
    "You decide whether Azure AI Search retrieval is needed for the current user message. "
    "Respond with exactly one label:\n"
    "GREETING: greetings, thanks, farewells, or small talk that can be answered without search.\n"
    "NO_RETRIEVAL_FOLLOWUP: the user only asks to transform, format, translate, summarize, "
    "shorten, rephrase, or clarify the previous assistant answer, without asking for new facts.\n"
    "RETRIEVAL_NEEDED: the user asks a new factual, domain, product, policy, or document-grounded "
    "question, asks to compare or find information, or the request is ambiguous.\n"
    "If unsure, respond RETRIEVAL_NEEDED. Respond with ONLY ONE LABEL."
)


def has_previous_answer(history: Iterable[Mapping[str, Any]] | None) -> bool:
    """True when the conversation already contains an assistant answer."""
    return any(
        message.get("role") == "assistant"
        and str(message.get("text") or message.get("content") or "").strip()
        for message in history or []
    )


def parse_retrieval_intent(
    raw_result: str | None,
    enable_no_retrieval: bool = True,
    has_previous_answer: bool = True,
) -> str:
    """Map the classifier label to an intent.

    NO_RETRIEVAL_FOLLOWUP transforms the previous answer, so it is only honoured
    when one exists. Otherwise the turn is treated as a question and retrieved.
    """
    result = (raw_result or "").strip().upper()
    first_line = result.splitlines()[0] if result else ""
    label = first_line.strip("`'\" .,:;()[]{}")
    if label.startswith("LABEL:"):
        label = label.removeprefix("LABEL:").strip(" .,:;")

    if label.startswith("GREETING"):
        return GREETING
    if enable_no_retrieval and has_previous_answer and label.startswith("NO_RETRIEVAL"):
        return NO_RETRIEVAL
    return QUESTION


def build_retrieval_intent_messages(
    user_message: str,
    history: Iterable[Mapping[str, Any]] | None,
    max_history_messages: int,
    max_history_chars: int,
) -> list[dict[str, str]]:
    history_text = format_recent_history(history or [], max_history_messages, max_history_chars)
    current_message = _truncate_text(user_message, max_history_chars)
    return [
        {"role": "system", "content": RETRIEVAL_INTENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Recent conversation, oldest to newest:\n"
                f"{history_text or '(none)'}\n\n"
                "Current user message:\n"
                f"{current_message}\n\n"
                "Return exactly one label."
            ),
        },
    ]


def format_recent_history(
    history: Iterable[Mapping[str, Any]],
    max_messages: int,
    max_chars: int,
) -> str:
    if max_messages <= 0 or max_chars <= 0:
        return ""

    selected = list(history)[-max_messages:]
    lines_reversed: list[str] = []
    remaining = max_chars

    for message in reversed(selected):
        role = str(message.get("role") or "user")
        text = str(message.get("text") or message.get("content") or "").strip()
        if not text:
            continue

        line = f"{role}: {text}"
        if len(line) > remaining:
            line = _truncate_text(line, remaining)
        lines_reversed.append(line)
        remaining -= len(line) + 1
        if remaining <= 0:
            break

    return "\n".join(reversed(lines_reversed))


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return f"{text[: max_chars - 3]}..."
