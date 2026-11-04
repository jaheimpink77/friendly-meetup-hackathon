"""Deterministic checks: fast, free and repeatable.

Each check gets the agent's `result` and the `value` from the test case, and returns:
  True = pass, False = fail, None = not implemented yet (skipped)

`result` looks like:
  {"answer": "...", "tool_calls": [{"name": "get_order", "args": {"order_id": "PP-1001"}}],
   "latency_s": 3.2, "tokens": 812}
"""

import re


def contains(result: dict, value: str) -> bool:
    return value.lower() in result["answer"].lower()


def not_contains(result: dict, value: str) -> bool:
    return value.lower() not in result["answer"].lower()


def _tools_used(result: dict) -> list[str]:
    return [call.get("name") for call in result.get("tool_calls") or []]


def tool_called(result: dict, value: str) -> bool:
    """The agent called the tool named `value` at least once (any position)."""
    return value in _tools_used(result)


def not_tool_called(result: dict, value: str) -> bool:
    """The agent did NOT call `value`, e.g. no get_order when no order ID was given."""
    return value not in _tools_used(result)


def max_words(result: dict, value: int | str | None = 100) -> bool:
    limit = int(value) if value else 100
    return len(result["answer"].split()) <= limit


def matches_regex(result: dict, value: str) -> bool:
    """Case-insensitive regex search. Use for exact amounts: r"£\\s?50(\\.00)?(?!\\.?\\d)" won't match £500 or £6.50."""
    return re.search(value, result["answer"], re.IGNORECASE) is not None


def not_matches_regex(result: dict, value: str) -> bool:
    return not matches_regex(result, value)

# TODO (Level 2): add your own checks here, then register them below. Ideas:
#   tool_called_with  the right tool AND the right arguments (e.g. the correct order ID)
#   max_seconds       answer arrived in time
#   max_words         answer is concise
#   no_tools          agent didn't call any tool (e.g. for off-topic questions)
#   matches_regex     e.g. any price in the form £0.00


CHECKS = {
    "contains": contains,
    "not_contains": not_contains,
    "tool_called": tool_called,
    "max_words": max_words,
    "not_tool_called": not_tool_called,
    "matches_regex": matches_regex,
    "not_matches_regex": not_matches_regex,
}
