"""Deterministic checks: fast, free and repeatable.

Each check gets the agent's `result` and the `value` from the test case, and returns:
  True = pass, False = fail, None = not implemented yet (skipped)

`result` looks like:
  {"answer": "...", "tool_calls": [{"name": "get_order", "args": {"order_id": "PP-1001"}}],
   "latency_s": 3.2, "tokens": 812}
"""


def contains(result: dict, value: str) -> bool:
    return value.lower() in result["answer"].lower()


def not_contains(result: dict, value: str) -> bool:
    return value.lower() not in result["answer"].lower()


def tool_called(result: dict, value: str) -> bool | None:
    if result["tool_calls"][0]["name"] == value:
        return True
    return None


def max_words(result: dict, value: int | str | None = 100) -> bool:
    limit = int(value) if value else 100
    return len(result["answer"].split()) <= limit

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
}
