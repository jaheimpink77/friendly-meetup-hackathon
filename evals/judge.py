"""LLM-as-judge: a second model grades each answer against rubric.md."""

import json
import sys
from pathlib import Path

from openai import APIError

from agent.llm import client_for, sampling_kwargs

RUBRIC = (Path(__file__).parent / "rubric.md").read_text()

# 4 = "correct, minor omissions". 3 means the customer is missing something they need.
PASS_SCORE = 4

RETRY_NUDGE = (
    'Your previous reply was not valid. Reply with ONLY a JSON object like '
    '{"score": 4, "reason": "one sentence"} where score is an integer from 1 to 5.'
)


def build_prompt(case: dict, result: dict) -> str:
    # Show which tools were called and with what arguments (e.g. the right order ID).
    # Tool outputs aren't recorded, so the judge relies on `expected` for the facts.
    tool_calls = json.dumps(result.get("tool_calls") or [], ensure_ascii=False)
    return f"""{RUBRIC}

## Customer question
{case["input"]}

## Expected behaviour
{case["expected"]}

## Tools the assistant called (name and arguments only)
{tool_calls}

## Assistant's answer
<answer>
{result.get("answer", "")}
</answer>
"""


# json_schema (not json_object): Anthropic's OpenAI-compatible API only accepts this type; Ollama supports it too.
VERDICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}


def parse_verdict(content: str | None) -> dict:
    """Raise ValueError/KeyError/TypeError if the judge's reply isn't a usable grade."""
    data = json.loads(content or "")
    score = int(data["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"score {score} out of range")
    return {"score": score, "passed": score >= PASS_SCORE, "reason": str(data.get("reason", ""))}


def warn(case: dict, message: str) -> None:
    print(f"⚠️  Judge failed on {case.get('id', '?')}: {message}", file=sys.stderr)


def judge(case: dict, result: dict) -> dict | None:
    """Return {"score": 1-5, "passed": bool, "reason": "..."}, or None to skip."""
    if not case.get("expected"):
        return None

    client, model = client_for("JUDGE")
    messages = [{"role": "user", "content": build_prompt(case, result)}]

    for attempt in range(2):  # small local models occasionally return bad JSON; retry once
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=VERDICT_FORMAT,
                **sampling_kwargs("JUDGE"),  # temperature=0 where supported: same answer -> same grade
            )
        except APIError as e:
            # Don't let a judge timeout/connection error crash the whole run.
            warn(case, f"API error: {e}")
            return None

        content = response.choices[0].message.content
        try:
            return parse_verdict(content)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            if attempt == 0:
                # Retrying the identical request at temperature 0 just repeats the mistake,
                # so show the model its bad reply and ask again.
                messages += [
                    {"role": "assistant", "content": content or ""},
                    {"role": "user", "content": RETRY_NUDGE},
                ]
            else:
                warn(case, f"unusable reply ({e}): {content!r}")

    return None  # judge couldn't produce a usable grade: leave unscored rather than fail the agent
