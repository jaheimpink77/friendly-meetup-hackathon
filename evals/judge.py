"""LLM-as-judge: a second model grades each answer against rubric.md."""

import json
from pathlib import Path

from agent.llm import client_for

RUBRIC = (Path(__file__).parent / "rubric.md").read_text()

# 4 = "correct, minor omissions". 3 means the customer is missing something they need.
PASS_SCORE = 4


def build_prompt(case: dict, result: dict) -> str:
    # Show the judge what the tools returned, so it can tell real facts from made-up ones.
    tool_calls = json.dumps(result.get("tool_calls") or [], ensure_ascii=False)
    return f"""{RUBRIC}

## Customer question
{case["input"]}

## Expected behaviour
{case["expected"]}

## Tools the assistant called
{tool_calls}

## Assistant's answer
{result.get("answer", "")}
"""


def judge(case: dict, result: dict) -> dict | None:
    """Return {"score": 1-5, "passed": bool, "reason": "..."}, or None to skip."""
    if not case.get("expected"):
        return None

    client, model = client_for("JUDGE")
    prompt = build_prompt(case, result)

    for _attempt in range(2):  # small local models occasionally return bad JSON; retry once
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,  # same answer -> same grade, so runs are comparable
            )
            data = json.loads(response.choices[0].message.content)
            score = max(1, min(5, int(data["score"])))
            return {"score": score, "passed": score >= PASS_SCORE, "reason": str(data.get("reason", ""))}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    return None  # judge couldn't produce a usable grade: leave unscored rather than fail the agent
