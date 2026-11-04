import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def client_for(role: str) -> tuple[OpenAI, str]:
    """Return (client, model) for "AGENT" or "JUDGE", configured from .env."""
    client = OpenAI(
        base_url=os.environ[f"{role}_BASE_URL"],
        api_key=os.environ.get(f"{role}_API_KEY") or "not-needed",
    )
    return client, os.environ[f"{role}_MODEL"]


def sampling_kwargs(role: str) -> dict:
    """temperature=0 for repeatable runs, except on Anthropic (Claude 5 models reject it)."""
    if "api.anthropic.com" in os.environ.get(f"{role}_BASE_URL", ""):
        return {}
    return {"temperature": 0}
