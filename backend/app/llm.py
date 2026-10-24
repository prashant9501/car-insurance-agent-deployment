"""One place to choose the OpenAI chat model used by every agent and tool.

gpt-4o and gpt-4o-mini are shut down in the OpenAI API on 14 January 2027.
OpenAI lists gpt-6-luna as the replacement for both, so every agent uses it.

gpt-6-luna is a reasoning model, which changes two settings compared with gpt-4o:
  - It rejects `temperature` (only the default is allowed), so we never pass one.
  - `reasoning_effort="none"` skips the thinking step. That keeps it as fast as
    gpt-4o did, and it is the setting OpenAI documents for function calling.

Both can be changed in backend/.env without touching code, e.g. to try a larger
model: OPENAI_CHAT_MODEL=gpt-6-sol. For a model that is not a reasoning model,
set OPENAI_REASONING_EFFORT= (empty) so the setting is not sent.
"""
import os

from langchain_openai import ChatOpenAI

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna")
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none")


def get_chat_model() -> ChatOpenAI:
    """Return the chat model used by the supervisor, the three agents and the vision tool."""
    model_settings = {"model": CHAT_MODEL}
    if REASONING_EFFORT:
        model_settings["reasoning_effort"] = REASONING_EFFORT
    return ChatOpenAI(**model_settings)
