"""Supervisor — top-level router.

Classifies user intent and decides which specialist agent should respond.
Uses the shared chat model (app.llm) with reasoning turned off, which is fast
and cheap enough for a classification task.
"""
import json
from langchain_core.messages import SystemMessage, AIMessage

from app.llm import get_chat_model
from app.workflows.state import AgentState


SUPERVISOR_PROMPT = """You are the supervisor of a car insurance AI system. Your only job is to classify
the user's intent and decide which specialist agent should handle their request.

You have 3 specialist agents:

1. **quote** — handles requests for premium quotes, vehicle insurance pricing, IDV/NCB calculations.
   Examples: "I want a quote for my Maruti Swift", "How much will insurance cost for a 2-year old Creta?",
             "Get me a premium for my car", "What's the IDV for a Honda City?"

2. **claims** — handles claim filing, damage assessment, policy verification for claims, payout estimates.
   Examples: "I had an accident", "I want to file a claim", "Help me with a claim",
             "My car got damaged", "I uploaded a photo of the damage"

3. **policy_qa** — answers questions about policy terms, coverage, exclusions, IRDAI rules,
   process questions, NCB rules, what's covered/not covered.
   Examples: "What is NCB?", "Is flood damage covered?", "How do I transfer my policy?",
             "What does IDV mean?", "What's covered under comprehensive insurance?"

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{"intent": "quote" | "claim" | "policy_qa", "reasoning": "<one short sentence>"}

If the user's request is unclear or doesn't fit, default to "policy_qa".
"""


def supervisor_node(state: AgentState) -> dict:
    """Classify user intent and set the next_agent for routing."""
    llm = get_chat_model()

    # Get the latest user message
    user_messages = [m for m in state["messages"] if getattr(m, "type", None) == "human"]
    if not user_messages:
        return {"intent": "policy_qa", "next_agent": "policy_qa"}

    latest_user_msg = user_messages[-1].content

    # A photo uploaded with THIS message is unambiguously a claim.
    # (A photo from an earlier turn must not force every later message into claims.)
    if state.get("image_uploaded_this_turn"):
        return {"intent": "claim", "next_agent": "claims"}

    # Always classify, but tell the classifier which flow is in progress and what the
    # agent last asked, so short answers ("yesterday, near Lonavala") stay in that flow
    # while a new question ("What is NCB?") switches to the right agent.
    classifier_input = build_classifier_input(state, latest_user_msg)
    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT + CONTINUATION_RULES),
        {"role": "user", "content": classifier_input},
    ])
    content = response.text.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        decision = json.loads(content)
        intent = decision.get("intent", "policy_qa")
    except json.JSONDecodeError:
        intent = "policy_qa"

    # The prompt names the agents "quote / claims / policy_qa" but asks for the intents
    # "quote / claim / policy_qa", and the model sometimes answers with the agent name.
    # Accept both spellings instead of silently falling back to policy_qa.
    intent = INTENT_ALIASES.get(str(intent).strip().lower(), intent)
    if intent not in ("quote", "claim", "policy_qa"):
        intent = "policy_qa"

    return {"intent": intent, "next_agent": _intent_to_agent(intent)}


# Spellings the classifier may return, mapped to the intent names the graph uses
INTENT_ALIASES = {
    "claims": "claim",
    "claim": "claim",
    "quotes": "quote",
    "quote": "quote",
    "policy": "policy_qa",
    "policy_qa": "policy_qa",
}


CONTINUATION_RULES = """
CONTEXT RULES:
- You may be told which flow is already in progress and the assistant's last message.
- If the new message answers or continues that flow (a date, a place, a pincode, a policy
  number, a car model, yes/no, "the first plan"), keep the same intent.
  Example: during a claim, "It happened yesterday near Lonavala" is claim.
- If the new message asks something different, choose the intent that fits the new message,
  even if a claim or quote is in progress. Example: during a claim, "What is NCB?" is policy_qa.
"""


def build_classifier_input(state: AgentState, latest_user_message: str) -> str:
    """Build the text the intent classifier sees: current flow, last reply, new message."""
    current_intent = state.get("intent")
    if current_intent not in ("quote", "claim", "policy_qa"):
        return latest_user_message

    # Keep the END of the last reply: agents put their follow-up question last
    # (e.g. a claim summary that ends with "Could you share the date and place?").
    last_assistant_reply = ""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.content:
            last_assistant_reply = message.content[-800:]
            break

    return (
        f"Flow in progress: {current_intent}\n"
        f"Assistant's last message: {last_assistant_reply or '(none)'}\n"
        f"New user message: {latest_user_message}"
    )


def _intent_to_agent(intent: str) -> str:
    return {"quote": "quote", "claim": "claims", "policy_qa": "policy_qa"}.get(intent, "policy_qa")


def route_from_supervisor(state: AgentState) -> str:
    """Conditional edge function — returns the name of the next node."""
    return state.get("next_agent") or "policy_qa"
