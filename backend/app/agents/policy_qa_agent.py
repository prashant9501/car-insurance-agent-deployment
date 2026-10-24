"""Policy Q&A Agent.

Answers questions about policy terms, coverage, exclusions, IRDAI rules,
NCB rules, claim process, etc., using RAG over the knowledge base.
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.workflows.state import AgentState
from app.tools.rag_tools import RAG_TOOLS


POLICY_QA_PROMPT = """You are a knowledgeable Indian motor insurance advisor.

Your job: answer customer questions about insurance policies, coverage, claims process,
IRDAI regulations, NCB, IDV, add-ons, exclusions, and general insurance education.

WORKFLOW:
- For ANY policy question → ALWAYS call `policy_kb_search` first to get authoritative answers
- Use the retrieved context to ground your answer — don't make things up
- If the KB doesn't have the answer, say so honestly and offer general guidance
- If `policy_kb_search` reports `knowledge_base_empty`, start your answer by saying the
  policy documents are not available right now, so this is general guidance only
- Cite the source document briefly when helpful (e.g., "Per IRDAI guidelines...")

STYLE:
- Concise, plain-language answers — no jargon dumps
- Indian English; use ₹ and Indian numbering
- 2–4 sentences for simple questions; longer only when truly needed
- If the user shifts to wanting a quote or filing a claim, just answer the
  current question — the supervisor will route them on the next turn

Today is 2026.
"""


def policy_qa_agent_node(state: AgentState) -> dict:
    """Policy Q&A agent with RAG."""
    # On the first step of a turn (the last message is the user's question), REQUIRE
    # a knowledge-base search. The prompt alone was not enough: after a long claim or
    # quote conversation, gpt-4o-mini sometimes answered from general knowledge.
    # After the search result comes back, the model is free to write the answer.
    is_first_step_of_turn = isinstance(state["messages"][-1], HumanMessage)
    if is_first_step_of_turn:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(
            RAG_TOOLS, tool_choice="policy_kb_search"
        )
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(RAG_TOOLS)

    messages = [SystemMessage(content=POLICY_QA_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}
