"""FastAPI entrypoint for Car Insurance AI."""
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from app.tools.rag_tools import ensure_knowledge_base_loaded
from app.workflows.main_graph import get_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("car-insurance-agent")


# ─── App setup ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the policy knowledge base into ChromaDB when the app starts.

    Ingestion only runs when the collection is empty, so restarts are fast.
    A missing or invalid OpenAI key is logged, not fatal: the app still starts,
    and the knowledge base is loaded on the first policy question instead.
    """
    chunk_count = ensure_knowledge_base_loaded()
    if chunk_count:
        log.info(f"Knowledge base ready: {chunk_count} chunks")
    yield


app = FastAPI(
    title="Car Insurance AI",
    description="Multi-agent car insurance assistant powered by LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ─── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    damage_image_b64: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    intent: str | None
    reply: str
    tool_calls: list[dict]


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the chat UI."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "service": "car-insurance-agent"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "car-insurance-agent"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to the multi-agent system and get a response."""
    if not req.message.strip() and not req.damage_image_b64:
        raise HTTPException(400, "Empty message")

    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    graph = get_graph()

    # Build the input state. `image_uploaded_this_turn` is reset on every request,
    # so only a request that actually carries a photo forces the Claims agent.
    # The photo itself stays in state until it is analysed, so a claim can span turns.
    message_text = req.message.strip() or "I've uploaded a damage photo."
    if req.damage_image_b64 and req.message.strip():
        # Say so in the message itself: the model trusts what the customer says more
        # than a note in its instructions. Without this, gpt-6-luna sometimes asked
        # for a photo the customer had just sent.
        message_text = f"{message_text} (Photo of the damage attached.)"

    input_state: dict = {
        "messages": [HumanMessage(content=message_text)],
        "image_uploaded_this_turn": bool(req.damage_image_b64),
    }
    if req.damage_image_b64:
        input_state["damage_image_b64"] = req.damage_image_b64

    log.info(f"[{session_id}] message='{req.message[:80]}' "
             f"image={'yes' if req.damage_image_b64 else 'no'}")

    try:
        result = await graph.ainvoke(input_state, config=config)
    except Exception as error:
        log.exception("Graph invocation failed")
        raise HTTPException(500, f"Agent error: {type(error).__name__}: {error}")

    current_turn_messages = get_current_turn_messages(result["messages"])

    # The reply is the last AI message with text in THIS turn (not an earlier turn).
    reply_text = ""
    for message in reversed(current_turn_messages):
        if isinstance(message, AIMessage) and message.text:
            reply_text = message.text
            break

    # Tool calls made in this turn only, so the UI trace matches the reply.
    tool_calls_summary = []
    for message in current_turn_messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls_summary.append({"name": tool_call["name"], "args": tool_call["args"]})

    return ChatResponse(
        session_id=session_id,
        intent=result.get("intent"),
        reply=reply_text or "I'm here to help. Could you tell me more?",
        tool_calls=tool_calls_summary,
    )


def get_current_turn_messages(messages: list) -> list:
    """Return the messages produced since the latest user message.

    The checkpointer returns the whole conversation, so without this the UI
    would show replies and tool calls from earlier turns.
    """
    for position in range(len(messages) - 1, -1, -1):
        if isinstance(messages[position], HumanMessage):
            return messages[position + 1:]
    return messages


@app.post("/api/reset")
async def reset_session(session_id: str):
    """Forget a conversation so the next message starts a fresh session."""
    checkpointer = get_graph().checkpointer
    cleared = False
    if hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(session_id)
        cleared = True
    return {"session_id": session_id, "status": "reset", "cleared": cleared}
