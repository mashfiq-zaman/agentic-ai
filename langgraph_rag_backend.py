

import asyncio
import atexit
import os
import sys
import tempfile
import threading
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "chatbot.db")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

model = ChatGoogleGenerativeAI(model=MODEL_NAME)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

SERVERS = {
    "mashfiq-tools": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [os.path.join(HERE, "mcp_server.py")],
    },
    "fetch": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "mcp_server_fetch"],
    },
}


# ---------------------------------------------------------------------------
# Background event loop
# ---------------------------------------------------------------------------

_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, name="mcp-loop", daemon=True)
_thread.start()


def _run(coro):
    """Run a coroutine on the background loop and block until it finishes."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


_SENTINEL = object()


async def _wrap(make_agen):
    return make_agen()


def _sync_iter(make_agen):
    """Turn an async generator into a normal (blocking) generator."""
    agen = _run(_wrap(make_agen))

    async def _next():
        try:
            return await agen.__anext__()
        except StopAsyncIteration:
            return _SENTINEL

    while True:
        item = _run(_next())
        if item is _SENTINEL:
            return
        yield item


# ---------------------------------------------------------------------------
# PDF ingestion — one FAISS retriever per chat thread
#
# In memory only: restarting the app clears every index. Fine for development;
# for anything permanent, persist the FAISS store to disk per thread.
# ---------------------------------------------------------------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Index an uploaded PDF and attach the retriever to this chat thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        docs = PyPDFLoader(temp_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "No text could be extracted. If this is a scanned PDF it needs OCR first."
            )

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        key = str(thread_id)
        summary = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_RETRIEVERS[key] = retriever
        _THREAD_METADATA[key] = summary
        return summary
    finally:
        # FAISS holds its own copy of the text, so the temp file can go.
        try:
            os.remove(temp_path)
        except OSError:
            pass


def has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def document_info(thread_id: str) -> Optional[dict]:
    return _THREAD_METADATA.get(str(thread_id))


# ---------------------------------------------------------------------------
# The RAG tool
#
# `config` is injected by LangChain and is NOT shown to the model, so the
# thread id comes from the graph itself. Asking the LLM to pass a thread_id
# argument works until the one time it forgets, and then the tool silently
# searches the wrong document.
# ---------------------------------------------------------------------------

@tool
def search_uploaded_pdf(query: str, config: RunnableConfig) -> dict:
    """Search the PDF the user uploaded in this chat and return relevant passages.

    Use this for any question about the uploaded document, its contents, or
    anything the user refers to as "the PDF", "the paper", "this document" or
    "the file".
    """
    thread_id = str((config.get("configurable") or {}).get("thread_id", ""))
    retriever = _THREAD_RETRIEVERS.get(thread_id)

    if retriever is None:
        return {
            "error": "No document is indexed for this chat. Ask the user to upload a PDF.",
            "query": query,
        }

    docs = retriever.invoke(query)
    return {
        "query": query,
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
        "passages": [
            {"page": d.metadata.get("page"), "text": d.page_content} for d in docs
        ],
    }


# ---------------------------------------------------------------------------
# State and graph
# ---------------------------------------------------------------------------

class chatstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _system_prompt(thread_id: str) -> SystemMessage:
    info = document_info(thread_id)
    if info:
        doc_line = (
            f"A PDF named '{info['filename']}' ({info['chunks']} chunks from "
            f"{info['documents']} pages) is indexed for this chat. For any question "
            f"about it, call search_uploaded_pdf and answer only from the passages it "
            f"returns. If the passages do not contain the answer, say so rather than "
            f"guessing."
        )
    else:
        doc_line = (
            "No PDF is indexed for this chat yet. If the user asks about a document, "
            "ask them to upload one with the sidebar uploader."
        )

    return SystemMessage(
        content=(
            "You are a helpful research assistant.\n\n"
            f"{doc_line}\n\n"
            "You also have web search, page fetching, stock prices and a calculator. "
            "Use them when they help, and say which source an answer came from."
        )
    )


async def _build():
    client = MultiServerMCPClient(SERVERS)

    try:
        mcp = await client.get_tools()
    except Exception as exc:
        print(f"[warning] MCP servers unavailable ({type(exc).__name__}: {exc}). "
              f"Running with the PDF tool only.")
        mcp = []

    # THE FIX: one list, containing both. Previously the graph was compiled
    # from the MCP tools alone, so the RAG tool was never reachable.
    all_tools = [search_uploaded_pdf, *mcp]

    model_with_tools = model.bind_tools(all_tools)

    async def chat_node(state: chatstate, config: RunnableConfig):
        """LLM node that may answer or request a tool call."""
        thread_id = str((config.get("configurable") or {}).get("thread_id", ""))
        messages = [_system_prompt(thread_id), *state["messages"]]
        response = await model_with_tools.ainvoke(messages, config=config)
        return {"messages": response}

    graph = StateGraph(chatstate)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(all_tools))

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    cm = AsyncSqliteSaver.from_conn_string(DB_PATH)
    checkpointer = await cm.__aenter__()

    return graph.compile(checkpointer=checkpointer), checkpointer, cm, all_tools


chatbot, checkpoint, _saver_cm, agent_tools = _run(_build())

TOOL_NAMES = [t.name for t in agent_tools]


@atexit.register
def _shutdown():
    try:
        _run(_saver_cm.__aexit__(None, None, None))
    except Exception:
        pass
    _loop.call_soon_threadsafe(_loop.stop)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_of(message) -> str:
    """Plain text out of a message whose content may be a list of blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ---------------------------------------------------------------------------
# Public sync API
# ---------------------------------------------------------------------------

def stream_events(user_input: str, thread_id: str):
    """Yield raw (message_chunk, metadata) pairs, like chatbot.stream()."""
    config = {"configurable": {"thread_id": str(thread_id)}}

    def make_agen():
        return chatbot.astream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        )

    yield from _sync_iter(make_agen)


def stream_chat(user_input: str, thread_id: str):
    """Yield just the assistant's text. Drop-in for st.write_stream."""
    for chunk, _meta in stream_events(user_input, thread_id):
        piece = text_of(chunk)
        if piece:
            yield piece


def invoke_chat(user_input: str, thread_id: str) -> str:
    """Non-streaming: the whole reply as one string."""
    config = {"configurable": {"thread_id": str(thread_id)}}
    result = _run(chatbot.ainvoke({"messages": [HumanMessage(content=user_input)]}, config=config))
    return text_of(result["messages"][-1])


def retrive_all_threads() -> list:
    """Every thread_id that has been checkpointed, most recent first."""
    async def _list():
        seen = {}
        async for ck in checkpoint.alist(None):
            tid = ck.config["configurable"]["thread_id"]
            seen.setdefault(tid, ck.checkpoint.get("ts"))
        return sorted(seen, key=lambda t: seen[t] or "", reverse=True)

    return _run(_list())


def load_conversation(thread_id: str) -> list:
    """Messages stored for a thread, for rebuilding the UI after a refresh."""
    config = {"configurable": {"thread_id": str(thread_id)}}
    state = _run(chatbot.aget_state(config))
    return state.values.get("messages", []) if state and state.values else []

def delete_thread(thread_id: str) -> None:
    """Delete a conversation: its checkpoints, and anything indexed for it."""
    tid = str(thread_id)

    async def _delete():
        deleter = getattr(checkpoint, "adelete_thread", None)
        if deleter is not None:
            await deleter(tid)
            return
        # Older langgraph versions have no adelete_thread — clear the rows ourselves.
        for table in ("checkpoints", "writes"):
            try:
                await checkpoint.conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?", (tid,)
                )
            except Exception:
                pass
        await checkpoint.conn.commit()

    _run(_delete())
    _THREAD_RETRIEVERS.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)


if __name__ == "__main__":
    print(f"Tools loaded ({len(TOOL_NAMES)}): {', '.join(TOOL_NAMES)}\n")
    tid = "cli-test"
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        print("AI: ", end="", flush=True)
        for piece in stream_chat(q, tid):
            print(piece, end="", flush=True)
        print("\n")