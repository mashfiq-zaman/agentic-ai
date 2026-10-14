# Agentic AI

A LangGraph agent in a Streamlit chat UI. Tools come from MCP servers rather than
being written into the agent, and each conversation can have its own PDF indexed
for retrieval.

Three files: a tool server, an agent, and a UI. None of them imports the other two
in a circle, and the agent runs on its own from the terminal without Streamlit.

## What it does

- **Chat with tool use** — the model decides when to call a tool, and the UI shows
  which one is running.
- **Per-chat PDF retrieval** — upload a PDF and it is chunked, embedded and indexed
  into *that conversation only*. Switching chats switches documents.
- **Persistent conversations** — every thread is checkpointed to SQLite, so chats
  survive a restart and can be reopened or deleted from the sidebar.
- **Streaming replies** — tokens appear as they are generated.

## Tools

Four come from `mcp_server.py` (calculator, DuckDuckGo search, stock quote via
Alpha Vantage, weather via WeatherAPI). Web page fetching comes from the
third-party `mcp-server-fetch`. PDF search is a local LangChain tool, because it
needs access to the retriever belonging to the current chat.

Adding a tool means adding an MCP server to the `SERVERS` dict — no agent code
changes.

## Layout

| File | Role |
|---|---|
| `mcp_server.py` | FastMCP server exposing four tools over stdio |
| `langgraph_rag_backend.py` | The agent: graph, PDF ingestion, checkpointing, public sync API |
| `streaming_frontend_mcp.py` | Streamlit UI. Imports the backend, contains no agent logic |

## How a few parts work

**Async agent, sync UI.** LangGraph's async API and the MCP client both need an
event loop; Streamlit scripts are synchronous. The backend starts one event loop
on a daemon thread at import and pushes coroutines onto it with
`asyncio.run_coroutine_threadsafe`. `_sync_iter` does the same for async
generators so `st.write_stream` can consume the token stream. Everything the
frontend touches is an ordinary blocking function.

**The PDF tool gets its thread id from the graph, not the model.** `config:
RunnableConfig` is injected by LangChain and is never shown to the LLM, so the
tool reads `thread_id` from there. If the model had to pass it as an argument, the
one time it guessed wrong the tool would quietly search a different conversation's
document.

**One tool list.** The graph binds `[search_uploaded_pdf, *mcp_tools]` to the model
and builds `ToolNode` from the same list. These must be the same list — binding one
set and routing another is how a tool ends up advertised but unreachable.

## Setup

```bash
git clone https://github.com/mashfiq-zaman/agentic-ai.git
cd agentic-ai

python -m venv myvenv
myvenv\Scripts\activate          # Windows
# source myvenv/bin/activate     # macOS / Linux

pip install -r requirements.txt

copy .env.example .env           # Windows
# cp .env.example .env           # macOS / Linux
```

Open `.env` and fill in the keys.

| Variable | Required | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | yes | Gemini, the agent model |
| `ALPHAVANTAGE_API_KEY` | no | the stock quote tool |
| `WEATHER_API_KEY` | no | the weather tool |
| `GEMINI_MODEL` | no | defaults to `gemini-3.5-flash` |

Without the optional keys the app still runs; those two tools just return an error
when called.

## Running

```bash
streamlit run streaming_frontend_mcp.py
```

The MCP servers are launched as subprocesses by the backend — you do not start them
yourself.

To talk to the agent without the UI, which is how the tools were tested:

```bash
python langgraph_rag_backend.py
```

It prints the tools it loaded and drops into a terminal chat loop.

## Notes

- Embeddings run locally through `sentence-transformers/all-MiniLM-L6-v2`. The
  first run downloads about 90MB.
- Indexed PDFs live in memory. Restarting the app clears them, while the chat text
  itself survives in `chatbot.db`.
- Scanned PDFs produce no text and are rejected with a message; they would need OCR
  first.
- `chatbot.db` is gitignored. Delete it to wipe all conversations.
- If the MCP servers fail to start, the backend prints a warning and runs with the
  PDF tool alone rather than crashing.

---

Built by Mashfiq Uz Zaman
