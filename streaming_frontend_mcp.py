import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    TOOL_NAMES,
    delete_thread,
    document_info,
    ingest_pdf,
    load_conversation,
    retrive_all_threads,
    stream_events,
    text_of,
)

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

USER_AVATAR = "🧑‍💻"
AI_AVATAR = "🤖"


# *********************** utility functions ************

def geenrate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    thread_id = str(thread_id)
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def reset_chat():
    st.session_state['thread_id'] = geenrate_thread_id()
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def to_display(messages):
    """Stored LangGraph messages -> {'role','content'} dicts for the UI."""
    out = []
    for message in messages:
        if isinstance(message, ToolMessage):          # tool output is not a bubble
            continue
        role = 'user' if isinstance(message, HumanMessage) else 'assistant'
        content = text_of(message)
        if not content.strip():                       # AI msg that only calls a tool
            continue
        out.append({'role': role, 'content': content})
    return out


def avatar_for(role):
    return USER_AVATAR if role == 'user' else AI_AVATAR


# ****************** session setup **********

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = geenrate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs'] = {}

if 'pending_delete' not in st.session_state:
    st.session_state['pending_delete'] = None

add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state['thread_id'])
thread_docs = st.session_state['ingested_docs'].setdefault(thread_key, {})
info = document_info(thread_key)


# ******************* sidebar **************

with st.sidebar:
    st.title("🤖 Agentic AI Chatbot")
    st.caption("built by Mashfiq")

    if st.button("➕  New Chat", use_container_width=True, type="secondary"):
        reset_chat()
        st.rerun()

    st.divider()

    # ---- PDF upload (indexes into THIS thread only)
    st.subheader("Document")

    if info:
        st.success(f"**{info['filename']}**", icon="📄")
        st.caption(f"{info['chunks']} chunks indexed from {info['documents']} pages")
    else:
        st.info("No PDF in this chat yet.", icon="📭")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF for this chat",
        type=['pdf'],
        label_visibility="collapsed",
        help="The PDF is indexed into this conversation only. Start a New Chat to use a different one.",
    )

    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.caption(f"`{uploaded_pdf.name}` already processed.")
        else:
            with st.status("Indexing PDF…", expanded=True) as box:
                try:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=uploaded_pdf.name,
                    )
                    thread_docs[uploaded_pdf.name] = summary
                    box.update(label="PDF indexed", state="complete", expanded=False)
                    st.rerun()
                except Exception as exc:
                    box.update(label=f"{exc}", state="error", expanded=True)

    st.divider()

    # ---- past conversations
    st.subheader("Chats")

    threads = st.session_state['chat_threads'][::-1]
    if not threads:
        st.caption("No past conversations yet.")

    # ---- confirmation step, shown only while a delete is pending
    pending = st.session_state['pending_delete']
    if pending:
        st.warning("Delete this chat? This cannot be undone.", icon="⚠️")
        yes, no = st.columns(2)
        if yes.button("Delete", key="confirm-delete", type="primary", use_container_width=True):
            delete_thread(pending)
            st.session_state['chat_threads'] = [
                t for t in st.session_state['chat_threads'] if t != pending
            ]
            st.session_state['ingested_docs'].pop(str(pending), None)
            st.session_state['pending_delete'] = None
            if str(pending) == thread_key:
                reset_chat()          # the open chat is gone, so start a fresh one
            st.rerun()
        if no.button("Cancel", key="cancel-delete", use_container_width=True):
            st.session_state['pending_delete'] = None
            st.rerun()

    for i, thread_id in enumerate(threads):
        is_current = thread_id == thread_key
        open_col, del_col = st.columns([5, 1], gap="small")

        if open_col.button(
            f"💬  Chat {len(threads) - i}",
            key=f"thread-{thread_id}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            st.session_state['thread_id'] = thread_id
            st.session_state['message_history'] = to_display(load_conversation(thread_id))
            st.session_state['pending_delete'] = None
            st.rerun()

        if del_col.button(
            "🗑",
            key=f"delete-{thread_id}",
            use_container_width=True,
            help="Delete this chat",
        ):
            st.session_state['pending_delete'] = thread_id
            st.rerun()

    st.divider()

    with st.expander(f"🔧  {len(TOOL_NAMES)} tools connected"):
        for name in TOOL_NAMES:
            st.markdown(f"- `{name}`")


# ********************** MAIN UI **************

st.title("Agentic AI Chatbot")

if info:
    st.caption(f"📄 Reading **{info['filename']}** · plus web search, page fetch, stocks and a calculator")
else:
    st.caption("Web search, page fetch, stock prices and a calculator. Upload a PDF in the sidebar to ask about a document.")

st.divider()

# ---- empty state
if not st.session_state['message_history']:
    with st.chat_message('assistant', avatar=AI_AVATAR):
        st.markdown(
            "Hi — ask me anything.\n\n"
            "- Upload a PDF in the sidebar and ask about it\n"
            "- Or try *“what is Nvidia trading at?”* or *“summarise this page: …”*"
        )

for message in st.session_state['message_history']:
    with st.chat_message(message['role'], avatar=avatar_for(message['role'])):
        st.markdown(message['content'])

user_input = st.chat_input('Ask about the PDF, or anything else')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user', avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message('assistant', avatar=AI_AVATAR):
        status_holder = {'box': None}

        def ai_only_stream():
            for message_chunk, metadata in stream_events(user_input, thread_key):
                # Lazily create & update the SAME status container when a tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, 'name', 'tool')
                    if status_holder['box'] is None:
                        status_holder['box'] = st.status(
                            f'🔧 Using `{tool_name}` …', expanded=False
                        )
                    else:
                        status_holder['box'].update(
                            label=f'🔧 Using `{tool_name}` …',
                            state='running',
                            expanded=False,
                        )

                if isinstance(message_chunk, AIMessage):
                    text = text_of(message_chunk)
                    if text:
                        yield text

                try:
                   ai_message = st.write_stream(ai_only_stream())
                except Exception as exc:
                  if "429" in str(exc) or "RateLimit" in type(exc).__name__:
                     ai_message = "I've hit the daily API limit. Please try again later."
                  else:
                     ai_message = f"Something went wrong: {type(exc).__name__}"
                  st.error(ai_message)

        if status_holder['box'] is not None:
            status_holder['box'].update(
                label='Tools finished', state='complete', expanded=False
            )

    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message}
    )
