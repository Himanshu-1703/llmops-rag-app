import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from frontend.api_client import check_dependency_health, list_uploaded_files, stream_chat

load_dotenv()

st.set_page_config(
    page_title="CampusX Doubt Solver",
    page_icon="🎓",
    layout="wide",
)

SESSIONS_FILE = Path(os.getenv("SESSIONS_FILE_PATH", "sessions.json"))
_rename_llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)


# ── Session persistence (frontend-only, never sent to the backend) ─────────────
def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def persist_chat(session_id: str) -> None:
    st.session_state.sessions_meta[session_id]["messages"] = st.session_state.chats[session_id]
    save_sessions(st.session_state.sessions_meta)


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a chat session "
                        "based on the user's first question. Return only the "
                        "title, no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
        "messages": [],
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    return sid


# ── Bootstrap ────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {
        sid: meta.get("messages", [])
        for sid, meta in st.session_state.sessions_meta.items()
    }
if "health" not in st.session_state:
    st.session_state.health = {
        "llm_health": None,
        "llm_error": None,
        "retriever_health": None,
        "retriever_error": None,
        "overall_health": None,
    }
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(), key=lambda s: s["created_at"]
        )
        st.session_state.active_session_id = latest["id"]
        st.session_state.chats.setdefault(latest["id"], [])
    else:
        st.session_state.active_session_id = create_session()

active_sid = st.session_state.active_session_id
st.session_state.chats.setdefault(active_sid, [])


# ── Top bar: title + health signals ─────────────────────────────────────────────
def render_dot(status: str | None) -> str:
    if status == "healthy":
        return "🟢"
    if status == "unhealthy":
        return "🔴"
    return "⚪"


def run_health_check() -> None:
    try:
        with st.spinner("Checking backend health…"):
            data = check_dependency_health()
        st.session_state.health.update(data)
    except requests.exceptions.RequestException as e:
        st.session_state.health.update(
            {
                "llm_health": "unhealthy",
                "llm_error": str(e),
                "retriever_health": "unhealthy",
                "retriever_error": str(e),
                "overall_health": "unhealthy",
            }
        )


title_col, llm_col, retriever_col, overall_col = st.columns([6, 2, 2, 1])

with title_col:
    st.title("🎓 CampusX Doubt Solver for Insider's Program")

with llm_col:
    if st.button(f"{render_dot(st.session_state.health['llm_health'])} Check LLM", use_container_width=True):
        run_health_check()
        st.rerun()
    if st.session_state.health["llm_error"]:
        st.caption(st.session_state.health["llm_error"][:80])

with retriever_col:
    if st.button(
        f"{render_dot(st.session_state.health['retriever_health'])} Check Retriever",
        use_container_width=True,
    ):
        run_health_check()
        st.rerun()
    if st.session_state.health["retriever_error"]:
        st.caption(st.session_state.health["retriever_error"][:80])

with overall_col:
    with st.container(border=True):
        st.markdown(
            f"<div style='text-align:center; line-height:1.6;'>"
            f"{render_dot(st.session_state.health['overall_health'])}"
            f"<div style='font-size:0.75rem; color:var(--text-color-light,#808495);'>Overall</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "🔍 **Ask a doubt** and get an instant answer from the CampusX knowledge base."
)
st.divider()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.active_session_id = create_session()
        st.rerun()
    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == active_sid
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                st.session_state.active_session_id = sid
                st.session_state.chats.setdefault(sid, [])
                st.rerun()

    st.divider()
    st.markdown("## 📄 Loaded Documents")
    try:
        files_data = list_uploaded_files()
        with st.expander(f"{files_data['unique_files']} file(s) indexed", expanded=False):
            for filename in files_data["filenames"]:
                st.markdown(f"- {filename}")
    except requests.exceptions.RequestException:
        st.caption("Document list not available — is the backend running?")


# ── Chat display ─────────────────────────────────────────────────────────────
for msg in st.session_state.chats[active_sid]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask your doubt…"):
    is_first_message = len(st.session_state.chats[active_sid]) == 0

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
    persist_chat(active_sid)

    with st.chat_message("assistant"):
        status = st.status("Generating response…", expanded=False)
        try:
            response_text = st.write_stream(stream_chat(prompt))
            status.update(label="Response generated", state="complete")
        except requests.exceptions.RequestException as e:
            response_text = f"Could not reach the backend: {e}"
            st.error(response_text)
            status.update(label="Failed to generate response", state="error")

    st.session_state.chats[active_sid].append(
        {"role": "assistant", "content": response_text}
    )
    persist_chat(active_sid)

    if is_first_message:
        rename_session(active_sid, prompt)
        st.rerun()
