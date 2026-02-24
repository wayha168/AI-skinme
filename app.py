"""
Skin Assistant Chat — Streamlit UI for the AI skincare bot.
Run: streamlit run app.py  (from project root; ensure src is on PYTHONPATH or pip install -e .)
"""
import sys
from pathlib import Path

# Prefer new package under src/
_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st

try:
    from skin_assistant.services import ChatService
    _chat = ChatService()
    def get_reply(msg, conversation_history=None, use_llm=True):
        return _chat.get_reply(msg, conversation_history=conversation_history, use_llm=use_llm)
except ImportError:
    from skin_chatbot import get_reply

st.set_page_config(
    page_title="Skin Assistant",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a clean chat look
st.markdown("""
<style>
    .stApp { max-width: 720px; margin: 0 auto; }
    [data-testid="stChatMessage"] {
        padding: 0.75rem 1rem;
        border-radius: 1rem;
    }
    [data-testid="stChatMessage"] p { margin: 0.25em 0; }
    .assistant-avatar { font-size: 1.5rem; }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stChatMessage"]) {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and short intro
st.title("✨ Skin Assistant")
st.caption("Ask about ingredients, skin concerns, or product recommendations. I use your skincare database to answer.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Optional: LLM toggle in sidebar (if you add one) or use env only
use_llm = True  # Set OPENAI_API_KEY to use GPT; otherwise retrieval-only

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about ingredients or skincare..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking..."):
            reply = get_reply(
                prompt,
                conversation_history=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ],
                use_llm=use_llm,
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
