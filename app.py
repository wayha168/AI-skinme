"""
Skin Assistant Chat — Streamlit UI for the AI skincare bot.
Run: streamlit run app.py  (from project root; ensure src is on PYTHONPATH or pip install -e .)
Supports text chat and optional selfie upload for skin condition analysis.
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
    def get_reply(msg, conversation_history=None, use_llm=True, use_database=False):
        return _chat.get_reply(
            msg,
            conversation_history=conversation_history,
            use_llm=use_llm,
            use_database=use_database,
        )
    try:
        from skin_assistant.models.skin_condition_trainer import predict_skin_condition_from_image
        _has_skin_predictor = True
    except ImportError:
        predict_skin_condition_from_image = None
        _has_skin_predictor = False
except ImportError:
    _has_skin_predictor = False
    predict_skin_condition_from_image = None
    try:
        from skin_chatbot import get_reply
    except ImportError:
        def get_reply(msg, conversation_history=None, use_llm=True, use_database=False):
            return "Chat service not available. Install skin_assistant or skin_chatbot."

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
st.caption("Ask about ingredients, skin concerns, or product recommendations. You can also upload a selfie for skin analysis.")

# Initialize chat history and image state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "detected_condition" not in st.session_state:
    st.session_state.detected_condition = None  # (label, confidence) or None

# Sidebar: options
with st.sidebar:
    use_llm = st.checkbox("Use LLM (GPT) for replies", value=True, help="Requires OPENAI_API_KEY")
    use_database = st.checkbox(
        "Check with database (skinme_db)",
        value=False,
        help="Query MySQL for product recommendations when configured (MYSQL_* env)",
    )

# Image upload for skin analysis (optional)
st.subheader("📷 Skin selfie (optional)")
uploaded = st.file_uploader(
    "Upload a photo of your skin for condition analysis",
    type=["jpg", "jpeg", "png", "webp"],
    key="skin_upload",
    help="Upload a clear selfie; we'll try to detect skin condition and use it for recommendations.",
)
if uploaded is not None:
    st.session_state.uploaded_image = uploaded
    if _has_skin_predictor and predict_skin_condition_from_image:
        with st.spinner("Analyzing skin..."):
            condition, conf = predict_skin_condition_from_image(uploaded)
        if condition:
            st.session_state.detected_condition = (condition, conf)
            st.success(f"Detected: **{condition}** ({conf:.0%} confidence)")
            st.caption("Ask below for product recommendations, or describe your concern.")
        else:
            st.session_state.detected_condition = None
            st.info("No skin condition model found. Train one with: `python main.py train-skin-condition`")
    else:
        st.session_state.detected_condition = None
        st.info("Install PyTorch and train a skin condition model for analysis: `python main.py train-skin-condition`")
    st.image(uploaded, caption="Your photo", use_container_width=True)
else:
    st.session_state.uploaded_image = None
    st.session_state.detected_condition = None

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="✨" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about ingredients or skincare..."):
    # Build message: if we have a detected condition, prepend it for better recommendations
    effective_prompt = prompt
    if st.session_state.detected_condition:
        condition_label, _ = st.session_state.detected_condition
        effective_prompt = f"My skin condition is {condition_label}. {prompt}"
    st.session_state.messages.append({"role": "user", "content": effective_prompt})

    with st.chat_message("user"):
        if st.session_state.uploaded_image is not None:
            st.image(st.session_state.uploaded_image, use_container_width=True)
        st.markdown(prompt)
        if st.session_state.detected_condition:
            cond, conf = st.session_state.detected_condition
            st.caption(f"Using detected condition: {cond} ({conf:.0%})")

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking..."):
            reply = get_reply(
                effective_prompt,
                conversation_history=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ],
                use_llm=use_llm,
                use_database=use_database,
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
