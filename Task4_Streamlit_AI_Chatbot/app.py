
import os
import base64
import textwrap
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR.parent / "Task3_Multi_Tool_AI_Agent" / ".env")
load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Chat Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


@st.cache_resource
def get_model(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        streaming=True
    )


# =========================================================
# 2. ROBOT IMAGE
# =========================================================

robot_path = BASE_DIR / "robot.png"

if robot_path.is_file():
    robot_data = base64.b64encode(
        robot_path.read_bytes()
    ).decode("utf-8")

    robot_html = f"""
    <img class="hero-art"
         src="data:image/png;base64,{robot_data}"
         alt="AI Chat Studio robot">
    """
else:
    robot_html = '<div class="hero-art robot-fallback">🤖</div>'


# =========================================================
# 3. PREMIUM DARK THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 80% 0%, #172253 0%, transparent 38%),
        linear-gradient(145deg, #071225, #09152b 60%, #10143a);
    color: #f5f7ff;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

.block-container {
    max-width: 1800px !important;
    width: 100% !important;
    padding: .6rem 1rem 6rem !important;
}

[data-testid="stMain"] {
    min-width: 0 !important;
}


/* =====================================================
   WHITE GLOWING SIDEBAR TOGGLE
   ===================================================== */

.st-key-sidebar_toggle button {
    width: 56px !important;
    min-width: 56px !important;
    height: 56px !important;
    min-height: 56px !important;

    padding: 0 !important;
    margin: .1rem 0 1.1rem auto !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    background: #ffffff !important;
    color: #17116c !important;

    border: 2px solid #e4ddff !important;
    border-radius: 13px !important;

    box-shadow: 0 0 18px rgba(117,74,255,.65) !important;

    transition: .2s ease !important;
}

.st-key-sidebar_toggle button:hover {
    background: #ffffff !important;
    color: #17116c !important;
    border-color: #8057ff !important;
    box-shadow: 0 0 28px rgba(117,74,255,.95) !important;
    transform: scale(1.04);
}

.st-key-sidebar_toggle button p {
    color: #17116c !important;
    font-size: 29px !important;
    font-weight: 900 !important;
    margin: 0 !important;
    line-height: 1 !important;
}


/* =====================================================
   SIDEBAR DESIGN
   ===================================================== */

.side-brand {
    text-align: center;
    padding: .6rem .2rem 1.25rem;
}

.side-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 13px #7651ff);
}

.side-name {
    color: white;
    font-size: 1.35rem;
    font-weight: 850;
    margin: .55rem 0 .35rem;
}

.side-sub {
    color: #b3c9fa;
    font-size: .88rem;
}

.side-rule {
    height: 1px;
    background: #294675;
    margin: 1rem 0 1.35rem;
}

.side-label {
    color: #8eafff;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: 1.6px;
    margin: 1.1rem 0 .7rem;
}

.side-count {
    color: #a9b8d5;
    font-size: .84rem;
    margin: .35rem 0 .85rem;
}

.side-note {
    margin-top: 1.6rem;
    padding: 1.1rem;
    border: 1px solid #754aff;
    border-radius: 18px;

    background:
        radial-gradient(circle at 100% 100%, #451bb4, transparent 55%),
        linear-gradient(145deg, #15264c, #191549);
}

.side-note h3 {
    color: white;
    font-size: 1.08rem;
    margin: .65rem 0;
}

.side-note p {
    color: #c2d1ef;
    font-size: .88rem;
    line-height: 1.6;
    margin: 0;
}

.rail-logo {
    text-align: center;
    font-size: 2.1rem;
    margin: .5rem 0 1.6rem;
    filter: drop-shadow(0 0 12px #7651ff);
}

.rail-icon {
    text-align: center;
    color: #e9e4ff;
    font-size: 1.55rem;
    margin: 1.9rem 0;
}

.rail-rule {
    height: 1px;
    background: #294675;
    margin: 1.25rem .2rem;
}


/* Sidebar buttons */

.st-key-new_chat button,
.st-key-clear_chat button {
    width: 100% !important;
    min-height: 49px !important;

    background: linear-gradient(145deg, #122743, #10203b) !important;
    color: #f6f8ff !important;

    border: 1px solid #294a78 !important;
    border-radius: 12px !important;
}

.st-key-new_chat button:hover,
.st-key-clear_chat button:hover {
    background: #1b3055 !important;
    color: white !important;
    border-color: #8057ff !important;
}


/* =====================================================
   HERO BANNER
   ===================================================== */

.hero {
    position: relative;
    overflow: hidden;

    min-height: 185px;

    padding: 1.55rem 210px 1.35rem 1.7rem;

    border: 1px solid #28519b;
    border-radius: 22px;

    background:
        radial-gradient(ellipse at 90% 0%, rgba(0,174,255,.85), transparent 33%),
        radial-gradient(ellipse at 75% 110%, rgba(112,45,255,.8), transparent 48%),
        linear-gradient(110deg, #0b1733, #101c55 55%, #1c32a4);

    box-shadow: 0 14px 38px rgba(0,0,0,.2);

    margin: .1rem 0 1.5rem;
}

.hero-tag {
    display: inline-block;
    padding: .4rem .8rem;

    border: 1px solid #5266a5;
    border-radius: 99px;

    background: rgba(255,255,255,.08);

    color: #edf0ff;
    font-size: .72rem;
    font-weight: 750;
    letter-spacing: .6px;

    margin-bottom: .8rem;
}

.hero h1 {
    color: white !important;
    font-size: clamp(1.8rem, 3vw, 2.65rem);
    font-weight: 850;
    margin: 0 0 .6rem;
    letter-spacing: -1px;
}

.hero-sub {
    color: #d4e0ff;
    font-size: 1rem;
    margin-bottom: 1.1rem;
}

.hero-features {
    display: flex;
    flex-wrap: wrap;
    gap: .65rem 1.2rem;

    color: #eaf0ff;
    font-size: .8rem;
}

.hero-art {
    position: absolute;
    right: 1.5rem;
    top: 50%;
    transform: translateY(-50%);

    width: 175px;
    height: 175px;

    object-fit: contain;
    filter: drop-shadow(0 0 16px rgba(56,170,255,.4));
}

.robot-fallback {
    font-size: 5rem;
    text-align: center;
}


/* =====================================================
   PROMPT SECTION
   ===================================================== */

.section-heading {
    color: #f5f7ff;
    font-size: 1.1rem;
    font-weight: 800;
    margin: .7rem 0 .5rem;
}

.section-caption {
    color: #a9b8d5;
    font-size: .83rem;
    margin-bottom: .9rem;
}

.st-key-prompt_concept button,
.st-key-prompt_code button,
.st-key-prompt_summary button,
.st-key-prompt_ideas button {
    width: 100% !important;
    min-height: 66px !important;

    white-space: normal !important;

    background: linear-gradient(145deg, #122541, #101b35) !important;
    color: #f2f5ff !important;

    border: 1px solid #294675 !important;
    border-radius: 14px !important;

    padding: .8rem !important;
    font-size: .88rem !important;
}

.st-key-prompt_concept button:hover,
.st-key-prompt_code button:hover,
.st-key-prompt_summary button:hover,
.st-key-prompt_ideas button:hover {
    color: white !important;
    border-color: #8262ff !important;

    background: linear-gradient(145deg, #192d53, #201a55) !important;
}


/* =====================================================
   CHAT MESSAGES
   ===================================================== */

[data-testid="stChatMessage"] {
    background: #11223e !important;
    border: 1px solid #254574 !important;
    border-radius: 16px !important;

    padding: .85rem 1rem !important;
    margin-bottom: .7rem !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] code {
    color: #f4f7ff !important;
    line-height: 1.65;
}


/* Empty conversation */

.empty-state {
    border: 1px solid #254574;
    border-radius: 18px;

    padding: 2rem 1.5rem;

    text-align: center;

    background: linear-gradient(145deg, #0d1c34, #10172f);

    margin: .8rem 0 1rem;
}

.empty-title {
    color: white;
    font-size: 1.3rem;
    font-weight: 800;
    margin: .5rem 0;
}

.empty-state p {
    color: #b3c7ea;
    margin: 0;
}


/* =====================================================
   CHAT INPUT - WHITE BOX, BLACK TEXT
   ===================================================== */

[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: #071225 !important;
}

[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid #8055ff !important;
    border-radius: 15px !important;

    box-shadow: 0 0 20px rgba(112,64,255,.15) !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    background: #ffffff !important;

    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;

    caret-color: #6d28d9 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background: #7040ff !important;
    color: white !important;
    border-radius: 9px !important;
}

#MainMenu, footer {
    visibility: hidden;
}


/* Responsive design */

@media(max-width: 900px) {
    .block-container {
        padding: .5rem .55rem 6rem !important;
    }

    .hero {
        padding: 1.2rem 95px 1.2rem 1rem;
    }

    .hero-art {
        width: 85px;
        height: 85px;
        right: .4rem;
    }

    .hero-features {
        font-size: .72rem;
        gap: .45rem;
    }

    .st-key-sidebar_toggle button {
        width: 46px !important;
        min-width: 46px !important;
        height: 46px !important;
        min-height: 46px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. SIDEBAR AND MAIN LAYOUT
# =========================================================

if st.session_state.sidebar_open:
    sidebar_col, main_col = st.columns(
        [1.22, 5.78],
        gap="medium"
    )
else:
    sidebar_col, main_col = st.columns(
        [0.32, 6.68],
        gap="small"
    )


# =========================================================
# 5. SIDEBAR
# =========================================================

with sidebar_col:

    toggle_text = "≪" if st.session_state.sidebar_open else "≫"

    if st.button(
        toggle_text,
        key="sidebar_toggle",
        help="Close sidebar" if st.session_state.sidebar_open else "Open sidebar"
    ):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

    if st.session_state.sidebar_open:

        st.html("""
        <div class="side-brand">
            <div class="side-icon">🤖</div>
            <div class="side-name">AI Chat Studio ✨</div>
            <div class="side-sub">⚡ Powered by Gemini</div>
        </div>

        <div class="side-rule"></div>

        <div class="side-label">WORKSPACE</div>
        """)

        if st.button(
            "💬  New chat",
            key="new_chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

        st.html(f"""
        <div class="side-label">YOUR CONVERSATION</div>
        <div class="side-count">
            Messages in this chat: {len(st.session_state.messages)}
        </div>
        """)

        if st.button(
            "🗑️  Clear conversation",
            key="clear_chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.rerun()

        st.html("""
        <div class="side-note">
            <div style="font-size:2rem;">💡</div>
            <h3>Think Smarter<br>Build Faster</h3>
            <p>
                Your AI companion for learning, coding,
                exploring ideas and bringing projects to life.
            </p>
        </div>
        """)

    else:

        st.html("""
        <div class="rail-logo">🤖</div>
        <div class="rail-rule"></div>
        <div class="rail-icon">💬</div>
        <div class="rail-icon">🗑️</div>
        <div class="rail-rule"></div>
        <div class="rail-icon">💡</div>
        """)


# =========================================================
# 6. MAIN DASHBOARD
# =========================================================

with main_col:

    # Hero header

    st.html(textwrap.dedent(f"""
    <div class="hero">
        <div class="hero-tag">
            ✨ YOUR PERSONAL AI WORKSPACE
        </div>

        <h1>AI Chat Studio ✨</h1>

        <div class="hero-sub">
            Think smarter. Explore ideas. Build something amazing.
        </div>

        <div class="hero-features">
            <span>🌈 Powered by Gemini</span>
            <span>🧠 Conversation memory</span>
            <span>⚡ Streaming responses</span>
            <span>🛡️ Simple and secure</span>
        </div>

        {robot_html}
    </div>
    """))


    # Prompt suggestions

    st.html("""
    <div class="section-heading">💡 Try these prompts</div>
    <div class="section-caption">
        Choose a starting point or type your own question below.
    </div>
    """)

    prompt_cols = st.columns(4, gap="small")

    prompts = [
        (
            "🎓  Explain a concept",
            "Explain a difficult concept in simple words with examples.",
            "prompt_concept"
        ),
        (
            "💻  Write code",
            "Help me write Python code and explain it step by step.",
            "prompt_code"
        ),
        (
            "📄  Summarise content",
            "Help me summarise a long document into clear key points.",
            "prompt_summary"
        ),
        (
            "💡  Brainstorm ideas",
            "Suggest five creative AI project ideas for a student.",
            "prompt_ideas"
        )
    ]

    for col, (label, prompt, key) in zip(prompt_cols, prompts):

        with col:
            if st.button(
                label,
                key=key,
                use_container_width=True
            ):
                st.session_state.pending_prompt = prompt
                st.rerun()


    # Conversation heading

    st.html("""
    <div class="section-heading">💬 Your conversation</div>
    """)


    # Empty state

    if not st.session_state.messages:

        st.html("""
        <div class="empty-state">
            <div style="font-size:3rem;">🌌</div>

            <div class="empty-title">
                Where shall we start?
            </div>

            <p>
                Ask a question, explore an idea or let AI help with your next project.
            </p>
        </div>
        """)


    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Chat input

    user_prompt = st.chat_input("Ask me anything...")


# =========================================================
# 7. GEMINI STREAMING CHAT
# =========================================================

if st.session_state.pending_prompt:

    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


if user_prompt:

    if not API_KEY:

        st.error(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            "Task3_Multi_Tool_AI_Agent/.env or "
            "Task4_Streamlit_AI_Chatbot/.env."
        )

        st.stop()


    # Store user message

    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })


    try:

        llm = get_model(API_KEY)

        conversation = []

        for message in st.session_state.messages:

            if message["role"] == "user":
                conversation.append(
                    HumanMessage(content=message["content"])
                )
            else:
                conversation.append(
                    AIMessage(content=message["content"])
                )


        # Display current response

        with main_col:

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):

                response_placeholder = st.empty()
                full_response = ""

                for chunk in llm.stream(conversation):

                    content = chunk.content

                    if isinstance(content, str):
                        full_response += content

                    elif isinstance(content, list):

                        for part in content:

                            if (
                                isinstance(part, dict)
                                and part.get("type") == "text"
                            ):
                                full_response += part.get("text", "")

                    response_placeholder.markdown(
                        full_response + " ▌"
                    )


                if not full_response.strip():
                    full_response = (
                        "I couldn't generate a text response. Please try again."
                    )

                response_placeholder.markdown(full_response)


        # Save assistant response

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })


    except Exception as error:

        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and st.session_state.messages[-1]["content"] == user_prompt
        ):
            st.session_state.messages.pop()


        st.error(
            "I couldn't generate a response. Please try again later. "
            "If the problem continues, check your Gemini API quota "
            "and model availability."
        )

        with st.expander("Technical details"):
            st.code(str(error))


    st.rerun()
