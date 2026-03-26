"""
streamlit_app.py
───────────────────────────────────────────────────────────────
Lunar — KONE IT & Workplace Support Assistant
Beautiful Streamlit UI for the Agentic AI Chatbot
───────────────────────────────────────────────────────────────
"""

import json
import os
import time
import requests
import ollama
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Config ────────────────────────────────────────────────────
LLM_MODEL        = os.getenv("model", "llama3.2:1b")
CLASSIFIED_FILE  = "data/all_classified.json"
SNOW_BASE_URL    = os.getenv("SNOW_BASE_URL", "").rstrip("/")
SNOW_USERNAME    = os.getenv("SNOW_USERNAME", "")
SNOW_PASSWORD    = os.getenv("SNOW_PASSWORD", "")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Lunar — KONE Support",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0f1e;
    color: #e8eaf6;
    overflow: hidden;
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1428 0%, #111827 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
    padding: 0;
}

[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem !important;
}

/* ── Main area ── */
.main-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #0a0f1e;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(59, 130, 246, 0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(99, 179, 237, 0.05) 0%, transparent 50%);
}

/* ── Header ── */
.genie-header {
    padding: 1.5rem 2.5rem 1rem;
    border-bottom: 1px solid rgba(99, 179, 237, 0.1);
    background: rgba(13, 20, 40, 0.8);
    backdrop-filter: blur(10px);
    display: flex;
    align-items: center;
    gap: 1rem;
}

.genie-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #90cdf4, #bee3f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}

.genie-subtitle {
    font-size: 0.8rem;
    color: #718096;
    margin: 0;
    font-weight: 300;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #48bb78;
    border-radius: 50%;
    animation: pulse 2s infinite;
    flex-shrink: 0;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(72, 187, 120, 0.4); }
    50% { opacity: 0.8; box-shadow: 0 0 0 6px rgba(72, 187, 120, 0); }
}

/* ── Chat messages ── */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 0.5rem 3rem;
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.message-row {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    animation: fadeSlideIn 0.3s ease;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.message-row.user { flex-direction: row-reverse; }

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}

.avatar.genie {
    background: linear-gradient(135deg, #1a365d, #2a4a7f);
    border: 1px solid rgba(99, 179, 237, 0.3);
}

.avatar.user {
    background: linear-gradient(135deg, #1a202c, #2d3748);
    border: 1px solid rgba(255,255,255,0.1);
}

.bubble {
    max-width: 70%;
    padding: 1rem 1.25rem;
    border-radius: 16px;
    line-height: 1.65;
    font-size: 0.92rem;
}

.bubble.genie {
    background: linear-gradient(135deg, #0d1f3c, #132240);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-top-left-radius: 4px;
    color: #e2e8f0;
}

.bubble.user {
    background: linear-gradient(135deg, #1a365d, #1e4080);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-top-right-radius: 4px;
    color: #e8f4fd;
    text-align: right;
}

.bubble strong { color: #90cdf4; }
.bubble h3 { color: #63b3ed; font-family: 'Syne', sans-serif; }

/* ── Ticket badge ── */
.ticket-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a4731, #276749);
    border: 1px solid #48bb78;
    color: #9ae6b4;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* ── Input area ── */
.input-area {
    padding: 1.25rem 3rem 1.5rem;
    border-top: 1px solid rgba(99, 179, 237, 0.1);
    background: rgba(13, 20, 40, 0.9);
    backdrop-filter: blur(10px);
}

/* ── Streamlit input overrides ── */
.stTextInput input {
    background: #000000 !important;
    border: 1px solid rgba(99, 179, 237, 0.25) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    padding: 0.85rem 1.25rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}

.stTextInput input:focus {
    border-color: rgba(99, 179, 237, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1) !important;
}

.stTextInput input::placeholder { color: #4a5568 !important; }

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #1a365d, #2a4a7f) !important;
    border: 1px solid rgba(99, 179, 237, 0.3) !important;
    color: #90cdf4 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #2a4a7f, #3a5a9f) !important;
    border-color: rgba(99, 179, 237, 0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Sidebar elements ── */
.sidebar-section {
    margin-bottom: 1.5rem;
}

.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.75rem;
}

.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stat-label { font-size: 0.78rem; color: #718096; }
.stat-value { font-size: 0.9rem; font-weight: 600; color: #90cdf4; font-family: 'Syne', sans-serif; }

.quick-btn {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99, 179, 237, 0.12);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: #a0aec0;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
    text-align: left;
}

.quick-btn:hover {
    background: rgba(99, 179, 237, 0.08);
    border-color: rgba(99, 179, 237, 0.3);
    color: #e2e8f0;
}

/* ── Thinking indicator ── */
.thinking {
    display: flex;
    gap: 4px;
    padding: 0.5rem 0;
    align-items: center;
}

.thinking span {
    width: 6px; height: 6px;
    background: #63b3ed;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.4); }
</style>
""", unsafe_allow_html=True)


# ── Load classified data ───────────────────────────────────────
@st.cache_data
def load_classified():
    if not os.path.exists(CLASSIFIED_FILE):
        return []
    with open(CLASSIFIED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Find similar incidents ────────────────────────────────────
def find_similar(query: str, incidents: list, top_k: int = 5) -> list:
    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored = []
    for inc in incidents:
        text = " ".join([
            inc.get("category", ""),
            inc.get("sub_category", ""),
            inc.get("summary", ""),
            inc.get("root_cause", ""),
            inc.get("original_short_description", ""),
            inc.get("type", ""),
        ]).lower()
        word_score = sum(1 for w in query_words if w in text)
        phrase_score = 3 if query_lower in text else 0
        total = word_score + phrase_score
        if total > 0:
            scored.append((total, inc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [inc for _, inc in scored[:top_k]]


# ── Format context ────────────────────────────────────────────
def format_context(similar: list) -> str:
    if not similar:
        return "No similar past incidents found."
    lines = []
    for i, inc in enumerate(similar, 1):
        lines.append(f"""
Past Incident #{i}:
  Number     : {inc.get('number', 'N/A')}
  Summary    : {inc.get('summary', 'N/A')}
  Category   : {inc.get('category', 'N/A')} -> {inc.get('sub_category', 'N/A')}
  Priority   : {inc.get('priority', 'N/A')}
  Root Cause : {inc.get('root_cause', 'N/A')}
  Team       : {inc.get('assigned_team', 'N/A')}
  Self-Help  : {inc.get('self_help_possible', 'N/A')}
  Steps      : {inc.get('self_help_steps', 'N/A')}
""")
    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Lunar — an expert AI assistant for IT and workplace support at KONE Corporation.
You are highly intelligent, empathetic, and domain-specific.

YOUR PERSONALITY:
- Friendly, professional, and confident
- You explain your reasoning clearly
- You ask clarifying questions when the problem is unclear
- You never make up information — if unsure, say so
- You always think step-by-step before answering

YOUR CAPABILITIES:
1. Classify the employee's issue (Incident or Service Request)
2. Identify the category and sub-category
3. Determine priority based on business impact
4. Suggest self-help steps FIRST before escalating
5. Route to the correct team if self-help doesn't work
6. Offer to create a ServiceNow ticket

RESPONSE FORMAT — always structure your answers like this:

Understanding Your Issue
[Brief restatement of what you understood]

Self-Help Steps (Try these first)
[Numbered steps the employee can try themselves]

Root Cause Analysis
[Probable reason this is happening]

If Issue Persists — Route To
[Team name and why]

Create a Ticket?
[Ask if they want you to create a ServiceNow ticket]

IMPORTANT RULES:
- Always suggest self-help steps FIRST
- Only suggest ticket creation if self-help is unlikely to work or they ask
- If the problem is vague, ask 1-2 clarifying questions before answering
- Base your answers on the past incident history provided
- Keep responses clear and easy to understand for non-technical employees
"""


# ── Ask Lunar ─────────────────────────────────────────────────
def ask_lunar(user_message: str, history: list, context: str) -> str:
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT + f"\n\nRELEVANT PAST INCIDENTS:\n{context}"
    }]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


# ── Create ticket ─────────────────────────────────────────────
def create_ticket(short_desc: str, description: str) -> dict:
    if not SNOW_BASE_URL or not SNOW_USERNAME or not SNOW_PASSWORD:
        return {"success": False, "message": "ServiceNow credentials not configured in .env"}
    url = f"{SNOW_BASE_URL}/api/now/table/incident"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "short_description": short_desc,
        "description": description,
        "caller_id": SNOW_USERNAME,
    }
    try:
        r = requests.post(url, auth=(SNOW_USERNAME, SNOW_PASSWORD), headers=headers, json=payload, timeout=10)
        if r.status_code == 201:
            result = r.json().get("result", {})
            return {"success": True, "number": result.get("number", "N/A")}
        return {"success": False, "message": f"Error {r.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "ticket_created" not in st.session_state:
    st.session_state.ticket_created = None
if "last_context" not in st.session_state:
    st.session_state.last_context = ""
if "total_chats" not in st.session_state:
    st.session_state.total_chats = 0

incidents = load_classified()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <p style="font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800;
                  background: linear-gradient(135deg, #63b3ed, #90cdf4);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                  margin: 0 0 0.2rem 0;">Lunar</p>
        <p style="font-size: 0.75rem; color: #4a5568; margin: 0;">KONE IT Support Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Knowledge Base</div>', unsafe_allow_html=True)
    total = len(incidents)
    incidents_count = sum(1 for i in incidents if i.get("source") == "Incident")
    sr_count = sum(1 for i in incidents if i.get("source") == "Service Request")

    st.markdown(f"""
    <div class="stat-card"><span class="stat-label">Total Records</span><span class="stat-value">{total:,}</span></div>
    <div class="stat-card"><span class="stat-label">Incidents</span><span class="stat-value">{incidents_count:,}</span></div>
    <div class="stat-card"><span class="stat-label">Service Requests</span><span class="stat-value">{sr_count:,}</span></div>
    <div class="stat-card"><span class="stat-label">Chats This Session</span><span class="stat-value">{st.session_state.total_chats}</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<br><div class="sidebar-label">Quick Questions</div>', unsafe_allow_html=True)

    quick_questions = [
        "My account is locked",
        "WiFi not connecting",
        "Need a new laptop",
        "Outlook not working",
        "Need SAP access",
        "Printer offline",
    ]

    for q in quick_questions:
        if st.button(q, key=f"quick_{q}"):
            st.session_state["quick_input"] = q
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.ticket_created = None
        st.session_state.last_context = ""
        st.rerun()

    st.markdown(f"""
    <br>
    <div style="font-size: 0.7rem; color: #2d3748; text-align: center;">
        Model: {LLM_MODEL}<br>
        Running 100% locally
    </div>
    """, unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="genie-header">
    <div class="status-dot"></div>
    <div>
        <p class="genie-title">Lunar</p>
        <p class="genie-subtitle">KONE IT & Workplace Support · Powered by local AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat messages ─────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 2rem; color: #2d3748;">
            <p style="font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #4a5568; margin-bottom: 0.3rem;">
                Hi! I'm Lunar, your KONE support assistant.
            </p>
            <p style="font-size: 0.85rem; color: #2d3748; margin: 0;">
                Describe your IT issue or service request and I'll help you resolve it.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f"""
                <div class="message-row user">
                    <div class="avatar user">U</div>
                    <div class="bubble user">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-row">
                    <div class="avatar genie">L</div>
                    <div class="bubble genie">{content}</div>
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.ticket_created:
            st.markdown(f"""
            <div style="text-align:center; margin-top: 1rem;">
                <span class="ticket-badge">Ticket Created: {st.session_state.ticket_created}</span>
            </div>
            """, unsafe_allow_html=True)


# ── Input area ────────────────────────────────────────────────
st.markdown('<div class="input-area">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([7, 1, 1])

with col1:
    default_val = st.session_state.pop("quick_input", "")
    user_input = st.text_input(
        "",
        value=default_val,
        placeholder="Describe your issue or request... (e.g. My VPN is not connecting)",
        key="chat_input",
        label_visibility="collapsed"
    )

with col2:
    send_clicked = st.button("Send", use_container_width=True)

with col3:
    ticket_clicked = st.button("Create Ticket", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Handle send ───────────────────────────────────────────────
if (send_clicked or user_input) and user_input.strip():
    query = user_input.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.total_chats += 1

    # Find similar incidents
    similar = find_similar(query, incidents)
    context = format_context(similar)
    st.session_state.last_context = context

    # Get Lunar response
    with st.spinner("Lunar is thinking..."):
        response = ask_lunar(query, st.session_state.history[:-1], context)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append({"role": "assistant", "content": response})

    st.rerun()


# ── Handle ticket creation ────────────────────────────────────
if ticket_clicked:
    if st.session_state.messages:
        last_user = next(
            (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
            "Support ticket"
        )
        with st.spinner("Creating ServiceNow ticket..."):
            result = create_ticket(
                short_desc=last_user[:100],
                description=f"Ticket created via Lunar.\n\nIssue: {last_user}\n\nContext: {st.session_state.last_context[:500]}"
            )
        if result["success"]:
            st.session_state.ticket_created = result["number"]
            st.success(f"Ticket created: {result['number']}")
        else:
            st.error(f"{result['message']}")
        st.rerun()
    else:
        st.warning("Please describe your issue first!")

st.markdown('</div>', unsafe_allow_html=True)