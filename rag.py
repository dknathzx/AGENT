"""
rag.py
───────────────────────────────────────────────────────────────
Genie-like Agentic AI Chatbot for IT & Service Request Support

Features:
  • Understands context deeply
  • Asks clarifying questions when needed
  • Gives structured, confident answers
  • Remembers full conversation history
  • Explains its reasoning
  • Suggests self-help steps first
  • Offers to create a ServiceNow ticket
  • Classifies & routes to the right team
  • Works for both IT and non-IT employees
───────────────────────────────────────────────────────────────
"""

import json
import os
import requests
import ollama
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Config ────────────────────────────────────────────────────
LLM_MODEL        = os.getenv("model", "llama3.2:1b")
EMBED_MODEL      = os.getenv("embed_model", "nomic-embed-text")
CLASSIFIED_FILE  = "data/all_classified.json"
CHAT_LOG_FILE    = "data/chat_history.json"
SNOW_BASE_URL    = os.getenv("SNOW_BASE_URL", "")
SNOW_USERNAME    = os.getenv("SNOW_USERNAME", "")
SNOW_PASSWORD    = os.getenv("SNOW_PASSWORD", "")

# ── Load classified incidents ─────────────────────────────────
def load_classified() -> list[dict]:
    if not os.path.exists(CLASSIFIED_FILE):
        print(f"⚠️  {CLASSIFIED_FILE} not found. Run classify.py first!")
        return []
    with open(CLASSIFIED_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Find similar incidents by keyword scoring ─────────────────
def find_similar(query: str, incidents: list[dict], top_k: int = 5) -> list[dict]:
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

        # Score by word overlap + exact phrase match bonus
        word_score = sum(1 for w in query_words if w in text)
        phrase_score = 3 if query_lower in text else 0
        total_score = word_score + phrase_score

        if total_score > 0:
            scored.append((total_score, inc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [inc for _, inc in scored[:top_k]]

# ── Format similar incidents for context ──────────────────────
def format_context(similar: list[dict]) -> str:
    if not similar:
        return "No similar past incidents found."
    lines = []
    for i, inc in enumerate(similar, 1):
        lines.append(f"""
Past Incident #{i}:
  Number       : {inc.get('number', 'N/A')}
  Summary      : {inc.get('summary', 'N/A')}
  Category     : {inc.get('category', 'N/A')} → {inc.get('sub_category', 'N/A')}
  Priority     : {inc.get('priority', 'N/A')}
  Root Cause   : {inc.get('root_cause', 'N/A')}
  Team         : {inc.get('assigned_team', 'N/A')}
  Self-Help    : {inc.get('self_help_possible', 'N/A')}
  Steps        : {inc.get('self_help_steps', 'N/A')}
""")
    return "\n".join(lines)

# ── Build system prompt ───────────────────────────────────────
SYSTEM_PROMPT = """You are Genie — an expert AI assistant for IT and workplace support at KONE Corporation.
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

🔍 **Understanding Your Issue**
[Brief restatement of what you understood]

💡 **Self-Help Steps** (Try these first)
[Numbered steps the employee can try themselves]

🧠 **Root Cause Analysis**
[Probable reason this is happening]

👥 **If Issue Persists — Route To**
[Team name and why]

🎫 **Create a Ticket?**
[Ask if they want you to create a ServiceNow ticket]

IMPORTANT RULES:
- Always suggest self-help steps FIRST
- Only suggest ticket creation if self-help is unlikely to work or they ask
- If the problem is vague, ask 1-2 clarifying questions before answering
- Base your answers on the past incident history provided
- Keep responses clear and easy to understand for non-technical employees
"""

# ── Ask LLM ───────────────────────────────────────────────────
def ask_genie(user_message: str, conversation_history: list, context: str) -> str:
    # Build messages with full conversation history
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\nRELEVANT PAST INCIDENTS FROM COMPANY HISTORY:\n{context}"
        }
    ]

    # Add full conversation history for memory
    for turn in conversation_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        options={"temperature": 0.3},  # slight creativity for natural responses
    )
    return response["message"]["content"].strip()

# ── Create ServiceNow Ticket ──────────────────────────────────
def create_snow_ticket(short_desc: str, description: str, category: str = "", assigned_group: str = "") -> dict:
    if not SNOW_BASE_URL or not SNOW_USERNAME or not SNOW_PASSWORD:
        return {"success": False, "message": "ServiceNow credentials not configured in .env"}

    url = f"{SNOW_BASE_URL}/api/now/table/incident"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    payload = {
        "short_description": short_desc,
        "description": description,
        "category": category,
        "assignment_group": assigned_group,
        "caller_id": SNOW_USERNAME,
    }

    try:
        response = requests.post(
            url,
            auth=(SNOW_USERNAME, SNOW_PASSWORD),
            headers=headers,
            json=payload,
            timeout=10
        )
        if response.status_code == 201:
            result = response.json().get("result", {})
            return {
                "success": True,
                "number": result.get("number", "N/A"),
                "sys_id": result.get("sys_id", ""),
                "message": f"✅ Ticket created: {result.get('number', 'N/A')}"
            }
        else:
            return {"success": False, "message": f"Failed: {response.status_code} — {response.text}"}
    except Exception as e:
        return {"success": False, "message": f"Error creating ticket: {e}"}

# ── Save chat log ─────────────────────────────────────────────
def save_chat_log(history: list):
    os.makedirs("data", exist_ok=True)
    log = {
        "session": datetime.now().isoformat(),
        "model": LLM_MODEL,
        "turns": history
    }
    existing = []
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r") as f:
            existing = json.load(f)
    existing.append(log)
    with open(CHAT_LOG_FILE, "w") as f:
        json.dump(existing, f, indent=2)

# ── Main chatbot loop ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("🧞 Genie — KONE IT & Workplace Support Assistant")
    print(f"   Model: {LLM_MODEL}")
    print("=" * 60)
    print("  I can help you with IT issues, service requests,")
    print("  and workplace support. Type 'exit' to quit.")
    print("  Type 'new' to start a fresh conversation.")
    print("=" * 60)
    print()

    # Load classified incidents
    incidents = load_classified()
    if incidents:
        print(f"📚 Loaded {len(incidents)} classified incidents from history.\n")
    else:
        print("⚠️  No classified incidents loaded. Answers will be based on general knowledge only.\n")

    conversation_history = []
    last_bot_response = ""
    last_context = ""

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n🧞 Genie: Goodbye! Have a great day. 👋")
            if conversation_history:
                save_chat_log(conversation_history)
                print(f"💾 Chat saved to {CHAT_LOG_FILE}")
            break

        if user_input.lower() == "new":
            if conversation_history:
                save_chat_log(conversation_history)
            conversation_history = []
            last_bot_response = ""
            last_context = ""
            print("\n🧞 Genie: Starting a fresh conversation. How can I help you?\n")
            continue

        # ── Check if user wants to create a ticket ──
        if any(word in user_input.lower() for word in ["yes create", "create ticket", "raise ticket", "log ticket", "yes please", "go ahead"]):
            if last_bot_response:
                print("\n🧞 Genie: Creating your ServiceNow ticket...\n")
                result = create_snow_ticket(
                    short_desc=user_input,
                    description=f"Ticket created via Genie chatbot.\n\nUser message: {user_input}\n\nContext: {last_context}",
                )
                if result["success"]:
                    print(f"🧞 Genie: {result['message']}")
                    print(f"   Your ticket number is: {result['number']}")
                    print(f"   The right team has been notified. You'll receive updates via email.\n")
                else:
                    print(f"🧞 Genie: {result['message']}\n")

                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": result["message"]})
                continue

        # ── Find similar past incidents ──
        similar = find_similar(user_input, incidents)
        context = format_context(similar)
        last_context = context

        # ── Get Genie response ──
        print("\n🧞 Genie: ", end="", flush=True)
        response = ask_genie(user_input, conversation_history, context)
        print(response)
        print()

        # ── Update conversation memory ──
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        last_bot_response = response


if __name__ == "__main__":
    main()