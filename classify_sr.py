"""
classify_sr.py
───────────────────────────────────────────────────────────────
Batch-classifies every service request in data/service_requests.json
using a local LLaMA model via Ollama.

For every service request it produces:
  • type            – Service Request
  • category        – Standard ITSM SR category
  • sub_category    – Specific sub-category
  • priority        – P1-Critical → P4-Low
  • root_cause      – Probable cause (1-2 sentences)
  • assigned_team   – Which team should handle it
  • self_help_possible – Yes | No
  • self_help_steps – Step-by-step fix if self-help is possible
  • summary         – One-line summary

Checkpointing: saves after EVERY service request so if it crashes,
just re-run and it resumes from where it left off.
───────────────────────────────────────────────────────────────
"""

import json
import os
import time
import ollama
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Config ────────────────────────────────────────────────────
LLM_MODEL      = os.getenv("model", "llama3.2:1b")
INPUT_FILE     = "data/service_requests.json"
OUTPUT_FILE    = "data/service_requests_classified.json"
PROGRESS_FILE  = "data/classify_sr_progress.json"
MAX_RETRIES    = 3
RETRY_DELAY    = 2  # seconds between retries

# ── Standard ServiceNow Service Request Categories ────────────
SR_CATEGORIES = """
SERVICE REQUEST CATEGORIES:
  1. Access Request       – New user access, Role change, VPN access, System permissions,
                           Application access, Shared mailbox, Distribution group
  2. Hardware Request     – New laptop, Desktop, Monitor, Docking station, Keyboard,
                           Mouse, Headset, Webcam, USB devices, Cables
  3. Software Request     – Install software, New license, Upgrade request,
                           Software removal, Plugin installation
  4. Onboarding           – New employee IT setup, AD account creation, Email provisioning,
                           Equipment setup, System access for new joiner
  5. Offboarding          – Account disable, Equipment return, Data handover,
                           License revocation, Email forwarding setup
  6. Mobile Device        – Phone setup, MDM enrollment, SIM card request,
                           Mobile app access, Device replacement
  7. Facilities           – Desk setup, Parking access, Badge access,
                           Meeting room AV setup, Office equipment
  8. Training & Knowledge – Tool training request, Documentation request,
                           How-to guide, User manual, Knowledge base article
  9. Configuration Change – System configuration, Network change, Firewall rule,
                           DNS update, Group policy change, Server configuration
  10. Communication Setup – Distribution list creation, Shared mailbox setup,
                            Teams channel creation, Zoom license, Conference bridge
  11. Data & Reporting    – Data export request, Report creation, Dashboard access,
                            Analytics setup, BI tool access
  12. ERP & Business Apps – SAP access, Oracle setup, Salesforce license,
                            ServiceNow access, Custom internal app access
"""

# ── Standard Teams ────────────────────────────────────────────
TEAMS = """
  - Service Desk (L1)                  – First line, general requests
  - Network Services Team              – Network, VPN, Firewall, DNS changes
  - Desktop Support Team (L2)          – Hardware, OS, Software on endpoints
  - Identity & Access Management (IAM) – Access requests, Passwords, MFA, SSO
  - Security Operations (SOC)          – Security-related requests, Phishing
  - Infrastructure & Server Team       – Server, VM, Cloud, Backup requests
  - Database Administration (DBA)      – Database access, reporting requests
  - Application Support Team           – Business apps, ERP, Salesforce, SAP
  - Telephony & Collaboration Team     – Phones, Teams, Zoom, Conferencing
  - HR & Facilities Team               – Onboarding, Offboarding, Facilities
  - Change & Configuration Team        – Config changes, releases, deployments
"""

# ── Helpers ───────────────────────────────────────────────────
def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Classifier ────────────────────────────────────────────────
def classify_sr(sr: dict) -> dict:
    """
    Sends one service request to LLaMA and returns a structured classification dict.
    Retries up to MAX_RETRIES times on failure.
    """
    prompt = f"""You are an expert ServiceNow ITSM analyst with 15 years of experience.
Carefully analyze the service request below and classify it using the standard ITSM taxonomy.
Think step by step before answering.
Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERVICE REQUEST DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Number           : {sr.get('number', 'N/A')}
Short Description: {sr.get('short_description', 'N/A')}
Description      : {sr.get('description', 'N/A')}
State            : {sr.get('state', 'N/A')}
Assignment Group : {sr.get('assignment_group', 'N/A')}
Resolution Notes : {sr.get('close_notes', 'N/A')}
Request Type     : {sr.get('type', 'N/A')}
Service          : {sr.get('business_service', 'N/A')}
Service Offering : {sr.get('service_offering', 'N/A')}
Work Notes       : {sr.get('work_notes_list', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STANDARD SERVICE REQUEST CATEGORIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{SR_CATEGORIES}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TEAMS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{TEAMS}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P1-Critical : Urgent business blocker, affects many users, executive request
P2-High     : Important request, time-sensitive, affects team productivity
P3-Medium   : Standard request, normal business timeline
P4-Low      : Non-urgent, nice to have, no immediate business impact

Return ONLY this exact JSON:
{{
  "number": "{sr.get('number', '')}",
  "original_short_description": "{sr.get('short_description', '').replace('"', "'")}",
  "type": "Service Request",
  "category": "exact category name from the list above",
  "sub_category": "specific sub-category",
  "priority": "P1-Critical or P2-High or P3-Medium or P4-Low",
  "priority_reason": "one sentence explaining why this priority",
  "root_cause": "probable reason this request was raised in 1-2 clear sentences",
  "assigned_team": "exact team name from the list above",
  "assignment_reason": "one sentence explaining why this team",
  "self_help_possible": "Yes or No",
  "self_help_steps": "numbered step-by-step instructions if self_help_possible is Yes, else empty string",
  "summary": "one clear sentence summarizing the service request and its resolution"
}}"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = response["message"]["content"].strip()

            # ── Clean up response ──
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            # Extract JSON block
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            return json.loads(raw)

        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


def fallback_entry(sr: dict, idx: int) -> dict:
    """Return a safe fallback classification when LLM fails."""
    return {
        "number": sr.get("number", f"sr_{idx}"),
        "original_short_description": sr.get("short_description", ""),
        "type": "Service Request",
        "category": "Unknown",
        "sub_category": "Unknown",
        "priority": "P3-Medium",
        "priority_reason": "Could not classify automatically",
        "root_cause": "Could not determine root cause automatically",
        "assigned_team": "Service Desk (L1)",
        "assignment_reason": "Default assignment — could not classify",
        "self_help_possible": "No",
        "self_help_steps": "",
        "summary": sr.get("short_description", "Unknown service request")
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🤖 ServiceNow Service Request Classifier")
    print(f"   Model  : {LLM_MODEL}")
    print(f"   Input  : {INPUT_FILE}")
    print(f"   Output : {OUTPUT_FILE}")
    print("=" * 60)

    service_requests = load_json(INPUT_FILE, [])
    if not service_requests:
        print(f"❌ No service requests found in {INPUT_FILE}")
        return

    print(f"📄 Total service requests: {len(service_requests)}")

    # Load checkpoint and existing output
    progress = load_json(PROGRESS_FILE, {})
    output   = load_json(OUTPUT_FILE, [])

    already_done = set(progress.keys())
    if already_done:
        print(f"🔁 Resuming — {len(already_done)} already classified, skipping them...")
    else:
        print("🚀 Starting fresh...")

    print("-" * 60)

    success_count = 0
    error_count   = 0
    skip_count    = len(already_done)

    for idx, sr in enumerate(service_requests):
        sr_number = sr.get("number", f"sr_{idx}")

        # ── Skip already done ──
        if sr_number in already_done:
            continue

        try:
            classified = classify_sr(sr)
            output.append(classified)

            # ✅ Checkpoint after every service request
            progress[sr_number] = {
                "done": True,
                "category": classified.get("category", "?"),
                "timestamp": datetime.now().isoformat()
            }
            save_json(PROGRESS_FILE, progress)
            save_json(OUTPUT_FILE, output)

            success_count += 1
            total_done = skip_count + success_count
            pct = round((total_done / len(service_requests)) * 100, 1)

            print(f"  ✅ [{total_done}/{len(service_requests)}] ({pct}%) "
                  f"{sr_number} → {classified.get('type','?')} | "
                  f"{classified.get('category','?')} | "
                  f"{classified.get('priority','?')} | "
                  f"Team: {classified.get('assigned_team','?')}")

        except Exception as e:
            error_count += 1
            fallback = fallback_entry(sr, idx)
            output.append(fallback)

            progress[sr_number] = {"done": True, "category": "Unknown", "timestamp": datetime.now().isoformat()}
            save_json(PROGRESS_FILE, progress)
            save_json(OUTPUT_FILE, output)

            print(f"  ⚠️  [{idx+1}/{len(service_requests)}] {sr_number} → Error: {e} — saved fallback")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"🎉 Classification complete!")
    print(f"   ✅ Classified  : {success_count}")
    print(f"   ⏭️  Skipped    : {skip_count}")
    print(f"   ❌ Errors      : {error_count}")
    print(f"   📂 Output      : {OUTPUT_FILE}")
    print("=" * 60)

    # Clean up progress file
    if len(progress) >= len(service_requests):
        os.remove(PROGRESS_FILE)
        print("🧹 Progress checkpoint cleaned up.")


if __name__ == "__main__":
    main()