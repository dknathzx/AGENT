"""
classify.py
───────────────────────────────────────────────────────────────
Batch-classifies every incident in data/incidents.json using a
local LLaMA model via Ollama.

For every incident it produces:
  • type            – Incident | Service Request
  • category        – Standard ITSM category
  • sub_category    – Specific sub-category
  • priority        – P1-Critical → P4-Low
  • root_cause      – Probable cause (1-2 sentences)
  • assigned_team   – Which team should handle it
  • self_help_possible – Yes | No
  • self_help_steps – Step-by-step fix if self-help is possible
  • summary         – One-line summary

Checkpointing: saves after EVERY incident so if it crashes,
just re-run and it resumes from where it left off.
───────────────────────────────────────────────────────────────
"""

import json
import os
import time
import ollama
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────
LLM_MODEL      = os.getenv("model", "llama3.2:1b")
INPUT_FILE     = "data/incidents.json"
OUTPUT_FILE    = "data/incidents_classified.json"
PROGRESS_FILE  = "data/classify_progress.json"
MAX_RETRIES    = 3
RETRY_DELAY    = 2  # seconds between retries

# ── Standard ServiceNow Categories ───────────────────────────
CATEGORIES = """
INCIDENT CATEGORIES:
  1. Network              – VPN, WiFi, LAN, Firewall, DNS, Proxy, Connectivity
  2. Hardware             – Laptop, Desktop, Printer, Monitor, Keyboard, Mouse, Scanner
  3. Software             – App crash, Installation, License, Update, Patch, Performance
  4. Access & Identity    – Password reset, Account locked, MFA, Permissions, SSO
  5. Email & Communication– Outlook, Teams, Zoom, Calendar, Distribution list, SharePoint
  6. Storage & Backup     – OneDrive, File server, Backup failure, Disk space, SharePoint
  7. Security             – Virus, Malware, Phishing, Data breach, Suspicious activity
  8. Database             – Query failure, Connection error, Slow performance, Corruption
  9. Server & Infrastructure – Server down, CPU/Memory high, Disk, VM, Cloud infra
  10. Telephony           – IP Phone, Softphone, Conference bridge, Voicemail
  11. Printing            – Print queue stuck, Driver issue, Network printer offline
  12. ERP & Business Apps – SAP, Oracle, Salesforce, ServiceNow, custom internal apps
  13. Operating System    – Windows, macOS, Linux, BSOD, Boot failure, Update failure

SERVICE REQUEST CATEGORIES:
  1. Access Request       – New user access, Role change, VPN access, System permissions
  2. Hardware Request     – New laptop, Monitor, Docking station, Keyboard, Headset
  3. Software Request     – Install software, New license, Upgrade request
  4. Onboarding           – New employee IT setup, AD account, Email, Equipment provisioning
  5. Offboarding          – Account disable, Equipment return, Data handover
  6. Mobile Device        – Phone setup, MDM enrollment, SIM card, Mobile app access
  7. Facilities           – Desk setup, Parking, Badge access, Meeting room AV
  8. Training & Knowledge – Tool training request, Documentation, How-to guide
  9. Configuration Change – System config, Network change, Firewall rule, DNS update
"""

# ── Standard Teams ────────────────────────────────────────────
TEAMS = """
  - Service Desk (L1)                  – First line, general issues
  - Network Services Team              – Network, VPN, Firewall, DNS
  - Desktop Support Team (L2)          – Hardware, OS, Software on endpoints
  - Identity & Access Management (IAM) – Access, Passwords, MFA, SSO
  - Security Operations (SOC)          – Security incidents, threats, phishing
  - Infrastructure & Server Team       – Servers, VMs, Cloud, Backup
  - Database Administration (DBA)      – Database issues
  - Application Support Team           – Business apps, ERP, Salesforce, SAP
  - Telephony & Collaboration Team     – Phones, Teams, Zoom, Conferencing
  - HR & Facilities Team               – Onboarding, offboarding, facilities
  - Change & Configuration Team        – Config changes, releases
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
def classify_incident(inc: dict) -> dict:
    """
    Sends one incident to LLaMA and returns a structured classification dict.
    Retries up to MAX_RETRIES times on failure.
    """
    prompt = f"""You are an expert ServiceNow ITSM analyst with 15 years of experience.
Carefully analyze the incident below and classify it using the standard ITSM taxonomy.
Think step by step before answering.
Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INCIDENT DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Number           : {inc.get('number', 'N/A')}
Short Description: {inc.get('short_description', 'N/A')}
Description      : {inc.get('description', 'N/A')}
State            : {inc.get('state', 'N/A')}
Assignment Group : {inc.get('assignment_group', 'N/A')}
Resolution Code  : {inc.get('close_code', 'N/A')}
Incident Type    : {inc.get('incident_type', 'N/A')}
Service          : {inc.get('business_service', 'N/A')}
Service Offering : {inc.get('service_offering', 'N/A')}
Work Notes       : {inc.get('work_notes_list', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STANDARD CATEGORIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{CATEGORIES}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TEAMS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{TEAMS}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIORITY RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P1-Critical : Complete service outage, security breach, affects many users
P2-High     : Major functionality broken, workaround exists but painful
P3-Medium   : Partial issue, workaround exists, moderate impact
P4-Low      : Minor issue, cosmetic, single user, no business impact

Return ONLY this exact JSON:
{{
  "number": "{inc.get('number', '')}",
  "original_short_description": "{inc.get('short_description', '').replace('"', "'")}",
  "type": "Incident or Service Request",
  "category": "exact category name from the list above",
  "sub_category": "specific sub-category",
  "priority": "P1-Critical or P2-High or P3-Medium or P4-Low",
  "priority_reason": "one sentence explaining why this priority",
  "root_cause": "probable root cause in 1-2 clear sentences",
  "assigned_team": "exact team name from the list above",
  "assignment_reason": "one sentence explaining why this team",
  "self_help_possible": "Yes or No",
  "self_help_steps": "numbered step-by-step instructions if self_help_possible is Yes, else empty string",
  "summary": "one clear sentence summarizing the incident and its resolution"
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


def fallback_entry(inc: dict, idx: int) -> dict:
    """Return a safe fallback classification when LLM fails."""
    return {
        "number": inc.get("number", f"inc_{idx}"),
        "original_short_description": inc.get("short_description", ""),
        "type": "Incident",
        "category": "Unknown",
        "sub_category": "Unknown",
        "priority": "P3-Medium",
        "priority_reason": "Could not classify automatically",
        "root_cause": "Could not determine root cause automatically",
        "assigned_team": "Service Desk (L1)",
        "assignment_reason": "Default assignment — could not classify",
        "self_help_possible": "No",
        "self_help_steps": "",
        "summary": inc.get("short_description", "Unknown incident")
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🤖 ServiceNow Incident Classifier")
    print(f"   Model  : {LLM_MODEL}")
    print(f"   Input  : {INPUT_FILE}")
    print(f"   Output : {OUTPUT_FILE}")
    print("=" * 60)

    incidents = load_json(INPUT_FILE, [])
    if not incidents:
        print(f"❌ No incidents found in {INPUT_FILE}")
        return

    print(f"📄 Total incidents: {len(incidents)}")

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

    for idx, inc in enumerate(incidents):
        inc_number = inc.get("number", f"inc_{idx}")

        # ── Skip already done ──
        if inc_number in already_done:
            continue

        try:
            classified = classify_incident(inc)
            output.append(classified)

            # ✅ Checkpoint after every incident
            progress[inc_number] = {
                "done": True,
                "category": classified.get("category", "?"),
                "timestamp": datetime.now().isoformat()
            }
            save_json(PROGRESS_FILE, progress)
            save_json(OUTPUT_FILE, output)

            success_count += 1
            total_done = skip_count + success_count
            pct = round((total_done / len(incidents)) * 100, 1)

            print(f"  ✅ [{total_done}/{len(incidents)}] ({pct}%) "
                  f"{inc_number} → {classified.get('type','?')} | "
                  f"{classified.get('category','?')} | "
                  f"{classified.get('priority','?')} | "
                  f"Team: {classified.get('assigned_team','?')}")

        except Exception as e:
            error_count += 1
            fallback = fallback_entry(inc, idx)
            output.append(fallback)

            progress[inc_number] = {"done": True, "category": "Unknown", "timestamp": datetime.now().isoformat()}
            save_json(PROGRESS_FILE, progress)
            save_json(OUTPUT_FILE, output)

            print(f"  ⚠️  [{idx+1}/{len(incidents)}] {inc_number} → Error: {e} — saved fallback")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"🎉 Classification complete!")
    print(f"   ✅ Classified  : {success_count}")
    print(f"   ⏭️  Skipped    : {skip_count}")
    print(f"   ❌ Errors      : {error_count}")
    print(f"   📂 Output      : {OUTPUT_FILE}")
    print("=" * 60)

    # Clean up progress file
    if len(progress) >= len(incidents):
        os.remove(PROGRESS_FILE)
        print("🧹 Progress checkpoint cleaned up.")


if __name__ == "__main__":
    from datetime import datetime
    main()