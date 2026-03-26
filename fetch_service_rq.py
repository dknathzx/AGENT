"""
fetch_snow_service_requests.py
───────────────────────────────────────────────────────────────
Fetches Service Requests from ServiceNow API and saves them
to data/service_requests.json

Uses same credentials as incidents from .env:
  SNOW_BASE_URL
  SNOW_USERNAME
  SNOW_PASSWORD
  SNOW_CLIENT_ID
  SNOW_CLIENT_SECRET

Fetches same columns as incidents:
  number, short_description, description, state,
  assignment_group, close_code, incident_type,
  business_service, service_offering, work_notes_list

Checkpointing: saves progress so if it crashes, resume safely.
───────────────────────────────────────────────────────────────
"""

import json
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────
SNOW_BASE_URL      = os.getenv("SNOW_BASE_URL", "").rstrip("/")
SNOW_USERNAME      = os.getenv("SNOW_USERNAME", "")
SNOW_PASSWORD      = os.getenv("SNOW_PASSWORD", "")
SNOW_CLIENT_ID     = os.getenv("SNOW_CLIENT_ID", "")
SNOW_CLIENT_SECRET = os.getenv("SNOW_CLIENT_SECRET", "")

OUTPUT_FILE        = "data/service_requests.json"
PROGRESS_FILE      = "data/sr_fetch_progress.json"

FETCH_LIMIT        = 1000   # total records to fetch
BATCH_SIZE         = 100    # records per API call
RETRY_LIMIT        = 3
RETRY_DELAY        = 3      # seconds between retries

# ── Same fields as incidents ──────────────────────────────────
FIELDS = ",".join([
    "number",
    "short_description",
    "description",
    "state",
    "assignment_group",
    "close_code",
    "incident_type",
    "business_service",
    "service_offering",
    "work_notes_list",
    "opened_at",
    "closed_at",
    "priority",
    "category",
    "subcategory",
    "caller_id",
    "cmdb_ci",
    "u_service_type",
])

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

# ── OAuth Token ───────────────────────────────────────────────
def get_oauth_token() -> str:
    """Get OAuth token if CLIENT_ID and CLIENT_SECRET are available."""
    if not SNOW_CLIENT_ID or not SNOW_CLIENT_SECRET:
        return ""

    url = f"{SNOW_BASE_URL}/oauth_token.do"
    payload = {
        "grant_type": "password",
        "client_id": SNOW_CLIENT_ID,
        "client_secret": SNOW_CLIENT_SECRET,
        "username": SNOW_USERNAME,
        "password": SNOW_PASSWORD,
    }

    try:
        response = requests.post(url, data=payload, timeout=15)
        if response.status_code == 200:
            token = response.json().get("access_token", "")
            print(f"✅ OAuth token obtained successfully.")
            return token
        else:
            print(f"⚠️  OAuth failed ({response.status_code}), falling back to Basic Auth.")
            return ""
    except Exception as e:
        print(f"⚠️  OAuth error: {e}, falling back to Basic Auth.")
        return ""

# ── Build headers ─────────────────────────────────────────────
def get_headers(token: str = "") -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

# ── Fetch one batch ───────────────────────────────────────────
def fetch_batch(offset: int, token: str = "") -> list[dict]:
    """Fetch one batch of service requests from ServiceNow."""
    url = f"{SNOW_BASE_URL}/api/now/table/sc_request"

    params = {
        "sysparm_limit": BATCH_SIZE,
        "sysparm_offset": offset,
        "sysparm_fields": FIELDS,
        "sysparm_order_by": "opened_at",
        "sysparm_display_value": "true",
    }

    auth = None if token else (SNOW_USERNAME, SNOW_PASSWORD)

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = requests.get(
                url,
                headers=get_headers(token),
                auth=auth,
                params=params,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json().get("result", [])
            elif response.status_code == 401:
                print(f"❌ Authentication failed. Check your credentials in .env")
                return []
            elif response.status_code == 403:
                print(f"❌ Access denied. Check your ServiceNow permissions.")
                return []
            else:
                print(f"  ⚠️  Attempt {attempt}/{RETRY_LIMIT} — Status {response.status_code}: {response.text[:200]}")
                if attempt < RETRY_LIMIT:
                    time.sleep(RETRY_DELAY)

        except requests.exceptions.ConnectionError:
            print(f"  ⚠️  Attempt {attempt}/{RETRY_LIMIT} — Connection error. Retrying...")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.Timeout:
            print(f"  ⚠️  Attempt {attempt}/{RETRY_LIMIT} — Timeout. Retrying...")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt}/{RETRY_LIMIT} — Error: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    return []

# ── Normalize record ──────────────────────────────────────────
def normalize(record: dict) -> dict:
    """Normalize service request record to match incident format."""
    def val(field):
        v = record.get(field, "")
        if isinstance(v, dict):
            return v.get("display_value", "") or v.get("value", "")
        return str(v) if v else ""

    return {
        "number":             val("number"),
        "short_description":  val("short_description"),
        "description":        val("description"),
        "state":              val("state"),
        "assignment_group":   val("assignment_group"),
        "close_code":         val("close_code"),
        "incident_type":      "Service Request",
        "business_service":   val("business_service"),
        "service_offering":   val("service_offering"),
        "work_notes_list":    val("work_notes_list"),
        "opened_at":          val("opened_at"),
        "closed_at":          val("closed_at"),
        "priority":           val("priority"),
        "category":           val("category"),
        "subcategory":        val("subcategory"),
        "caller_id":          val("caller_id"),
        "cmdb_ci":            val("cmdb_ci"),
        "record_type":        "Service Request",
    }

# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("📥 ServiceNow Service Request Fetcher")
    print(f"   URL      : {SNOW_BASE_URL}")
    print(f"   Output   : {OUTPUT_FILE}")
    print(f"   Limit    : {FETCH_LIMIT} records")
    print("=" * 60)

    # Validate credentials
    if not SNOW_BASE_URL:
        print("❌ SNOW_BASE_URL not set in .env")
        return
    if not SNOW_USERNAME or not SNOW_PASSWORD:
        print("❌ SNOW_USERNAME or SNOW_PASSWORD not set in .env")
        return

    # Load checkpoint
    progress = load_json(PROGRESS_FILE, {"fetched_offsets": [], "total": 0})
    all_records = load_json(OUTPUT_FILE, [])
    fetched_offsets = set(progress.get("fetched_offsets", []))

    if fetched_offsets:
        print(f"🔁 Resuming — {len(all_records)} records already fetched, continuing...")
    else:
        print("🚀 Starting fresh fetch...")

    # Get OAuth token if available
    token = get_oauth_token()
    if not token:
        print("🔑 Using Basic Auth (USERNAME + PASSWORD)")

    print("-" * 60)

    total_fetched = len(all_records)

    for offset in range(0, FETCH_LIMIT, BATCH_SIZE):

        # Skip already fetched offsets
        if offset in fetched_offsets:
            continue

        print(f"  📦 Fetching batch offset {offset} to {offset + BATCH_SIZE}...")
        batch = fetch_batch(offset, token)

        if not batch:
            print(f"  ⚠️  Empty batch at offset {offset} — stopping.")
            break

        # Normalize and save
        normalized = [normalize(r) for r in batch]
        all_records.extend(normalized)
        total_fetched += len(normalized)

        # ✅ Checkpoint after every batch
        fetched_offsets.add(offset)
        progress["fetched_offsets"] = list(fetched_offsets)
        progress["total"] = total_fetched
        save_json(PROGRESS_FILE, progress)
        save_json(OUTPUT_FILE, all_records)

        print(f"  ✅ Batch saved — Total so far: {total_fetched} service requests")

        # Stop if we got less than batch size (end of records)
        if len(batch) < BATCH_SIZE:
            print(f"  📭 Reached end of records.")
            break

    print("\n" + "=" * 60)
    print(f"🎉 Fetch complete!")
    print(f"   📄 Total service requests fetched: {total_fetched}")
    print(f"   📂 Saved to: {OUTPUT_FILE}")
    print("=" * 60)

    # Clean up progress file
    if total_fetched > 0:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("🧹 Progress checkpoint cleaned up.")


if __name__ == "__main__":
    main()