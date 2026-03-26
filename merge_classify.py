"""
merge_classified.py
───────────────────────────────────────────────────────────────
Merges incidents_classified.json and service_requests_classified.json
into one file: data/all_classified.json
───────────────────────────────────────────────────────────────
"""

import json
import os

INCIDENTS_FILE        = "data/incidents_classified.json"
SERVICE_REQUESTS_FILE = "data/service_requests_classified.json"
OUTPUT_FILE           = "data/all_classified.json"


def load_json(path: str) -> list:
    if not os.path.exists(path):
        print(f"⚠️  File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("🔀 Merging Classified Incidents + Service Requests")
    print("=" * 60)

    incidents = load_json(INCIDENTS_FILE)
    service_requests = load_json(SERVICE_REQUESTS_FILE)

    print(f"📄 Incidents classified       : {len(incidents)}")
    print(f"📄 Service Requests classified: {len(service_requests)}")

    # Tag each record with its source
    for inc in incidents:
        inc["source"] = "Incident"

    for sr in service_requests:
        sr["source"] = "Service Request"

    # Merge both
    merged = incidents + service_requests

    # Save
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Merged total : {len(merged)} records")
    print(f"📂 Saved to     : {OUTPUT_FILE}")
    print("=" * 60)
    print("\n🚀 Now run: python rag.py")


if __name__ == "__main__":
    main()