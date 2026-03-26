import requests
import json
import os
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()

SNOW_BASE_URL = os.getenv("SNOW_BASE_URL")
SNOW_USERNAME = os.getenv("SNOW_USERNAME")
SNOW_PASSWORD = os.getenv("SNOW_PASSWORD")

def fetch_incidents():
    url = f"{SNOW_BASE_URL}/api/now/table/incident"
    headers = {"Accept": "application/json"}
    params = {
        "sysparm_fields": (
            "number,"
            "short_description,"
            "description,"
            "state,"
            "assignment_group,"
            "close_code,"           # Resolution code
            "incident_type,"        # Incident type  (u_incident_type in some instances)
            "business_service,"     # Service
            "service_offering,"     # Service offering
            "work_notes_list"       # Work notes list
        ),
        "sysparm_limit": 1000,      # ✅ 1000 incidents
        "sysparm_display_value": "true",
        "sysparm_query": "ORDERBYDESCsys_created_on"  # latest 1000
    }

    response = requests.get(
        url,
        auth=HTTPBasicAuth(SNOW_USERNAME, SNOW_PASSWORD),
        headers=headers,
        params=params
    )

    if response.status_code == 200:
        results = response.json().get("result", [])
        print(f"✅ Fetched {len(results)} incidents")
        return results
    else:
        print(f"❌ Failed: {response.status_code} - {response.text}")
        return []

if __name__ == "__main__":
    incidents = fetch_incidents()
    os.makedirs("data", exist_ok=True)
    with open("data/incidents.json", "w") as f:
        json.dump(incidents, f, indent=2)
    print("✅ Saved to data/incidents.json")