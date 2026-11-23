"""Try to fetch the Mendeley Data API for direct file download links."""
import requests, json, re

# Mendeley Data API v1 (public, no auth needed for public datasets)
api_url = "https://data.mendeley.com/api/datasets/y58jknpgs8/2"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

r = requests.get(api_url, headers=headers, timeout=20)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type','?')}")

if r.status_code == 200:
    try:
        data = r.json()
        print(json.dumps(data, indent=2)[:3000])
    except:
        print("Not JSON. First 1000 chars:")
        print(r.text[:1000])
else:
    print(r.text[:500])

# Also try the files endpoint
print("\n--- Trying files endpoint ---")
files_url = "https://data.mendeley.com/api/datasets/y58jknpgs8/2/files"
r2 = requests.get(files_url, headers=headers, timeout=20)
print(f"Status: {r2.status_code}")
if r2.status_code == 200:
    try:
        data2 = r2.json()
        print(json.dumps(data2, indent=2)[:3000])
    except:
        print(r2.text[:1000])
