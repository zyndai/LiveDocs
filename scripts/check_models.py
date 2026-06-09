"""Diagnose Gemini embedding access: REST API test + available model list."""
import os
import requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not set in .env")
    raise SystemExit(1)

model = "gemini-embedding-2"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

print(f"Testing REST embed: {url}")
resp = requests.post(
    url,
    json={"content": {"parts": [{"text": "hello world"}]}, "outputDimensionality": 768},
    params={"key": api_key},
    timeout=15,
)
print(f"  HTTP {resp.status_code}")
if resp.status_code == 200:
    vals = resp.json()["embedding"]["values"]
    print(f"  OK — dim={len(vals)}, first 3: {vals[:3]}")
else:
    print(f"  FAIL: {resp.text}")
    print("\nListing available embedding models via SDK:")
    from google import genai
    client = genai.Client(api_key=api_key)
    for m in client.models.list():
        if "embed" in m.name.lower():
            print(f"  {m.name}  —  {getattr(m, 'display_name', '')}")
