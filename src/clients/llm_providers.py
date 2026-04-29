# %%writefile {str(PROJECT_ROOT / "src/clients/llm_providers.py")}
import requests
import os
from groq import Groq
from google import genai

def check_available_models():

    print("="*25 + "GOOGLE MODELS" + "="*25)
    google_client = genai.Client()
    for m in google_client.models.list():
        print(m.name)

    print("="*25 + "GROQ MODELS" + "="*25)
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    for m in response.json()['data']:
        print(m['id'])

def check_my_limits(provider='groq', model='qwen/qwen3-32b'):

    print(f"\n-- Checking ratelimits for {model} [provider: {provider}]--")

    if provider == 'groq':
        client = Groq()
        response = client.chat.completions.with_raw_response.create(
            messages=[{"role": "user", "content": "Ratelimit check, please answer just with 'OK'"}],
            model=model
        )

    # I don't think google yields those headers
    # elif provider == 'google':
    #     response =

    # Extract headers
    headers = response.headers

    print(f"Tokens Per Minute (TPM) Limit: {headers.get('x-ratelimit-limit-tokens')}")
    print(f"Remaining Tokens this minute: {headers.get('x-ratelimit-remaining-tokens')}")
    print(f"Requests Per Day (RPD) Limit: {headers.get('x-ratelimit-limit-requests')}")
    print(f"Remaining Requests today:     {headers.get('x-ratelimit-remaining-requests')}")
    print(f"Time until reset (tokens):    {headers.get('x-ratelimit-reset-tokens')}")
