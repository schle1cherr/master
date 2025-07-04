import os
import requests

# Alternativ kannst du den Key hier direkt einfügen – sicherer ist: .env verwenden
HF_API_KEY = os.getenv("HF_API_KEY")

# Prüfen, ob Key geladen wurde
if not HF_API_KEY:
    raise ValueError("❌ Kein Hugging Face API Key gefunden! Bitte in .env setzen oder direkt eintragen.")

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# Beispielprompt
data = {
    "inputs": "Erkläre mir in einem Satz die Bedeutung von Digitalisierung in kleinen Kommunen.",
    "parameters": {
        "temperature": 0.3,
        "max_new_tokens": 150,
        "do_sample": False
    }
}

# Anfrage absenden
response = requests.post(HF_API_URL, headers=headers, json=data)

# Ausgabe prüfen
print(f"Status Code: {response.status_code}")
try:
    print("Antwort:", response.json())
except Exception as e:
    print("Fehler beim Parsen:", e)
    print("Rohantwort:", response.text)
