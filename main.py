"""
main.py – Einstiegspunkt und Steuerzentrale des KI-RAG-Prototyps

Dieses Modul stellt die FastAPI-Schnittstelle für alle Kernfunktionalitäten bereit:
- Laden und Verarbeiten von Dokumenten,
- Aufbau des Vektorindex,
- Beantwortung von Nutzeranfragen auf Basis eines Retrieval-Augmented Generation (RAG) Workflows.

Die Konzeption folgt dem Prinzip der klaren Trennung zwischen Web-Service (API), Dokumentenmanagement und Modellinteraktion.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import os
import traceback
import requests
from utils.retriever import build_hybrid_retriever
from utils.loader import load_documents_from_folder
from utils.vectorstore import build_vectorstore_from_docs, load_vectorstore
from langchain_core.documents import Document

# Laden der Umgebungsvariablen (u.a. API-Schlüssel) aus .env-Datei.
load_dotenv()

# Zentrale Modellkonfiguration: Trennung von Schlüssel, Endpunkt und Modellnamen ermöglicht flexible Anpassungen.
MODEL_CONFIG = {
    "api_key": os.getenv("GROQ_API_KEY"),
    "api_url": "https://api.groq.com/openai/v1/chat/completions",
    "model_name": "meta-llama/llama-4-scout-17b-16e-instruct"
}

# Initialisierung der FastAPI-Anwendung.
app = FastAPI()

# CORS-Konfiguration, um Anfragen aus beliebigen Quellen (z.B. externes Frontend) zu ermöglichen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health-Check-Endpunkt, um den Serverstatus zu prüfen.
@app.get("/")
def root():
    return {"message": "KI-Dokumentenserver läuft"}

# Endpunkt zur Anzeige einer Vorschau aller geladenen Dokumente.
@app.get("/test-dokumente")
def test_dokumente():
    """
    Lädt alle Dokumente aus dem Standardverzeichnis und gibt eine Vorschau der Inhalte zurück.
    Dient der Qualitätssicherung der Dokumentenverarbeitung.
    """
    try:
        docs = load_documents_from_folder()
        return {
            "anzahl_dokumente": len(docs),
            "vorschau": [doc.page_content[:300] for doc in docs]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

# Endpunkt zum (Neu-)Aufbau des Vektorindex aus den aktuell verfügbaren Dokumenten.
@app.get("/build")
def build_store():
    """
    Erzeugt den Vektorstore auf Basis aller verfügbaren Dokumente.
    Ermöglicht Aktualisierungen bei geänderten oder neuen Inhalten.
    """
    try:
        docs = load_documents_from_folder()
        build_vectorstore_from_docs(docs)
        return {"message": "Vektorstore erfolgreich gebaut und gespeichert"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

# Endpunkt zur Suche der relevantesten Dokumente (Top-Kandidaten für die spätere Kontextgenerierung).
@app.get("/query")
def query(question: str):
    """
    Gibt die relevantesten Dokumente zum Suchbegriff zurück.
    Dient der Vorschau und Transparenz für den Nutzer.
    """
    try:
        vs = load_vectorstore()
        docs = vs.similarity_search(question, k=6)
        return {"antwortkontext": [doc.page_content for doc in docs]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

# Eingabemodell für Nutzerfragen, validiert über Pydantic (Stabilität und Nachvollziehbarkeit).
class FrageInput(BaseModel):
    question: str

# Initialisierung des hybriden Retrievers (semantisch + keyword-basiert) beim Serverstart.
retriever = build_hybrid_retriever(k=6)

# Zentrale API für KI-gestützte Antworten, basierend auf dem Retrieval-Augmented Generation (RAG)-Prinzip.
@app.post("/ask")
def ask(frage: FrageInput):
    """
    Verarbeitet Nutzerfragen und generiert Antworten auf Basis der durchsuchten Dokumente und dem LLM.
    Wissenschaftlich und fachlich nachvollziehbare Kontextbildung durch explizites Retrieval.
    """
    try:
        question = frage.question
        docs: List[Document] = retriever.get_relevant_documents(question)

        if not docs:
            return {"antwort": "Dazu liegt mir keine verlässliche Information vor.", "quellen": []}

        # Kontextaufbau: Kürzeste relevante Dokumente werden bevorzugt, um möglichst viel Informationsdichte in das Modell zu geben (Prompt-Limit).
        sorted_docs = sorted(docs, key=lambda d: len(d.page_content))
        context = ""
        total_chars = 0
        included_docs = []
        seen_sources = set()

        for doc in sorted_docs:
            # Deduplizierung nach Quelle und Seite – identische Dokumente werden nur einmal übergeben.
            src_id = (doc.metadata.get("source"), doc.metadata.get("page_number"))
            if src_id in seen_sources:
                continue

            doc_text = doc.page_content.strip()
            if total_chars + len(doc_text) <= 4000:  # Promptgröße beschränkt für das LLM
                context += doc_text + "\n"
                total_chars += len(doc_text)
                included_docs.append(doc)
                seen_sources.add(src_id)
            else:
                break

        # Debug-Ausgabe (optional, zur Protokollierung der verwendeten Dokumente)
        print("📦 Kontextlänge:", len(context))
        print("📄 Übergebene Dokumente:")
        for d in included_docs:
            print(
                f" - {d.metadata.get('source')} | Seite {d.metadata.get('page_number')} | "
                f"§ {d.metadata.get('paragraph', '-')}"
            )
            print("   Auszug:", d.page_content[:120].replace('\n', ' '), "\n")

        if not included_docs:
            return {"antwort": "Dazu liegt mir keine verlässliche Information vor.", "quellen": []}

        # Systemprompt steuert das Antwortverhalten des LLM: 
        # Wissenschaftlich, sachlich, nur belegbare Zahlen/Begriffe ausgeben, Antworten strukturiert präsentieren.
        messages = [
            {
                "role": "system",
                "content": (
                    "Deine Aufgabe ist es, Fachfragen mithilfe amtlicher Dokumente zu beantworten. "
                    "Berücksichtige auch Anleitungen, Prozessbeschreibungen, Beispiele und praktische Hinweise. "
                    "Wenn Anleitungen oder Schritte enthalten sind, gib diese strukturiert wieder. "
                    "Wenn im Kontext ein nummerierter Abschnitt wie '17. Spielapparatesteuer' genannt wird, beziehe dich explizit darauf."
                    "Gehe bei Gebühren und Steuerbeträgen äußerst sorgfältig vor. Verlasse dich nur auf explizit genannte Werte in der Satzung. Werte wie 84 €, 132 € oder 700 € dürfen nur verwendet werden, wenn diese exakt so im Text stehen."
                    "Formatiere deine Antwort, wenn möglich, tabellarisch oder in nummerierten Punkten. Beispiel: 1. Betrag: …, 2. Gültigkeit: …, 3. Quelle: …"
                )
            },
            {
                "role": "user",
                "content": f"Kontext:\n{context.strip()}\n\nFrage: {question}"
            }
        ]

        # Anfrage an das LLM über die Groq-API. Modellparameter sind so gewählt, dass die Antwort deterministisch, konsistent und nachvollziehbar bleibt:
        # - max_tokens = 1024 (Länge der Antwort begrenzt)
        # - temperature = 0.0 (keine zufälligen Antworten, maximale Wiederholbarkeit)
        # - top_p = 0.9 (gewisse sprachliche Varianz ohne Kontrollverlust)
        response = requests.post(
            MODEL_CONFIG["api_url"],
            headers={
                "Authorization": f"Bearer {MODEL_CONFIG['api_key']}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_CONFIG["model_name"],
                "messages": messages,
                "temperature": 0.0,
                "top_p": 0.9,
                "max_tokens": 1024,
                "stream": False
            }
        )

        if response.status_code != 200:
            # Fehlerausgabe und Rückgabe aller verwendeten Quellen.
            return {
                "antwort": f"Fehler bei Anfrage: {response.status_code} - {response.text}",
                "quellen": [f"{doc.metadata.get('source', 'unbekannt')} (Seite {doc.metadata.get('page_number', '-')})"
                            for doc in included_docs]
            }

        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Quellenangabe: Nachvollziehbarkeit und wissenschaftliche Sorgfalt.
        quellen = []
        for doc in included_docs:
            seite = doc.metadata.get("page_number", "-")
            para = doc.metadata.get("paragraph")
            quelle = f"{doc.metadata.get('source')} (Seite {seite}"
            if para:
                quelle += f", § {para}"
            quelle += ")"
            quellen.append(quelle)

        return {
            "antwort": answer,
            "quellen": quellen
        }

    except Exception as e:
        # Fehlerhandling gibt neben der Fehlermeldung immer auch das Traceback zur transparenten Diagnose zurück.
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
