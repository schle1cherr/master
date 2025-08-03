# main.py – Einstiegspunkt des KI-RAG-Prototyps für Dokumentenrecherche im kommunalen Kontext
# Diese Datei implementiert die FastAPI-Serverlogik und kapselt zentrale Abläufe:
# Dokumentenverarbeitung, Vektorindex-Erstellung, Hybrid-Retrieval und LLM-gestützte Antwortgenerierung.

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

# Laden der Umgebungsvariablen (.env) – z.B. für API-Schlüssel
load_dotenv()

# Modell-Konfiguration getrennt definiert (bessere Wartbarkeit)
MODEL_CONFIG = {
    "api_key": os.getenv("GROQ_API_KEY"),
    "api_url": "https://api.groq.com/openai/v1/chat/completions",
    "model_name": "meta-llama/llama-4-scout-17b-16e-instruct"
}

# Initialisierung FastAPI-Anwendung
app = FastAPI()

# CORS erlauben (nötig für Kommunikation mit externem Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health-Check-Endpunkt
@app.get("/")
def root():
    # Gibt Statusnachricht zurück, nützlich für Monitoring/Deployment
    return {"message": "KI-Dokumentenserver läuft"}

# Endpunkt zur Dokumentenvorschau
@app.get("/test-dokumente")
def test_dokumente():
    # Zeigt Vorschau der geladenen Dokumente, dient der Überprüfung des Imports und zur Qualitätssicherung
    try:
        docs = load_documents_from_folder()
        return {
            "anzahl_dokumente": len(docs),
            "vorschau": [doc.page_content[:300] for doc in docs]
        }
    except Exception as e:
        # Fehlerbehandlung für Transparenz und Debugging
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

# Endpunkt: Vektorindex neu erstellen
@app.get("/build")
def build_store():
    # Erstellt den FAISS-Vektorstore aus allen verfügbaren Dokumenten
    try:
        docs = load_documents_from_folder()
        build_vectorstore_from_docs(docs)
        return {"message": "Vektorstore erfolgreich gebaut und gespeichert"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

# Endpunkt: Dokumenten-Retrieval (Top 6 relevante)
@app.get("/query")
def query(question: str):
    # Gibt relevanteste Dokumente zum Suchbegriff zurück, macht die Kontextbildung transparent
    try:
        vs = load_vectorstore()
        docs = vs.similarity_search(question, k=6)
        return {"antwortkontext": [doc.page_content for doc in docs]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

# Anfrage-Datenmodell (Frage als JSON-Objekt), validiert Eingabe
class FrageInput(BaseModel):
    question: str

# Initialisierung Hybrid-Retriever: FAISS (dense) + BM25 (sparse), optimale Suchqualität für Verwaltungstexte
retriever = build_hybrid_retriever(k=6)

# Zentrale POST-Route für Nutzerfragen: Retrieval & Antwortgenerierung (RAG-Prinzip)
@app.post("/ask")
def ask(frage: FrageInput):
    # Verarbeitet eine Nutzerfrage und liefert eine dokumentengestützte, KI-generierte Antwort zurück
    try:
        question = frage.question
        docs: List[Document] = retriever.get_relevant_documents(question)

        if not docs:
            # Keine relevante Information gefunden, Rückmeldung an Nutzer
            return {"antwort": "Dazu liegt mir keine verlässliche Information vor.", "quellen": []}

        # Kontextaufbau: Kürzeste relevante Dokumente zuerst (maximale Informationsdichte bei LLM-Längenlimit)
        sorted_docs = sorted(docs, key=lambda d: len(d.page_content))
        context = ""
        total_chars = 0
        included_docs = []
        seen_sources = set()

        for doc in sorted_docs:
            # Duplikate nach Quelle/Seite vermeiden, pro Chunk einmal ins Kontextfenster aufnehmen
            src_id = (doc.metadata.get("source"), doc.metadata.get("page_number"))
            if src_id in seen_sources:
                continue

            doc_text = doc.page_content.strip()
            if total_chars + len(doc_text) <= 4000:  # LLM Prompt Size Limit
                context += doc_text + "\n"
                total_chars += len(doc_text)
                included_docs.append(doc)
                seen_sources.add(src_id)
            else:
                break

        # Debug-Ausgabe für Analyse und Transparenz (optional)
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

        # Prompt-Definition für das LLM:
        # Strikte Regeln zur Antwortgenerierung, um faktische, nachvollziehbare und formatierte Ausgaben zu erzwingen
        messages = [
            {
                "role": "system",
                "content": (
                    "Deine Aufgabe ist es, Fachfragen mithilfe amtlicher Dokumente zu beantworten. "
                    "Berücksichtige auch Anleitungen, Prozessbeschreibungen, Beispiele und praktische Hinweise. "
                    "Wenn Anleitungen oder Schritte enthalten sind, gib diese strukturiert wieder. "
                    "Wenn im Kontext ein nummerierter Abschnitt wie '17. Spielapparatesteuer' genannt wird, beziehe dich explizit darauf. "
                    "Gehe bei Gebühren und Steuerbeträgen äußerst sorgfältig vor. Verlasse dich nur auf explizit genannte Werte in der Satzung. "
                    "Werte wie 84 €, 132 € oder 700 € dürfen nur verwendet werden, wenn diese exakt so im Text stehen. "
                    "Formatiere deine Antwort, wenn möglich, tabellarisch oder in nummerierten Punkten. Beispiel: 1. Betrag: …, 2. Gültigkeit: …, 3. Quelle: …"
                )
            },
            {
                "role": "user",
                "content": f"Kontext:\n{context.strip()}\n\nFrage: {question}"
            }
        ]

        # Anfrage an das Groq-LLM mit definierter Parameter-Setzung:
        # temperature=0.0 (max. Konsistenz), top_p=0.9 (geringe Varianz), max_tokens=1024 (Antwortlänge begrenzen)
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
            # Fehlerantwort mit Quellauflistung für Nachvollziehbarkeit
            return {
                "antwort": f"Fehler bei Anfrage: {response.status_code} - {response.text}",
                "quellen": [f"{doc.metadata.get('source', 'unbekannt')} (Seite {doc.metadata.get('page_number', '-')})"
                            for doc in included_docs]
            }

        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Quellenformatierung: Transparente Rückverfolgung der Antwortinhalte
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
        # Transparente Fehlerausgabe inkl. Traceback für Debugging und Fehlerdiagnose
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
