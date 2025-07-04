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

# Hier lesen wir die Variablen aus der .env ein
load_dotenv()

#Hier definieren wir das Modell welches wir nutzen möchten und die API (GROQ), der API Schlüssel für GROQ ist in der .env gespeichert und wir hier abgerufen.
MODEL_CONFIG = {
    "api_key": os.getenv("GROQ_API_KEY"),
    "api_url": "https://api.groq.com/openai/v1/chat/completions",
    "model_name": "meta-llama/llama-4-scout-17b-16e-instruct"
}

app = FastAPI()

#Hier wird das Cross-Origin Resource Sharing kurz CORS definiert, um mit dem Frontend (index.html) kommunizieren zu können.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Kurzer Health Check
@app.get("/")
def root():
    return {"message": "KI-Dokumentenserver läuft"}

# Vorschau aller geladenen Dokumente um zu prüfen ob alle Dokumente und ihre Inhalte erkannt wurden
@app.get("/test-dokumente")
def test_dokumente():
    try:
        docs = load_documents_from_folder()
        return {
            "anzahl_dokumente": len(docs),
            "vorschau": [doc.page_content[:300] for doc in docs]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

#Bauen des Vektorstores mit /build, weil simpel
@app.get("/build")
def build_store():
    try:
        docs = load_documents_from_folder()
        build_vectorstore_from_docs(docs)
        return {"message": "Vektorstore erfolgreich gebaut und gespeichert"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

#Vorschau der relevantesten Dokumente
@app.get("/query")
def query(question: str):
    try:
        vs = load_vectorstore()
        docs = vs.similarity_search(question, k=12)
        return {"antwortkontext": [doc.page_content for doc in docs]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})

#Datensatzmodell für POST-Anfragen
class FrageInput(BaseModel):
    question: str

#Initialisiere Vektorstore bei Serverstart #k= erklären
retriever = build_hybrid_retriever(k=10)

#Haupt-Endpunkt für KI-Fragen
@app.post("/ask")
def ask(frage: FrageInput):
    try:
        question = frage.question
        docs: List[Document] = retriever.get_relevant_documents(question)

        if not docs:
            return {"antwort": "Dazu liegt mir keine verlässliche Information vor.", "quellen": []}

        # Dokumente sortieren (kürzeste zuerst), dann kontextuell deduplizieren
        sorted_docs = sorted(docs, key=lambda d: len(d.page_content))
        context = ""
        total_chars = 0
        included_docs = []
        seen_sources = set()

        for doc in sorted_docs:
            # Quelle + Seite identifizieren
            src_id = (doc.metadata.get("source"), doc.metadata.get("page_number"))
            
            # Gleiche Quelle + Seite nur 1× berücksichtigen
            if src_id in seen_sources:
                continue

            doc_text = doc.page_content.strip()
            if total_chars + len(doc_text) <= 4000:
                context += doc_text + "\n"
                total_chars += len(doc_text)
                included_docs.append(doc)
                seen_sources.add(src_id)
            else:
                break


                    # 🔍 Debug: Kontextausgabe vor API-Aufruf
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

        #Prompt
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

        

        #Anfrage an Groq API
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
            return {
                "antwort": f"Fehler bei Anfrage: {response.status_code} - {response.text}",
                "quellen": [f"{doc.metadata.get('source', 'unbekannt')} (Seite {doc.metadata.get('page_number', '-')})"
                            for doc in included_docs]
            }

        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        #Quellenformatierung
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
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})
