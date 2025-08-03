# vectorstore.py – Verwaltung des FAISS-Vektorindex und Embeddings für semantisches Retrieval
# Kapselt Aufbau, Speicherung, Erweiterung und Laden eines persistenten FAISS-Indexes
# Wissenschaftliche Zielsetzung: Skalierbarkeit, Effizienz und Reproduzierbarkeit der semantischen Suche

from __future__ import annotations
import os
from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.loader import load_documents_from_folder

# Zentrale Konfigurationen – für einfache Anpassung und Nachvollziehbarkeit
VECTORSTORE_PATH = Path("vectorstore")      # Ordner für FAISS-Index und Metadaten
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Modellwahl: bewährt für deutsche Fachtexte

# --- Hilfsfunktionen ---

def _get_embeddings() -> HuggingFaceEmbeddings:
    # Initialisiert (oder cached) das Embedding-Modell.
    # Vorteil: Leicht um weitere Modelle oder Konfigurationen erweiterbar.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _ensure_path(path: Path):
    # Erstellt den Zielordner für den Vektorstore, falls er noch nicht existiert.
    path.mkdir(parents=True, exist_ok=True)

# --- Hauptfunktionen (öffentliche API) ---

def build_vectorstore_from_docs(docs: List[Document], append: bool = False) -> None:
    # Baut einen neuen FAISS-Vektorindex aus Dokumenten oder erweitert einen bestehenden Index.
    # :param docs: Liste aus langchain Documents (Chunks mit Metadaten)
    # :param append: True  ➜ Index wird erweitert
    #                False ➜ Index wird neu erstellt (überschreibt alten Index)
    if not docs:
        print("Keine Dokumente übergeben – Abbruch.")
        return

    _ensure_path(VECTORSTORE_PATH)
    embeddings = _get_embeddings()

    if append and (VECTORSTORE_PATH / "index.faiss").exists():
        # Index erweitern (empfohlen bei laufender Dokumentenpflege)
        print("Bestehenden Vectorstore laden und erweitern …")
        db = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        db.add_documents(docs)
    else:
        # Neuen Index anlegen (Initialisierung oder Reset)
        print("Neuen Vectorstore anlegen …")
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(VECTORSTORE_PATH)
    print(f"Vectorstore gespeichert unter: {VECTORSTORE_PATH.resolve()}")

def load_vectorstore() -> FAISS:
    # Lädt den gespeicherten FAISS-Vektorindex mit den zugehörigen Embeddings.
    embeddings = _get_embeddings()
    db = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print("Vectorstore erfolgreich geladen.")
    return db

# --- CLI-Funktionalität: Script kann direkt ausgeführt werden (z.B. für Batch-Build auf Server) ---

if __name__ == "__main__":
    import argparse

    # Ermöglicht Kommandozeilen-Nutzung: z.B. python -m utils.vectorstore --docs-path=data --append
    parser = argparse.ArgumentParser(
        description="Erstellt oder erweitert den FAISS-Vectorstore aus Dokumenten."
    )
    parser.add_argument(
        "--docs-path",
        default="docs",
        help="Ordner mit Dokumenten (PDF, DOCX, XLSX …)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Vorhandenen Index erweitern statt überschreiben",
    )
    args = parser.parse_args()

    # Dokumente laden
    docs = load_documents_from_folder(args.docs_path)
    print(f"Geladene Dokumenten-Chunks: {len(docs)}")

    # Index bauen/erweitern
    build_vectorstore_from_docs(docs, append=args.append)
