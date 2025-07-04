# utils/vectorstore.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.loader import load_documents_from_folder


#Konfiguration
VECTORSTORE_PATH = Path("vectorstore")           #Ordner für FAISS-Dateien
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  #frei anpassbar



#Helferfunktionen
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Initialisiert (oder cached) das Embedding-Modell."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _ensure_path(path: Path):
    """Legt den Pfad an, falls er noch nicht existiert."""
    path.mkdir(parents=True, exist_ok=True)


#Öffentliche API
def build_vectorstore_from_docs(docs: List[Document], append: bool = False) -> None:
    """
    Erstellt einen neuen FAISS-Index oder erweitert einen vorhandenen.
    :param docs: Liste aus langchain Documents
    :param append: True  ➜ bestehender Index wird erweitert  
                   False ➜ Index wird überschrieben
    """
    if not docs:
        print("Keine Dokumente übergeben – Abbruch.")
        return

    _ensure_path(VECTORSTORE_PATH)
    embeddings = _get_embeddings()

    if append and (VECTORSTORE_PATH / "index.faiss").exists():
        #Index erweitern
        print("Bestehenden Vectorstore laden und erweitern …")
        db = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        db.add_documents(docs)
    else:
        #Neuen Index erstellen
        print("Neuen Vectorstore anlegen …")
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(VECTORSTORE_PATH)
    print(f"Vectorstore gespeichert unter: {VECTORSTORE_PATH.resolve()}")


def load_vectorstore() -> FAISS:
    """Lädt den gespeicherten Vectorstore."""
    embeddings = _get_embeddings()
    db = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print("Vectorstore erfolgreich geladen.")
    return db



#Convenience-CLI:  python -m utils.vectorstore  (index neu bauen)
if __name__ == "__main__":
    import argparse

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

    #Dokumente laden
    docs = load_documents_from_folder(args.docs_path)
    print(f"Geladene Dokumenten-Chunks: {len(docs)}")

    #Index bauen/erweitern
    build_vectorstore_from_docs(docs, append=args.append)
