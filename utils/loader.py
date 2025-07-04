#utils/loader.py
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from utils.extractors import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_xlsx,
)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xlsm", ".xls"}

def load_documents_from_folder(folder: str = "data") -> List[Document]:
    """
    Lädt ALLE unterstützten Dateien aus dem angegebenen Ordner (rekursiv),
    wandelt sie per Extraktor in LangChain-Document-Objekte um
    und gibt eine konsolidierte Liste zurück.
    """
    folder_path = Path(folder)
    all_docs: List[Document] = []

    if not folder_path.exists():
        print(f"Ordner '{folder_path.resolve()}' existiert nicht!")
        return []

    print(f"Durchsuche Ordner '{folder_path.resolve()}' …")
    for file_path in folder_path.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        # Datei nach Typ verarbeiten
        if file_path.suffix.lower() == ".pdf":
            docs = extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            docs = extract_text_from_docx(file_path)
        elif file_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            docs = extract_text_from_xlsx(file_path)
        else:
            # Dieser Zweig wird durch SUPPORTED_SUFFIXES eigentlich nie erreicht
            print(f"Überspringe nicht unterstützte Datei: {file_path.name}")
            continue

        if not docs:
            print(f"Keine Inhalte extrahiert aus: {file_path.name}")
            continue

        print(f"Verarbeitet: {file_path.name} – {len(docs)} Abschnitte")
        all_docs.extend(docs)

    print(f"Gesamtdokumente/Chunks geladen: {len(all_docs)}")
    return all_docs
