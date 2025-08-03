# loader.py – Dokumenten-Loader für RAG-Systeme
# Lädt alle unterstützten Dokumente aus einem Zielordner,
# extrahiert und segmentiert sie zu normierten Document-Objekten.
# Designziel: Robustheit, Rückmeldung, Übersicht über den Ladeprozess.

from pathlib import Path
from typing import List
from langchain_core.documents import Document

from utils.extractors import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_xlsx,
)

# Unterstützte Dateiendungen zentral als Set definiert (effiziente Suche, leichte Erweiterung)
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xlsm", ".xls"}

def load_documents_from_folder(folder: str = "data") -> List[Document]:
    # Lädt alle Dateien aus dem angegebenen Verzeichnis (rekursiv), 
    # verarbeitet sie je nach Typ und gibt eine konsolidierte Liste von Document-Objekten zurück.

    folder_path = Path(folder)
    all_docs: List[Document] = []

    # Prüfe, ob Zielordner existiert. Verhindert Fehler und sorgt für sauberes Feedback.
    if not folder_path.exists():
        print(f"Ordner '{folder_path.resolve()}' existiert nicht!")
        return []

    print(f"Durchsuche Ordner '{folder_path.resolve()}' …")

    # Durchsuche alle Dateien im Verzeichnis und Unterverzeichnissen
    for file_path in folder_path.rglob("*"):
        # Ignoriere Nicht-Dateien und nicht unterstützte Endungen (effizient durch Set)
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        # Dateityp-Dispatch: Verarbeite Datei mit dem passenden Extraktor
        if file_path.suffix.lower() == ".pdf":
            docs = extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            docs = extract_text_from_docx(file_path)
        elif file_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            docs = extract_text_from_xlsx(file_path)
        else:
            # Sollte durch SUPPORTED_SUFFIXES nie eintreten – Fail-Safe-Branch für Robustheit
            print(f"Überspringe nicht unterstützte Datei: {file_path.name}")
            continue

        # Rückmeldung bei leerer Extraktion (z. B. leere oder fehlerhafte Datei)
        if not docs:
            print(f"Keine Inhalte extrahiert aus: {file_path.name}")
            continue

        # Fortschritts-Feedback für Monitoring und Transparenz
        print(f"Verarbeitet: {file_path.name} – {len(docs)} Abschnitte")
        all_docs.extend(docs)

    print(f"Gesamtdokumente/Chunks geladen: {len(all_docs)}")
    return all_docs
