from pathlib import Path
import docx
import openpyxl
from PyPDF2 import PdfReader
from langchain_core.documents import Document
import re

from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
from langchain_core.documents import Document
import re

def extract_text_from_pdf(file_path: Path) -> List[Document]:
    try:
        reader = PdfReader(str(file_path))
        documents = []

        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            lines = raw_text.splitlines()

            # Zeilen zusammenführen, wenn sie logisch zusammengehören
            merged_lines = []
            buffer = ""
            for line in lines:
                clean = line.strip()
                if not clean:
                    continue
                if re.match(r"^[a-zäöüß]", clean, re.IGNORECASE) and buffer:
                    buffer += " " + clean
                else:
                    if buffer:
                        merged_lines.append(buffer.strip())
                    buffer = clean
            if buffer:
                merged_lines.append(buffer.strip())

            # Chunking anhand von Abschnittsnummern oder Bulletpoints
            chunks = []
            chunk_buffer = ""
            for line in merged_lines:
                if re.match(r"^\d{1,2}\.\s+\w+", line) or re.match(r"^\d+\s+\w+", line):
                    if chunk_buffer:
                        chunks.append(chunk_buffer.strip())
                    chunk_buffer = line
                else:
                    chunk_buffer += " " + line
            if chunk_buffer:
                chunks.append(chunk_buffer.strip())

            for chunk in chunks:
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path.name,
                            "page_number": page_num + 1
                        }
                    )
                )

        return documents
    except Exception as e:
        print(f"Fehler beim Verarbeiten von PDF {file_path.name}: {e}")
        return []

def extract_text_from_docx(file_path: Path) -> list[Document]:
    try:
        doc = docx.Document(str(file_path))
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return [Document(page_content=full_text.strip(), metadata={"source": str(file_path)})]
    except Exception as e:
        print(f"Fehler beim DOCX {file_path.name}: {e}")
        return []

def extract_text_from_xlsx(file_path: Path) -> list[Document]:
    try:
        wb = openpyxl.load_workbook(str(file_path), data_only=True)
        text = ""
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = [str(cell) for cell in row if cell is not None]
                if row_text:
                    text += " | ".join(row_text) + "\n"
        return [Document(page_content=text.strip(), metadata={"source": str(file_path)})]
    except Exception as e:
        print(f"Fehler beim XLSX {file_path.name}: {e}")
        return []

def extract_all_documents_from_folder(folder_path: Path) -> list[Document]:
    all_docs = []
    for file_path in folder_path.rglob("*"):
        if file_path.suffix.lower() == ".pdf":
            all_docs.extend(extract_text_from_pdf(file_path))
        elif file_path.suffix.lower() == ".docx":
            all_docs.extend(extract_text_from_docx(file_path))
        elif file_path.suffix.lower() == ".xlsx":
            all_docs.extend(extract_text_from_xlsx(file_path))
    return all_docs
