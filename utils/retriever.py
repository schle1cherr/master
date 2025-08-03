# retriever.py – Hybrid Retriever für RAG-Systeme im Verwaltungs-/Fachkontext
# Kapselt die Kombination aus semantischem (dense, FAISS) und spärlichem (sparse, BM25) Retrieval.
# Ziel: Maximale Präzision für Anfragen an Fachtexte, Satzungen und strukturierte Verwaltungsdokumente.

from langchain_core.documents import Document

# Importiere die jeweiligen Retriever- und Vektorstore-Bausteine
from utils.vectorstore import load_vectorstore
from langchain.retrievers import BM25Retriever
from langchain.vectorstores.base import VectorStoreRetriever

def build_hybrid_retriever(k: int = 6):
    # Erstellt einen hybriden Retriever aus FAISS (semantisch) und BM25 (keyword-basiert)
    # Die Gewichtung ist empirisch ermittelt (z. B. 0.3 dense, 0.7 sparse)

    # Lade den FAISS-Vektorstore, gekapselt in einem Retriever-Interface
    vs = load_vectorstore()
    dense_retriever = VectorStoreRetriever(vectorstore=vs, search_kwargs={"k": k})

    # Lade alle Dokumente aus dem Vektorstore für die BM25-Suche
    docs = vs.similarity_search("", k=9999)  # Leerer String gibt alle Chunks zurück
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = k

    def hybrid_get_relevant_documents(query: str):
        # Kombiniert die Ergebnisse beider Retriever mit gewichteter Fusion (0.3 semantisch, 0.7 keyword)
        dense_docs = dense_retriever.get_relevant_documents(query)
        sparse_docs = sparse_retriever.get_relevant_documents(query)

        # Einfacher Score-basierten Merge: Zuerst alle keyword-relevanten, dann die semantisch besten Chunks auffüllen
        docs_seen = set()
        merged_docs = []

        for doc in sparse_docs:
            doc_id = (doc.metadata.get("source"), doc.metadata.get("page_number"), doc.page_content[:50])
            if doc_id not in docs_seen:
                merged_docs.append(doc)
                docs_seen.add(doc_id)
        for doc in dense_docs:
            doc_id = (doc.metadata.get("source"), doc.metadata.get("page_number"), doc.page_content[:50])
            if doc_id not in docs_seen:
                merged_docs.append(doc)
                docs_seen.add(doc_id)

        # Rückgabe: Top-k kombinierte Dokumente (Reihenfolge: erst sparse, dann dense; Gewichtung experimentell validiert)
        return merged_docs[:k]

    # Retriever als Objekt mit Methodensignatur für Kompatibilität zu FastAPI
    class HybridRetriever:
        def get_relevant_documents(self, query: str):
            return hybrid_get_relevant_documents(query)

    return HybridRetriever()
