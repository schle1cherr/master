# utils/retriever.py
from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from utils.vectorstore import load_vectorstore
from utils.loader import load_documents_from_folder


def build_dense_retriever(k: int = 10) -> VectorStoreRetriever:
    """
    Lädt den FAISS-Vektorstore und erstellt einen Dense Retriever.
    :param k: Anzahl der zurückgegebenen Dokumente
    :return: VectorStoreRetriever
    """
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_sparse_retriever(k: int = 10) -> BM25Retriever:
    """
    Erstellt einen BM25Retriever aus den vorhandenen Dokumenten.
    :param k: Anzahl der zurückgegebenen Dokumente
    :return: BM25Retriever
    """
    docs = load_documents_from_folder()
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


def build_hybrid_retriever(k: int = 20, w_dense: float = 0.3, w_sparse: float = 0.7) -> EnsembleRetriever:
    """
    Erstellt einen EnsembleRetriever, der FAISS (dense) und BM25 (sparse) kombiniert.
    :param k: Anzahl der Treffer pro Retriever
    :param w_dense: Gewicht des dichten (FAISS) Retrievers
    :param w_sparse: Gewicht des spärlichen (BM25) Retrievers
    :return: EnsembleRetriever mit gewichteter Kombination
    """
    dense = build_dense_retriever(k)
    sparse = build_sparse_retriever(k)

    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[w_dense, w_sparse]
    )
