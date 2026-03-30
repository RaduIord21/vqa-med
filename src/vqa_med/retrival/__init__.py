"""Retrieval module for RAG."""
from .knowledge_base import MedicalKnowledgeBase
from .retriever import MedicalRetriever

__all__ = ["MedicalKnowledgeBase", "MedicalRetriever"]