"""
Medical retriever for VQA with RAG.
"""
from typing import List, Dict, Optional
from pathlib import Path

from .knowledge_base import MedicalKnowledgeBase


class MedicalRetriever:
    """
    Retriever that combines question and image context for medical VQA.
    """
    
    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase,
        top_k: int = 3,
        rerank: bool = False,
    ):
        """
        Args:
            knowledge_base: Medical knowledge base
            top_k: Number of documents to retrieve
            rerank: Whether to rerank results (future enhancement)
        """
        self.kb = knowledge_base
        self.top_k = top_k
        self.rerank = rerank
        
        print(f"Medical Retriever initialized:")
        print(f"  Top-K: {top_k}")
        print(f"  Reranking: {rerank}")
    
    def retrieve(
        self,
        question: str,
        image_caption: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant medical knowledge.
        
        Args:
            question: VQA question
            image_caption: Optional image caption for context
            
        Returns:
            List of retrieved documents
        """
        # Combine question and caption for better retrieval
        if image_caption:
            query = f"{question} Context: {image_caption}"
        else:
            query = question
        
        # Retrieve
        results = self.kb.search(query, top_k=self.top_k)
        
        # TODO: Add reranking if enabled
        
        return results
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[{i}] {doc['text']}")
        
        return " ".join(context_parts)