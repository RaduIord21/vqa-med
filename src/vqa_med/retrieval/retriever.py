"""
Medical retriever for VQA with RAG.
Enhanced with visual-aware retrieval capabilities.
"""
from typing import List, Dict, Optional
from pathlib import Path
import torch
import numpy as np

from .knowledge_base import MedicalKnowledgeBase


class MedicalRetriever:
    """
    Retriever that combines question, image, and context for medical VQA.
    Supports visual-aware retrieval using image embeddings.
    """
    
    def __init__(
        self,
        knowledge_base: MedicalKnowledgeBase,
        top_k: int = 3,
        rerank: bool = False,
        use_visual_context: bool = True,
        visual_weight: float = 0.3,
    ):
        """
        Args:
            knowledge_base: Medical knowledge base
            top_k: Number of documents to retrieve
            rerank: Whether to rerank results
            use_visual_context: Whether to use image features for retrieval
            visual_weight: Weight for visual features in hybrid retrieval (0-1)
        """
        self.kb = knowledge_base
        self.top_k = top_k
        self.rerank = rerank
        self.use_visual_context = use_visual_context
        self.visual_weight = visual_weight
        
        print(f"Medical Retriever initialized:")
        print(f"  Top-K: {top_k}")
        print(f"  Reranking: {rerank}")
        print(f"  Visual-aware retrieval: {use_visual_context}")
        if use_visual_context:
            print(f"  Visual weight: {visual_weight}")
    
    def retrieve(
        self,
        question: str,
        image_caption: Optional[str] = None,
        image_features: Optional[torch.Tensor] = None,
        visual_weight: Optional[float] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant medical knowledge with optional visual context and multi-query expansion.
        
        Args:
            question: VQA question
            image_caption: Optional image caption for context
            image_features: Optional image embeddings for visual-aware retrieval
            visual_weight: Optional override for visual weight (0-1)
            
        Returns:
            List of retrieved documents
        """
        # Use provided visual weight or default
        original_weight = self.visual_weight
        if visual_weight is not None:
            self.visual_weight = visual_weight

        # Combine question and caption
        if image_caption:
            query = f"{question} Image: {image_caption}"
        else:
            query = question
        
        # Multi-query retrieval for better coverage
        all_results = {}  # Use dict to avoid duplicates with key=doc_text
        
        # Query 1: Original question
        results1 = self.kb.search(query, top_k=self.top_k * 2)
        for r in results1:
            all_results[r['text']] = r
        
        # Query 2: Expanded query with medical synonyms/context
        expanded_query = self._expand_query(question)
        if expanded_query != query:
            results2 = self.kb.search(expanded_query, top_k=self.top_k)
            for r in results2:
                if r['text'] not in all_results:
                    all_results[r['text']] = r
        
        results = list(all_results.values())
        
        # Visual-aware retrieval if image features provided
        if self.use_visual_context and image_features is not None:
            results = self._rerank_with_visual(results, image_features)
        
        # Keep only top-k
        results = results[:self.top_k]
        
        # Restore original visual weight
        self.visual_weight = original_weight
        
        return results
    
    def _rerank_with_visual(
        self,
        results: List[Dict],
        image_features: torch.Tensor,
    ) -> List[Dict]:
        """
        Rerank retrieved documents using visual image features.
        
        Higher visual similarity + lower text distance = better ranking.
        
        Args:
            results: Initial retrieval results with scores
            image_features: Image embeddings [batch_size, dim]
            
        Returns:
            Reranked results
        """
        try:
            if not results:
                return results

            # Get image embeddings from knowledge base embedder
            image_emb = image_features.detach().cpu().numpy().astype('float32')

            # Convert vision features to a single 2D embedding [1, dim].
            # Common input shape from ViT is [B, num_patches, dim].
            if image_emb.ndim == 3:
                # Mean-pool over patch tokens, keep batch dimension.
                image_emb = image_emb.mean(axis=1)
            if image_emb.ndim == 1:
                image_emb = image_emb[None, :]
            elif image_emb.ndim == 2 and image_emb.shape[0] > 1:
                # If more than one sample sneaks in, average to one vector.
                image_emb = image_emb.mean(axis=0, keepdims=True)
            
            # Get document embeddings for comparison
            doc_texts = [r['text'] for r in results]
            doc_embeddings = self.kb.embedder.encode(
                doc_texts,
                convert_to_numpy=True
            ).astype('float32')

            # Align dimensions if vision and text embedding spaces differ
            if image_emb.shape[1] != doc_embeddings.shape[1]:
                image_emb = self._align_embedding_dim(image_emb, doc_embeddings.shape[1])
            
            # Compute visual similarity (negative distance -> higher similarity is better)
            # Using cosine similarity between image and document embeddings
            from sklearn.metrics.pairwise import cosine_similarity
            visual_similarity = cosine_similarity(image_emb, doc_embeddings)[0]
            
            # Normalize scores (lower text distance is better, higher visual similarity is better)
            text_scores = np.array([r['score'] for r in results])
            text_scores = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min() + 1e-8)
            
            visual_scores = (visual_similarity - visual_similarity.min()) / (visual_similarity.max() - visual_similarity.min() + 1e-8)
            
            # Hybrid ranking: combine text and visual scores
            # Lower text score is better, higher visual score is better
            combined_scores = (1 - self.visual_weight) * (1 - text_scores) + self.visual_weight * visual_scores
            
            # Rerank results
            sorted_indices = np.argsort(-combined_scores)
            reranked_results = [results[i] for i in sorted_indices]
            
            # Add reranking scores to metadata
            for idx, i in enumerate(sorted_indices):
                reranked_results[idx]['rerank_score'] = float(combined_scores[i])
            
            return reranked_results
            
        except Exception as e:
            print(f"Warning: Visual reranking failed ({e}). Using original results.")
            return results

    @staticmethod
    def _align_embedding_dim(embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Align embedding dimensionality with deterministic pooling/interpolation.

        This is used to compare ViT image embeddings (e.g. 768-dim) with
        sentence-transformer embeddings (e.g. 384-dim) for cosine similarity.
        """
        current_dim = embedding.shape[1]
        if current_dim == target_dim:
            return embedding

        if current_dim > target_dim:
            # If divisible, average contiguous groups for stable downsampling.
            if current_dim % target_dim == 0:
                factor = current_dim // target_dim
                return embedding.reshape(embedding.shape[0], target_dim, factor).mean(axis=2)

            # Fallback: linear interpolation down to target dimension.
            src_idx = np.linspace(0, current_dim - 1, num=current_dim)
            dst_idx = np.linspace(0, current_dim - 1, num=target_dim)
            return np.stack([
                np.interp(dst_idx, src_idx, row) for row in embedding
            ]).astype('float32')

        # If smaller, zero-pad to target size.
        pad = np.zeros((embedding.shape[0], target_dim - current_dim), dtype=embedding.dtype)
        return np.concatenate([embedding, pad], axis=1)
    
    def retrieve_with_fallback(
        self,
        question: str,
        image_caption: Optional[str] = None,
        image_features: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """
        Retrieve with fallback queries for better coverage.
        
        If initial retrieval returns few results, try expanding the query.
        """
        results = self.retrieve(question, image_caption, image_features)
        
        # If we got few results, try alternative queries
        if len(results) < self.top_k:
            # Try broader search without specific terms
            broad_question = self._broaden_query(question)
            if broad_question != question:
                additional = self.kb.search(broad_question, top_k=self.top_k - len(results))
                # Remove duplicates
                result_texts = {r['text'] for r in results}
                for doc in additional:
                    if doc['text'] not in result_texts:
                        results.append(doc)
                        if len(results) >= self.top_k:
                            break
        
        return results[:self.top_k]
    
    @staticmethod
    def _expand_query(question: str) -> str:
        """
        Expand query with medical synonyms and related terms for better retrieval coverage.
        
        Examples:
        - "Is there pneumonia?" -> "pneumonia infiltrate consolidation infection"
        - "Where is the abnormality?" -> "abnormality finding lesion pathology"
        """
        medical_synonyms = {
            'pneumonia': 'pneumonia infiltrate consolidation infection opacity',
            'abnormality': 'abnormality finding lesion pathology',
            'fracture': 'fracture break fracture line fragment',
            'mass': 'mass lesion nodule tumor growth',
            'nodule': 'nodule mass lesion opacity finding',
            'edema': 'edema swelling fluid accumulation pulmonary',
            'effusion': 'effusion fluid accumulation collection pleural',
            'hemorrhage': 'hemorrhage bleeding hematoma blood',
            'enlarged': 'enlarged enlarged dilated hypertrophied big',
            'normal': 'normal unremarkable no abnormality clear',
            'chest': 'chest thorax pulmonary lung thoracic',
            'abdomen': 'abdomen abdominal belly stomach visceral',
            'heart': 'heart cardiac cardiac chamber ventricle',
            'bone': 'bone skeletal osseous bony',
            'liver': 'liver hepatic hepatomegaly abdominal',
            'kidney': 'kidney renal nephric urinary',
            'lung': 'lung pulmonary respiratory air',
        }
        
        expanded = question.lower()
        for term, synonyms in medical_synonyms.items():
            if term in expanded:
                # Replace first occurrence with synonyms
                expanded = expanded.replace(term, synonyms, 1)
                return expanded
        
        return question
    
    @staticmethod
    def _broaden_query(question: str) -> str:
        """Create a broader version of the question by removing specific terms."""
        # Simple broadening: keep only important words
        stop_words = {'is', 'the', 'a', 'an', 'are', 'in', 'on', 'at', 'to', 'for', 'of', 'by', 'with'}
        words = question.lower().split()
        important_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        if len(important_words) < len(words):
            return " ".join(important_words[:3])  # Return top 3 important words
        return question
    
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
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics."""
        return {
            'total_documents': len(self.kb.documents),
            'top_k': self.top_k,
            'visual_aware': self.use_visual_context,
        }