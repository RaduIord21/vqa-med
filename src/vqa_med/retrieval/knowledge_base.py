"""
Medical knowledge base for RAG.
Stores and indexes medical text for retrieval.
"""
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Install with: pip install faiss-cpu")

from sentence_transformers import SentenceTransformer


class MedicalKnowledgeBase:
    """
    Vector database for medical knowledge.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flatl2",
        metric: str = "cosine",
    ):
        """
        Args:
            embedding_model: HuggingFace sentence transformer model
            index_type: FAISS index type ('flatl2', 'ivfflat', 'hnsw')
            metric: Similarity metric ('cosine' or 'l2')
        """
        self.embedding_model_name = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        self.index_type = index_type
        self.metric = metric.lower()
        if self.metric not in {"cosine", "l2"}:
            raise ValueError("metric must be one of: 'cosine', 'l2'")

        self.index = None
        self.documents = []
        self.metadata = []
        
        print(f"Knowledge Base initialized:")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Index type: {index_type}")
        print(f"  Metric: {self.metric}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for cleaner indexing and stronger recall."""
        text = text.strip().replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        # Keep punctuation but normalize repeated separators/noise.
        text = re.sub(r"[\t\r]+", " ", text)
        return text

    @staticmethod
    def _is_valid_text(text: str, min_words: int = 5) -> bool:
        """Filter noisy entries that are too short to help retrieval."""
        if not text:
            return False
        return len(text.split()) >= min_words

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 80,
        chunk_overlap: int = 15,
    ) -> List[str]:
        """
        Chunk text by words with overlap to improve retrieval granularity.

        Args:
            text: Input document
            chunk_size: Max words per chunk
            chunk_overlap: Word overlap between consecutive chunks
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        step = max(1, chunk_size - chunk_overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = words[start:start + chunk_size]
            if not chunk:
                continue
            chunks.append(" ".join(chunk))
            if start + chunk_size >= len(words):
                break
        return chunks

    def prepare_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        chunk_size: int = 80,
        chunk_overlap: int = 15,
        deduplicate: bool = True,
        min_words: int = 5,
    ) -> tuple[List[str], List[Dict], Dict]:
        """Clean, chunk, and deduplicate raw documents before indexing."""
        if metadata is None:
            metadata = [{"doc_id": i} for i in range(len(documents))]

        if len(metadata) != len(documents):
            raise ValueError("metadata length must match documents length")

        prepared_docs: List[str] = []
        prepared_meta: List[Dict] = []
        seen = set()

        stats = {
            "input_documents": len(documents),
            "dropped_short_or_empty": 0,
            "deduplicated": 0,
            "chunked_documents": 0,
            "output_chunks": 0,
        }

        for doc_idx, (text, meta) in enumerate(zip(documents, metadata)):
            normalized = self._normalize_text(str(text))
            if not self._is_valid_text(normalized, min_words=min_words):
                stats["dropped_short_or_empty"] += 1
                continue

            chunks = self._chunk_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if len(chunks) > 1:
                stats["chunked_documents"] += 1

            for chunk_id, chunk in enumerate(chunks):
                if deduplicate:
                    dedup_key = chunk.lower()
                    if dedup_key in seen:
                        stats["deduplicated"] += 1
                        continue
                    seen.add(dedup_key)

                chunk_meta = dict(meta)
                chunk_meta.update({
                    "original_doc_id": doc_idx,
                    "chunk_id": chunk_id,
                    "num_chunks": len(chunks),
                    "word_count": len(chunk.split()),
                })

                prepared_docs.append(chunk)
                prepared_meta.append(chunk_meta)

        stats["output_chunks"] = len(prepared_docs)
        return prepared_docs, prepared_meta, stats
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32,
        chunk_size: int = 80,
        chunk_overlap: int = 15,
        deduplicate: bool = True,
        min_words: int = 5,
    ):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
            batch_size: Batch size for embedding
            chunk_size: Max words per chunk for long documents
            chunk_overlap: Word overlap between chunks
            deduplicate: Remove duplicate chunks during ingestion
            min_words: Minimum words required to keep a chunk
        """
        documents, metadata, prep_stats = self.prepare_documents(
            documents=documents,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            deduplicate=deduplicate,
            min_words=min_words,
        )

        if not documents:
            raise ValueError("No valid documents remained after preprocessing")
        
        print(f"\nAdding {len(documents)} documents to knowledge base...")
        
        # Generate embeddings
        embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding documents"):
            batch = documents[i:i + batch_size]
            batch_embeddings = self.embedder.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')

        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Create or update FAISS index
        if self.index is None:
            self._create_index(embeddings.shape[1])

        if self.index_type == "ivfflat" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        print(f"✓ Added {len(documents)} documents")
        print(f"✓ Total documents: {len(self.documents)}")
        print(f"✓ Preprocessing stats: {prep_stats}")
    
    def _create_index(self, dimension: int):
        """Create FAISS index."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        if self.index_type == "flatl2":
            # Exact search index
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)
        
        elif self.index_type == "ivfflat":
            # IVF index (faster for large datasets)
            if self.metric == "cosine":
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT
                )
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, dimension, 100, faiss.METRIC_L2
                )
            # Note: Needs training after adding documents
        
        elif self.index_type == "hnsw":
            # HNSW index (fast approximate search)
            metric = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
            self.index = faiss.IndexHNSWFlat(dimension, 32, metric)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        print(f"Created FAISS index: {self.index_type}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'score', 'metadata'
        """
        if self.index is None or len(self.documents) == 0:
            print("Warning: Knowledge base is empty!")
            return []
        
        # Embed query
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx < len(self.documents):  # Valid index
                raw_score = float(dist)
                # Keep `score` as a distance-like value for backward compatibility.
                if self.metric == "cosine":
                    distance_like_score = 1.0 - raw_score
                    similarity_score = raw_score
                    score_type = "cosine_distance"
                else:
                    distance_like_score = raw_score
                    similarity_score = 1.0 / (1.0 + raw_score)
                    score_type = "l2_distance"

                results.append({
                    'text': self.documents[idx],
                    'score': distance_like_score,
                    'raw_score': raw_score,
                    'similarity_score': similarity_score,
                    'score_type': score_type,
                    'metadata': self.metadata[idx],
                })
        
        return results
    
    def save(self, save_dir: Path):
        """Save knowledge base to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save index
        if self.index is not None:
            faiss.write_index(self.index, str(save_dir / "faiss.index"))
        
        # Save documents and metadata
        with open(save_dir / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(save_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save config
        config = {
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'num_documents': len(self.documents),
        }
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Knowledge base saved to: {save_dir}")
    
    @classmethod
    def load(cls, load_dir: Path):
        """Load knowledge base from disk."""
        load_dir = Path(load_dir)
        
        # Load config
        with open(load_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        kb = cls(
            embedding_model=config['embedding_model'],
            index_type=config['index_type'],
            metric=config.get('metric', 'cosine'),
        )
        
        # Load index
        if (load_dir / "faiss.index").exists():
            kb.index = faiss.read_index(str(load_dir / "faiss.index"))
        
        # Load documents and metadata
        with open(load_dir / "documents.pkl", 'rb') as f:
            kb.documents = pickle.load(f)
        
        with open(load_dir / "metadata.pkl", 'rb') as f:
            kb.metadata = pickle.load(f)
        
        print(f"✓ Knowledge base loaded from: {load_dir}")
        print(f"✓ Documents: {len(kb.documents)}")
        
        return kb
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        return {
            'num_documents': len(self.documents),
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'metric': self.metric,
            'model': self.embedding_model_name,
            'avg_doc_words': (
                float(np.mean([len(doc.split()) for doc in self.documents])) if self.documents else 0.0
            ),
        }