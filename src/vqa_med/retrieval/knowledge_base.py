"""
Medical knowledge base for RAG.
Stores and indexes medical text for retrieval.
"""
import json
import pickle
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
    ):
        """
        Args:
            embedding_model: HuggingFace sentence transformer model
            index_type: FAISS index type ('flatl2', 'ivfflat', 'hnsw')
        """
        self.embedding_model_name = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
        
        print(f"Knowledge Base initialized:")
        print(f"  Embedding model: {embedding_model}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Index type: {index_type}")
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32,
    ):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of text documents
            metadata: Optional metadata for each document
            batch_size: Batch size for embedding
        """
        if metadata is None:
            metadata = [{'doc_id': i} for i in range(len(documents))]
        
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
        
        # Create or update FAISS index
        if self.index is None:
            self._create_index(embeddings.shape[1])
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        print(f"✓ Added {len(documents)} documents")
        print(f"✓ Total documents: {len(self.documents)}")
    
    def _create_index(self, dimension: int):
        """Create FAISS index."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        if self.index_type == "flatl2":
            # Simple flat L2 index (exact search)
            self.index = faiss.IndexFlatL2(dimension)
        
        elif self.index_type == "ivfflat":
            # IVF index (faster for large datasets)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            # Note: Needs training after adding documents
        
        elif self.index_type == "hnsw":
            # HNSW index (fast approximate search)
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        
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
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append({
                    'text': self.documents[idx],
                    'score': float(dist),
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
            index_type=config['index_type']
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
            'model': self.embedding_model_name,
        }