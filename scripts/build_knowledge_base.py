"""
Build medical knowledge base from text sources.
"""
import argparse
from pathlib import Path
import json

from vqa_med.retrieval import MedicalKnowledgeBase
from vqa_med.config import config


def load_medical_texts(source_file: Path) -> list:
    """
    Load medical texts from file.
    
    Expected format (JSON):
    [
        {"text": "...", "source": "...", "topic": "..."},
        ...
    ]
    
    Or simple text file (one document per line).
    """
    source_file = Path(source_file)
    
    if source_file.suffix == '.json':
        with open(source_file, 'r') as f:
            data = json.load(f)
        
        documents = [item['text'] for item in data]
        metadata = [{'source': item.get('source', 'unknown'), 
                    'topic': item.get('topic', 'general')} 
                   for item in data]
    
    elif source_file.suffix == '.txt':
        with open(source_file, 'r') as f:
            documents = [line.strip() for line in f if line.strip()]
        
        metadata = [{'source': str(source_file), 'line': i} 
                   for i in range(len(documents))]
    
    else:
        raise ValueError(f"Unsupported file format: {source_file.suffix}")
    
    return documents, metadata


def create_sample_medical_knowledge():
    """Create sample medical knowledge for testing."""
    documents = [
        # Radiology basics
        "CT scans use X-rays to create detailed cross-sectional images of the body.",
        "MRI uses magnetic fields and radio waves to produce detailed images of organs and tissues.",
        "X-rays are a form of electromagnetic radiation used to view bones and some organs.",
        "Ultrasound uses high-frequency sound waves to create images of internal body structures.",
        
        # Anatomical planes
        "The sagittal plane divides the body into left and right halves.",
        "The coronal plane divides the body into front and back portions.",
        "The axial plane divides the body into upper and lower sections.",
        "The transverse plane is perpendicular to the long axis of the body.",
        
        # Common findings
        "Pneumothorax is the presence of air in the pleural space causing lung collapse.",
        "Pleural effusion is the accumulation of fluid in the pleural space around the lungs.",
        "Cardiomegaly refers to an enlarged heart visible on chest X-ray.",
        "Pulmonary edema is fluid accumulation in the lungs often seen as cloudy areas on imaging.",
        
        # Organs
        "The lungs are paired organs located in the thoracic cavity responsible for gas exchange.",
        "The heart is a muscular organ that pumps blood throughout the body.",
        "The liver is the largest internal organ located in the upper right abdomen.",
        "The kidneys are bean-shaped organs that filter blood and produce urine.",
        
        # Abnormalities
        "A mass is an abnormal collection of tissue that may be benign or malignant.",
        "Nodules are small round growths that can appear in various organs.",
        "Fractures are breaks in bone continuity visible on X-ray imaging.",
        "Lesions are areas of abnormal tissue change visible on medical imaging.",
        
        # Colors and contrast
        "Hyperdense areas appear brighter or whiter on CT scans.",
        "Hypodense areas appear darker on CT scans.",
        "Air appears black on X-rays and CT scans.",
        "Bone appears white on X-rays due to high density.",
        "Contrast agents enhance visibility of blood vessels and certain tissues.",
        
        # Positions
        "Anterior refers to the front of the body.",
        "Posterior refers to the back of the body.",
        "Superior means toward the head or upper part.",
        "Inferior means toward the feet or lower part.",
        "Lateral means away from the midline.",
        "Medial means toward the midline.",
        
        # Imaging modalities details
        "FLAIR MRI suppresses fluid signal to better visualize brain lesions.",
        "T1-weighted MRI provides good anatomical detail with fat appearing bright.",
        "T2-weighted MRI shows fluid as bright and is good for detecting edema.",
        "Contrast-enhanced imaging uses agents to improve tissue differentiation.",
    ]
    
    metadata = [{'source': 'sample', 'topic': 'radiology'} for _ in documents]
    
    return documents, metadata


def parse_args():
    parser = argparse.ArgumentParser(description='Build Medical Knowledge Base')
    parser.add_argument('--source_file', type=str, default=None,
                        help='Path to medical text source file (JSON or TXT)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save knowledge base')
    parser.add_argument('--embedding_model', type=str, 
                        default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Sentence transformer model for embeddings')
    parser.add_argument('--use_sample', action='store_true',
                        help='Use sample medical knowledge (for testing)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Building Medical Knowledge Base")
    print("=" * 60)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.paths.data_root / "knowledge" / "medical_kb"
    
    # Load documents
    if args.use_sample:
        print("\nUsing sample medical knowledge...")
        documents, metadata = create_sample_medical_knowledge()
    elif args.source_file:
        print(f"\nLoading from: {args.source_file}")
        documents, metadata = load_medical_texts(Path(args.source_file))
    else:
        print("ERROR: Provide --source_file or use --use_sample")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Create knowledge base
    print("\nCreating knowledge base...")
    kb = MedicalKnowledgeBase(
        embedding_model=args.embedding_model,
        index_type='flatl2'
    )
    
    # Add documents
    kb.add_documents(documents, metadata)
    
    # Save
    kb.save(output_dir)
    
    # Test retrieval
    print("\n" + "=" * 60)
    print("Testing Retrieval")
    print("=" * 60)
    
    test_queries = [
        "What imaging modality was used?",
        "Where is the abnormality located?",
        "Is there any fluid in the lungs?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = kb.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. [Score: {result['score']:.4f}] {result['text'][:80]}...")
    
    print("\n" + "=" * 60)
    print("✓ Knowledge base built successfully!")
    print("=" * 60)
    print(f"Saved to: {output_dir}")
    print(f"Total documents: {len(documents)}")


if __name__ == "__main__":
    main()