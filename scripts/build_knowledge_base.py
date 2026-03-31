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
    """Create comprehensive medical knowledge for RAG."""
    documents = [
        # === IMAGING MODALITIES ===
        # X-ray
        "Chest X-ray is the most commonly performed imaging examination, useful for evaluating the lungs, heart, and mediastinum.",
        "X-rays are produced by passing electrical current through a vacuum tube, creating electromagnetic radiation.",
        "X-rays pass through soft tissues but are absorbed by dense structures like bone, appearing white on radiographs.",
        "Air appears black on X-rays due to low density, while bone appears white due to high density.",
        "Lateral chest X-ray provides a side view of the chest useful for detecting mediastinal abnormalities.",
        "Frontal or anteroposterior chest X-ray is the standard projection in chest radiography.",
        
        # CT Scan
        "CT scans use X-rays to create detailed cross-sectional images, allowing visualization of internal structures.",
        "CT provides superior soft tissue contrast compared to conventional X-rays.",
        "Hyperdense areas on CT appear brighter or whiter and represent dense tissue like bone or acute blood.",
        "Hypodense areas on CT appear darker and represent less dense tissue like fat or air.",
        "CT with intravenous contrast provides better visualization of blood vessels and vascular pathology.",
        "Helical CT allows for fast scanning with thin slices, improving image quality and reducing motion artifacts.",
        
        # MRI
        "MRI uses magnetic fields and radio waves to produce detailed images without using ionizing radiation.",
        "T1-weighted MRI provides good anatomical detail, with fat and contrast appearing bright.",
        "T2-weighted MRI shows fluid as bright and is excellent for detecting edema and abnormal tissue.",
        "FLAIR MRI suppresses cerebrospinal fluid signal to better visualize brain lesions and gray matter abnormalities.",
        "Diffusion-weighted imaging (DWI) in MRI helps detect acute stroke and restricted water diffusion.",
        "MRI is superior to CT for soft tissue contrast and detecting lesions in the brain and spinal cord.",
        "Gadolinium contrast in MRI crosses the blood-brain barrier when inflammation disrupts it.",
        
        # Ultrasound
        "Ultrasound uses high-frequency sound waves to create images of internal body structures in real-time.",
        "Ultrasound is non-invasive, does not use ionizing radiation, and is safe in pregnancy.",
        "Ecogenicity describes the brightness of tissue on ultrasound, with bone and calcifications appearing bright.",
        "Doppler ultrasound measures blood flow in vessels and helps identify vascular abnormalities.",
        "Transducer frequency affects image quality, with higher frequencies providing better resolution but less depth.",
        
        # === ANATOMICAL STRUCTURES ===
        # Thorax
        "The lungs are paired organs located in the thoracic cavity on either side of the mediastinum.",
        "The right lung has three lobes (upper, middle, lower) while the left lung has two lobes (upper, lower).",
        "The trachea divides at the carina into the right and left main bronchi.",
        "The mediastinum is the central compartment of the chest containing the heart, esophagus, and major vessels.",
        "The pleural space is the potential space between the visceral and parietal pleura.",
        "The heart is a four-chambered muscular organ located in the left anterior mediastinum.",
        "The left ventricle is the main pumping chamber of the heart and appears larger than the right ventricle.",
        "The right atrium receives deoxygenated blood from the superior and inferior vena cava.",
        
        # Abdomen
        "The liver is the largest internal organ, occupying the right upper quadrant and extending across the epigastrium.",
        "The spleen is located in the left upper quadrant, posterior to the stomach and ribs.",
        "The kidneys are bean-shaped organs located retroperitoneally on either side of the vertebral column.",
        "The pancreas is located in the retroperitoneum, anterior to the right kidney and posterior to the stomach.",
        "The stomach is located in the left upper quadrant below the diaphragm.",
        "The gallbladder is a small pear-shaped organ in the right upper quadrant containing bile.",
        "The small bowel consists of the duodenum, jejunum, and ileum.",
        
        # Brain
        "The brain is protected by the skull and suspended in cerebrospinal fluid.",
        "The cerebral cortex is the outer layer of the brain responsible for consciousness and cognition.",
        "The basal ganglia are deep brain structures involved in motor control and emotion.",
        "The cerebellum is located posteriorly and is responsible for coordination and balance.",
        "The corpus callosum is the largest white matter tract connecting the right and left hemispheres.",
        "The ventricles are fluid-filled cavities in the brain containing cerebrospinal fluid.",
        
        # === PATHOLOGY FINDINGS ===
        # Lung pathology
        "Pneumothorax is the presence of air in the pleural space, causing the lung to collapse.",
        "Spontaneous pneumothorax occurs without prior lung disease, often in young tall males.",
        "Tension pneumothorax compresses the heart and mediastinum, causing hemodynamic compromise.",
        "Pleural effusion is the accumulation of fluid in the pleural space around the lungs.",
        "Transudative effusion results from systemic conditions like heart failure and has low protein content.",
        "Exudative effusion is due to inflammation or malignancy and has high protein content.",
        "Hemothorax is blood in the pleural space, often from trauma or malignancy.",
        "Pulmonary edema is fluid in the lungs, appearing as diffuse opacities on chest X-ray.",
        "Cardiogenic pulmonary edema occurs due to elevated hydrostatic pressure from heart failure.",
        "Non-cardiogenic pulmonary edema results from increased vascular permeability or decreased plasma colloid osmotic pressure.",
        
        # Cardiac pathology
        "Cardiomegaly refers to cardiac enlargement visible on chest X-ray when the cardiothoracic ratio exceeds 0.5.",
        "Atrial fibrillation causes irregular heart rhythm and increased risk of stroke.",
        "Myocardial infarction causes damage to the heart muscle and can be detected by ECG changes and elevated troponin.",
        "Heart failure results in inability of the heart to pump adequate blood, causing pulmonary and systemic congestion.",
        "Valvular disease affects blood flow through the heart chambers and can be assessed by echocardiography.",
        
        # Bone and skeletal pathology
        "Fractures are breaks in bone continuity visible as lucent lines on X-rays.",
        "Comminuted fractures involve multiple fragments and indicate high-energy trauma.",
        "Pathological fractures occur in abnormal bone due to osteoporosis, metastases, or other disease.",
        "Osteoporosis is decreased bone density increasing fracture risk, seen as decreased radiographic density.",
        "Osteoarthritis causes cartilage loss and joint space narrowing visible on X-rays.",
        
        # Mass and lesion pathology
        "Masses are abnormal collections of tissue that may be benign or malignant.",
        "Benign masses typically have well-defined borders and slow growth.",
        "Malignant masses have irregular borders, rapid growth, and may invade surrounding structures.",
        "Nodules are round or oval lesions typically less than 3 centimeters in size.",
        "Ground-glass nodules have partial opacity without obscuring vessels, suggesting early pneumonia or atypical adenocarcinoma.",
        "Cavitary lesions have an air-filled center and thin wall, commonly seen in tuberculosis.",
        "Lesions are areas of abnormal tissue visible on imaging and may require biopsy for diagnosis.",
        
        # Inflammatory and infectious pathology
        "Pneumonia is inflammation of the lungs causing consolidation appearing as dense opacities on chest X-ray.",
        "Bacterial pneumonia typically appears as lobar or segmental consolidation.",
        "Viral pneumonia often presents with interstitial or bilateral patterns.",
        "Aspiration pneumonia typically affects dependent lung portions due to gravity.",
        "Tuberculosis commonly affects the apical and posterior segments of the upper lobes.",
        "Interstitial lung disease causes reticular or reticulonodular pattern throughout the lungs.",
        "Bronchiectasis causes bronchial dilation with bronchus-to-artery ratio greater than 1.",
        
        # === IMAGING SIGNS AND PATTERNS ===
        # Densities and patterns
        "Consolidation is homogeneous opacity replacing air in lungs, indicating pneumonia or pulmonary edema.",
        "Infiltrate refers to abnormal material accumulating in lung tissue.",
        "Opacification is the appearance of increased brightness on imaging.",
        "Lucency refers to decreased density or darkness on imaging.",
        "Ground-glass opacification shows hazy opacity without obscuring vessels.",
        "Reticular pattern appears as a net-like grid of opacities.",
        "Nodular pattern shows multiple small rounded opacities.",
        "Reticulonodular pattern combines reticular and nodular components.",
        
        # Anatomical signs
        "Air bronchogram sign indicates bronchi filled with air surrounded by consolidated lung.",
        "Silhouette sign occurs when adjacent structures with similar density obscure the border between them.",
        "Halo sign is ground-glass opacity surrounding a central nodule, often seen in invasive aspergillosis.",
        "Target sign is a nodule with central lucency and peripheral consolidation.",
        "Atoll sign is ring-like consolidation surrounding a focal ground-glass opacity.",
        "Reversed halo sign is a central ground-glass area surrounded by consolidation.",
        
        # === ANATOMY POSITION AND DIRECTION ===
        "Anterior refers to the front or ventral surface of the body.",
        "Posterior refers to the back or dorsal surface of the body.",
        "Superior means toward the head or upper part of the body.",
        "Inferior means toward the feet or lower part of the body.",
        "Lateral means toward the side or away from the midline.",
        "Medial means toward the center or midline of the body.",
        "Proximal means closer to the origin or trunk of the body.",
        "Distal means farther from the origin or trunk of the body.",
        "Cranial or cephalic means toward the head.",
        "Caudal means toward the tail or lower body.",
        
        # Anatomical planes
        "The sagittal plane divides the body into left and right halves, running front to back.",
        "The midsagittal plane is the vertical plane through the midline dividing left and right equally.",
        "The coronal or frontal plane divides the body into front (anterior) and back (posterior).",
        "The axial, transverse, or horizontal plane divides the body into upper and lower sections.",
        "Parasagittal planes are vertical planes parallel to the midsagittal plane.",
        
        # === COMMON FINDINGS BY LOCATION ===
        # Right upper quadrant
        "Right upper quadrant contains the right lobe of liver, right kidney, gallbladder, and right colon.",
        "Pneumatosis in the right upper quadrant may indicate bowel wall air from intestinal perforation.",
        "Hepatomegaly is enlargement of the liver extending below the right costal margin.",
        
        # Left upper quadrant
        "Left upper quadrant contains the spleen, left kidney, stomach, and left colon.",
        "Splenic laceration causes free fluid (blood) in the left upper quadrant.",
        "Gastric distension fills the left upper quadrant with air.",
        
        # Right lower quadrant
        "Right lower quadrant contains the right colon, right ovary/testis, appendix, and ileal loops.",
        "Appendicitis causes right lower quadrant pain and may show appendiceal thickening on CT.",
        "Inguinal hernia appears as bowel loops extending through the inguinal canal.",
        
        # Left lower quadrant
        "Left lower quadrant contains the left colon, left ovary/testis and sigmoid.",
        "Diverticulitis affects the sigmoid colon causing thickening and inflammation.",
        "Ovarian pathology may present with mass or cyst in the left lower quadrant.",
        
        # === COMMON DISEASES AND CONDITIONS ===
        "Congestive heart failure causes bilateral pulmonary edema and cardiomegaly.",
        "Pneumonia causes consolidation, air bronchogram, and may be lobar or diffuse.",
        "Chronic obstructive pulmonary disease causes hyperinflation with flattened diaphragms.",
        "Asthma presents with normal imaging during remission or shows hyperinflation during attacks.",
        "Pulmonary embolism may show wedge-shaped infarction or atelectasis.",
        "Aortic aneurysm appears as widening of the mediastinum on chest X-ray.",
        "Pneumoperitoneum (free air in abdomen) appears as lucent shadow under the diaphragm.",
        "Bowel obstruction shows dilated bowel loops with air-fluid levels.",
    ]
    
    metadata = [{'source': 'comprehensive_medical_kb', 'topic': 'radiology'} for _ in documents]
    
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