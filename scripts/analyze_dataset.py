"""
Dataset analysis script.
Analyzes VQA-RAD dataset characteristics, distributions, and statistics.
"""

import os


os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
from tqdm import tqdm

from vqa_med.config import config


class DatasetAnalyzer:
    """Analyze VQA dataset characteristics."""
    
    def __init__(self, data_csv: Path, image_dir: Path):
        self.data_csv = Path(data_csv)
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(data_csv)
        
        print(f"Loaded dataset: {len(self.df)} samples")
    
    def analyze_basic_stats(self):
        """Analyze basic dataset statistics."""
        stats = {
            'total_samples': len(self.df),
            'unique_images': self.df['image_path'].nunique(),
            'unique_questions': self.df['question'].nunique(),
            'unique_answers': self.df['answer'].nunique(),
            'avg_questions_per_image': len(self.df) / self.df['image_path'].nunique(),
        }
        
        # Question length stats
        question_lengths = self.df['question'].str.split().str.len()
        stats['question_length'] = {
            'mean': question_lengths.mean(),
            'median': question_lengths.median(),
            'min': question_lengths.min(),
            'max': question_lengths.max(),
            'std': question_lengths.std(),
        }
        
        # Answer length stats
        answer_lengths = self.df['answer'].str.split().str.len()
        stats['answer_length'] = {
            'mean': answer_lengths.mean(),
            'median': answer_lengths.median(),
            'min': answer_lengths.min(),
            'max': answer_lengths.max(),
            'std': answer_lengths.std(),
        }
        
        return stats
    
    def analyze_question_types(self):
        """Analyze question type distribution."""
        if 'question_type' not in self.df.columns:
            return None
        
        qt_dist = self.df['question_type'].value_counts().to_dict()
        
        # Analyze answer distribution per question type
        qt_answer_dist = {}
        for qt in self.df['question_type'].unique():
            qt_df = self.df[self.df['question_type'] == qt]
            qt_answer_dist[qt] = {
                'total_samples': len(qt_df),
                'unique_answers': qt_df['answer'].nunique(),
                'top_answers': qt_df['answer'].value_counts().head(10).to_dict(),
            }
        
        return {
            'distribution': qt_dist,
            'answer_analysis': qt_answer_dist,
        }
    
    def analyze_answer_types(self):
        """Analyze answer type distribution."""
        if 'answer_type' not in self.df.columns:
            return None
        
        at_dist = self.df['answer_type'].value_counts().to_dict()
        
        # Analyze per answer type
        at_answer_dist = {}
        for at in self.df['answer_type'].unique():
            at_df = self.df[self.df['answer_type'] == at]
            at_answer_dist[at] = {
                'total_samples': len(at_df),
                'unique_answers': at_df['answer'].nunique(),
                'top_answers': at_df['answer'].value_counts().head(10).to_dict(),
            }
        
        return {
            'distribution': at_dist,
            'answer_analysis': at_answer_dist,
        }
    
    def analyze_answer_distribution(self, top_n: int = 50):
        """Analyze answer distribution and imbalance."""
        answer_counts = self.df['answer'].value_counts()
        
        # Class imbalance metrics
        total = len(self.df)
        imbalance_ratio = answer_counts.iloc[0] / answer_counts.iloc[-1]
        
        # Gini coefficient (measure of inequality)
        counts = answer_counts.values
        n = len(counts)
        gini = (2 * np.sum((np.arange(1, n+1)) * np.sort(counts))) / (n * np.sum(counts)) - (n + 1) / n
        
        stats = {
            'total_unique_answers': len(answer_counts),
            'most_common_answer': answer_counts.index[0],
            'most_common_count': int(answer_counts.iloc[0]),
            'most_common_percentage': 100.0 * answer_counts.iloc[0] / total,
            'least_common_answer': answer_counts.index[-1],
            'least_common_count': int(answer_counts.iloc[-1]),
            'imbalance_ratio': float(imbalance_ratio),
            'gini_coefficient': float(gini),
            'top_n_answers': answer_counts.head(top_n).to_dict(),
        }
        
        # Coverage analysis (how many answers cover X% of data)
        cumsum = answer_counts.cumsum()
        for pct in [50, 75, 90, 95, 99]:
            threshold = total * pct / 100
            n_answers = (cumsum <= threshold).sum() + 1
            stats[f'answers_for_{pct}pct_coverage'] = int(n_answers)
        
        return stats
    
    def analyze_images(self, sample_size: int = 100):
        """Analyze image characteristics."""
        print("\nAnalyzing images (sampling)...")
        
        # Sample images
        unique_images = self.df['image_path'].unique()
        sample_images = np.random.choice(
            unique_images,
            size=min(sample_size, len(unique_images)),
            replace=False
        )
        
        sizes = []
        aspects = []
        modes = []
        
        for img_name in tqdm(sample_images, desc="Loading images"):
            img_path = self.image_dir / img_name
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    sizes.append(img.size)
                    aspects.append(img.size[0] / img.size[1])
                    modes.append(img.mode)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if not sizes:
            return None
        
        widths, heights = zip(*sizes)
        
        stats = {
            'num_sampled': len(sizes),
            'width': {
                'mean': np.mean(widths),
                'median': np.median(widths),
                'min': np.min(widths),
                'max': np.max(widths),
                'std': np.std(widths),
            },
            'height': {
                'mean': np.mean(heights),
                'median': np.median(heights),
                'min': np.min(heights),
                'max': np.max(heights),
                'std': np.std(heights),
            },
            'aspect_ratio': {
                'mean': np.mean(aspects),
                'median': np.median(aspects),
                'min': np.min(aspects),
                'max': np.max(aspects),
            },
            'modes': dict(Counter(modes)),
        }
        
        return stats
    
    def analyze_question_patterns(self):
        """Analyze common question patterns and structures."""
        # Extract question words (first word)
        question_words = self.df['question'].str.split().str[0].str.lower()
        question_word_dist = question_words.value_counts().to_dict()
        
        # Common bigrams
        from collections import defaultdict
        bigram_counts = defaultdict(int)
        
        for question in self.df['question']:
            words = question.lower().split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigram_counts[bigram] += 1
        
        top_bigrams = dict(sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        
        return {
            'question_words': question_word_dist,
            'top_bigrams': top_bigrams,
        }
    
    def generate_report(self, output_dir: Path):
        """Generate comprehensive analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Generating Dataset Analysis Report")
        print("=" * 60)
        
        report = {}
        
        # Basic statistics
        print("\n1. Basic Statistics...")
        report['basic_stats'] = self.analyze_basic_stats()
        
        # Question types
        print("2. Question Types...")
        report['question_types'] = self.analyze_question_types()
        
        # Answer types
        print("3. Answer Types...")
        report['answer_types'] = self.analyze_answer_types()
        
        # Answer distribution
        print("4. Answer Distribution...")
        report['answer_distribution'] = self.analyze_answer_distribution()
        
        # Question patterns
        print("5. Question Patterns...")
        report['question_patterns'] = self.analyze_question_patterns()
        
        # Image analysis
        print("6. Image Characteristics...")
        report['image_stats'] = self.analyze_images(sample_size=100)
        
        # Convert numpy types to native Python types
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        report = convert_to_serializable(report)
        
        # Save report
        with open(output_dir / 'dataset_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_dir / 'dataset_analysis.json'}")
        
        return report

    def plot_visualizations(self, output_dir: Path):
        """Generate visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        # 1. Question type distribution
        if 'question_type' in self.df.columns:
            plt.figure(figsize=(12, 6))
            qt_counts = self.df['question_type'].value_counts()
            plt.bar(range(len(qt_counts)), qt_counts.values)
            plt.xticks(range(len(qt_counts)), qt_counts.index, rotation=45, ha='right')
            plt.xlabel('Question Type')
            plt.ylabel('Count')
            plt.title('Question Type Distribution')
            plt.tight_layout()
            plt.savefig(output_dir / 'question_type_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Answer type distribution
        if 'answer_type' in self.df.columns:
            plt.figure(figsize=(10, 6))
            at_counts = self.df['answer_type'].value_counts()
            plt.bar(range(len(at_counts)), at_counts.values)
            plt.xticks(range(len(at_counts)), at_counts.index, rotation=45, ha='right')
            plt.xlabel('Answer Type')
            plt.ylabel('Count')
            plt.title('Answer Type Distribution')
            plt.tight_layout()
            plt.savefig(output_dir / 'answer_type_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Top answers distribution
        plt.figure(figsize=(14, 6))
        top_answers = self.df['answer'].value_counts().head(30)
        plt.bar(range(len(top_answers)), top_answers.values)
        plt.xticks(range(len(top_answers)), top_answers.index, rotation=45, ha='right')
        plt.xlabel('Answer')
        plt.ylabel('Frequency')
        plt.title('Top 30 Most Frequent Answers')
        plt.tight_layout()
        plt.savefig(output_dir / 'top_answers.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Answer frequency distribution (log scale)
        plt.figure(figsize=(12, 6))
        answer_counts = self.df['answer'].value_counts().values
        plt.hist(answer_counts, bins=50, edgecolor='black')
        plt.xlabel('Frequency')
        plt.ylabel('Number of Answers')
        plt.title('Answer Frequency Distribution')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / 'answer_frequency_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Question length distribution
        plt.figure(figsize=(10, 6))
        question_lengths = self.df['question'].str.split().str.len()
        plt.hist(question_lengths, bins=30, edgecolor='black')
        plt.xlabel('Question Length (words)')
        plt.ylabel('Frequency')
        plt.title('Question Length Distribution')
        plt.axvline(question_lengths.mean(), color='red', linestyle='--', label=f'Mean: {question_lengths.mean():.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'question_length_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 6. Answer length distribution
        plt.figure(figsize=(10, 6))
        answer_lengths = self.df['answer'].str.split().str.len()
        plt.hist(answer_lengths, bins=range(1, int(answer_lengths.max()) + 2), edgecolor='black')
        plt.xlabel('Answer Length (words)')
        plt.ylabel('Frequency')
        plt.title('Answer Length Distribution')
        plt.axvline(answer_lengths.mean(), color='red', linestyle='--', label=f'Mean: {answer_lengths.mean():.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'answer_length_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 7. Questions per image distribution
        questions_per_image = self.df.groupby('image_path').size()
        plt.figure(figsize=(10, 6))
        plt.hist(questions_per_image, bins=range(1, int(questions_per_image.max()) + 2), edgecolor='black')
        plt.xlabel('Number of Questions per Image')
        plt.ylabel('Number of Images')
        plt.title('Questions per Image Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / 'questions_per_image.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 8. Question word frequency
        question_words = self.df['question'].str.split().str[0].str.lower()
        top_qwords = question_words.value_counts().head(15)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_qwords)), top_qwords.values)
        plt.xticks(range(len(top_qwords)), top_qwords.index, rotation=45, ha='right')
        plt.xlabel('Question Starting Word')
        plt.ylabel('Frequency')
        plt.title('Top Question Starting Words')
        plt.tight_layout()
        plt.savefig(output_dir / 'question_words.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")
    
    def print_summary(self, report: dict):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("Dataset Analysis Summary")
        print("=" * 60)
        
        # Basic stats
        basic = report['basic_stats']
        print(f"\nBasic Statistics:")
        print(f"  Total samples: {basic['total_samples']:,}")
        print(f"  Unique images: {basic['unique_images']:,}")
        print(f"  Unique questions: {basic['unique_questions']:,}")
        print(f"  Unique answers: {basic['unique_answers']:,}")
        print(f"  Avg questions per image: {basic['avg_questions_per_image']:.2f}")
        
        print(f"\n  Question length: {basic['question_length']['mean']:.1f} ± {basic['question_length']['std']:.1f} words")
        print(f"  Answer length: {basic['answer_length']['mean']:.1f} ± {basic['answer_length']['std']:.1f} words")
        
        # Question types
        if report['question_types']:
            print(f"\nQuestion Type Distribution:")
            for qt, count in sorted(report['question_types']['distribution'].items(), key=lambda x: x[1], reverse=True):
                pct = 100.0 * count / basic['total_samples']
                print(f"  {qt:20s}: {count:5d} ({pct:5.1f}%)")
        
        # Answer types
        if report['answer_types']:
            print(f"\nAnswer Type Distribution:")
            for at, count in sorted(report['answer_types']['distribution'].items(), key=lambda x: x[1], reverse=True):
                pct = 100.0 * count / basic['total_samples']
                print(f"  {at:20s}: {count:5d} ({pct:5.1f}%)")
        
        # Answer distribution
        ans_dist = report['answer_distribution']
        print(f"\nAnswer Distribution Analysis:")
        print(f"  Total unique answers: {ans_dist['total_unique_answers']}")
        print(f"  Most common: '{ans_dist['most_common_answer']}' ({ans_dist['most_common_percentage']:.1f}%)")
        print(f"  Imbalance ratio: {ans_dist['imbalance_ratio']:.1f}:1")
        print(f"  Gini coefficient: {ans_dist['gini_coefficient']:.3f}")
        
        print(f"\n  Coverage:")
        for pct in [50, 75, 90, 95]:
            n = ans_dist[f'answers_for_{pct}pct_coverage']
            print(f"    {n} answers cover {pct}% of data")
        
        # Top answers
        print(f"\n  Top 10 Answers:")
        for i, (ans, count) in enumerate(list(ans_dist['top_n_answers'].items())[:10], 1):
            pct = 100.0 * count / basic['total_samples']
            print(f"    {i:2d}. {ans:20s}: {count:5d} ({pct:5.1f}%)")
        
        # Question patterns
        qpatterns = report['question_patterns']
        print(f"\n  Top Question Words:")
        for word, count in list(qpatterns['question_words'].items())[:10]:
            pct = 100.0 * count / basic['total_samples']
            print(f"    {word:15s}: {count:5d} ({pct:5.1f}%)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze VQA Dataset')
    
    parser.add_argument('--data_csv', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results')
    parser.add_argument('--no_images', action='store_true',
                        help='Skip image analysis (faster)')
    
    return parser.parse_args()


def main():
    """Main analysis function."""
    args = parse_args()
    
    print("=" * 60)
    print("VQA Dataset Analysis")
    print("=" * 60)
    
    # Paths
    if args.data_csv:
        data_csv = Path(args.data_csv)
    else:
        data_csv = config.paths.processed_data / "vqa_rad_closed.csv"
    
    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        image_dir = config.paths.raw_data / "VQA-RAD" / "images"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.paths.outputs_dir / "dataset_analysis"
    
    if not data_csv.exists():
        print(f"ERROR: Data file not found at {data_csv}")
        return
    
    # Create analyzer
    analyzer = DatasetAnalyzer(data_csv, image_dir)
    
    # Generate report
    report = analyzer.generate_report(output_dir)
    
    # Generate visualizations
    analyzer.plot_visualizations(output_dir)
    
    # Print summary
    analyzer.print_summary(report)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()