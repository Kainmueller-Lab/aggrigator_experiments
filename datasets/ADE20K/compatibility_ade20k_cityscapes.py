import os
import numpy as np
import json
from pathlib import Path
from PIL import Image
from collections import Counter
import argparse
from tqdm import tqdm

def load_ade20k_sample_names(image_path):
    """Load all ADE20K sample names from the image directory."""
    image_path = Path(image_path)
    
    # Check if it's validation or test directory
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    sample_names = []
    for f in os.listdir(image_path):
        if f.endswith('.jpg'):
            sample_names.append(f.split('.')[0])
    
    return sorted(sample_names)

def load_mask(mask_path, sample_name):
    """Load a single mask file."""
    mask_file = mask_path / f"{sample_name}.png"
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    mask = np.array(Image.open(mask_file))
    return mask

def analyze_sample_classes(mask, exact_matches):
    """
    Analyze a sample mask to determine class distribution.
    
    Args:
        mask: numpy array of shape (H, W) containing class indices
        exact_matches: dictionary mapping ADE20K classes to Cityscapes classes
    
    Returns:
        dict: Analysis results containing class statistics
    """
    # Get unique classes in the mask
    unique_classes = np.unique(mask)
    total_pixels = mask.size
    
    # Count pixels for each class
    class_counts = {}
    for class_id in unique_classes:
        class_counts[class_id] = np.sum(mask == class_id)
    
    # Analyze matching classes
    matching_classes = set(exact_matches.keys())
    present_matching_classes = set(unique_classes) & matching_classes
    
    # Calculate coverage of matching classes
    matching_pixels = sum(class_counts.get(cls, 0) for cls in present_matching_classes)
    matching_coverage = matching_pixels / total_pixels
    
    # Calculate coverage of non-matching classes (excluding background/ignore)
    non_matching_classes = set(unique_classes) - matching_classes - {0, 255}  # Exclude background and ignore
    non_matching_pixels = sum(class_counts.get(cls, 0) for cls in non_matching_classes)
    non_matching_coverage = non_matching_pixels / total_pixels
    
    return {
        'total_classes': len(unique_classes),
        'matching_classes': list(present_matching_classes),
        'non_matching_classes': list(non_matching_classes),
        'matching_coverage': matching_coverage,
        'non_matching_coverage': non_matching_coverage,
        'class_counts': class_counts,
        'total_pixels': total_pixels
    }

def filter_samples_by_criteria(samples_analysis, criteria):
    """
    Filter samples based on specified criteria.
    
    Args:
        samples_analysis: dict with sample_name -> analysis results
        criteria: dict with filtering criteria
    
    Returns:
        dict: Filtered samples with their analysis
    """
    filtered_samples = {}
    
    min_matching_coverage = criteria.get('min_matching_coverage', 0.3)
    max_non_matching_coverage = criteria.get('max_non_matching_coverage', 0.7)
    min_matching_classes = criteria.get('min_matching_classes', 3)
    max_total_classes = criteria.get('max_total_classes', 20)
    
    for sample_name, analysis in samples_analysis.items():
        # Check if sample meets criteria
        if (analysis['matching_coverage'] >= min_matching_coverage and
            analysis['non_matching_coverage'] <= max_non_matching_coverage and
            len(analysis['matching_classes']) >= min_matching_classes and
            analysis['total_classes'] <= max_total_classes):
            
            filtered_samples[sample_name] = analysis
    
    return filtered_samples

def save_filtered_samples(filtered_samples, output_path, criteria):
    """Save filtered sample names to text files - one clean, one with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create paths for both files
    clean_output_path = output_path
    detailed_output_path = f"{output_path}_matching_info"
    
    # Sort samples by matching coverage (descending) and then by non-matching coverage (ascending)
    sorted_samples = sorted(
        filtered_samples.items(),
        key=lambda x: (-x[1]['matching_coverage'], x[1]['non_matching_coverage'])
    )
    
    # Save clean file with just sample names
    with open(clean_output_path, 'w') as f:
        for sample_name, analysis in sorted_samples:
            f.write(f"{sample_name}.png\n")
    
    # Save detailed file with metadata and statistics
    with open(detailed_output_path, 'w') as f:
        # Write header with criteria and statistics
        f.write(f"# ADE20K samples filtered for Cityscapes class compatibility\n")
        f.write(f"# Filtering criteria:\n")
        f.write(f"# - min_matching_coverage: {criteria.get('min_matching_coverage', 0.3)}\n")
        f.write(f"# - max_non_matching_coverage: {criteria.get('max_non_matching_coverage', 0.7)}\n")
        f.write(f"# - min_matching_classes: {criteria.get('min_matching_classes', 3)}\n")
        f.write(f"# - max_total_classes: {criteria.get('max_total_classes', 20)}\n")
        f.write(f"# Total samples found: {len(filtered_samples)}\n")
        f.write(f"#\n")
        f.write(f"# Format: sample_name.png (matching_coverage, non_matching_coverage, matching_classes_count)\n")
        f.write(f"#\n")
        
        # Write sample names with statistics as comments
        for sample_name, analysis in sorted_samples:
            f.write(f"{sample_name}.png  # match_cov={analysis['matching_coverage']:.3f}, "
                   f"non_match_cov={analysis['non_matching_coverage']:.3f}, "
                   f"match_classes={len(analysis['matching_classes'])}, "
                   f"total_classes={analysis['total_classes']}\n")
    
    print(f"Saved {len(filtered_samples)} filtered samples to:")
    print(f"  Clean list: {clean_output_path}")
    print(f"  Detailed list: {detailed_output_path}")

def print_statistics(samples_analysis, filtered_samples, exact_matches):
    """Print comprehensive statistics about the analysis."""
    print("\n" + "="*60)
    print("ANALYSIS STATISTICS")
    print("="*60)
    
    print(f"Total samples analyzed: {len(samples_analysis)}")
    print(f"Samples meeting criteria: {len(filtered_samples)}")
    print(f"Filtering success rate: {len(filtered_samples)/len(samples_analysis)*100:.1f}%")
    
    if not filtered_samples:
        print("No samples met the filtering criteria!")
        return
    
    # Statistics for filtered samples
    matching_coverages = [analysis['matching_coverage'] for analysis in filtered_samples.values()]
    non_matching_coverages = [analysis['non_matching_coverage'] for analysis in filtered_samples.values()]
    matching_classes_counts = [len(analysis['matching_classes']) for analysis in filtered_samples.values()]
    
    print("\nFiltered Samples Statistics:")
    print(f"Matching coverage - Mean: {np.mean(matching_coverages):.3f}, "
          f"Std: {np.std(matching_coverages):.3f}, "
          f"Range: [{np.min(matching_coverages):.3f}, {np.max(matching_coverages):.3f}]")
    print(f"Non-matching coverage - Mean: {np.mean(non_matching_coverages):.3f}, "
          f"Std: {np.std(non_matching_coverages):.3f}, "
          f"Range: [{np.min(non_matching_coverages):.3f}, {np.max(non_matching_coverages):.3f}]")
    print(f"Matching classes count - Mean: {np.mean(matching_classes_counts):.1f}, "
          f"Range: [{np.min(matching_classes_counts)}, {np.max(matching_classes_counts)}]")
    
    # Class frequency analysis
    print("\nClass Frequency in Filtered Samples:")
    all_matching_classes = []
    for analysis in filtered_samples.values():
        all_matching_classes.extend(analysis['matching_classes'])
    
    class_frequency = Counter(all_matching_classes)
    for ade_class, count in class_frequency.most_common():
        percentage = count / len(filtered_samples) * 100
        print(f"  ADE20K class {ade_class}: {count} samples ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze ADE20K dataset for Cityscapes class compatibility')
    parser.add_argument('--image_path', type=str, default='/fast/AG_Kainmueller/data/ADEChallengeData2016/images/validation/', 
                        help='Path to ADE20K images directory')
    parser.add_argument('--mask_path', type=str, default='/fast/AG_Kainmueller/data/ADEChallengeData2016/annotations/validation/', 
                        help='Path to ADE20K masks/annotations directory')
    parser.add_argument('--output_path', type=str, default='/fast/AG_Kainmueller/data/GTA_ValUES_splits/ADE20k_id_test',
                        help='Output file path for filtered sample names')
    parser.add_argument('--min_matching_coverage', type=float, default=0.9,
                        help='Minimum coverage of matching classes (default: 0.3)')
    parser.add_argument('--max_non_matching_coverage', type=float, default=0.1,
                        help='Maximum coverage of non-matching classes (default: 0.7)')
    parser.add_argument('--min_matching_classes', type=int, default=3,
                        help='Minimum number of matching classes (default: 3)')
    parser.add_argument('--max_total_classes', type=int, default=20,
                        help='Maximum total number of classes (default: 20)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to analyze (for testing)')
    
    args = parser.parse_args()
    
    # Define the exact matches mapping (ADE20K -> Cityscapes)
    exact_matches = {
        0: 3,   # wall -> wall
        1: 2,   # building -> building
        2: 10,  # sky -> sky
        6: 0,   # road -> road
        11: 1,  # sidewalk -> sidewalk
        12: 11, # person -> person
        20: 13, # car -> car
        32: 4,  # fence -> fence
        80: 15, # bus -> bus
        83: 14, # truck -> truck
        93: 5,  # pole -> pole
        127: 18, # bicycle -> bicycle
        136: 6,  # traffic light -> traffic light
    }
    
    # Load sample names
    print("Loading ADE20K sample names...")
    sample_names = load_ade20k_sample_names(args.image_path)
    
    if args.max_samples:
        sample_names = sample_names[:args.max_samples]
        print(f"Limiting analysis to {len(sample_names)} samples for testing")
    
    print(f"Found {len(sample_names)} samples to analyze")
    
    # Analyze each sample
    print("Analyzing samples...")
    samples_analysis = {}
    mask_path = Path(args.mask_path)
    
    for sample_name in tqdm(sample_names, desc="Processing samples"):
        try:
            mask = load_mask(mask_path, sample_name)
            analysis = analyze_sample_classes(mask, exact_matches)
            samples_analysis[sample_name] = analysis
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
    
    # Filter samples based on criteria
    criteria = {
        'min_matching_coverage': args.min_matching_coverage,
        'max_non_matching_coverage': args.max_non_matching_coverage,
        'min_matching_classes': args.min_matching_classes,
        'max_total_classes': args.max_total_classes
    }
    
    print("Filtering samples...")
    filtered_samples = filter_samples_by_criteria(samples_analysis, criteria)
    
    # Save results
    save_filtered_samples(filtered_samples, args.output_path, criteria)
    
    # Print statistics
    print_statistics(samples_analysis, filtered_samples, exact_matches)
    
    print(f"\nAnalysis complete! Check {args.output_path} for filtered sample names.")

if __name__ == "__main__":
    main()