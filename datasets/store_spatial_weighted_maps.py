import argparse
import logging
import os
import time
import numpy as np
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict, Any

from spatial_decomposition import spatial_decomposition #TODO - later it will be part of the methods of aggrigator and will import from it
from aggrigator.uncertainty_maps import UncertaintyMap
from evaluation.data_utils import load_unc_maps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- Configuration Functions ----
def get_variation_names() -> Dict[str, str]:
    """Get alternative variation names for datasets without clear ID/OOD distinction."""
    return {
        'lizard': 'LizardData',
    }

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute and store uncertainty maps by a certain spatial map',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    ) 
    parser.add_argument('--uq_path', type=str, 
        default='/home/vanessa/Documents/data/uncertainty_arctique_v1-0-corrected_14/',
        help='Path to uncertainty evaluation results'
    )
    parser.add_argument('--task', type=str, default='instance', 
        choices=['fgbg', 'instance', 'semantic'], help='Task type'
    )
    parser.add_argument('--variation', type=str, default='nuclei_intensity', 
        choices=['nuclei_intensity', 'blood_cells', 'malignancy', 'texture'], help='Variation type'
    )
    parser.add_argument('--model_noise', type=int, default=0, help='Model noise level')
    parser.add_argument('--decomp', type=str, default='pu', choices=['pu', 'au', 'eu'], help='Decomposition component')
    parser.add_argument('--uq_methods', 
                        type=str, default='tta,softmax,ensemble,dropout', help='Comma-separated list of UQ methods to evaluate'
    )
    parser.add_argument('--dataset_name', type=str, default='arctique', 
        choices=['arctique', 'lidc', 'lizard'], help='Dataset name'
    )
    parser.add_argument('--image_noise', type=str, default='0_00,1_00', help='Comma-separated list of image noise levels')
    parser.add_argument('--window_size', type=int, default=3, help='Pixel window size for spatial map filter calculation')
    parser.add_argument('--spatial_measure', type=str, default='eds', 
                        choices=['eds', 'moran'], help='Spatial measure type to weigh uncertainty maps'
    )
    parser.add_argument('--n_workers', type=int, default=4, help='Number of parallel workers (default: auto-detect)')
    
    return parser.parse_args()

# ---- Utility Functions ----
def create_map_filename(task: str, model_noise: int, variation: str, data_noise: str, 
                       uq_method: str, decomp: str, spatial_level: str, spatial_measure: str) -> str:
    """Create filename following the established nomenclature."""
    return f"{task}_noise_{model_noise}_{variation}_{data_noise}_{uq_method}_{decomp}_{spatial_level}_{spatial_measure}"

def save_spatial_maps(high_maps: List[np.ndarray], low_maps: List[np.ndarray], 
                     output_path: Path, task: str, model_noise: int, variation: str, 
                     data_noise: str, uq_method: str, decomp: str, spatial_measure: str) -> None:
    """Save high and low spatial maps to the output directory."""
    
    # Create filenames
    high_filename = create_map_filename(task, model_noise, variation, data_noise, 
                                       uq_method, decomp, "high", spatial_measure)
    low_filename = create_map_filename(task, model_noise, variation, data_noise, 
                                      uq_method, decomp, "low", spatial_measure)
    
    # Save high maps
    high_path = output_path / f"{high_filename}.npy"
    if high_maps:
        np.save(high_path, high_maps)
        logger.info(f"Saved high spatial maps to: {high_path}")
    
    # Save low maps
    low_path = output_path / f"{low_filename}.npy"
    if low_maps:
        np.save(low_path, low_maps)
        logger.info(f"Saved low spatial maps to: {low_path}")

# ---- Faster Computation ----

def process_single_map(unc_map, window_size, spatial_measure):
    """Process a single uncertainty map - designed for parallel execution."""
    try:
        return spatial_decomposition(unc_map, window_size=window_size, spatial_measure=spatial_measure)
    except Exception as e:
        print(f"Error processing map {unc_map.name}: {e}")
        return None

def compute_spatial_maps_parallel(
    uncertainty_maps: List[UncertaintyMap], 
    window_size: int,
    spatial_measure: str = "eds",
    n_workers: int = None
) -> Tuple[Any, Any]:
    """
    Compute spatial decomposition maps in parallel.
    
    Args:
        uncertainty_maps: List of UncertaintyMap objects
        window_size: Window size for spatial decomposition
        spatial_measure: Spatial measure type
        n_workers: Number of parallel workers (None = auto-detect)
        
    Returns:
        Tuple of (high_measure_map, low_measure_map)
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(uncertainty_maps))
    
    logger.info(f"Computing spatial maps with {n_workers} workers")
    logger.info(f"Parameters: window_size={window_size}, measure={spatial_measure}")
    
    # Create partial function with fixed parameters
    process_func = partial(process_single_map, 
                          window_size=window_size, 
                          spatial_measure=spatial_measure)
    
    start_time = time.time()
    
    try:
        # Use multiprocessing Pool
        with mp.Pool(n_workers) as pool:
            decomposition_results = pool.map(process_func, uncertainty_maps)
        
        # Filter out None results (failed processing)
        decomposition_results = [r for r in decomposition_results if r is not None]
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed:.2f} seconds")
        logger.info(f"Successfully processed {len(decomposition_results)}/{len(uncertainty_maps)} maps")
        
        if decomposition_results:
            # Transpose to separate the different components
            high_measure_maps, low_measure_maps, local_weights, spatial_mass_ratios = zip(*decomposition_results)
            return list(hmp.array for hmp in high_measure_maps), list(lmp.array for lmp in low_measure_maps)
        else:
            logger.warning("No decomposition results generated")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in parallel spatial decomposition: {e}")
        raise

# Update your main processing function
def process_uncertainty_maps_fast(
    uq_maps_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    uq_method: str,
    noise_level: str
) -> Tuple[Any, Any]:
    """Fast processing with parallel execution."""
    logger.info(f"Processing UQ method: {uq_method}, noise level: {noise_level}")
    
    # Load uncertainty maps
    unc_maps = load_unc_maps(
        uq_maps_path, args.task, args.model_noise, args.variation, 
        noise_level, uq_method, args.decomp, args.dataset_name, False
    )
    
    if unc_maps is None or len(unc_maps) == 0:
        logger.warning(f"No uncertainty maps loaded for {uq_method}, {noise_level}")
        return None, None
    
    # Create UncertaintyMap objects
    uncertainty_maps = [
        UncertaintyMap(array=array, mask=None, name=f"{uq_method}_{noise_level}_{i}") 
        for i, array in enumerate(unc_maps)
    ]
    
    logger.info(f"Created {len(uncertainty_maps)} uncertainty map objects")
    
    # Use parallel processing
    high_measure_map, low_measure_map = compute_spatial_maps_parallel(
        uncertainty_maps, 
        args.window_size,
        args.spatial_measure,
        args.n_workers
    )
    
    # Save the spatial maps
    if high_measure_map is not None and low_measure_map is not None:
        save_spatial_maps(
            high_measure_map, low_measure_map, output_path,
            args.task, args.model_noise, args.variation, noise_level,
            uq_method, args.decomp, args.spatial_measure
        )
    
    return high_measure_map, low_measure_map

# ---- Debugging Functions ----

def debug_data_sizes(uq_maps_path, args, uq_method, noise_level):
    """Debug function to compare data sizes between datasets."""
    logger.info("=== DATA SIZE DEBUG ===")
    
    # Load uncertainty maps
    unc_maps = load_unc_maps(
        uq_maps_path, args.task, args.model_noise, args.variation, 
        noise_level, uq_method, args.decomp, args.dataset_name, False
    )
    
    if unc_maps is None or len(unc_maps) == 0:
        logger.warning("No maps loaded!")
        return
    
    logger.info(f"Number of maps: {len(unc_maps)}")
    
    # Analyze each map
    total_size = 0
    for i, unc_map in enumerate(unc_maps[:10]):  # Check first 5 maps
        shape = unc_map.shape
        dtype = unc_map.dtype
        size_mb = unc_map.nbytes / 1024**2
        total_size += size_mb
        
        logger.info(f"Map {i+1}: Shape={shape}, dtype={dtype}, size={size_mb:.2f}MB")
        logger.info(f"  Min/Max values: {unc_map.min():.6f}/{unc_map.max():.6f}")
        logger.info(f"  Has NaN: {np.any(np.isnan(unc_map))}")
        logger.info(f"  Has Inf: {np.any(np.isinf(unc_map))}")
        logger.info(f"  Unique values: {len(np.unique(unc_map))}")
    
    # Estimate total data size
    avg_size = total_size / min(5, len(unc_maps))
    estimated_total = avg_size * len(unc_maps)
    logger.info(f"Average map size: {avg_size:.2f}MB")
    logger.info(f"Estimated total data size: {estimated_total:.2f}MB ({estimated_total/1024:.2f}GB)")
    
    # Memory requirements for spatial decomposition
    # Spatial decomposition typically creates multiple copies of data
    estimated_memory_needed = estimated_total * 4  # Conservative estimate
    logger.info(f"Estimated memory needed for spatial decomposition: {estimated_memory_needed:.2f}MB ({estimated_memory_needed/1024:.2f}GB)")
    
    return unc_maps

# Add this to your process function
def process_uncertainty_maps_with_debug(
    uq_maps_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    uq_method: str,
    noise_level: str
) -> Tuple[Any, Any]:
    """Process with detailed debugging."""
    logger.info(f"DEBUG: Processing {uq_method}, {noise_level}")
    
    # Debug data sizes first
    unc_maps = debug_data_sizes(uq_maps_path, args, uq_method, noise_level)
    
    if unc_maps is None or len(unc_maps) == 0:
        return None, None
    
    # Create UncertaintyMap objects
    uncertainty_maps = [
        UncertaintyMap(array=array, mask=None, name=f"{uq_method}_{noise_level}_{i}") 
        for i, array in enumerate(unc_maps)
    ]
    
    logger.info(f"Created {len(uncertainty_maps)} uncertainty map objects")
    
    # Test with just ONE map first
    logger.info("=== TESTING WITH SINGLE MAP ===")
    test_map = uncertainty_maps[0]
    logger.info(f"Testing spatial decomposition on single map: {test_map.array.shape}")
    
    try:
        start_time = time.time()
        result = spatial_decomposition(
            test_map, 
            window_size=args.window_size,
            spatial_measure=args.spatial_measure
        )
        elapsed = time.time() - start_time
        logger.info(f"Single map processing took: {elapsed:.2f} seconds")
        
        # If single map works, estimate total time
        estimated_total_time = elapsed * len(uncertainty_maps)
        logger.info(f"Estimated total processing time: {estimated_total_time:.2f} seconds ({estimated_total_time/60:.2f} minutes)")
        
        if estimated_total_time > 3600:  # More than 1 hour
            logger.warning(f"Processing might take a very long time: {estimated_total_time/3600:.2f} hours")
            logger.info("Consider processing in smaller batches or with different parameters")
        
    except Exception as e:
        logger.error(f"Single map test failed: {e}")
        return None, None
    
    # If single map test passed, proceed with all maps
    logger.info("=== PROCESSING ALL MAPS ===")
    high_measure_map, low_measure_map = compute_spatial_maps(
        uncertainty_maps, 
        args.window_size,
        args.spatial_measure
    )
    
    # Save the spatial maps
    if high_measure_map is not None and low_measure_map is not None:
        save_spatial_maps(
            high_measure_map, low_measure_map, output_path,
            args.task, args.model_noise, args.variation, noise_level,
            uq_method, args.decomp, args.spatial_measure
        )
    
    return high_measure_map, low_measure_map

# ---- Computation Functions ----

def compute_spatial_maps(
    uncertainty_maps: List[UncertaintyMap], 
    window_size: int,
    spatial_measure: str = "eds"
) -> Tuple[Any, Any]:
    """
    Compute spatial decomposition maps for uncertainty maps.
    
    Args:
        uncertainty_maps: List of UncertaintyMap objects
        window_size: Window size for spatial decomposition
        spatial_measure: Spatial measure type
        
    Returns:
        Tuple of (high_measure_map, low_measure_map)
    """
    logger.info(f"Computing spatial maps with window_size={window_size}, measure={spatial_measure}")
    
    try:        
        # Apply spatial decomposition to each uncertainty map
        decomposition_results = [
            spatial_decomposition(
                unc_map, 
                window_size=window_size, 
                spatial_measure=spatial_measure
            )
            for unc_map in uncertainty_maps
        ]
        
        # Unpack results (assuming spatial_decomposition returns 4 values)
        if decomposition_results:
            # Transpose to separate the different components
            high_measure_maps, low_measure_maps, local_weights, spatial_mass_ratios = zip(*decomposition_results)
            high_measure_maps, low_measure_maps = list(high_measure_maps), list(low_measure_maps)
            return list(hmp.array for hmp in high_measure_maps), list(lmp.array for lmp in low_measure_maps)
        else:
            logger.warning("No decomposition results generated")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in spatial decomposition: {e}")
        raise

def process_uncertainty_maps(
    uq_maps_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    uq_method: str,
    noise_level: str
) -> Tuple[Any, Any]:
    """Process uncertainty maps for a specific UQ method and noise level."""
    logger.info(f"Processing UQ method: {uq_method}, noise level: {noise_level}")
    
    # Load uncertainty maps
    unc_maps = load_unc_maps(
        uq_maps_path, 
        args.task, 
        args.model_noise, 
        args.variation, 
        noise_level,
        uq_method, 
        args.decomp, 
        args.dataset_name, 
        False
    )
    
    if unc_maps is None or len(unc_maps) == 0:
        logger.warning(f"No uncertainty maps loaded for {uq_method}, {noise_level}")
        return None, None
    
    # Create UncertaintyMap objects
    uncertainty_maps = [
        UncertaintyMap(array=array, mask=None, name=f"{uq_method}_{noise_level}_{i}") 
        for i, array in enumerate(unc_maps)
    ]
    
    logger.info(f"Created {len(uncertainty_maps)} uncertainty map objects")
    
    # Compute spatial maps
    high_measure_map, low_measure_map = compute_spatial_maps(
        uncertainty_maps, 
        args.window_size,
        args.spatial_measure
    )
    
    # Save the spatial maps
    if high_measure_map is not None and low_measure_map is not None:
        save_spatial_maps(
            high_measure_map, low_measure_map, output_path,
            args.task, args.model_noise, args.variation, noise_level,
            uq_method, args.decomp, args.spatial_measure
        )
    
    return high_measure_map, low_measure_map

def main():
    """Main execution function."""
    # Parse and validate arguments
    args = parse_arguments()
    
    # Convert string path to Path object
    args.uq_path = Path(args.uq_path)
        
    # Parse lists from comma-separated strings
    uq_methods = [method.strip() for method in args.uq_methods.split(',')]
    image_noise_levels = [noise.strip() for noise in args.image_noise.split(',')]
        
    logger.info(f"Processing {len(uq_methods)} UQ methods and {len(image_noise_levels)} noise levels")
        
    # Handle variation name mapping
    if not args.variation:
        alt_names = get_variation_names()
        if args.dataset_name in alt_names:
            args.variation = alt_names[args.dataset_name]
            logger.info(f"Using variation name: {args.variation}")
        else:
            raise ValueError(f"No variation specified and no default for dataset: {args.dataset_name}")
        
    # Set paths
    uq_maps_path = args.uq_path / "UQ_maps"
    output_path = args.uq_path / "UQ_spatial"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Created output directory: {output_path}")
        
    if not uq_maps_path.exists():
        raise FileNotFoundError(f"UQ maps path does not exist: {uq_maps_path}")
        
    # Process each combination of UQ method and noise level
    results = {}
        
    for uq_method in uq_methods:
        results[uq_method] = {}
        
        for noise_level in image_noise_levels:
            logger.info(f"Processing: {uq_method} with noise level {noise_level}")
            
            try:
                high_map, low_map = process_uncertainty_maps_fast(
                    uq_maps_path, output_path, args, uq_method, noise_level
                )
                                
                results[uq_method][noise_level] = {
                    'high_measure_map': high_map,
                    'low_measure_map': low_map
                }
                
                logger.info(f"Successfully processed {uq_method} - {noise_level}")
                
            except Exception as e:
                logger.error(f"Failed to process {uq_method} - {noise_level}: {e}")
                continue
    
    logger.info("Processing completed successfully")
    logger.info(f"All spatial maps saved to: {output_path}")
    
    return results
    
if __name__ == "__main__":
    main()