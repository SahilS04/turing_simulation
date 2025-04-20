"""
pattern_quantification.py

This module provides tools for quantifying and analyzing Turing patterns from
simulation data. It includes functions for calculating pattern metrics, 
performing spatial frequency analysis, and classifying pattern types.

Functions:
- load_pattern_data: Load pattern data from CSV files
- calculate_pattern_metrics: Calculate basic metrics for a pattern
- calculate_power_spectrum: Calculate the power spectrum of a pattern
- classify_pattern_type: Classify pattern as spots, stripes, or mixed
- segment_pattern: Segment a pattern into distinct features
- calculate_feature_metrics: Calculate metrics for segmented features
- visualize_pattern: Visualize pattern data with various metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import filters, measure, segmentation, feature, morphology
import os
import glob
from matplotlib.colors import LinearSegmentedColormap

def load_pattern_data(file_path):
    """
    Load pattern data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing pattern data
        
    Returns:
    --------
    numpy.ndarray
        2D array containing pattern data
    """
    try:
        # Load data from CSV file
        data = np.loadtxt(file_path, delimiter=',')
        return data
    except Exception as e:
        print(f"Error loading pattern data: {e}")
        return None

def calculate_pattern_metrics(pattern_data):
    """
    Calculate basic metrics for a pattern.
    
    Parameters:
    -----------
    pattern_data : numpy.ndarray
        2D array containing pattern data
        
    Returns:
    --------
    dict
        Dictionary containing pattern metrics
    """
    # Check if pattern_data is valid
    if pattern_data is None or pattern_data.size == 0:
        return {}
    
    # Calculate basic statistics
    mean = np.mean(pattern_data)
    std_dev = np.std(pattern_data)
    min_val = np.min(pattern_data)
    max_val = np.max(pattern_data)
    
    # Calculate contrast
    if max_val + min_val != 0:
        contrast = (max_val - min_val) / (max_val + min_val)
    else:
        contrast = 0
    
    # Calculate gradient magnitude (edge strength)
    gradient_x = ndimage.sobel(pattern_data, axis=0)
    gradient_y = ndimage.sobel(pattern_data, axis=1)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    mean_gradient = np.mean(gradient_magnitude)
    
    # Calculate threshold-based metrics
    threshold = filters.threshold_otsu(pattern_data)
    binary = pattern_data > threshold
    
    # Count features (connected components)
    labeled, num_features = ndimage.label(binary)
    
    # Calculate average feature size
    if num_features > 0:
        avg_feature_size = np.sum(binary) / num_features
    else:
        avg_feature_size = 0
    
    # Calculate feature density
    feature_density = num_features / pattern_data.size
    
    # Return metrics as dictionary
    metrics = {
        'Mean': mean,
        'StdDev': std_dev,
        'Min': min_val,
        'Max': max_val,
        'Contrast': contrast,
        'MeanGradient': mean_gradient,
        'NumFeatures': num_features,
        'AvgFeatureSize': avg_feature_size,
        'FeatureDensity': feature_density
    }
    
    return metrics

def calculate_power_spectrum(pattern_data):
    """
    Calculate the power spectrum of a pattern using FFT.
    
    Parameters:
    -----------
    pattern_data : numpy.ndarray
        2D array containing pattern data
        
    Returns:
    --------
    tuple
        (power_spectrum, radial_profile, dominant_wavelength)
        - power_spectrum: 2D array containing the power spectrum
        - radial_profile: 1D array containing the radially averaged power spectrum
        - dominant_wavelength: Dominant wavelength in the pattern
    """
    # Check if pattern_data is valid
    if pattern_data is None or pattern_data.size == 0:
        return None, None, None
    
    # Get dimensions
    ny, nx = pattern_data.shape
    
    # Subtract mean to remove DC component
    pattern_centered = pattern_data - np.mean(pattern_data)
    
    # Apply window function to reduce edge effects
    window = np.outer(np.hanning(ny), np.hanning(nx))
    pattern_windowed = pattern_centered * window
    
    # Compute 2D FFT
    fft2 = np.fft.fft2(pattern_windowed)
    
    # Shift zero frequency to center
    fft2_shifted = np.fft.fftshift(fft2)
    
    # Calculate power spectrum (magnitude squared)
    power_spectrum = np.abs(fft2_shifted)**2
    
    # Calculate radial profile (average power at each frequency)
    y, x = np.indices((ny, nx))
    center = (ny // 2, nx // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Convert r to integer bins
    r_int = r.astype(int)
    
    # Create radial profile
    radial_profile = np.zeros(min(nx, ny) // 2)
    for i in range(len(radial_profile)):
        mask = r_int == i
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    # Find dominant frequency (excluding DC component)
    # We start from index 1 to skip the DC component (index 0)
    peaks, _ = find_peaks(radial_profile[1:], height=0)
    
    # Add 1 to peaks to account for the skipped DC component
    peaks = peaks + 1
    
    # If no peaks found, return 0 as dominant wavelength
    if len(peaks) == 0:
        dominant_wavelength = 0
    else:
        # Get the peak with maximum power
        dominant_freq_idx = peaks[np.argmax(radial_profile[peaks])]
        
        # Convert to wavelength (grid units)
        # Wavelength = grid_size / frequency
        if dominant_freq_idx > 0:
            dominant_wavelength = min(nx, ny) / dominant_freq_idx
        else:
            dominant_wavelength = 0
    
    return power_spectrum, radial_profile, dominant_wavelength

def classify_pattern_type(pattern_data, threshold=None):
    """
    Classify pattern as spots, stripes, or mixed.
    
    Parameters:
    -----------
    pattern_data : numpy.ndarray
        2D array containing pattern data
    threshold : float, optional
        Threshold for binarization. If None, Otsu's method is used.
        
    Returns:
    --------
    str
        Pattern type: 'spots', 'stripes', 'mixed', or 'none'
    """
    # Check if pattern_data is valid
    if pattern_data is None or pattern_data.size == 0:
        return 'none'
    
    # Normalize pattern data to [0, 1]
    pattern_norm = (pattern_data - np.min(pattern_data)) / (np.max(pattern_data) - np.min(pattern_data))
    
    # Determine threshold if not provided
    if threshold is None:
        threshold = filters.threshold_otsu(pattern_norm)
    
    # Binarize the pattern
    binary = pattern_norm > threshold
    
    # Label connected components
    labeled, num_features = ndimage.label(binary)
    
    if num_features == 0:
        return 'none'
    
    # Calculate properties of each region
    regions = measure.regionprops(labeled)
    
    # Calculate eccentricity and area for each region
    eccentricities = [region.eccentricity for region in regions]
    areas = [region.area for region in regions]
    
    # Calculate metrics
    mean_eccentricity = np.mean(eccentricities)
    mean_area = np.mean(areas)
    
    # Calculate perimeter-to-area ratio for each region
    perimeter_area_ratios = []
    for region in regions:
        perimeter = region.perimeter
        area = region.area
        if area > 0:
            perimeter_area_ratios.append(perimeter / np.sqrt(area))
    
    mean_perimeter_area_ratio = np.mean(perimeter_area_ratios) if perimeter_area_ratios else 0
    
    # Classify based on eccentricity and perimeter-to-area ratio
    # Spots have low eccentricity (more circular) and low perimeter-to-area ratio
    # Stripes have high eccentricity (more elongated) and high perimeter-to-area ratio
    if mean_eccentricity < 0.5 and mean_perimeter_area_ratio < 4.0:
        return 'spots'
    elif mean_eccentricity > 0.8 or mean_perimeter_area_ratio > 5.0:
        return 'stripes'
    else:
        return 'mixed'

def segment_pattern(pattern_data, threshold=None):
    """
    Segment a pattern into distinct features.
    
    Parameters:
    -----------
    pattern_data : numpy.ndarray
        2D array containing pattern data
    threshold : float, optional
        Threshold for binarization. If None, Otsu's method is used.
        
    Returns:
    --------
    numpy.ndarray
        Labeled array where each feature has a unique integer label
    """
    # Check if pattern_data is valid
    if pattern_data is None or pattern_data.size == 0:
        return None
    
    # Normalize pattern data to [0, 1]
    pattern_norm = (pattern_data - np.min(pattern_data)) / (np.max(pattern_data) - np.min(pattern_data))
    
    # Determine threshold if not provided
    if threshold is None:
        threshold = filters.threshold_otsu(pattern_norm)
    
    # Binarize the pattern
    binary = pattern_norm > threshold
    
    # Apply morphological operations to clean up the binary image
    binary = morphology.remove_small_holes(binary, area_threshold=5)
    binary = morphology.remove_small_objects(binary, min_size=5)
    
    # Label connected components
    labeled, _ = ndimage.label(binary)
    
    return labeled

def calculate_feature_metrics(labeled_pattern):
    """
    Calculate metrics for segmented features.
    
    Parameters:
    -----------
    labeled_pattern : numpy.ndarray
        Labeled array where each feature has a unique integer label
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metrics for each feature
    """
    # Check if labeled_pattern is valid
    if labeled_pattern is None or labeled_pattern.size == 0:
        return pd.DataFrame()
    
    # Calculate region properties
    regions = measure.regionprops(labeled_pattern)
    
    # Extract metrics for each region
    metrics = []
    for region in regions:
        metrics.append({
            'Label': region.label,
            'Area': region.area,
            'Perimeter': region.perimeter,
            'Eccentricity': region.eccentricity,
            'MajorAxisLength': region.major_axis_length,
            'MinorAxisLength': region.minor_axis_length,
            'Orientation': region.orientation,
            'Circularity': 4 * np.pi * region.area / (region.perimeter**2) if region.perimeter > 0 else 0,
            'CentroidX': region.centroid[1],
            'CentroidY': region.centroid[0]
        })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df

def visualize_pattern(pattern_data, output_dir=None, filename_prefix='pattern'):
    """
    Visualize pattern data with various metrics.
    
    Parameters:
    -----------
    pattern_data : numpy.ndarray
        2D array containing pattern data
    output_dir : str, optional
        Directory to save visualization files. If None, figures are not saved.
    filename_prefix : str, optional
        Prefix for saved filenames
        
    Returns:
    --------
    dict
        Dictionary containing figure objects
    """
    # Check if pattern_data is valid
    if pattern_data is None or pattern_data.size == 0:
        return {}
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store figures
    figures = {}
    
    # 1. Visualize original pattern
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im1 = ax1.imshow(pattern_data, cmap='viridis')
    plt.colorbar(im1, ax=ax1, label='Concentration')
    ax1.set_title('Original Pattern')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    figures['original'] = fig1
    
    if output_dir is not None:
        fig1.savefig(os.path.join(output_dir, f'{filename_prefix}_original.png'), dpi=300)
    
    # 2. Visualize power spectrum
    power_spectrum, radial_profile, dominant_wavelength = calculate_power_spectrum(pattern_data)
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im2 = ax2.imshow(np.log10(power_spectrum + 1), cmap='inferno')
    plt.colorbar(im2, ax=ax2, label='Log Power')
    ax2.set_title(f'Power Spectrum (Dominant Wavelength: {dominant_wavelength:.2f})')
    ax2.set_xlabel('Frequency X')
    ax2.set_ylabel('Frequency Y')
    figures['power_spectrum'] = fig2
    
    if output_dir is not None:
        fig2.savefig(os.path.join(output_dir, f'{filename_prefix}_power_spectrum.png'), dpi=300)
    
    # 3. Visualize radial profile
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(radial_profile)
    ax3.set_title('Radial Power Spectrum')
    ax3.set_xlabel('Frequency (radial)')
    ax3.set_ylabel('Power')
    ax3.grid(True)
    
    # Find and mark peaks
    peaks, _ = find_peaks(radial_profile[1:], height=0)
    peaks = peaks + 1  # Adjust for skipped DC component
    
    if len(peaks) > 0:
        ax3.plot(peaks, radial_profile[peaks], 'ro')
        
        # Mark dominant peak
        dominant_peak = peaks[np.argmax(radial_profile[peaks])]
        ax3.plot(dominant_peak, radial_profile[dominant_peak], 'go', markersize=10)
        ax3.annotate(f'Dominant: {dominant_peak}', 
                    xy=(dominant_peak, radial_profile[dominant_peak]),
                    xytext=(dominant_peak + 5, radial_profile[dominant_peak]),
                    arrowprops=dict(facecolor='green', shrink=0.05))
    
    figures['radial_profile'] = fig3
    
    if output_dir is not None:
        fig3.savefig(os.path.join(output_dir, f'{filename_prefix}_radial_profile.png'), dpi=300)
    
    # 4. Visualize segmentation
    labeled = segment_pattern(pattern_data)
    
    # Create a colormap for visualization
    # We use a modified tab20 colormap with black for background (label 0)
    cmap = plt.cm.get_cmap('tab20', np.max(labeled) + 1)
    cmap_colors = cmap(np.arange(np.max(labeled) + 1))
    cmap_colors[0] = [0, 0, 0, 1]  # Set background to black
    custom_cmap = LinearSegmentedColormap.from_list('custom_tab20', cmap_colors)
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    im4 = ax4.imshow(labeled, cmap=custom_cmap)
    plt.colorbar(im4, ax=ax4, label='Feature Label')
    ax4.set_title(f'Pattern Segmentation (Features: {np.max(labeled)})')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    figures['segmentation'] = fig4
    
    if output_dir is not None:
        fig4.savefig(os.path.join(output_dir, f'{filename_prefix}_segmentation.png'), dpi=300)
    
    # 5. Visualize feature metrics
    metrics_df = calculate_feature_metrics(labeled)
    
    if not metrics_df.empty:
        # Create scatter plot of feature properties
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        scatter = ax5.scatter(metrics_df['Area'], metrics_df['Eccentricity'], 
                             c=metrics_df['Circularity'], cmap='viridis', 
                             s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax5, label='Circularity')
        ax5.set_title('Feature Properties')
        ax5.set_xlabel('Area')
        ax5.set_ylabel('Eccentricity')
        ax5.grid(True)
        figures['feature_metrics'] = fig5
        
        if output_dir is not None:
            fig5.savefig(os.path.join(output_dir, f'{filename_prefix}_feature_metrics.png'), dpi=300)
    
    # 6. Visualize gradient magnitude (edge strength)
    gradient_x = ndimage.sobel(pattern_data, axis=0)
    gradient_y = ndimage.sobel(pattern_data, axis=1)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    im6 = ax6.imshow(gradient_magnitude, cmap='magma')
    plt.colorbar(im6, ax=ax6, label='Gradient Magnitude')
    ax6.set_title('Edge Strength')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    figures['gradient'] = fig6
    
    if output_dir is not None:
        fig6.savefig(os.path.join(output_dir, f'{filename_prefix}_gradient.png'), dpi=300)
    
    # Close all figures to free memory
    for fig in figures.values():
        plt.close(fig)
    
    return figures

def analyze_pattern_evolution(pattern_dir, pattern_prefix='A_t', output_dir=None):
    """
    Analyze the evolution of patterns over time.
    
    Parameters:
    -----------
    pattern_dir : str
        Directory containing pattern CSV files
    pattern_prefix : str, optional
        Prefix for pattern filenames
    output_dir : str, optional
        Directory to save analysis results. If None, results are not saved.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metrics for each time step
    """
    # Check if pattern_dir exists
    if not os.path.isdir(pattern_dir):
        print(f"Error: Directory {pattern_dir} does not exist")
        return pd.DataFrame()
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all pattern files
    pattern_files = sorted(glob.glob(os.path.join(pattern_dir, f'{pattern_prefix}*.csv')))
    
    if len(pattern_files) == 0:
        print(f"Error: No pattern files found in {pattern_dir} with prefix {pattern_prefix}")
        return pd.DataFrame()
    
    # Initialize list to store metrics
    metrics_list = []
    
    # Process each pattern file
    for file_path in pattern_files:
        # Extract time step from filename
        filename = os.path.basename(file_path)
        time_step = int(filename.replace(pattern_prefix, '').replace('.csv', ''))
        
        # Load pattern data
        pattern_data = load_pattern_data(file_path)
        
        if pattern_data is None:
            continue
        
        # Calculate metrics
        metrics = calculate_pattern_metrics(pattern_data)
        
        # Calculate power spectrum
        _, _, dominant_wavelength = calculate_power_spectrum(pattern_data)
        
        # Classify pattern type
        pattern_type = classify_pattern_type(pattern_data)
        
        # Add time step and additional metrics
        metrics['TimeStep'] = time_step
        metrics['DominantWavelength'] = dominant_wavelength
        metrics['PatternType'] = pattern_type
        
        # Add to list
        metrics_list.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by time step
    metrics_df = metrics_df.sort_values('TimeStep')
    
    # Save to CSV if output_dir is provided
    if output_dir is not None and not metrics_df.empty:
        metrics_df.to_csv(os.path.join(output_dir, 'pattern_evolution.csv'), index=False)
        
        # Create plots of metrics over time
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot standard deviation over time
        axes[0].plot(metrics_df['TimeStep'], metrics_df['StdDev'], 'b-', linewidth=2)
        axes[0].set_ylabel('Standard Deviation')
        axes[0].set_title('Pattern Evolution')
        axes[0].grid(True)
        
        # Plot dominant wavelength over time
        axes[1].plot(metrics_df['TimeStep'], metrics_df['DominantWavelength'], 'g-', linewidth=2)
        axes[1].set_ylabel('Dominant Wavelength')
        axes[1].grid(True)
        
        # Plot number of features over time
        axes[2].plot(metrics_df['TimeStep'], metrics_df['NumFeatures'], 'r-', linewidth=2)
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Number of Features')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pattern_evolution.png'), dpi=300)
        plt.close(fig)
    
    return metrics_df

def compare_patterns(pattern_files, labels=None, output_dir=None):
    """
    Compare multiple patterns and visualize differences.
    
    Parameters:
    -----------
    pattern_files : list
        List of file paths to pattern CSV files
    labels : list, optional
        List of labels for each pattern. If None, filenames are used.
    output_dir : str, optional
        Directory to save comparison results. If None, results are not saved.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metrics for each pattern
    """
    # Check if pattern_files is valid
    if not pattern_files:
        print("Error: No pattern files provided")
        return pd.DataFrame()
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Use filenames as labels if not provided
    if labels is None:
        labels = [os.path.basename(file_path) for file_path in pattern_files]
    
    # Initialize list to store metrics
    metrics_list = []
    
    # Process each pattern file
    for i, file_path in enumerate(pattern_files):
        # Load pattern data
        pattern_data = load_pattern_data(file_path)
        
        if pattern_data is None:
            continue
        
        # Calculate metrics
        metrics = calculate_pattern_metrics(pattern_data)
        
        # Calculate power spectrum
        _, _, dominant_wavelength = calculate_power_spectrum(pattern_data)
        
        # Classify pattern type
        pattern_type = classify_pattern_type(pattern_data)
        
        # Add label and additional metrics
        metrics['Label'] = labels[i]
        metrics['DominantWavelength'] = dominant_wavelength
        metrics['PatternType'] = pattern_type
        
        # Add to list
        metrics_list.append(metrics)
        
        # Visualize pattern if output_dir is provided
        if output_dir is not None:
            visualize_pattern(pattern_data, output_dir, f'pattern_{i}')
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Save to CSV if output_dir is provided
    if output_dir is not None and not metrics_df.empty:
        metrics_df.to_csv(os.path.join(output_dir, 'pattern_comparison.csv'), index=False)
        
        # Create bar plots for key metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot standard deviation
        axes[0, 0].bar(metrics_df['Label'], metrics_df['StdDev'])
        axes[0, 0].set_title('Standard Deviation')
        axes[0, 0].set_xticklabels(metrics_df['Label'], rotation=45, ha='right')
        
        # Plot dominant wavelength
        axes[0, 1].bar(metrics_df['Label'], metrics_df['DominantWavelength'])
        axes[0, 1].set_title('Dominant Wavelength')
        axes[0, 1].set_xticklabels(metrics_df['Label'], rotation=45, ha='right')
        
        # Plot number of features
        axes[1, 0].bar(metrics_df['Label'], metrics_df['NumFeatures'])
        axes[1, 0].set_title('Number of Features')
        axes[1, 0].set_xticklabels(metrics_df['Label'], rotation=45, ha='right')
        
        # Plot feature density
        axes[1, 1].bar(metrics_df['Label'], metrics_df['FeatureDensity'])
        axes[1, 1].set_title('Feature Density')
        axes[1, 1].set_xticklabels(metrics_df['Label'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pattern_comparison.png'), dpi=300)
        plt.close(fig)
    
    return metrics_df

def analyze_parameter_sweep(sweep_dir, pattern_prefix='A_final', output_dir=None):
    """
    Analyze patterns from a parameter sweep.
    
    Parameters:
    -----------
    sweep_dir : str
        Directory containing subdirectories with pattern data for each parameter value
    pattern_prefix : str, optional
        Prefix for pattern filenames
    output_dir : str, optional
        Directory to save analysis results. If None, results are not saved.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing metrics for each parameter value
    """
    # Check if sweep_dir exists
    if not os.path.isdir(sweep_dir):
        print(f"Error: Directory {sweep_dir} does not exist")
        return pd.DataFrame()
    
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all subdirectories
    subdirs = [d for d in os.listdir(sweep_dir) if os.path.isdir(os.path.join(sweep_dir, d))]
    
    if len(subdirs) == 0:
        print(f"Error: No subdirectories found in {sweep_dir}")
        return pd.DataFrame()
    
    # Initialize list to store metrics
    metrics_list = []
    
    # Process each subdirectory
    for subdir in subdirs:
        # Extract parameter value from directory name
        # Assuming directory name format is "DA_X.X"
        param_value = float(subdir.split('_')[1]) if '_' in subdir else 0.0
        
        # Find pattern file
        pattern_file = os.path.join(sweep_dir, subdir, f'{pattern_prefix}.csv')
        
        if not os.path.isfile(pattern_file):
            print(f"Warning: Pattern file not found in {os.path.join(sweep_dir, subdir)}")
            continue
        
        # Load pattern data
        pattern_data = load_pattern_data(pattern_file)
        
        if pattern_data is None:
            continue
        
        # Calculate metrics
        metrics = calculate_pattern_metrics(pattern_data)
        
        # Calculate power spectrum
        _, _, dominant_wavelength = calculate_power_spectrum(pattern_data)
        
        # Classify pattern type
        pattern_type = classify_pattern_type(pattern_data)
        
        # Add parameter value and additional metrics
        metrics['ParameterValue'] = param_value
        metrics['DominantWavelength'] = dominant_wavelength
        metrics['PatternType'] = pattern_type
        
        # Try to load metadata file if it exists
        metadata_file = os.path.join(sweep_dir, subdir, 'metadata.txt')
        if os.path.isfile(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=')
                            key = key.strip()
                            value = value.strip()
                            
                            # Add metadata to metrics
                            if key in ['alpha', 'beta', 'DA', 'DB', 'D_ratio']:
                                try:
                                    metrics[key] = float(value)
                                except ValueError:
                                    metrics[key] = value
            except Exception as e:
                print(f"Error reading metadata file: {e}")
        
        # Add to list
        metrics_list.append(metrics)
        
        # Visualize pattern if output_dir is provided
        if output_dir is not None:
            pattern_output_dir = os.path.join(output_dir, subdir)
            visualize_pattern(pattern_data, pattern_output_dir, 'pattern')
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by parameter value
    if 'ParameterValue' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('ParameterValue')
    
    # Save to CSV if output_dir is provided
    if output_dir is not None and not metrics_df.empty:
        metrics_df.to_csv(os.path.join(output_dir, 'parameter_sweep_analysis.csv'), index=False)
        
        # Create plots of metrics vs parameter value
        if 'ParameterValue' in metrics_df.columns:
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Plot standard deviation vs parameter value
            axes[0].plot(metrics_df['ParameterValue'], metrics_df['StdDev'], 'bo-', linewidth=2)
            axes[0].set_ylabel('Standard Deviation')
            axes[0].set_title('Pattern Metrics vs Parameter Value')
            axes[0].grid(True)
            
            # Plot dominant wavelength vs parameter value
            axes[1].plot(metrics_df['ParameterValue'], metrics_df['DominantWavelength'], 'go-', linewidth=2)
            axes[1].set_ylabel('Dominant Wavelength')
            axes[1].grid(True)
            
            # Plot number of features vs parameter value
            axes[2].plot(metrics_df['ParameterValue'], metrics_df['NumFeatures'], 'ro-', linewidth=2)
            axes[2].set_xlabel('Parameter Value')
            axes[2].set_ylabel('Number of Features')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'parameter_sweep_metrics.png'), dpi=300)
            plt.close(fig)
            
            # Create scatter plot of wavelength vs D_ratio if available
            if 'D_ratio' in metrics_df.columns and 'DominantWavelength' in metrics_df.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(metrics_df['D_ratio'], metrics_df['DominantWavelength'], s=50)
                ax.set_xlabel('D_ratio (DA/DB)')
                ax.set_ylabel('Dominant Wavelength')
                ax.set_title('Wavelength vs Diffusion Ratio')
                ax.set_xscale('log')
                ax.grid(True)
                
                # Add trend line
                if len(metrics_df) > 1:
                    try:
                        # Use log of D_ratio for fitting
                        x = np.log10(metrics_df['D_ratio'])
                        y = metrics_df['DominantWavelength']
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        
                        # Generate points for trend line
                        x_trend = np.linspace(min(x), max(x), 100)
                        y_trend = p(x_trend)
                        
                        # Plot trend line
                        ax.plot(10**x_trend, y_trend, 'r--', linewidth=2)
                        
                        # Add equation to plot
                        equation = f'λ ≈ {p[0]:.2f} log(D_ratio) + {p[1]:.2f}'
                        ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
                               fontsize=12, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    except Exception as e:
                        print(f"Error fitting trend line: {e}")
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'wavelength_vs_diffusion_ratio.png'), dpi=300)
                plt.close(fig)
    
    return metrics_df

def main():
    """
    Main function for command-line usage.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pattern Quantification Tools')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze pattern command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single pattern')
    analyze_parser.add_argument('--file', type=str, required=True, help='Path to pattern CSV file')
    analyze_parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    # Analyze evolution command
    evolution_parser = subparsers.add_parser('evolution', help='Analyze pattern evolution')
    evolution_parser.add_argument('--dir', type=str, required=True, help='Directory containing pattern CSV files')
    evolution_parser.add_argument('--prefix', type=str, default='A_t', help='Prefix for pattern filenames')
    evolution_parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    # Compare patterns command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple patterns')
    compare_parser.add_argument('--files', type=str, nargs='+', required=True, help='Paths to pattern CSV files')
    compare_parser.add_argument('--labels', type=str, nargs='+', help='Labels for each pattern')
    compare_parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    # Analyze parameter sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Analyze parameter sweep')
    sweep_parser.add_argument('--dir', type=str, required=True, help='Directory containing parameter sweep results')
    sweep_parser.add_argument('--prefix', type=str, default='A_final', help='Prefix for pattern filenames')
    sweep_parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'analyze':
        # Load pattern data
        pattern_data = load_pattern_data(args.file)
        
        if pattern_data is None:
            print(f"Error: Could not load pattern data from {args.file}")
            return
        
        # Calculate metrics
        metrics = calculate_pattern_metrics(pattern_data)
        
        # Print metrics
        print("Pattern Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Calculate power spectrum
        _, _, dominant_wavelength = calculate_power_spectrum(pattern_data)
        print(f"Dominant Wavelength: {dominant_wavelength:.2f}")
        
        # Classify pattern type
        pattern_type = classify_pattern_type(pattern_data)
        print(f"Pattern Type: {pattern_type}")
        
        # Visualize pattern
        visualize_pattern(pattern_data, args.output)
        
    elif args.command == 'evolution':
        # Analyze pattern evolution
        metrics_df = analyze_pattern_evolution(args.dir, args.prefix, args.output)
        
        if metrics_df.empty:
            print("Error: Could not analyze pattern evolution")
            return
        
        # Print summary
        print("Pattern Evolution Summary:")
        print(f"  Number of time steps: {len(metrics_df)}")
        print(f"  Initial StdDev: {metrics_df['StdDev'].iloc[0]:.4f}")
        print(f"  Final StdDev: {metrics_df['StdDev'].iloc[-1]:.4f}")
        print(f"  Final Dominant Wavelength: {metrics_df['DominantWavelength'].iloc[-1]:.2f}")
        print(f"  Final Pattern Type: {metrics_df['PatternType'].iloc[-1]}")
        
    elif args.command == 'compare':
        # Compare patterns
        metrics_df = compare_patterns(args.files, args.labels, args.output)
        
        if metrics_df.empty:
            print("Error: Could not compare patterns")
            return
        
        # Print summary
        print("Pattern Comparison Summary:")
        print(metrics_df[['Label', 'StdDev', 'DominantWavelength', 'PatternType']])
        
    elif args.command == 'sweep':
        # Analyze parameter sweep
        metrics_df = analyze_parameter_sweep(args.dir, args.prefix, args.output)
        
        if metrics_df.empty:
            print("Error: Could not analyze parameter sweep")
            return
        
        # Print summary
        print("Parameter Sweep Summary:")
        if 'ParameterValue' in metrics_df.columns:
            print(metrics_df[['ParameterValue', 'StdDev', 'DominantWavelength', 'PatternType']])
        else:
            print(metrics_df)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
