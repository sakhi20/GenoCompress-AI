import numpy as np
from typing import Dict, List
from collections import Counter
import Levenshtein  # For edit distance (install via: pip install python-Levenshtein)
from scipy.stats import pearsonr # For correlation
import logging

logging.basicConfig(level=logging.INFO)

def calculate_compression_metrics(original_size_bytes: int, compressed_size_bytes: int, original_sequence: str, compressed_sequence: str) -> Dict[str, float]:
    """
    Calculates compression performance metrics.

    Args:
        original_size_bytes: Size of the original data in bytes.
        compressed_size_bytes: Size of the compressed data in bytes.
        original_sequence: The original sequence (string).
        compressed_sequence: The compressed sequence (string) - used only for BPB.

    Returns:
        A dictionary containing the calculated metrics.
    """
    metrics = {}
    metrics['original_size_bytes'] = original_size_bytes
    metrics['compressed_size_bytes'] = compressed_size_bytes
    metrics['compression_ratio'] = original_size_bytes / compressed_size_bytes
    metrics['space_savings'] = ((original_size_bytes - compressed_size_bytes) / original_size_bytes) * 100

    if original_sequence:
        total_bases = len(original_sequence)
        compressed_size_bits = compressed_size_bytes * 8
        metrics['bits_per_base'] = compressed_size_bits / total_bases
    else:
        metrics['bits_per_base'] = None  # Or some other default

    return metrics

def calculate_reconstruction_accuracy(original_sequence: str, reconstructed_sequence: str) -> Dict[str, float]:
    """
    Calculates reconstruction accuracy metrics.

    Args:
        original_sequence: The original genomic sequence (string).
        reconstructed_sequence: The reconstructed genomic sequence (string).

    Returns:
        A dictionary containing the calculated metrics.
    """
    metrics = {}
    if not original_sequence or not reconstructed_sequence:
        metrics['base_level_accuracy'] = 0.0
        metrics['hamming_distance'] = 0
        metrics['edit_distance'] = 0
        return metrics

    min_length = min(len(original_sequence), len(reconstructed_sequence))
    correct_bases = sum(1 for a, b in zip(original_sequence[:min_length], reconstructed_sequence[:min_length]) if a == b)
    metrics['base_level_accuracy'] = (correct_bases / len(original_sequence)) * 100 if original_sequence else 0.0
    metrics['hamming_distance'] = sum(a != b for a, b in zip(original_sequence[:min_length], reconstructed_sequence[:min_length])) + abs(len(original_sequence) - len(reconstructed_sequence))
    metrics['edit_distance'] = Levenshtein.distance(original_sequence, reconstructed_sequence)

    return metrics



def calculate_quality_score_correlation(original_qualities: List[float], reconstructed_qualities: List[float]) -> Dict[str, float]:
    """
    Calculates the Pearson correlation coefficient between original and reconstructed quality scores.

    Args:
        original_qualities: List of original quality scores (floats).
        reconstructed_qualities: List of reconstructed quality scores (floats).

    Returns:
        A dictionary containing the correlation.
    """
    metrics = {}
    if not original_qualities or not reconstructed_qualities:
        metrics['quality_score_correlation'] = 0.0
        return metrics

    min_length = min(len(original_qualities), len(reconstructed_qualities)) #find the min length
    correlation, _ = pearsonr(original_qualities[:min_length], reconstructed_qualities[:min_length]) #calculate correlation
    metrics['quality_score_correlation'] = correlation
    return metrics


def calculate_gatk_concordance(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """
    Calculates variant calling concordance metrics.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.
        tn: Number of true negatives

    Returns:
        A dictionary containing the calculated metrics.
    """
    metrics = {}
    if (tp + fp) == 0:
        metrics['ppv'] = 0.0
    else:
        metrics['ppv'] = tp / (tp + fp)
    if (tp + fn) == 0:
        metrics['sensitivity'] = 0.0
    else:
        metrics['sensitivity'] = tp / (tp + fn)
    if (tn + fp) == 0:
        metrics['specificity'] = 0.0
    else:
        metrics['specificity'] = tn / (tn + fp)
    if (metrics['ppv'] + metrics['sensitivity']) == 0:
        metrics['f1_score'] = 0.0
    else:
        metrics['f1_score'] = 2 * (metrics['ppv'] * metrics['sensitivity']) / (metrics['ppv'] + metrics['sensitivity'])
    return metrics

def log_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Logs the calculated metrics.

    Args:
        metrics: A dictionary of metrics.
        title: A title for the log message.
    """
    logging.info(f"--- {title} ---")
    for key, value in metrics.items():
        logging.info(f"{key}: {value:.4f}")  # Format to 4 decimal places



if __name__ == '__main__':


    # Dummy data for demonstration
    original_size_bytes = 1000000
    compressed_size_bytes = 500000
    original_sequence = "ACGTACGTACGTACGT"
    reconstructed_sequence = "ACGTACCCTACGT"  # Make it shorter for demonstration
    original_qualities = [0.9, 0.8, 0.7, 0.9, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    reconstructed_qualities = [0.9, 0.7, 0.6, 0.9, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.8, 0.7, 0.6] # Make it shorter
    tp = 90
    fp = 10
    fn = 5
    tn = 990


    # Calculate and log metrics
    compression_metrics = calculate_compression_metrics(original_size_bytes, compressed_size_bytes, original_sequence, reconstructed_sequence)
    log_metrics(compression_metrics, "Compression Metrics")

    reconstruction_metrics = calculate_reconstruction_accuracy(original_sequence, reconstructed_sequence)
    log_metrics(reconstruction_metrics, "Reconstruction Accuracy Metrics")

    quality_metrics = calculate_quality_score_correlation(original_qualities, reconstructed_qualities)
    log_metrics(quality_metrics, "Quality Score Metrics")

    gatk_metrics = calculate_gatk_concordance(tp, fp, fn, tn)
    log_metrics(gatk_metrics, "GATK Concordance Metrics")
