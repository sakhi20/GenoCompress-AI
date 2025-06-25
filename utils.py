#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities module for genomic data compression pipeline.
Contains helper functions for file handling, visualization, etc.
"""

import os
import h5py
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_compressed_data(compressed_data, output_file):
    """
    Save compressed data to HDF5 format.
    
    Args:
        compressed_data (numpy.ndarray): Compressed genomic data
        output_file (str): Path to output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('compressed_data', data=compressed_data)
        # Store metadata
        f.attrs['date_created'] = str(datetime.now())
        f.attrs['shape'] = compressed_data.shape
        f.attrs['dtype'] = str(compressed_data.dtype)
    
    file_size = os.path.getsize(output_file)
    logger.info(f"Compressed data saved to {output_file} ({file_size} bytes)")

def load_compressed_data(input_file):
    """
    Load compressed data from HDF5 format.
    
    Args:
        input_file (str): Path to input file
        
    Returns:
        numpy.ndarray: Compressed genomic data
    """
    with h5py.File(input_file, 'r') as f:
        compressed_data = f['compressed_data'][:]
        logger.info(f"Loaded compressed data from {input_file}")
        logger.info(f"Shape: {compressed_data.shape}")
        logger.info(f"Created on: {f.attrs.get('date_created', 'unknown')}")
    
    return compressed_data

def plot_training_history(history, output_file):
    """
    Plot training history of the autoencoder.
    
    Args:
        history: Keras history object
        output_file (str): Path to output file
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Training history plot saved to {output_file}")

def plot_original_vs_reconstructed(original, reconstructed, n_samples=5, chunk_size=1000, output_file=None):
    """
    Plot comparison of original vs reconstructed sequences.
    
    Args:
        original (numpy.ndarray): Original one-hot encoded sequences
        reconstructed (numpy.ndarray): Reconstructed one-hot encoded sequences
        n_samples (int): Number of samples to plot
        chunk_size (int): Size of sequence chunks
        output_file (str): Path to output file
    """
    n_samples = min(n_samples, original.shape[0])
    
    plt.figure(figsize=(15, n_samples * 2))
    
    for i in range(n_samples):
        # Plot original sequence
        plt.subplot(n_samples, 2, i*2+1)
        plt.imshow(original[i].T, aspect='auto', cmap='viridis')
        plt.title(f"Original Sequence {i+1}")
        plt.ylabel("Base")
        if i == n_samples - 1:
            plt.xlabel("Position")
        
        # Plot reconstructed sequence
        plt.subplot(n_samples, 2, i*2+2)
        plt.imshow(reconstructed[i].T, aspect='auto', cmap='viridis')
        plt.title(f"Reconstructed Sequence {i+1}")
        if i == n_samples - 1:
            plt.xlabel("Position")
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Comparison plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory (str): Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    else:
        logger.info(f"Directory already exists: {directory}")

def config_to_string(config):
    """
    Convert a configuration dictionary to a formatted string.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        str: Formatted string representation
    """
    config_str = "Configuration:\n"
    for key, value in config.items():
        config_str += f"- {key}: {value}\n"
    
    return config_str

def count_bases(sequences):
    """
    Count the frequency of each base in the sequences.
    
    Args:
        sequences (list): List of DNA sequences
        
    Returns:
        dict: Dictionary with base counts
    """
    base_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'N': 0, 'Other': 0}
    
    for seq in sequences:
        for base in seq.upper():
            if base in base_counts:
                base_counts[base] += 1
            else:
                base_counts['Other'] += 1
    
    total_bases = sum(base_counts.values())
    
    logger.info("Base frequency:")
    for base, count in base_counts.items():
        percentage = (count / total_bases) * 100 if total_bases > 0 else 0
        logger.info(f"- {base}: {count} ({percentage:.2f}%)")
    
    return base_counts

def plot_base_distribution(base_counts, output_file):
    """
    Plot the distribution of bases.
    
    Args:
        base_counts (dict): Dictionary with base counts
        output_file (str): Path to output file
    """
    bases = list(base_counts.keys())
    counts = list(base_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(bases, counts)
    plt.title('Base Distribution')
    plt.xlabel('Base')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + (0.01 * max(counts)), f"{count:,}", 
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Base distribution plot saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    print("This module provides utility functions for the genomic data compression pipeline.")
    print("Import this module in your scripts to use its functionality.")