#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessor module for genomic data compression pipeline.
Handles sequence encoding, normalization, and batch generation.
"""

import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def one_hot_encode(sequences, chunk_size=1000, base_dict=None):
    """
    One-hot encode DNA sequences into a numerical format.
    
    Args:
        sequences (list): List of DNA sequences
        chunk_size (int): Size of sequence chunks to encode (default: 1000)
        base_dict (dict): Dictionary mapping bases to indices
        
    Returns:
        numpy.ndarray: One-hot encoded sequences
        dict: Dictionary mapping bases to indices
    """
    if base_dict is None:
        # Standard DNA bases + N for unknown
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Number of unique bases
    n_bases = len(base_dict)
    
    # Create chunks from sequences
    sequence_chunks = []
    for seq in sequences:
        # Convert the sequence to uppercase
        seq = seq.upper()
        
        # Split the sequence into chunks
        for i in range(0, len(seq), chunk_size):
            chunk = seq[i:i+chunk_size]
            # Only use chunks that are the correct size
            if len(chunk) == chunk_size:
                sequence_chunks.append(chunk)
    
    logger.info(f"Created {len(sequence_chunks)} chunks of size {chunk_size}")
    
    # Initialize one-hot encoded array
    one_hot = np.zeros((len(sequence_chunks), chunk_size, n_bases), dtype=np.float32)
    
    # Fill the one-hot encoded array
    for i, chunk in enumerate(sequence_chunks):
        for j, base in enumerate(chunk):
            if base in base_dict:
                one_hot[i, j, base_dict[base]] = 1.0
            else:
                # For any unexpected character, treat as N
                one_hot[i, j, base_dict['N']] = 1.0
    
    return one_hot, base_dict

def reverse_one_hot_encode(one_hot_sequences, base_dict=None):
    """
    Convert one-hot encoded sequences back to DNA sequences.
    
    Args:
        one_hot_sequences (numpy.ndarray): One-hot encoded sequences
        base_dict (dict): Dictionary mapping bases to indices
        
    Returns:
        list: List of DNA sequences
    """
    if base_dict is None:
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Create a reverse mapping from indices to bases
    reverse_dict = {v: k for k, v in base_dict.items()}
    
    sequences = []
    
    for i in range(one_hot_sequences.shape[0]):
        seq = ""
        for j in range(one_hot_sequences.shape[1]):
            # Get the index of the maximum value
            idx = np.argmax(one_hot_sequences[i, j])
            seq += reverse_dict[idx]
        sequences.append(seq)
    
    return sequences

def prepare_data_batches(encoded_data, batch_size=32, test_size=0.2, validation_size=0.1):
    """
    Prepare data batches for training, validation, and testing.
    
    Args:
        encoded_data (numpy.ndarray): One-hot encoded sequences
        batch_size (int): Size of batches
        test_size (float): Proportion of data to use for testing
        validation_size (float): Proportion of data to use for validation
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Split data into training and test sets
    train_val_data, test_data = train_test_split(
        encoded_data, test_size=test_size, random_state=42
    )
    
    # Split training data into training and validation sets
    train_data, val_data = train_test_split(
        train_val_data, test_size=validation_size, random_state=42
    )
    
    logger.info(f"Data split: train={train_data.shape[0]}, "
                f"validation={val_data.shape[0]}, test={test_data.shape[0]}")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for preprocessing genomic data.")
    print("Import this module in your scripts to use its functionality.")