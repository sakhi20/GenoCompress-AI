#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loader module for genomic data compression pipeline.
Handles FASTQ and FASTA file formats.
"""

import os
import numpy as np
from Bio import SeqIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_fasta(file_path):
    """
    Load sequences from a FASTA file.
    
    Args:
        file_path (str): Path to the FASTA file
        
    Returns:
        list: List of sequences
        list: List of sequence IDs
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading FASTA file: {file_path}")
    sequences = []
    seq_ids = []
    
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)
    
    logger.info(f"Loaded {len(sequences)} sequences from FASTA file")
    return sequences, seq_ids

def load_fastq(file_path):
    """
    Load sequences from a FASTQ file.
    
    Args:
        file_path (str): Path to the FASTQ file
        
    Returns:
        list: List of sequences
        list: List of sequence IDs
        list: List of quality scores
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading FASTQ file: {file_path}")
    sequences = []
    seq_ids = []
    quality_scores = []
    
    for record in SeqIO.parse(file_path, "fastq"):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)
        quality_scores.append(record.letter_annotations["phred_quality"])
    
    logger.info(f"Loaded {len(sequences)} sequences from FASTQ file")
    return sequences, seq_ids, quality_scores

def save_sequences(sequences, seq_ids, output_path, file_format="fasta", quality_scores=None):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO
    
    logger.info(f"Saving {len(sequences)} sequences to {output_path}")
    
    records = []
    for i, (seq, seq_id) in enumerate(zip(sequences, seq_ids)):
        if file_format == "fastq" and quality_scores is not None:
            # Ensure quality scores are provided for FASTQ
            record = SeqRecord(
                Seq(seq),
                id=seq_id,
                description="",
                letter_annotations={"phred_quality": quality_scores[i]}
            )
        else:
            record = SeqRecord(
                Seq(seq),
                id=seq_id,
                description=""
            )
        records.append(record)
    
    if file_format == "fasta":
        SeqIO.write(records, output_path, "fasta")
    elif file_format == "fastq":
        SeqIO.write(records, output_path, "fastq")
    else:
        raise ValueError("Unsupported file format. Use 'fasta' or 'fastq'.")
    
    logger.info(f"Sequences saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for loading genomic data.")
    print("Import this module in your scripts to use its functionality.")