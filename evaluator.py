#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluator module for genomic data compression pipeline.
Provides metrics for evaluating compression quality.
"""

import numpy as np
import logging
import subprocess
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_accuracy(original_sequences, reconstructed_sequences):
    """
    Calculate base-level accuracy between original and reconstructed sequences.
    
    Args:
        original_sequences (list): List of original DNA sequences
        reconstructed_sequences (list): List of reconstructed DNA sequences
        
    Returns:
        float: Accuracy score
    """
    # Join all sequences together for overall accuracy
    original_concat = ''.join(original_sequences)
    reconstructed_concat = ''.join(reconstructed_sequences)
    
    # Ensure equal length for comparison
    min_length = min(len(original_concat), len(reconstructed_concat))
    original_concat = original_concat[:min_length]
    reconstructed_concat = reconstructed_concat[:min_length]
    
    # Calculate accuracy
    correct = sum(1 for o, r in zip(original_concat, reconstructed_concat) if o == r)
    accuracy = correct / min_length
    
    logger.info(f"Base-level accuracy: {accuracy:.4f}")
    
    return accuracy

def calculate_compression_ratio(original_data, compressed_data):
    """
    Calculate compression ratio.
    
    Args:
        original_data (numpy.ndarray): Original one-hot encoded data
        compressed_data (numpy.ndarray): Compressed data
        
    Returns:
        float: Compression ratio
    """
    original_size = original_data.size * original_data.itemsize
    compressed_size = compressed_data.size * compressed_data.itemsize
    
    compression_ratio = original_size / compressed_size
    
    logger.info(f"Original size: {original_size} bytes")
    logger.info(f"Compressed size: {compressed_size} bytes")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    return compression_ratio

def calculate_base_accuracy_per_position(original_seqs, reconstructed_seqs, seq_length):
    """
    Calculate base accuracy at each position in the sequences.
    
    Args:
        original_seqs (list): List of original DNA sequences
        reconstructed_seqs (list): List of reconstructed DNA sequences
        seq_length (int): Length of sequences
        
    Returns:
        numpy.ndarray: Array of accuracy scores per position
    """
    accuracy_per_position = np.zeros(seq_length)
    
    for orig_seq, recon_seq in zip(original_seqs, reconstructed_seqs):
        for i in range(min(len(orig_seq), len(recon_seq), seq_length)):
            if orig_seq[i] == recon_seq[i]:
                accuracy_per_position[i] += 1
    
    # Normalize by number of sequences
    accuracy_per_position = accuracy_per_position / len(original_seqs)
    
    return accuracy_per_position

def plot_accuracy_by_position(accuracy_per_position, output_path):
    """
    Plot base accuracy by position.
    
    Args:
        accuracy_per_position (numpy.ndarray): Array of accuracy scores per position
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_per_position)
    plt.title('Base Accuracy by Position')
    plt.xlabel('Position')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Accuracy plot saved to {output_path}")

def run_gatk_variant_calling(reference_fasta, sample_fasta, output_vcf, gatk_path):
    """
    Run GATK variant calling to compare original and reconstructed sequences.
    
    Args:
        reference_fasta (str): Path to reference FASTA file (original)
        sample_fasta (str): Path to sample FASTA file (reconstructed)
        output_vcf (str): Path to output VCF file
        gatk_path (str): Path to GATK executable
        
    Returns:
        int: Return code from GATK process
    """
    # Create indices for reference
    index_cmd = f"samtools faidx {reference_fasta}"
    logger.info(f"Running: {index_cmd}")
    subprocess.run(index_cmd, shell=True, check=True)
    
    # Create dictionary for reference
    dict_cmd = f"gatk CreateSequenceDictionary -R {reference_fasta}"
    logger.info(f"Running: {dict_cmd}")
    subprocess.run(dict_cmd, shell=True, check=True)
    
    # Convert FASTA to BAM
    bam_output = sample_fasta.replace('.fasta', '.bam')
    sam_to_bam_cmd = f"gatk FastaToSam -F {sample_fasta} -O {bam_output}"
    logger.info(f"Running: {sam_to_bam_cmd}")
    subprocess.run(sam_to_bam_cmd, shell=True, check=True)
    
    # Sort BAM
    sorted_bam = bam_output.replace('.bam', '.sorted.bam')
    sort_cmd = f"samtools sort {bam_output} -o {sorted_bam}"
    logger.info(f"Running: {sort_cmd}")
    subprocess.run(sort_cmd, shell=True, check=True)
    
    # Index BAM
    index_bam_cmd = f"samtools index {sorted_bam}"
    logger.info(f"Running: {index_bam_cmd}")
    subprocess.run(index_bam_cmd, shell=True, check=True)
    
    # Run HaplotypeCaller
    haplotype_cmd = f"{gatk_path} HaplotypeCaller -R {reference_fasta} -I {sorted_bam} -O {output_vcf}"
    logger.info(f"Running: {haplotype_cmd}")
    
    try:
        process = subprocess.run(haplotype_cmd, shell=True, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"GATK variant calling failed: {e}")
        return e.returncode

def calculate_snp_concordance(orig_vcf, recon_vcf):
    """
    Calculate SNP concordance between original and reconstructed VCF files.
    
    Args:
        orig_vcf (str): Path to original VCF file
        recon_vcf (str): Path to reconstructed VCF file
        
    Returns:
        float: SNP concordance rate
    """
    # This is a simplified version - in practice, you might use specialized tools
    # like bcftools or vcftools for more comprehensive comparison
    
    # Read VCF files
    orig_snps = set()
    with open(orig_vcf, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                fields = line.strip().split('\t')
                chrom = fields[0]
                pos = fields[1]
                ref = fields[3]
                alt = fields[4]
                orig_snps.add((chrom, pos, ref, alt))
    
    recon_snps = set()
    with open(recon_vcf, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                fields = line.strip().split('\t')
                chrom = fields[0]
                pos = fields[1]
                ref = fields[3]
                alt = fields[4]
                recon_snps.add((chrom, pos, ref, alt))
    
    # Find common SNPs
    common_snps = orig_snps.intersection(recon_snps)
    
    # Calculate concordance
    if len(orig_snps) > 0:
        concordance = len(common_snps) / len(orig_snps)
    else:
        concordance = 0.0
    
    logger.info(f"Original SNPs: {len(orig_snps)}")
    logger.info(f"Reconstructed SNPs: {len(recon_snps)}")
    logger.info(f"Common SNPs: {len(common_snps)}")
    logger.info(f"SNP concordance: {concordance:.4f}")
    
    return concordance

def generate_evaluation_report(metrics, output_path):
    """
    Generate an evaluation report with all metrics.
    
    Args:
        metrics (dict): Dictionary of metrics
        output_path (str): Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("# Genomic Data Compression Evaluation Report\n\n")
        
        f.write("## Compression Metrics\n")
        f.write(f"- Compression Ratio: {metrics.get('compression_ratio', 'N/A')}x\n")
        f.write(f"- Original Size: {metrics.get('original_size', 'N/A')} bytes\n")
        f.write(f"- Compressed Size: {metrics.get('compressed_size', 'N/A')} bytes\n\n")
        
        f.write("## Accuracy Metrics\n")
        f.write(f"- Base-level Accuracy: {metrics.get('base_accuracy', 'N/A'):.4f}\n")
        snp_concordance = metrics.get('snp_concordance', 'N/A')
        if isinstance(snp_concordance, (int, float)):
            f.write(f"- SNP Concordance: {snp_concordance:.4f}\n\n")
        else:
            f.write(f"- SNP Concordance: {snp_concordance}\n\n")
        
        f.write("## Additional Information\n")
        f.write(f"- Number of Sequences: {metrics.get('num_sequences', 'N/A')}\n")
        f.write(f"- Sequence Length: {metrics.get('sequence_length', 'N/A')}\n")
        f.write(f"- Model Type: {metrics.get('model_type', 'Conv1D Autoencoder')}\n")
        
        if 'notes' in metrics:
            f.write(f"\n## Notes\n{metrics['notes']}\n")
    
    logger.info(f"Evaluation report saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for evaluating genomic data compression quality.")
    print("Import this module in your scripts to use its functionality.")