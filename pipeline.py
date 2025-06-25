#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main pipeline script for genomic data compression.
Implements the end-to-end workflow from data loading to evaluation with GATK integration.
"""

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard # type: ignore

# Import custom modules
from src.data_loader import load_fasta, load_fastq, save_sequences
from src.preprocessor import one_hot_encode, reverse_one_hot_encode, prepare_data_batches
from src.autoencoder_model import (create_autoencoder, train_autoencoder, save_models, load_trained_models,
                                   compress_sequences, reconstruct_sequences)
from src.evaluator import (calculate_accuracy, calculate_compression_ratio, calculate_base_accuracy_per_position,
                           plot_accuracy_by_position, generate_evaluation_report)
from src.utils import (save_compressed_data, load_compressed_data, plot_training_history,
                       plot_original_vs_reconstructed, ensure_directory_exists, config_to_string,
                       count_bases, plot_base_distribution)
from src.gatk_integration import GATKConfig, GATKProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Genomic Data Compression Pipeline')
    
    # Input and output options
    parser.add_argument('--input', type=str, required=True,
                      help='Input FASTA/FASTQ file path')
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory')
    parser.add_argument('--format', type=str, choices=['fasta', 'fastq'], default='fasta',
                      help='Input file format')
    
    # Model options
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory to save/load models')
    parser.add_argument('--compression-factor', type=int, default=18,
                      help='Compression factor')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Size of sequence chunks')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--validation-size', type=float, default=0.1,
                      help='Proportion of data to use for validation')
    
    # Mode options
    parser.add_argument('--mode', type=str, 
                      choices=['train', 'compress', 'decompress', 'evaluate', 'full'],
                      default='full', help='Pipeline mode')
    
    # GATK options
    parser.add_argument('--gatk-path', type=str, default='gatk',
                      help='Path to GATK executable')
    parser.add_argument('--reference-genome', type=str,
                      help='Path to reference genome FASTA file')
    parser.add_argument('--keep-intermediate', action='store_true',
                      help='Keep intermediate GATK files')
    parser.add_argument('--java-options', type=str, default='-Xmx4g -Xms2g',
                      help='Java options for GATK')
    
    # Other options
    parser.add_argument('--load-model', action='store_true',
                      help='Load pre-trained model')
    parser.add_argument('--gpu', action='store_true',
                      help='Use GPU for training')
    
    return parser.parse_args()

def setup_environment(args):
    """
    Set up the environment for the pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set up directories
    ensure_directory_exists(args.output_dir)
    ensure_directory_exists(args.model_dir)
    ensure_directory_exists(os.path.join(args.output_dir, 'compressed'))
    ensure_directory_exists(os.path.join(args.output_dir, 'reconstructed'))
    ensure_directory_exists(os.path.join(args.output_dir, 'evaluation'))
    ensure_directory_exists(os.path.join(args.output_dir, 'plots'))
    ensure_directory_exists(os.path.join(args.output_dir, 'gatk'))
    ensure_directory_exists(os.path.join(args.output_dir, 'tmp'))
    
    # Set up GPU
    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Using GPU(s): {gpus}")
            except RuntimeError as e:
                logger.error(f"Error setting up GPU: {e}")
        else:
            logger.warning("No GPUs found. Using CPU instead.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
        logger.info("Using CPU for computation")

def load_data(args):
    """
    Load and preprocess the genomic data.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (sequences, seq_ids, quality_scores)
    """
    if args.format == 'fasta':
        sequences, seq_ids = load_fasta(args.input)
        quality_scores = None
    else:  # fastq
        sequences, seq_ids, quality_scores = load_fastq(args.input)
    
    return sequences, seq_ids, quality_scores

def run_gatk_analysis(args, original_sequences, reconstructed_sequences):
    if not args.reference_genome:
        logger.warning("No reference genome provided. Skipping GATK analysis.")
        return {}
    
    # Create temporary FASTQ files directory
    ensure_directory_exists(os.path.join(args.output_dir, 'tmp'))
    
    # Create dummy quality scores (using 'I' = quality score 40)
    def create_dummy_qualities(sequences):
        return [['I'] * len(seq) for seq in sequences]
    
    # Save sequences as FASTQ with dummy quality scores
    original_fastq = os.path.join(args.output_dir, 'tmp', 'original.fastq')
    reconstructed_fastq = os.path.join(args.output_dir, 'tmp', 'reconstructed.fastq')
    
    # Convert dummy quality strings to numeric values
    def qualities_to_numeric(qualities):
        return [[ord(q) - 33 for q in qual] for qual in qualities]
    
    dummy_qualities_original = create_dummy_qualities(original_sequences)
    dummy_qualities_reconstructed = create_dummy_qualities(reconstructed_sequences)
    
    numeric_qualities_original = qualities_to_numeric(dummy_qualities_original)
    numeric_qualities_reconstructed = qualities_to_numeric(dummy_qualities_reconstructed)
    
    # Save sequences with numeric quality scores
    save_sequences(original_sequences, ['original'] * len(original_sequences),
                  original_fastq, file_format='fastq',
                  quality_scores=numeric_qualities_original)
    
    save_sequences(reconstructed_sequences, ['reconstructed'] * len(reconstructed_sequences),
                  reconstructed_fastq, file_format='fastq',
                  quality_scores=numeric_qualities_reconstructed)
    
    # Set up GATK processor
    gatk_config = GATKConfig(
        gatk_path=args.gatk_path,
        reference_genome=args.reference_genome,
        tmp_dir=os.path.join(args.output_dir, 'tmp'),
        java_options=args.java_options,
        keep_intermediate=args.keep_intermediate
    )
    
    try:
        gatk_processor = GATKProcessor(gatk_config)
        
        # Create BAM files directory
        ensure_directory_exists(os.path.join(args.output_dir, 'gatk'))
        original_bam = os.path.join(args.output_dir, 'gatk', 'original.bam')
        reconstructed_bam = os.path.join(args.output_dir, 'gatk', 'reconstructed.bam')
        
        # Align sequences to reference and generate BAMs
        gatk_processor.align_to_reference(original_fastq, original_bam)
        gatk_processor.align_to_reference(reconstructed_fastq, reconstructed_bam)
        
        # Calculate concordance metrics between BAMs
        metrics = gatk_processor.calculate_concordance_metrics(original_bam, reconstructed_bam)
        
        # Clean up temporary FASTQ files if not keeping intermediate files
        if not args.keep_intermediate:
            if os.path.exists(original_fastq):
                os.remove(original_fastq)
            if os.path.exists(reconstructed_fastq):
                os.remove(reconstructed_fastq)
        
        return metrics
    
    except Exception as e:
        logger.error(f"GATK analysis failed: {str(e)}")
        return {'gatk_error': str(e)}

def train_pipeline(args, sequences):
    """
    Train the autoencoder model.
    
    Args:
        args: Command line arguments
        sequences: List of sequences
        
    Returns:
        tuple: (autoencoder, encoder, decoder, history, base_dict, encoded_data, test_data)
    """
    # One-hot encode sequences
    logger.info("One-hot encoding sequences...")
    encoded_data, base_dict = one_hot_encode(
        sequences,
        chunk_size=args.chunk_size
    )
    
    # Prepare data batches
    train_data, val_data, test_data = prepare_data_batches(
        encoded_data,
        batch_size=args.batch_size,
        test_size=args.test_size,
        validation_size=args.validation_size
    )
    
    # Create or load models
    if args.load_model:
        logger.info("Loading pre-trained models...")
        autoencoder, encoder, decoder = load_trained_models(args.model_dir)
    else:
        logger.info("Creating new models...")
        input_shape = (args.chunk_size, len(base_dict))
        autoencoder, encoder, decoder = create_autoencoder(
            input_shape, compression_factor=args.compression_factor
        )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            filepath=os.path.join(args.model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        TensorBoard(log_dir=os.path.join(args.output_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S')))
    ]
    
    # Train the model
    history = train_autoencoder(
        autoencoder, train_data, val_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save models
    save_models(autoencoder, encoder, decoder, args.model_dir)
    
    # Plot training history
    plot_training_history(
        history,
        os.path.join(args.output_dir, 'plots', 'training_history.png')
    )
    
    # Evaluate on test data
    test_loss, test_accuracy = autoencoder.evaluate(test_data, test_data)
    logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    # Plot original vs reconstructed
    reconstructed_test = autoencoder.predict(test_data[:5])
    plot_original_vs_reconstructed(
        test_data[:5], reconstructed_test,
        n_samples=5,
        chunk_size=args.chunk_size,
        output_file=os.path.join(args.output_dir, 'plots', 'original_vs_reconstructed.png')
    )
    
    # Save test data and reconstructed data for later evaluation
    np.save(os.path.join(args.output_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(args.output_dir, 'reconstructed_test.npy'), reconstructed_test)
    
    return autoencoder, encoder, decoder, history, base_dict, encoded_data, test_data

def compress_pipeline(args, sequences, encoder, base_dict):
    """
    Compress the sequences using the trained encoder.
    
    Args:
        args: Command line arguments
        sequences: List of sequences
        encoder: Trained encoder model
        base_dict: Dictionary mapping bases to indices
        
    Returns:
        numpy.ndarray: Compressed data
    """
    # One-hot encode sequences
    logger.info("One-hot encoding sequences for compression...")
    encoded_data, base_dict = one_hot_encode(sequences, chunk_size=args.chunk_size, base_dict=base_dict)
    
    # Compress sequences
    compressed_data = compress_sequences(encoder, encoded_data)
    
    # Save compressed data
    compressed_file = os.path.join(args.output_dir, 'compressed', 'compressed_data.h5')
    save_compressed_data(compressed_data, compressed_file)
    
    # Calculate compression ratio
    compression_ratio = calculate_compression_ratio(encoded_data, compressed_data)
    
    return compressed_data, encoded_data, compression_ratio

def decompress_pipeline(args, compressed_data, decoder, base_dict):
    """
    Decompress the sequences using the trained decoder.
    
    Args:
        args: Command line arguments
        compressed_data: Compressed data
        decoder: Trained decoder model
        base_dict: Dictionary mapping bases to indices
        
    Returns:
        tuple: (reconstructed_data, reconstructed_sequences)
    """
    # Reconstruct sequences
    reconstructed_data = reconstruct_sequences(decoder, compressed_data)
    
    # Convert one-hot encoded data back to sequences
    logger.info("Converting reconstructed data back to sequences...")
    reconstructed_sequences = reverse_one_hot_encode(reconstructed_data, base_dict)
    
    # Save reconstructed sequences
    reconstructed_file = os.path.join(args.output_dir, 'reconstructed', 'reconstructed_sequences.fasta')
    # Create dummy sequence IDs for reconstructed sequences
    reconstructed_ids = [f'reconstructed_{i+1}' for i in range(len(reconstructed_sequences))]
    save_sequences(reconstructed_sequences, reconstructed_ids, reconstructed_file)
    
    return reconstructed_data, reconstructed_sequences

def evaluate_pipeline(args, original_sequences, reconstructed_sequences, encoded_data, compressed_data, reconstructed_data):
    """
    Evaluate the compression and reconstruction quality.
    
    Args:
        args: Command line arguments
        original_sequences: Original sequences
        reconstructed_sequences: Reconstructed sequences
        encoded_data: Original one-hot encoded data
        compressed_data: Compressed data
        reconstructed_data: Reconstructed one-hot encoded data
        
    Returns:
        dict: Evaluation metrics
    """
    # Calculate base-level accuracy
    accuracy = calculate_accuracy(original_sequences, reconstructed_sequences)
    
    # Calculate compression ratio
    compression_ratio = calculate_compression_ratio(encoded_data, compressed_data)
    
    # Calculate base accuracy per position
    accuracy_per_position = calculate_base_accuracy_per_position(
        original_sequences, reconstructed_sequences, args.chunk_size
    )
    
    # Plot accuracy by position
    plot_accuracy_by_position(
        accuracy_per_position,
        os.path.join(args.output_dir, 'plots', 'accuracy_by_position.png')
    )
    
    # Plot original vs reconstructed
    plot_original_vs_reconstructed(
        encoded_data[:5], reconstructed_data[:5],
        n_samples=5,
        chunk_size=args.chunk_size,
        output_file=os.path.join(args.output_dir, 'plots', 'detailed_comparison.png')
    )
    
    # Analyze base distribution
    base_counts = count_bases(original_sequences)
    plot_base_distribution(
        base_counts,
        os.path.join(args.output_dir, 'plots', 'base_distribution.png')
    )
    
    # Run GATK analysis if reference genome is provided
    gatk_metrics = {}
    if args.reference_genome:
        gatk_metrics = run_gatk_analysis(args, original_sequences, reconstructed_sequences)
    
    # Prepare metrics for report
    metrics = {
        'base_accuracy': accuracy,
        'compression_ratio': compression_ratio,
        'original_size': encoded_data.size * encoded_data.itemsize,
        'compressed_size': compressed_data.size * compressed_data.itemsize,
        'num_sequences': len(original_sequences),
        'sequence_length': args.chunk_size,
        'model_type': 'Conv1D Autoencoder',
        'notes': f"Compression factor: {args.compression_factor}",
        **gatk_metrics
    }
    
    # Generate evaluation report
    generate_evaluation_report(
        metrics,
        os.path.join(args.output_dir, 'evaluation', 'evaluation_report.md')
    )
    
    return metrics

def run_pipeline(args):
    """
    Run the genomic data compression pipeline.
    
    Args:
        args: Command line arguments
    """
    # Print configuration
    logger.info("Starting genomic data compression pipeline")
    logger.info(config_to_string(vars(args)))
    
    # Set up environment
    setup_environment(args)
    
    # Load data
    sequences, seq_ids, quality_scores = load_data(args)
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Create a dictionary to track base distribution
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    # Initialize variables
    autoencoder = None
    encoder = None
    decoder = None
    compressed_data = None
    encoded_data = None
    reconstructed_data = None
    reconstructed_sequences = None
    
    # Run the appropriate pipeline mode
    if args.mode in ['train', 'full']:
        autoencoder, encoder, decoder, history, base_dict, encoded_data, test_data = train_pipeline(args, sequences)
    
    if args.mode in ['compress', 'full']:
        if encoder is None and args.load_model:
            # Load models if not already loaded
            _, encoder, _ = load_trained_models(args.model_dir)
        
        compressed_data, encoded_data, compression_ratio = compress_pipeline(args, sequences, encoder, base_dict)
    
    if args.mode in ['decompress', 'full']:
        if decoder is None and args.load_model:
            # Load models if not already loaded
            _, _, decoder = load_trained_models(args.model_dir)
        
        if compressed_data is None and args.mode == 'decompress':
            # Load compressed data if not already available
            compressed_file = os.path.join(args.output_dir, 'compressed', 'compressed_data.h5')
            compressed_data = load_compressed_data(compressed_file)
        
        reconstructed_data, reconstructed_sequences = decompress_pipeline(args, compressed_data, decoder, base_dict)
    
    if args.mode in ['evaluate', 'full']:
        if reconstructed_sequences is None and args.mode == 'evaluate':
            # Need to run decompression first
            if compressed_data is None:
                compressed_file = os.path.join(args.output_dir, 'compressed', 'compressed_data.h5')
                compressed_data = load_compressed_data(compressed_file)
            
            if decoder is None:
                _, _, decoder = load_trained_models(args.model_dir)
            
            reconstructed_data, reconstructed_sequences = decompress_pipeline(args, compressed_data, decoder, base_dict)
        
        if encoded_data is None and args.mode == 'evaluate':
            # Encode original sequences for comparison
            encoded_data, _ = one_hot_encode(sequences, chunk_size=args.chunk_size, base_dict=base_dict)
        
        metrics = evaluate_pipeline(args, sequences, reconstructed_sequences, encoded_data, compressed_data, reconstructed_data)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)