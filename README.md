# AI-Driven Genomic Data Compression

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Optimizing Genomic Data Storage Using AI-Driven Compression Techniques**

An innovative deep learning-based approach to compress genomic sequences using convolutional autoencoders, achieving superior compression ratios while preserving biological fidelity.

## 🚀 Key Features

- **Superior Compression**: Achieves **18.9× compression ratio** (vs GZIP's 3.2×)
- **Biological Fidelity**: Preserves essential genomic features (GC content, read length)
- **Fast Processing**: 7.5× faster compression with GPU acceleration
- **Scalable**: Designed for large-scale genomic datasets
- **Lossy but Biologically Aware**: Maintains critical biological information

## 📊 Performance Comparison

| Metric | AI Autoencoder | GZIP |
|--------|----------------|------|
| Compression Ratio | **18.9×** | 3.2× |
| Compression Time | **4 sec/chunk** | 30 sec/file |
| Reconstruction Error (MSE) | 0.1749 | 0 (lossless) |
| GC Content Preservation | 99.6% | 100% |
| Base Call Accuracy | 92.3% | 100% |

## 🏗️ Architecture

The system uses a convolutional autoencoder architecture:

### Encoder
- **Input**: One-hot encoded DNA sequences (A, T, C, G)
- **Layers**: 
  - Conv1D (64 filters) → MaxPooling1D
  - Conv1D (32 filters) → MaxPooling1D  
  - Conv1D (16 filters)
- **Output**: Compressed latent vector

### Decoder
- **Layers**: Reverse convolution with upsampling
- **Output**: Reconstructed sequence probabilities

## 🛠️ Installation

### Prerequisites

```bash
# Python 3.10 or higher
python --version

# Required system packages
# For macOS with M1/M2:
brew install hdf5 c-blosc

# For Ubuntu/Debian:
sudo apt-get update
sudo apt-get install python3-dev python3-pip
```

### Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/genomic-compression.git
cd genomic-compression

# Install required packages
pip install -r requirements.txt
```

Create a `requirements.txt` file with:
```
tensorflow>=2.11.0
numpy>=1.21.0
pandas>=1.3.0
biopython>=1.79
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## 📁 Project Structure

```
├── autoencoder_model.py   # Conv1D encoder-decoder architecture (Keras)
├── pipeline.py            # End-to-end compression/decompression pipeline
├── preprocessor.py        # FASTA → one-hot encoded sequences
├── data_loader.py         # NCBI SRA data loading utilities
├── evaluator.py           # Compression ratio, GC content, base accuracy metrics
├── metrics.py             # Biological fidelity scoring functions
├── gatk_integration.py    # GATK toolkit interface for variant-aware preprocessing
├── utils.py               # Helper functions
├── encoder.h5             # Trained encoder weights
├── decoder.h5             # Trained decoder weights
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### 1. Prepare Your Data

```python
from preprocessor import prepare_sequences

# Load and preprocess FASTA file
sequences = prepare_sequences('genome.fasta', 
                             chunk_size=100000,
                             sequence_length=151)
```

### 2. Build and Train the Model

```python
from autoencoder_model import build_autoencoder

# Build the compression model
encoder, decoder, autoencoder = build_autoencoder(sequence_length=151)
autoencoder.fit(sequences, sequences, epochs=100, batch_size=32, validation_split=0.2)
```

### 3. Compress Genomic Data

```python
from pipeline import compress_sequences
from evaluator import calculate_compression_ratio

# Compress your genomic sequences
compressed_data = compress_sequences(encoder, sequences)
compression_ratio = calculate_compression_ratio(sequences, compressed_data)
print(f"Compression Ratio: {compression_ratio:.1f}×")
```

### 4. Decompress and Validate

```python
from pipeline import decompress_sequences
from evaluator import evaluate_biological_fidelity

# Decompress sequences
reconstructed = decompress_sequences(decoder, compressed_data)

# Validate biological fidelity
fidelity_metrics = evaluate_biological_fidelity(sequences, reconstructed)
print(f"GC Content Preservation: {fidelity_metrics['gc_preservation']:.1f}%")
```

## 📈 Usage Examples

### Python API

```python
from data_loader import load_sra_data
from preprocessor import prepare_sequences
from pipeline import compress_sequences, decompress_sequences
from evaluator import calculate_compression_ratio, evaluate_biological_fidelity

# Load data
raw_sequences = load_sra_data('SRR10971000')

# Preprocess
sequences = prepare_sequences(raw_sequences, sequence_length=151)

# Compress using pre-trained encoder
compressed_data = compress_sequences(encoder, sequences)

# Decompress and evaluate
reconstructed = decompress_sequences(decoder, compressed_data)
ratio = calculate_compression_ratio(sequences, compressed_data)
fidelity = evaluate_biological_fidelity(sequences, reconstructed)
print(f"Compression ratio: {ratio:.1f}× | GC preservation: {fidelity['gc_preservation']:.1f}%")
```

## 🔬 Dataset Information

The project was validated using:
- **Source**: NCBI SRA (Accession: SRR10971000)
- **Organism**: Homo sapiens
- **Type**: Whole Genome Sequencing (WGS)
- **Size**: 2.6 GB raw data
- **Read Length**: 151 base pairs
- **GC Content**: 47%

## 📊 Results & Validation

### Compression Performance
- **18.9× compression ratio** (vs GZIP's 3.2×)
- **7.5× faster processing** with GPU acceleration
- Effective on large-scale genomic datasets

### Biological Fidelity
- **99.6% GC content preservation** (0.2% deviation)
- **92.3% base call accuracy**
- **100% read length consistency**
- Suitable for non-clinical genomic analyses

## ⚠️ Limitations & Scope

- **Lossy compression**: 92.3% base call accuracy means this is NOT suitable for clinical diagnostics, rare variant detection (< 1% allele frequency), or any application requiring lossless reconstruction.
- **Baseline context**: GZIP (3.2×) is a general-purpose compressor not designed for genomic data. Purpose-built tools like CRAM (~5× on WGS) and SPRING are stronger references; this project focuses on demonstrating deep learning approaches rather than claiming state-of-the-art compression.
- **Dataset scale**: Validated on a single WGS sample (SRR10971000, 2.6 GB). Performance on diverse organisms or sequencing technologies is untested.

## ⚠️ Important Considerations

### When to Use
✅ **Recommended for:**
- Large-scale genomic storage
- Data transmission and archival
- Population genomics studies
- Research applications where slight data loss is acceptable

### When NOT to Use
❌ **Not recommended for:**
- Clinical diagnostics requiring 100% accuracy
- Rare variant detection (< 1% allele frequency)
- Applications requiring lossless compression
- Regulatory submissions requiring exact data preservation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Work

- **Variant-Sensitive Compression**: Preserve low-frequency variants
- **Adaptive Encoding**: Dynamic compression based on biological importance
- **Multi-modal Support**: Extend to RNA-seq, epigenomics data
- **Web Interface**: Browser-based compression tool

## 🙏 Acknowledgments

- NCBI SRA for providing genomic datasets
- TensorFlow and Keras teams for deep learning frameworks
- BioPython community for genomic data handling tools
- Open source genomics community
- **Dr. Paawan Sharma**, for his mentorship and expert guidance throughout the development of this project

---

**⭐ If this project helped you, please give it a star on GitHub!**
