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
genomic-compression/
├── data/
│   ├── raw/                    # Raw FASTA files
│   ├── processed/              # Preprocessed sequences
│   └── compressed/             # Compressed outputs
├── models/
│   ├── autoencoder.py         # Autoencoder architecture
│   ├── train.py              # Training script
│   └── saved_models/         # Trained model weights
├── src/
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── compression.py        # Compression pipeline
│   ├── evaluation.py         # Performance metrics
│   └── utils.py             # Helper functions
├── notebooks/
│   ├── exploration.ipynb     # Data exploration
│   └── results_analysis.ipynb # Results visualization
├── tests/
│   └── test_compression.py   # Unit tests
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Quick Start

### 1. Prepare Your Data

```python
from src.preprocessing import prepare_sequences

# Load and preprocess FASTA file
sequences = prepare_sequences('data/raw/genome.fasta', 
                             chunk_size=100000,
                             sequence_length=151)
```

### 2. Train the Model

```python
from models.train import train_autoencoder

# Train the compression model
model = train_autoencoder(sequences, 
                         epochs=100,
                         batch_size=32,
                         validation_split=0.2)
```

### 3. Compress Genomic Data

```python
from src.compression import compress_sequences

# Compress your genomic sequences
compressed_data = compress_sequences(model, sequences)
compression_ratio = calculate_compression_ratio(sequences, compressed_data)
print(f"Compression Ratio: {compression_ratio:.1f}×")
```

### 4. Decompress and Validate

```python
from src.compression import decompress_sequences
from src.evaluation import evaluate_biological_fidelity

# Decompress sequences
reconstructed = decompress_sequences(model, compressed_data)

# Validate biological fidelity
fidelity_metrics = evaluate_biological_fidelity(sequences, reconstructed)
print(f"GC Content Preservation: {fidelity_metrics['gc_preservation']:.1f}%")
```

## 📈 Usage Examples

### Command Line Interface

```bash
# Compress a FASTA file
python compress.py --input genome.fasta --output compressed.npy --model saved_models/autoencoder.h5

# Decompress back to FASTA
python decompress.py --input compressed.npy --output reconstructed.fasta --model saved_models/autoencoder.h5

# Evaluate compression performance
python evaluate.py --original genome.fasta --reconstructed reconstructed.fasta
```

### Python API

```python
from genomic_compression import GenomicCompressor

# Initialize compressor
compressor = GenomicCompressor(model_path='saved_models/autoencoder.h5')

# Compress
compressed_size = compressor.compress_file('genome.fasta', 'compressed.npy')

# Decompress  
compressor.decompress_file('compressed.npy', 'reconstructed.fasta')

# Get metrics
metrics = compressor.get_compression_metrics()
print(f"Compression ratio: {metrics['ratio']:.1f}×")
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

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/genomic-compression.git
cd genomic-compression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

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

---

**⭐ If this project helped you, please give it a star on GitHub!**
