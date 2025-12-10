# MiniCLIP: Text-Image Retrieval for Everyday Photos

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**EEP 596 Deep Learning Final Project**

A PyTorch implementation of CLIP-style contrastive learning for image-text retrieval on personal photo collections. This project implements a simplified version of OpenAI's CLIP model, enabling natural language search over image datasets.

![Retrieval Results](results/retrieval_results_lstm.png)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Pre-trained Models](#pre-trained-models)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)

## 🎯 Project Overview

MiniCLIP learns a joint embedding space for images and text captions using contrastive learning. Given a text query like "christmas tree" or "white chair", the model retrieves the most relevant images from the dataset.

### Key Contributions

1. **CLIP-style Contrastive Learning**: Implements InfoNCE loss with learnable temperature parameter
2. **Two Text Encoder Variants**: Compare LSTM vs Transformer architectures
3. **Transfer Learning**: Uses pretrained ResNet18 backbone for image encoding
4. **Proper Evaluation**: Image-based data splitting to prevent data leakage
5. **Data Augmentation**: Extensive augmentation for training on small datasets

## ✨ Features

- **Natural Language Image Search**: Query photos using everyday language
- **Multiple Text Encoders**: Choose between LSTM and Transformer architectures
- **Interactive Demo**: Real-time search interface for your photo collection
- **Comprehensive Metrics**: R@1, R@5, R@10, MRR, and Median Rank
- **Visualization Tools**: Training curves and retrieval result visualization

## 🏗️ Model Architecture

### Image Encoder
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Projection Head**: MLP with BatchNorm and dropout
- **Output**: 256-dimensional normalized embedding

### Text Encoders

#### LSTM Encoder
- Bidirectional LSTM with 2 layers
- Hidden dimension: 256
- Word embedding: 128 dimensions
- Uses packed sequences for efficiency

#### Transformer Encoder
- 2-layer Transformer with 4 attention heads
- Sinusoidal positional encoding
- CLS token for sequence representation
- Pre-LayerNorm for training stability

### Contrastive Loss
- Symmetric InfoNCE loss (image-to-text + text-to-image)
- Learnable temperature parameter (initialized at 0.07)

## 📊 Results

### Test Set Performance

| Model | R@1 | R@5 | R@10 | MRR | MedR |
|-------|-----|-----|------|-----|------|
| **LSTM** | **39.06%** | **76.56%** | **89.06%** | **0.547** | **2.0** |
| Transformer | 20.31% | 40.63% | 56.25% | 0.317 | 7.0 |
| Random Baseline | 3.13% | 15.63% | 31.25% | 0.127 | - |
| BoW Baseline | 9.38% | 20.31% | 26.56% | 0.162 | - |

The LSTM encoder achieves **12.5x improvement** over random baseline on R@1!

### Training Curves

#### LSTM Model
![LSTM Training](results/training_curves_lstm.png)

#### Transformer Model
![Transformer Training](results/training_curves_transformer.png)

### Model Comparison
![Metrics Comparison](results/metrics_comparison.png)

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/miniclip.git
cd miniclip
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

For CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. **Download pre-trained models**

Download the model weights from the link below and place them in the `checkpoints/` directory:

📥 **[Download Pre-trained Models (Google Drive)](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)**

Or use these direct links:
- [best_model_lstm.pth](https://drive.google.com/file/d/LSTM_FILE_ID)
- [best_model_transformer.pth](https://drive.google.com/file/d/TRANSFORMER_FILE_ID)

5. **Prepare your data**

Place your images in `data/images/` and your captions in `data/captions.json` with the following format:
```json
[
  {"file_name": "image_1.jpg", "caption": "white chair on dark hardwood floor"},
  {"file_name": "image_1.jpg", "caption": "dining chair on dark hardwood floor"},
  {"file_name": "image_2.jpg", "caption": "mug on white chair"}
]
```

## 🎮 How to Run

### Run the Demo

**Interactive Demo** (recommended):
```bash
python demo/demo.py --model_type lstm
```

**Batch Demo** (predefined queries):
```bash
python demo/demo.py --model_type lstm --batch
```

### Train a Model

```bash
# Train LSTM model
python src/main.py --model_type lstm --epochs 50

# Train Transformer model
python src/main.py --model_type transformer --epochs 50
```

### Training Options
```
--captions_json    Path to captions JSON file (default: data/captions.json)
--image_root       Path to images directory (default: data/images/)
--model_type       Text encoder: 'lstm' or 'transformer'
--epochs           Number of training epochs (default: 50)
--batch_size       Batch size (default: 32)
--lr               Learning rate (default: 3e-4)
--checkpoint_dir   Directory for model checkpoints
--results_dir      Directory for results and plots
```

## 📦 Expected Output

After running the demo with sample queries, you should see:

1. **Console output** showing top-k retrieved images with similarity scores
2. **Visualization window** displaying retrieved images for each query
3. **Saved results** in `results/` directory

Example output:
```
📸 Top 5 results for 'christmas tree':
   1. image_4.jpg (score: 0.425)
   2. image_5.jpg (score: 0.365)
   3. image_6.jpg (score: 0.287)
   4. image_12.jpg (score: 0.234)
   5. image_8.jpg (score: 0.198)
```

## 📁 Project Structure

```
miniclip/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration and hyperparameters
│   ├── model.py             # Model definitions (MiniCLIP, encoders)
│   ├── utils.py             # Utilities (tokenizer, dataset, metrics)
│   └── main.py              # Training script entry point
├── demo/
│   └── demo.py              # Interactive demo script
├── data/
│   ├── images/              # Image files (not included)
│   └── captions.json        # Image captions (sample included)
├── checkpoints/             # Saved model weights
│   ├── best_model_lstm.pth
│   └── best_model_transformer.pth
└── results/                 # Generated results
    ├── training_curves_lstm.png
    ├── training_curves_transformer.png
    ├── retrieval_results_lstm.png
    ├── metrics_lstm.json
    └── metrics_comparison.png
```

## ⚙️ Configuration

All hyperparameters are centralized in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | 224 | Input image size |
| `MAX_SEQ_LEN` | 32 | Maximum caption length |
| `MAX_VOCAB_SIZE` | 5000 | Maximum vocabulary size |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 3e-4 | AdamW learning rate |
| `EMBED_DIM` | 256 | Joint embedding dimension |
| `DROPOUT` | 0.2 | Dropout probability |
| `PATIENCE` | 10 | Early stopping patience |

## 🔬 Technical Details

### Data Splitting
- **70%** Training / **15%** Validation / **15%** Test
- Split by **images** (not captions) to prevent data leakage
- Multiple captions per image handled correctly in evaluation

### Training Strategy
- **Optimizer**: AdamW with weight decay (1e-4)
- **LR Schedule**: Cosine annealing with 5-epoch warmup
- **Gradient Clipping**: Max norm 1.0
- **Early Stopping**: Patience of 10 epochs

### Evaluation Metrics
- **R@K**: Recall at K (percentage of queries where correct image is in top-K)
- **MRR**: Mean Reciprocal Rank
- **MedR**: Median Rank of correct image

## 🙏 Acknowledgments

- **OpenAI CLIP**: Original CLIP paper and architecture inspiration
- **PyTorch**: Deep learning framework
- **torchvision**: Pretrained ResNet models
- **EEP 596 Course**: Deep Learning course materials and guidance

### References

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
2. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
3. He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.

## 📄 License

This project is for educational purposes as part of EEP 596 Deep Learning course.

---

**Author**: Soroush Raeisian  
**Course**: EEP 596 - Deep Learning  
**Date**: 2024
