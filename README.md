# MiniCLIP: Text-Image Retrieval for Everyday Photos

**Author**: Soroush Raeisian  
**Course**: University of Washington, EE P 596 Au 25: Practical Introduction to Deep Learning Applications and Theory 
**Date**: Autumn 2025

---

A PyTorch implementation of CLIP-style contrastive learning for image-text retrieval on personal photo collections. This project implements a simplified version of OpenAI's CLIP model, enabling natural language search over image datasets.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Expected Output](#expected-output)
- [Pre-trained Model Link](#pre-trained-model-link)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Challenges & Future Work](#challenges--future-work)
- [Acknowledgments](#acknowledgments)
---

## Project Overview

MiniCLIP learns a joint embedding space for images and text captions using contrastive learning. Given a text query like "christmas tree" or "white chair", the model retrieves the most relevant images from the dataset.

### Problem Statement
- Search personal photos with natural language queries
- Example: "white chair by window" → retrieve matching image

### Approach
- Map images and captions into a shared embedding space
- Compare BiLSTM baseline vs Transformer text encoders
- Use InfoNCE contrastive loss with learnable temperature

### Key Features
- **CLIP-style Contrastive Learning**: Implements symmetric InfoNCE loss with learnable temperature
- **Two Text Encoder Variants**: Compare BiLSTM vs Transformer architectures
- **Transfer Learning**: Uses pretrained ResNet-18 backbone for image encoding
- **Proper Evaluation**: Image-based data splitting to prevent data leakage
- **Interactive Demo**: Real-time search interface for photo collections

---

## Dataset

| Property | Value |
|----------|-------|
| Total Images | ~210 personal photos |
| Captions per Image | 2 |
| Total Pairs | 420 image-text pairs |
| Caption Length | 2-5 words each |
| Scenes | Desk, kitchen, living room, outdoor |

### Data Split (by image, not caption)
- **Train**: 70% (147 images, 294 captions)
- **Validation**: 15% (31 images, 62 captions)
- **Test**: 15% (32 images, 64 captions)

**Note**: Same image never appears in both train and test → prevents data leakage

---

## Model Architecture

### Image Encoder
- **Backbone**: ResNet-18 with pretrained ImageNet weights
- **Transfer Learning**: Freeze early layers (effective for small datasets)
- **Projection**: MLP (512 → 512 → 256) with BatchNorm, ReLU, and Dropout
- **Output**: 256-dimensional L2-normalized embedding

### Text Encoders

#### BiLSTM Encoder
- Bidirectional: reads left→right AND right→left
- 2 layers, hidden dimension 256
- Word embedding: 128 dimensions
- Output: concatenated forward and backward hidden states → 256-dim

#### Transformer Encoder
- 2-layer Transformer with 4 attention heads
- Sinusoidal positional encoding
- CLS token for sequence representation
- Pre-LayerNorm for training stability
- Output: 256-dimensional embedding

### Contrastive Loss (InfoNCE)
- Creates B×B similarity matrix for batch of B pairs
- Diagonal = positive pairs, Off-diagonal = negative pairs
- Symmetric loss: image-to-text + text-to-image
- Learnable temperature τ (initialized at 0.07)

![Model Architecture](data/miniclip_model.png)

---

## Results

### Test Set Performance

| Model | R@1 | R@5 | R@10 | MRR | MedR |
|-------|-----|-----|------|-----|------|
| **BiLSTM** | **37.50%** | **79.69%** | **89.06%** | **0.547** | **2.0** |
| Transformer | 21.88% | 53.12% | 70.31% | 0.368 | 5.0 |
| BoW Baseline | 9.38% | 20.31% | 26.56% | 0.162 | - |
| Random Baseline | 3.12% | 15.62% | 31.25% | 0.127 | - |

### Key Findings
- **BiLSTM achieves 12× improvement over random baseline**
- BiLSTM beats Transformer (37.5% vs 21.9% R@1)
- Large gap at R@5: 79.7% vs 53.1%
- MedR=2 means correct image usually in top 2!

### Metrics Explained
- **R@k**: Recall at K - correct image in top k results (higher = better)
- **MedR**: Median Rank of correct image (lower = better)
- **MRR**: Mean Reciprocal Rank (higher = better)

### Why BiLSTM > Transformer?
- BiLSTM has built-in sequential inductive bias
- Transformer must learn positional information from scratch
- With only 210 images + short captions, BiLSTM wins
- More data would likely help Transformer catch up

### Visualizations

#### Training Curves
| BiLSTM | Transformer |
|--------|-------------|
| ![LSTM Training](results/training_curves_lstm.png) | ![Transformer Training](results/training_curves_transformer.png) |

#### Model Comparison
![Model Comparison](results/model_comparison.png)

#### Embedding Space Analysis
| BiLSTM | Transformer |
|--------|-------------|
| ![Embedding Space LSTM](results/embedding_space_lstm.png) | ![Embedding Space Transformer](results/embedding_space_transformer.png) |

#### Similarity Matrix
| BiLSTM | Transformer |
|--------|-------------|
| ![Similarity Heatmap LSTM](results/similarity_heatmap_lstm.png) | ![Similarity Heatmap Transformer](results/similarity_heatmap_transformer.png) |

#### Retrieval Examples
| BiLSTM | Transformer |
|--------|-------------|
| ![Retrieval Results LSTM](results/retrieval_results_lstm.png) | ![Retrieval Results Transformer](results/retrieval_results_transformer.png) |

---

## Setup Instructions

### Prerequisites

- **Python**: 3.8 or higher (tested on 3.10, 3.11, 3.13)
- **Hardware, any one of the options below would work**: 
  - NVIDIA GPU with CUDA support
  - Apple Silicon Mac with MPS
  - CPU (slower, but works)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: ~2GB for packages and model weights

### Installation

#### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/miniclip.git
cd miniclip
```

#### Step 2: Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

#### Step 3: Install dependencies

**Option A: Using requirements.txt (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Manual installation**

For NVIDIA GPU (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy Pillow matplotlib scikit-learn
```

For Apple Silicon (M1/M2/M3):
```bash
pip install torch torchvision
pip install numpy Pillow matplotlib scikit-learn
```

For CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy Pillow matplotlib scikit-learn
```

#### Step 4: Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

Expected output (example for Apple Silicon):
```
PyTorch: 2.1.0
CUDA: False
MPS: True
```

#### Step 5: Download pre-trained models

Download the model weights from the link below and place them in the `checkpoints/` directory:

📥 **[Download Pre-trained Models (Google Drive)](https://drive.google.com/drive/folders/1yLb_3cQ5vGcfsbCwhPTvmY2QOovYccp_?usp=drive_link)**

Direct links:
- [best_model_lstm.pth](https://drive.google.com/file/d/1pxew-fhfUTFmkRXljLJlE2Pf2pnmaPhP/view?usp=drive_link)
- [best_model_transformer.pth](https://drive.google.com/file/d/17ExPU9v9nbk95UAr4Hnu7PlKMpaAMbFB/view?usp=drive_link)

#### Step 6: Prepare your data

Place images in `data/images/` and create `data/captions.json`:
```json
[
  {"file_name": "image_1.jpg", "caption": "white chair on dark hardwood floor"},
  {"file_name": "image_1.jpg", "caption": "dining chair on dark hardwood floor"},
  {"file_name": "image_2.jpg", "caption": "mug on white chair"}
]
```

---

## How to Run

### Run the Demo (Quickest Way)

**Interactive Mode** - type your own queries:
```bash
cd demo
python demo.py --model_type lstm
```

**Batch Mode** - run predefined queries:
```bash
cd demo
python demo.py --model_type lstm --batch
```

### Train a Model from Scratch

```bash
cd src
python main.py --model_type lstm
python main.py --model_type transformer
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | lstm | Text encoder: `lstm` or `transformer` |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--captions_json` | data/captions.json | Path to captions file |
| `--image_root` | data/images/ | Path to images directory |

---

## Expected Output

### Demo Output

When running the interactive demo, you should see:

```
============================================================
MiniCLIP Interactive Demo
============================================================
Device: mps
Model: lstm
Loaded 210 images
============================================================

Enter query (or 'quit' to exit): christmas tree

🔍 Query: christmas tree
📸 Top 5 results:
   1. image_4.jpg (score: 0.425)
   2. image_5.jpg (score: 0.365)
   3. image_6.jpg (score: 0.287)
   4. image_12.jpg (score: 0.234)
   5. image_8.jpg (score: 0.198)

[Visualization window opens showing retrieved images]
```

### Training Output

When training a model, you should see:

```
============================================================
MiniCLIP: Text-Image Retrieval
============================================================
Device: mps
Model: lstm
============================================================
[Data] Loaded 420 caption entries
[Tokenizer] Built vocabulary with 194 tokens
[Split] Train: 294 captions (147 images)
[Split] Val: 62 captions (31 images)
[Split] Test: 64 captions (32 images)

============================================================
TRAINING (LSTM)
============================================================
Epoch   1/50 | Loss: 3.5825 | Val R@1: 4.84% | LR: 6.00e-05
  → Saved best model (R@1: 4.84%)
Epoch   2/50 | Loss: 3.3619 | Val R@1: 11.29% | LR: 1.20e-04
  → Saved best model (R@1: 11.29%)
...

============================================================
TEST EVALUATION
============================================================
Test Results (LSTM):
  R@1:  37.50%
  R@5:  79.69%
  R@10: 89.06%
  MRR:  0.5469
  MedR: 2.0

🎯 12.0x improvement over random!
```

---

## Pre-trained Model Link

📥 **[Download Pre-trained Models (Google Drive)](https://drive.google.com/drive/folders/1yLb_3cQ5vGcfsbCwhPTvmY2QOovYccp_?usp=drive_link)**

| Model | File | Size | Val R@1 | Test R@1 |
|-------|------|------|---------|----------|
| BiLSTM | `best_model_lstm.pth` | ~54 MB | 25.8% | 37.5% |
| Transformer | `best_model_transformer.pth` | ~46 MB | 22.6% | 21.9% |

Place downloaded files in the `checkpoints/` directory.

---

## Project Structure

```
miniclip/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/
│   ├── config.py            # Configuration and hyperparameters
│   ├── model.py             # Model definitions (MiniCLIP, encoders)
│   ├── utils.py             # Utilities (tokenizer, dataset, metrics)
│   └── main.py              # Training script entry point
├── demo/
│   └── demo.py              # Interactive demo script
├── data/
│   ├── images/              # Image files (~210 photos)
│   ├── captions.json        # Image captions (420 entries)
│   └── miniclip_model.png  # Architecture diagram
├── checkpoints/             # Saved model weights
│   ├── best_model_lstm.pth
│   └── best_model_transformer.pth
└── results/                 # Generated results
    ├── training_curves_lstm.png
    ├── training_curves_transformer.png
    ├── similarity_heatmap_lstm.png
    ├── similarity_heatmap_transformer.png
    ├── embedding_space_lstm.png
    ├── embedding_space_transformer.png
    ├── retrieval_results_lstm.png
    ├── retrieval_results_transformer.png
    ├── model_comparison.png
    ├── metrics_lstm.json
    └── metrics_transformer.json
```

---

## Configuration

All hyperparameters are defined in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMAGE_SIZE` | 224 | Input image size |
| `MAX_SEQ_LEN` | 32 | Maximum caption length |
| `MAX_VOCAB_SIZE` | 5000 | Maximum vocabulary size |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 3e-4 | AdamW learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `EMBED_DIM` | 256 | Joint embedding dimension |
| `DROPOUT` | 0.2 | Dropout probability |
| `PATIENCE` | 10 | Early stopping patience |
| `INIT_TEMPERATURE` | 0.07 | Initial temperature for InfoNCE |

---

## Technical Details

### Training Strategy
- **Optimizer**: AdamW with weight decay (1e-4) and betas (0.9, 0.98)
- **LR Schedule**: Cosine annealing with 5-epoch linear warmup
- **Gradient Clipping**: Max norm 1.0
- **Early Stopping**: Patience of 10 epochs, monitoring validation R@1

### Training Results Summary
| Model | Best Epoch | Final Loss | Best Val R@1 |
|-------|------------|------------|--------------|
| BiLSTM | 16 | 0.53 | 25.8% |
| Transformer | 15 | 1.19 | 22.6% |

### Device Support
The code automatically detects and uses the best available device:
1. **CUDA** (NVIDIA GPU) - fastest
2. **MPS** (Apple Silicon) - good performance
3. **CPU** - slowest but always available

---

## Challenges & Future Work

### Challenges Encountered
- **Small dataset**: Only 210 images limits model capacity
- **Ambiguous captions**: Multiple images contain similar objects (e.g., "mug")
- **Hard cases**: Tiny objects, similar backgrounds cause confusion

### Future Directions
- **Scale up data**: More images and richer captions would help Transformer
- **Frozen CLIP encoder**: Use pretrained CLIP text encoder with custom image encoder
- **Bidirectional retrieval**: Add image→text retrieval (reverse direction)
- **Hard negative mining**: Improve training with challenging negative samples

---

## Acknowledgments

### Papers & Research

This project is inspired by and builds upon the following research:

| Paper | Authors | Year | Contribution |
|-------|---------|------|--------------|
| [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | Radford et al. (OpenAI) | 2021 | CLIP architecture, InfoNCE loss, temperature scaling |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. (Google) | 2017 | Transformer architecture, sinusoidal positional encoding |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | He et al. (Microsoft) | 2016 | ResNet architecture for image encoding |
| [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) | Hochreiter & Schmidhuber | 1997 | LSTM architecture for sequence modeling |
| [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) | Chen et al. (Google) | 2020 | SimCLR projection head design |

### Frameworks & Libraries

| Library | Version | Usage | License |
|---------|---------|-------|---------|
| [PyTorch](https://pytorch.org/) | ≥1.12.0 | Deep learning framework | BSD-3-Clause |
| [torchvision](https://pytorch.org/vision/) | ≥0.13.0 | Pretrained ResNet-18, transforms | BSD-3-Clause |
| [NumPy](https://numpy.org/) | ≥1.21.0 | Numerical computations | BSD-3-Clause |
| [Pillow](https://pillow.readthedocs.io/) | ≥8.0.0 | Image loading and processing | HPND |
| [Matplotlib](https://matplotlib.org/) | ≥3.5.0 | Visualization and plotting | PSF |
| [scikit-learn](https://scikit-learn.org/) | ≥1.0.0 | PCA for embedding visualization | BSD-3-Clause |

### Pretrained Models
- **ResNet-18**: Pretrained on [ImageNet-1K](https://www.image-net.org/) (1.2M images, 1000 classes)
  - Weights: `torchvision.models.ResNet18_Weights.DEFAULT`

### Dataset
- **Personal Photo Collection**: ~210 images with 420 manually written captions
  - Images collected from personal everyday scenes
  - Captions written by the author (2 captions per image)


