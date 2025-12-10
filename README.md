# MiniCLIP: Text-Image Retrieval for Everyday Photos

**Author**: Soroush Raeisian  
**Course**: University of Washington, EEP596: Practical Introduction to Deep Learning Applications and Theory 
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

![Model Architecture](data/miniclip_model_arch.png)

---

## Results

### Test Set Performance

| Model | R@1 | R@5 | R@10 | MRR | MedR |
|-------|-----|-----|------|-----|------|
| **BiLSTM** | **40.62%** | **73.44%** | **87.50%** | **0.558** | **2.0** |
| Transformer | 21.88% | 53.12% | 70.31% | 0.368 | 5.0 |
| BoW Baseline | 9.38% | 20.31% | 26.56% | 0.162 | - |
| Random Baseline | 3.12% | 15.62% | 31.25% | 0.127 | - |

### Key Findings
- **BiLSTM achieves 13× improvement over random baseline**
- BiLSTM beats Transformer (40.6% vs 21.9% R@1)
- Large gap at R@5: 73.4% vs 53.1%
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

Simply run:
```bash
cd demo
python demo.py
```

The demo provides an interactive menu:

1. **Select Model**: Choose between LSTM (recommended) or Transformer
2. **Select Mode**: Choose between Automatic or Interactive

#### Automatic Mode
- Randomly selects 5 captions from your dataset
- Runs retrieval on each query
- Saves results to `results/demo_results_{model}.png` and `.json`

#### Interactive Mode
- Type your own queries
- Results saved to `results/query_{n}_{query}_{model}.png`
- Type `quit` to exit

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

When running the demo, you will see a menu-driven interface:

``` bash
demo % python demo.py

============================================================
       MiniCLIP Demo - Text-to-Image Retrieval
         EE P 596 Deep Learning Final Project
============================================================

  Device: mps

----------------------------------------
  SELECT MODEL
----------------------------------------
  1. LSTM (Recommended - Better accuracy)
  2. Transformer
----------------------------------------
  Enter choice [1/2]: 1

----------------------------------------
  SELECT MODE
----------------------------------------
  1. Automatic (5 random captions from dataset)
  2. Interactive (Type your own queries)
----------------------------------------
  Enter choice [1/2]: 1

----------------------------------------
LOADING...
----------------------------------------
Loaded 420 captions
[Tokenizer] Built vocabulary with 194 tokens

  Loading LSTM model...
  ✓ Loaded from epoch 26, Val R@1: 27.42%

  Building image index (210 images)...
  ✓ Indexed 210 images

============================================================
  AUTOMATIC MODE - 5 Random Captions
============================================================

  Selected queries:
    1. "hand on side table"
    2. "tv"
    3. "santa side table"
    4. "ginger bread house"
    5. "espresso machine"

------------------------------------------------------------
  RESULTS
------------------------------------------------------------

Query: "hand on side table"
     ✓ 1. image_15.jpg (score: 0.703)
       2. image_13.jpg (score: 0.621)
       3. image_16.jpg (score: 0.592)
       4. image_12.jpg (score: 0.572)
       5. image_51.jpg (score: 0.398)

Query: "tv"
     ✓ 1. image_116.jpg (score: 0.693)
       2. image_96.jpg (score: 0.573)
       3. image_71.jpg (score: 0.249)
       4. image_114.jpg (score: 0.223)
       5. image_73.jpg (score: 0.203)

Query: "santa side table"
     ✓ 1. image_94.jpg (score: 0.573)
       2. image_60.jpg (score: 0.560)
       3. image_51.jpg (score: 0.525)
       4. image_17.jpg (score: 0.518)
       5. image_105.jpg (score: 0.423)

Query: "ginger bread house"
     ✓ 1. image_38.jpg (score: 0.679)
       2. image_37.jpg (score: 0.634)
       3. image_44.jpg (score: 0.540)
       4. image_13.jpg (score: 0.343)
       5. image_22.jpg (score: 0.323)

Query: "espresso machine"
     ✓ 1. image_39.jpg (score: 0.711)
       2. image_19.jpg (score: 0.574)
       3. image_40.jpg (score: 0.479)
       4. image_24.jpg (score: 0.454)
       5. image_21.jpg (score: 0.374)

============================================================
  DEMO COMPLETE!
============================================================

Results saved to:
     • ../results/demo_results_lstm.png
     • ../results/demo_results_lstm.json

demo % python demo.py

============================================================
       MiniCLIP Demo - Text-to-Image Retrieval
         EE P 596 Deep Learning Final Project
============================================================

  Device: mps

----------------------------------------
  SELECT MODEL
----------------------------------------
  1. LSTM (Recommended - Better accuracy)
  2. Transformer
----------------------------------------
  Enter choice [1/2]: 2

----------------------------------------
  SELECT MODE
----------------------------------------
  1. Automatic (5 random captions from dataset)
  2. Interactive (Type your own queries)
----------------------------------------
  Enter choice [1/2]: 1

----------------------------------------
LOADING...
----------------------------------------
Loaded 420 captions
[Tokenizer] Built vocabulary with 194 tokens

  Loading TRANSFORMER model...
/opt/anaconda3/lib/python3.13/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
  ✓ Loaded from epoch 8, Val R@1: 20.97%

  Building image index (210 images)...
  ✓ Indexed 210 images

============================================================
  AUTOMATIC MODE - 5 Random Captions
============================================================

  Selected queries:
    1. "unscented dog wipes"
    2. "thermostat"
    3. "white vases"
    4. "starbucks cup"
    5. "side table"

------------------------------------------------------------
  RESULTS
------------------------------------------------------------

Query: "unscented dog wipes"
     ✓ 1. image_198.jpg (score: 0.088)
       2. image_111.jpg (score: 0.084)
       3. image_103.jpg (score: 0.077)
       4. image_3.jpg (score: 0.071)
       5. image_126.jpg (score: 0.071)

Query: "thermostat"
     ✓ 1. image_91.jpg (score: 0.178)
       2. image_110.jpg (score: 0.168)
       3. image_143.jpg (score: 0.164)
       4. image_5.jpg (score: 0.163)
       5. image_7.jpg (score: 0.147)

Query: "white vases"
     ✓ 1. image_111.jpg (score: 0.172)
       2. image_136.jpg (score: 0.137)
       3. image_153.jpg (score: 0.137)
       4. image_112.jpg (score: 0.116)
       5. image_20.jpg (score: 0.112)

Query: "starbucks cup"
     ✓ 1. image_163.jpg (score: 0.180)
       2. image_53.jpg (score: 0.173)
       3. image_149.jpg (score: 0.165)
       4. image_147.jpg (score: 0.155)
       5. image_162.jpg (score: 0.153)

Query: "side table"
     ✓ 1. image_163.jpg (score: 0.207)
       2. image_53.jpg (score: 0.202)
       3. image_14.jpg (score: 0.201)
       4. image_165.jpg (score: 0.198)
       5. image_164.jpg (score: 0.182)

============================================================
  DEMO COMPLETE!
============================================================

Results saved to:
     • ../results/demo_results_transformer.png
     • ../results/demo_results_transformer.json




```

### Interactive Mode Output

``` bash
demo % python demo.py

============================================================
       MiniCLIP Demo - Text-to-Image Retrieval
         EE P 596 Deep Learning Final Project
============================================================

  Device: mps

----------------------------------------
  SELECT MODEL
----------------------------------------
  1. LSTM (Recommended - Better accuracy)
  2. Transformer
----------------------------------------
  Enter choice [1/2]: 1

----------------------------------------
  SELECT MODE
----------------------------------------
  1. Automatic (5 random captions from dataset)
  2. Interactive (Type your own queries)
----------------------------------------
  Enter choice [1/2]: 2

----------------------------------------
LOADING...
----------------------------------------
Loaded 420 captions
[Tokenizer] Built vocabulary with 194 tokens

  Loading LSTM model...
  ✓ Loaded from epoch 26, Val R@1: 27.42%

  Building image index (210 images)...
  ✓ Indexed 210 images

============================================================
  INTERACTIVE MODE
  Type a query to search, or 'quit' to exit
============================================================

Enter query: tree

Top 5 results:
     ✓ 1. image_55.jpg (score: 0.658)
       2. image_107.jpg (score: 0.604)
       3. image_9.jpg (score: 0.427)
       4. image_8.jpg (score: 0.413)
       5. image_6.jpg (score: 0.411)

Saved: ../results/query_1_tree_lstm.png

Enter query: dog

Top 5 results:
     ✓ 1. image_32.jpg (score: 0.623)
       2. image_31.jpg (score: 0.602)
       3. image_85.jpg (score: 0.439)
       4. image_69.jpg (score: 0.428)
       5. image_35.jpg (score: 0.419)

Saved: ../results/query_2_dog_lstm.png

Enter query: candle

Top 5 results:
     ✓ 1. image_61.jpg (score: 0.603)
       2. image_113.jpg (score: 0.559)
       3. image_59.jpg (score: 0.559)
       4. image_179.jpg (score: 0.535)
       5. image_130.jpg (score: 0.521)

Saved: ../results/query_3_candle_lstm.png

Enter query: exit


python demo.py

============================================================
       MiniCLIP Demo - Text-to-Image Retrieval
         EE P 596 Deep Learning Final Project
============================================================

  Device: mps

----------------------------------------
  SELECT MODEL
----------------------------------------
  1. LSTM (Recommended - Better accuracy)
  2. Transformer
----------------------------------------
  Enter choice [1/2]: 2

----------------------------------------
  SELECT MODE
----------------------------------------
  1. Automatic (5 random captions from dataset)
  2. Interactive (Type your own queries)
----------------------------------------
  Enter choice [1/2]: 2

----------------------------------------
LOADING...
----------------------------------------
Loaded 420 captions
[Tokenizer] Built vocabulary with 194 tokens

  Loading TRANSFORMER model...
/opt/anaconda3/lib/python3.13/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
  ✓ Loaded from epoch 8, Val R@1: 20.97%

  Building image index (210 images)...
  ✓ Indexed 210 images

============================================================
  INTERACTIVE MODE
  Type a query to search, or 'quit' to exit
============================================================

Enter query: brown dog

Top 5 results:
     ✓ 1. image_69.jpg (score: 0.110)
       2. image_4.jpg (score: 0.106)
       3. image_116.jpg (score: 0.101)
       4. image_106.jpg (score: 0.095)
       5. image_5.jpg (score: 0.090)

Saved: ../results/query_1_brown_dog_transformer.png

Enter query: christmas tree

Top 5 results:
     ✓ 1. image_56.jpg (score: 0.229)
       2. image_5.jpg (score: 0.225)
       3. image_4.jpg (score: 0.219)
       4. image_91.jpg (score: 0.191)
       5. image_6.jpg (score: 0.177)

Saved: ../results/query_2_christmas_tree_transformer.png

Enter query: pen

Top 5 results:
     ✓ 1. image_53.jpg (score: 0.184)
       2. image_81.jpg (score: 0.127)
       3. image_82.jpg (score: 0.126)
       4. image_85.jpg (score: 0.124)
       5. image_90.jpg (score: 0.120)

Saved: ../results/query_3_pen_transformer.png

Enter query: exit


```

### Training Output

When training a model, you should see:

``` bash
python main.py --model_type lstm
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
BASELINES
============================================================
Random baseline (N=32): R@1=3.12%
BoW baseline: R@1=9.38%

============================================================
MODEL
============================================================
Total parameters: 14,161,985
Trainable parameters: 11,970,049

============================================================
TRAINING (LSTM)
============================================================
Epoch   1/50 | Loss: 3.5825 | Val R@1: 4.84% | LR: 6.00e-05 | Temp: 14.28
  → Saved best model (R@1: 4.84%)
Epoch   2/50 | Loss: 3.3619 | Val R@1: 11.29% | LR: 1.20e-04 | Temp: 14.27
  → Saved best model (R@1: 11.29%)
Epoch   3/50 | Loss: 2.9434 | Val R@1: 19.35% | LR: 1.80e-04 | Temp: 14.26
  → Saved best model (R@1: 19.35%)
Epoch   4/50 | Loss: 2.5805 | Val R@1: 16.13% | LR: 2.40e-04 | Temp: 14.28
Epoch   5/50 | Loss: 2.0420 | Val R@1: 17.74% | LR: 3.00e-04 | Temp: 14.32
Epoch   6/50 | Loss: 1.7659 | Val R@1: 19.35% | LR: 3.00e-04 | Temp: 14.37
Epoch   7/50 | Loss: 1.3040 | Val R@1: 17.74% | LR: 2.99e-04 | Temp: 14.42
Epoch   8/50 | Loss: 1.1336 | Val R@1: 19.35% | LR: 2.97e-04 | Temp: 14.47
Epoch   9/50 | Loss: 0.9998 | Val R@1: 20.97% | LR: 2.94e-04 | Temp: 14.52
  → Saved best model (R@1: 20.97%)
Epoch  10/50 | Loss: 0.8709 | Val R@1: 22.58% | LR: 2.91e-04 | Temp: 14.56
  → Saved best model (R@1: 22.58%)
Epoch  11/50 | Loss: 0.8684 | Val R@1: 24.19% | LR: 2.87e-04 | Temp: 14.61
  → Saved best model (R@1: 24.19%)
Epoch  12/50 | Loss: 0.7595 | Val R@1: 24.19% | LR: 2.82e-04 | Temp: 14.65
Epoch  13/50 | Loss: 0.7587 | Val R@1: 22.58% | LR: 2.77e-04 | Temp: 14.68
Epoch  14/50 | Loss: 0.7413 | Val R@1: 22.58% | LR: 2.71e-04 | Temp: 14.72
Epoch  15/50 | Loss: 0.7074 | Val R@1: 22.58% | LR: 2.65e-04 | Temp: 14.75
Epoch  16/50 | Loss: 0.6457 | Val R@1: 25.81% | LR: 2.58e-04 | Temp: 14.78
  → Saved best model (R@1: 25.81%)
Epoch  17/50 | Loss: 0.6808 | Val R@1: 24.19% | LR: 2.50e-04 | Temp: 14.81
Epoch  18/50 | Loss: 0.6182 | Val R@1: 25.81% | LR: 2.42e-04 | Temp: 14.84
Epoch  19/50 | Loss: 0.5710 | Val R@1: 25.81% | LR: 2.34e-04 | Temp: 14.87
Epoch  20/50 | Loss: 0.6518 | Val R@1: 25.81% | LR: 2.25e-04 | Temp: 14.90
Epoch  21/50 | Loss: 0.5836 | Val R@1: 24.19% | LR: 2.16e-04 | Temp: 14.92
Epoch  22/50 | Loss: 0.5575 | Val R@1: 22.58% | LR: 2.06e-04 | Temp: 14.95
Epoch  23/50 | Loss: 0.5637 | Val R@1: 25.81% | LR: 1.96e-04 | Temp: 14.97
Epoch  24/50 | Loss: 0.5744 | Val R@1: 25.81% | LR: 1.86e-04 | Temp: 14.99
Epoch  25/50 | Loss: 0.5592 | Val R@1: 25.81% | LR: 1.76e-04 | Temp: 15.02
Epoch  26/50 | Loss: 0.5274 | Val R@1: 27.42% | LR: 1.66e-04 | Temp: 15.03
  → Saved best model (R@1: 27.42%)
Epoch  27/50 | Loss: 0.5591 | Val R@1: 25.81% | LR: 1.55e-04 | Temp: 15.05
Epoch  28/50 | Loss: 0.5073 | Val R@1: 24.19% | LR: 1.45e-04 | Temp: 15.06
Epoch  29/50 | Loss: 0.5578 | Val R@1: 25.81% | LR: 1.34e-04 | Temp: 15.08
Epoch  30/50 | Loss: 0.4747 | Val R@1: 25.81% | LR: 1.24e-04 | Temp: 15.09
Epoch  31/50 | Loss: 0.4653 | Val R@1: 25.81% | LR: 1.14e-04 | Temp: 15.11
Epoch  32/50 | Loss: 0.4802 | Val R@1: 24.19% | LR: 1.04e-04 | Temp: 15.12
Epoch  33/50 | Loss: 0.5070 | Val R@1: 24.19% | LR: 9.38e-05 | Temp: 15.13
Epoch  34/50 | Loss: 0.4220 | Val R@1: 24.19% | LR: 8.42e-05 | Temp: 15.15
Epoch  35/50 | Loss: 0.4850 | Val R@1: 25.81% | LR: 7.50e-05 | Temp: 15.16
Epoch  36/50 | Loss: 0.4415 | Val R@1: 25.81% | LR: 6.61e-05 | Temp: 15.17

Early stopping at epoch 36
Saved training curves to ../results/training_curves_lstm.png

============================================================
TEST EVALUATION
============================================================

Test Results (LSTM):
  R@1:  39.06%
  R@5:  71.88%
  R@10: 85.94%
  MRR:  0.5460
  MedR: 2.0

12.5x improvement over random!

Metrics to ../results/metrics_lstm.json
Saved retrieval visualization to ../results/retrieval_results_lstm.png
Saved similarity heatmap to ../results/similarity_heatmap_lstm.png
Saved embedding space to ../results/embedding_space_lstm.png
Need both metrics_lstm.json and metrics_transformer.json to compare
```
``` bash
python main.py --model_type transformer
============================================================
MiniCLIP: Text-Image Retrieval
============================================================
Device: mps
Model: transformer
============================================================
[Data] Loaded 420 caption entries
[Tokenizer] Built vocabulary with 194 tokens
[Split] Train: 294 captions (147 images)
[Split] Val: 62 captions (31 images)
[Split] Test: 64 captions (32 images)

============================================================
BASELINES
============================================================
Random baseline (N=32): R@1=3.12%
BoW baseline: R@1=9.38%

============================================================
MODEL
============================================================
/opt/anaconda3/lib/python3.13/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
Total parameters: 12,042,817
Trainable parameters: 9,850,881

============================================================
TRAINING (TRANSFORMER)
============================================================
Epoch   1/50 | Loss: 3.4728 | Val R@1: 3.23% | LR: 6.00e-05 | Temp: 14.28
  → Saved best model (R@1: 3.23%)
Epoch   2/50 | Loss: 3.4759 | Val R@1: 1.61% | LR: 1.20e-04 | Temp: 14.27
Epoch   3/50 | Loss: 3.4170 | Val R@1: 1.61% | LR: 1.80e-04 | Temp: 14.25
Epoch   4/50 | Loss: 3.3869 | Val R@1: 6.45% | LR: 2.40e-04 | Temp: 14.23
  → Saved best model (R@1: 6.45%)
Epoch   5/50 | Loss: 3.3124 | Val R@1: 8.06% | LR: 3.00e-04 | Temp: 14.20
  → Saved best model (R@1: 8.06%)
Epoch   6/50 | Loss: 3.1463 | Val R@1: 12.90% | LR: 3.00e-04 | Temp: 14.17
  → Saved best model (R@1: 12.90%)
Epoch   7/50 | Loss: 3.0197 | Val R@1: 16.13% | LR: 2.99e-04 | Temp: 14.17
  → Saved best model (R@1: 16.13%)
Epoch   8/50 | Loss: 2.9274 | Val R@1: 20.97% | LR: 2.97e-04 | Temp: 14.17
  → Saved best model (R@1: 20.97%)
Epoch   9/50 | Loss: 2.7796 | Val R@1: 20.97% | LR: 2.94e-04 | Temp: 14.18
Epoch  10/50 | Loss: 2.4999 | Val R@1: 20.97% | LR: 2.91e-04 | Temp: 14.20
Epoch  11/50 | Loss: 2.2476 | Val R@1: 19.35% | LR: 2.87e-04 | Temp: 14.23
Epoch  12/50 | Loss: 2.0688 | Val R@1: 20.97% | LR: 2.82e-04 | Temp: 14.28
Epoch  13/50 | Loss: 1.9582 | Val R@1: 20.97% | LR: 2.77e-04 | Temp: 14.32
Epoch  14/50 | Loss: 1.8058 | Val R@1: 20.97% | LR: 2.71e-04 | Temp: 14.36
Epoch  15/50 | Loss: 1.7748 | Val R@1: 20.97% | LR: 2.65e-04 | Temp: 14.40
Epoch  16/50 | Loss: 1.6594 | Val R@1: 20.97% | LR: 2.58e-04 | Temp: 14.44
Epoch  17/50 | Loss: 1.5010 | Val R@1: 20.97% | LR: 2.50e-04 | Temp: 14.48
Epoch  18/50 | Loss: 1.5279 | Val R@1: 19.35% | LR: 2.42e-04 | Temp: 14.51

Early stopping at epoch 18
Saved training curves to ../results/training_curves_transformer.png

============================================================
TEST EVALUATION
============================================================

Test Results (TRANSFORMER):
  R@1:  17.19%
  R@5:  39.06%
  R@10: 56.25%
  MRR:  0.2975
  MedR: 8.5

5.5x improvement over random!

Metrics to ../results/metrics_transformer.json
Saved retrieval visualization to ../results/retrieval_results_transformer.png
Saved similarity heatmap to ../results/similarity_heatmap_transformer.png
Saved embedding space to ../results/embedding_space_transformer.png
Saved model comparison to ../results/model_comparison.png

============================================================
MODEL COMPARISON SUMMARY
============================================================

LSTM:        R@1=39.1%  R@5=71.9%  MRR=0.546
Transformer: R@1=17.2%  R@5=39.1%  MRR=0.297

Winner: LSTM (+21.9% R@1)
```
---

## Pre-trained Model Link

📥 **[Download Pre-trained Models (Google Drive)](https://drive.google.com/drive/folders/1yLb_3cQ5vGcfsbCwhPTvmY2QOovYccp_?usp=drive_link)**

| Model | File | Size | Val R@1 | Test R@1 |
|-------|------|------|---------|----------|
| BiLSTM | `best_model_lstm.pth` | ~54 MB | 27.4% | 40.6% |
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
│   └── demo.py              # Menu-driven demo script
├── data/
│   ├── images/              # Image files (~210 photos)
│   ├── captions.json        # Image captions (420 entries)
│   └── miniclip_model_arch.png  # Architecture diagram
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
    ├── metrics_transformer.json
    ├── demo_results_lstm.png        # Demo output (automatic mode)
    ├── demo_results_lstm.json       # Demo output (automatic mode)
    ├── demo_results_transformer.png # Demo output (automatic mode)
    ├── demo_results_transformer.json# Demo output (automatic mode)
    └── query_*_{model}.png          # Demo output (interactive mode)
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
| BiLSTM | 26 | 0.54 | 27.4% |
| Transformer | 15 | 1.19 | 22.6% |

### Device Support
The code automatically detects and uses the best available device:
1. **CUDA** (NVIDIA GPU) - fastest
2. **MPS** (Apple Silicon) - good performance
3. **CPU** - slowest but always available

### Embedding Space Visualization
- **CUDA/CPU**: Uses t-SNE for better cluster separation
- **MPS (Apple Silicon)**: Uses PCA (t-SNE has compatibility issues on MPS)

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
| [scikit-learn](https://scikit-learn.org/) | ≥1.0.0 | t-SNE/PCA for embedding visualization | BSD-3-Clause |

### Pretrained Models
- **[ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)**: Pretrained on [ImageNet-1K](https://www.image-net.org/) (1.2M images, 1000 classes)
  - Weights: `torchvision.models.ResNet18_Weights.DEFAULT`

### Dataset
- **Personal Photo Collection**: ~210 images with 420 manually written captions
  - Images collected from personal everyday scenes
  - Captions written by the author (2 captions per image)


