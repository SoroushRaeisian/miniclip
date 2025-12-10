"""
MiniCLIP: Text-Image Retrieval for Everyday Photos
===================================================
A PyTorch implementation of CLIP-style contrastive learning for
image-text retrieval on personal photo collections.

EEP 596 Deep Learning Final Project
"""

from .config import Config, DEVICE
from .model import (
    MiniCLIP,
    ImageEncoder,
    LSTMTextEncoder,
    TransformerTextEncoder,
    PositionalEncoding,
    build_model,
)
from .utils import (
    SimpleTokenizer,
    MiniCLIPDataset,
    collate_fn,
    set_seed,
    get_train_transforms,
    get_val_transforms,
    load_data,
    split_data_by_images,
    compute_retrieval_metrics,
    evaluate_batch_metrics,
    compute_random_baseline,
    compute_bow_baseline,
    visualize_retrieval,
    plot_training_curves,
    plot_comparison,
)

__version__ = "1.0.0"
__author__ = "Soroush Raeisian"

__all__ = [
    # Config
    "Config",
    "DEVICE",
    # Models
    "MiniCLIP",
    "ImageEncoder",
    "LSTMTextEncoder",
    "TransformerTextEncoder",
    "PositionalEncoding",
    "build_model",
    # Utils
    "SimpleTokenizer",
    "MiniCLIPDataset",
    "collate_fn",
    "set_seed",
    "get_train_transforms",
    "get_val_transforms",
    "load_data",
    "split_data_by_images",
    "compute_retrieval_metrics",
    "evaluate_batch_metrics",
    "compute_random_baseline",
    "compute_bow_baseline",
    "visualize_retrieval",
    "plot_training_curves",
    "plot_comparison",
]
