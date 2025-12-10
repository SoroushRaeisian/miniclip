"""
MiniCLIP Configuration
======================
All hyperparameters and settings.
"""

import torch


class Config:
    # Paths
    CAPTIONS_JSON = "data/captions.json"
    IMAGE_ROOT = "data/images/"
    
    # Data parameters
    IMAGE_SIZE = 224
    MAX_SEQ_LEN = 32
    MAX_VOCAB_SIZE = 5000
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    PATIENCE = 10
    MAX_GRAD_NORM = 1.0
    
    # Model parameters
    EMBED_DIM = 256
    WORD_EMBED_DIM = 128
    LSTM_HIDDEN = 256
    LSTM_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 2
    DROPOUT = 0.2
    INIT_TEMPERATURE = 0.07
    
    # Transfer learning
    USE_PRETRAINED = True
    
    # Device
    @classmethod
    def get_device(cls):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    SEED = 42
    
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    @classmethod
    def to_dict(cls):
        return {
            'image_size': cls.IMAGE_SIZE,
            'max_seq_len': cls.MAX_SEQ_LEN,
            'max_vocab_size': cls.MAX_VOCAB_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'embed_dim': cls.EMBED_DIM,
            'dropout': cls.DROPOUT,
            'seed': cls.SEED,
        }


DEVICE = Config.get_device()
