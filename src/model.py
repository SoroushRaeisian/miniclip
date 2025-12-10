"""
MiniCLIP Model Definitions
==========================
Neural network architectures for image-text retrieval.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import Config


class ImageEncoder(nn.Module):
    """CNN-based image encoder using pretrained ResNet18."""

    def __init__(self, embed_dim=Config.EMBED_DIM, use_pretrained=True, dropout=Config.DROPOUT):
        super().__init__()
        
        if use_pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            backbone_dim = 512
            for param in list(self.backbone.parameters())[:-20]:
                param.requires_grad = False
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            backbone_dim = 512
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, backbone_dim),
            nn.BatchNorm1d(backbone_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, embed_dim),
        )
        
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.backbone(x)
        return self.projection(features)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMTextEncoder(nn.Module):
    """Bidirectional LSTM text encoder."""

    def __init__(self, vocab_size, embed_dim=Config.EMBED_DIM, word_embed_dim=Config.WORD_EMBED_DIM,
                 hidden_dim=Config.LSTM_HIDDEN, num_layers=2, dropout=Config.DROPOUT):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=word_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x, lengths):
        emb = self.embedding(x)
        emb = self.dropout(emb)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        
        _, (h_n, _) = self.lstm(packed)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        hidden = self.layer_norm(hidden)
        return self.projection(hidden)


class TransformerTextEncoder(nn.Module):
    """Transformer-based text encoder."""

    def __init__(self, vocab_size, embed_dim=Config.EMBED_DIM, word_embed_dim=Config.WORD_EMBED_DIM,
                 num_heads=Config.TRANSFORMER_HEADS, num_layers=Config.TRANSFORMER_LAYERS,
                 dropout=Config.DROPOUT, max_seq_len=Config.MAX_SEQ_LEN):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(word_embed_dim, max_seq_len + 1, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=word_embed_dim,
            nhead=num_heads,
            dim_feedforward=word_embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, word_embed_dim))
        self.layer_norm = nn.LayerNorm(word_embed_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(word_embed_dim, word_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(word_embed_dim, embed_dim),
        )

    def forward(self, x, lengths):
        B, T = x.shape
        
        emb = self.embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls_tokens, emb], dim=1)
        emb = self.pos_encoding(emb)
        
        pad_mask = torch.zeros(B, T + 1, dtype=torch.bool, device=x.device)
        for i, length in enumerate(lengths):
            pad_mask[i, length.item() + 1:] = True
        
        out = self.transformer(emb, src_key_padding_mask=pad_mask)
        cls_output = out[:, 0, :]
        
        cls_output = self.layer_norm(cls_output)
        return self.projection(cls_output)


class MiniCLIP(nn.Module):
    """CLIP-style contrastive model for image-text retrieval."""

    def __init__(self, img_enc, txt_enc, init_temperature=Config.INIT_TEMPERATURE):
        super().__init__()
        self.img_enc = img_enc
        self.txt_enc = txt_enc
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def forward(self, images, texts, lengths):
        img_embs = self.img_enc(images)
        txt_embs = self.txt_enc(texts, lengths)
        
        img_embs = F.normalize(img_embs, dim=-1)
        txt_embs = F.normalize(txt_embs, dim=-1)
        
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = txt_embs @ img_embs.t() * logit_scale
        
        labels = torch.arange(images.size(0), device=images.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return img_embs, txt_embs, loss

    def get_image_features(self, images):
        return F.normalize(self.img_enc(images), dim=-1)

    def get_text_features(self, texts, lengths):
        return F.normalize(self.txt_enc(texts, lengths), dim=-1)
