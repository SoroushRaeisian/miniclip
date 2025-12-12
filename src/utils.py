"""
MiniCLIP Utilities
==================
Tokenizer, dataset, evaluation metrics, and visualization.
"""

import os
import json
import random
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import Config


def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleTokenizer:
    """Simple word-level tokenizer."""

    def __init__(self, max_vocab_size=Config.MAX_VOCAB_SIZE):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}

    def _basic_tokenize(self, text):
        text = text.lower().strip()
        for ch in [",", ".", "!", "?", ":", ";", "'", '"', "(", ")", "[", "]"]:
            text = text.replace(ch, " ")
        return text.split()

    def build_vocab(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(self._basic_tokenize(t))
        
        most_common = counter.most_common(self.max_vocab_size - 4)
        for i, (w, _) in enumerate(most_common, start=4):
            self.word2idx[w] = i
            self.idx2word[i] = w
        
        print(f"[Tokenizer] Built vocabulary with {self.vocab_size} tokens")

    def encode(self, text, max_len):
        tokens = self._basic_tokenize(text)
        seq = [self.word2idx.get(tok, 1) for tok in tokens]
        
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))
        
        return torch.tensor(seq, dtype=torch.long)

    def texts_to_sequences(self, texts):
        sequences = []
        for t in texts:
            tokens = self._basic_tokenize(t)
            seq = [self.word2idx.get(tok, 1) for tok in tokens]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, seqs, max_len):
        batch_size = len(seqs)
        padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, seq in enumerate(seqs):
            length = min(len(seq), max_len)
            padded[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
            lengths[i] = max(length, 1)
        
        return padded, lengths

    @property
    def vocab_size(self):
        return len(self.word2idx)


def get_train_transforms(image_size=Config.IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size=Config.IMAGE_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])


class MiniCLIPDataset(Dataset):
    """Dataset for image-caption pairs."""

    def __init__(self, data, image_root, tokenizer, transform=None, max_seq_len=Config.MAX_SEQ_LEN):
        self.data = data
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        self.image_to_indices = {}
        for idx, item in enumerate(data):
            fname = item["file_name"]
            if fname not in self.image_to_indices:
                self.image_to_indices[fname] = []
            self.image_to_indices[fname].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_root, item["file_name"])
        caption = item["caption"]
        file_name = item["file_name"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Could not open {img_path}: {e}")
            img = Image.new("RGB", (Config.IMAGE_SIZE, Config.IMAGE_SIZE), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        seq = self.tokenizer.texts_to_sequences([caption])[0]
        padded, lengths = self.tokenizer.pad_sequences([seq], self.max_seq_len)
        txt_tensor = padded[0]
        length = lengths[0]

        return img, txt_tensor, length, caption, file_name


def collate_fn(batch):
    imgs, txts, lens, caps, fnames = zip(*batch)
    imgs = torch.stack(imgs)
    txts = torch.stack(txts)
    lens = torch.stack(lens)
    return imgs, txts, lens, list(caps), list(fnames)


def split_data_by_images(data, train_ratio=0.7, val_ratio=0.15):
    """Split data by images to prevent data leakage."""
    image_to_captions = {}
    for item in data:
        fname = item["file_name"]
        if fname not in image_to_captions:
            image_to_captions[fname] = []
        image_to_captions[fname].append(item)
    
    image_names = list(image_to_captions.keys())
    random.shuffle(image_names)
    
    n_images = len(image_names)
    n_train = int(train_ratio * n_images)
    n_val = int(val_ratio * n_images)
    
    train_images = image_names[:n_train]
    val_images = image_names[n_train:n_train + n_val]
    test_images = image_names[n_train + n_val:]
    
    train_data = [item for img in train_images for item in image_to_captions[img]]
    val_data = [item for img in val_images for item in image_to_captions[img]]
    test_data = [item for img in test_images for item in image_to_captions[img]]
    
    print(f"[Split] Train: {len(train_data)} captions ({len(train_images)} images)")
    print(f"[Split] Val: {len(val_data)} captions ({len(val_images)} images)")
    print(f"[Split] Test: {len(test_data)} captions ({len(test_images)} images)")
    
    return train_data, val_data, test_data


def load_data(captions_path):
    with open(captions_path, "r") as f:
        data = json.load(f)
    print(f"[Data] Loaded {len(data)} caption entries")
    return data


@torch.no_grad()
def compute_retrieval_metrics(model, loader, device):
    """Compute retrieval metrics (R@K, MRR, MedR)."""
    model.eval()
    
    all_img_embs = []
    all_txt_embs = []
    all_fnames = []
    
    for imgs, txts, lens, caps, fnames in loader:
        imgs = imgs.to(device)
        txts = txts.to(device)
        lens = lens.to(device)
        
        img_embs = model.get_image_features(imgs)
        txt_embs = model.get_text_features(txts, lens)
        
        all_img_embs.append(img_embs.cpu())
        all_txt_embs.append(txt_embs.cpu())
        all_fnames.extend(fnames)
    
    if not all_img_embs:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "MedR": 0.0}
    
    all_img_embs = torch.cat(all_img_embs, dim=0)
    all_txt_embs = torch.cat(all_txt_embs, dim=0)
    
    unique_fnames = list(dict.fromkeys(all_fnames))
    fname_to_idx = {fname: idx for idx, fname in enumerate(unique_fnames)}
    
    num_unique = len(unique_fnames)
    unique_img_embs = torch.zeros(num_unique, all_img_embs.size(1))
    img_counts = torch.zeros(num_unique)
    
    for i, fname in enumerate(all_fnames):
        idx = fname_to_idx[fname]
        unique_img_embs[idx] += all_img_embs[i]
        img_counts[idx] += 1
    
    unique_img_embs = unique_img_embs / img_counts.unsqueeze(1)
    unique_img_embs = F.normalize(unique_img_embs, dim=-1)
    
    sims = all_txt_embs @ unique_img_embs.t()
    
    ranks = []
    for i in range(sims.size(0)):
        correct_img_idx = fname_to_idx[all_fnames[i]]
        ranking = torch.argsort(sims[i], descending=True)
        rank = (ranking == correct_img_idx).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    return {
        "R@1": np.mean(ranks < 1) * 100,
        "R@5": np.mean(ranks < 5) * 100,
        "R@10": np.mean(ranks < 10) * 100,
        "MRR": np.mean(1.0 / (ranks + 1)),
        "MedR": np.median(ranks) + 1,
    }


@torch.no_grad()
def evaluate_batch_metrics(model, loader, device):
    """Quick within-batch evaluation for validation."""
    model.eval()
    
    r1_sum = 0.0
    r5_sum = 0.0
    n_samples = 0
    
    for imgs, txts, lens, caps, fnames in loader:
        imgs = imgs.to(device)
        txts = txts.to(device)
        lens = lens.to(device)
        
        img_embs = model.get_image_features(imgs)
        txt_embs = model.get_text_features(txts, lens)
        
        sims = txt_embs @ img_embs.t()
        batch_size = imgs.size(0)
        
        for i in range(batch_size):
            ranking = torch.argsort(sims[i], descending=True)
            rank = (ranking == i).nonzero(as_tuple=True)[0].item()
            if rank < 1:
                r1_sum += 1
            if rank < 5:
                r5_sum += 1
        
        n_samples += batch_size
    
    return {
        "R@1": r1_sum / max(n_samples, 1) * 100,
        "R@5": r5_sum / max(n_samples, 1) * 100,
    }


def compute_random_baseline(n_images):
    return {
        "R@1": 1.0 / n_images * 100,
        "R@5": min(5, n_images) / n_images * 100,
        "R@10": min(10, n_images) / n_images * 100,
        "MRR": sum(1.0 / i for i in range(1, n_images + 1)) / n_images,
    }


def compute_bow_baseline(dataset, tokenizer):
    """Bag-of-words baseline."""
    texts = [item["caption"] for item in dataset.data]
    fnames = [item["file_name"] for item in dataset.data]
    
    seqs = tokenizer.texts_to_sequences(texts)
    V = tokenizer.vocab_size
    N = len(seqs)
    
    if N == 0:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0}
    
    bows = np.zeros((N, V), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for idx in seq:
            if idx < V:
                bows[i, idx] += 1.0
    
    norms = np.linalg.norm(bows, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    bows_normalized = bows / norms
    
    sims = np.dot(bows_normalized, bows_normalized.T)
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=0.0)
    
    ranks = []
    for i in range(N):
        target_fname = fnames[i]
        same_image_indices = set(j for j, f in enumerate(fnames) if f == target_fname and j != i)
        
        if not same_image_indices:
            continue
        
        scores = sims[i].copy()
        scores[i] = -np.inf
        ranking = np.argsort(-scores)
        
        for rank, idx in enumerate(ranking):
            if idx in same_image_indices:
                ranks.append(rank)
                break
    
    if not ranks:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0}
    
    ranks = np.array(ranks)
    return {
        "R@1": float(np.mean(ranks < 1) * 100),
        "R@5": float(np.mean(ranks < 5) * 100),
        "R@10": float(np.mean(ranks < 10) * 100),
        "MRR": float(np.mean(1.0 / (ranks + 1))),
    }


def visualize_retrieval(model, dataset, device, num_queries=5, top_k=5, save_path="results/retrieval_results.png"):
    """Visualize retrieval results."""
    model.eval()
    
    # Create directory if needed
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    unique_fnames = list(dict.fromkeys([item["file_name"] for item in dataset.data]))
    transform = get_val_transforms(Config.IMAGE_SIZE)
    
    img_embs = []
    img_paths = []
    
    with torch.no_grad():
        for fname in unique_fnames:
            path = os.path.join(dataset.image_root, fname)
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                emb = model.get_image_features(img_tensor)
                img_embs.append(emb.cpu())
                img_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    
    if not img_embs:
        print("No images to visualize")
        return
    
    img_embs = torch.cat(img_embs, dim=0)
    
    query_indices = random.sample(range(len(dataset)), min(num_queries, len(dataset)))
    
    fig, axes = plt.subplots(num_queries, top_k + 1, figsize=(3 * (top_k + 1), 3 * num_queries))
    if num_queries == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for row, idx in enumerate(query_indices):
            item = dataset.data[idx]
            caption = item["caption"]
            correct_fname = item["file_name"]
            
            seq = dataset.tokenizer.texts_to_sequences([caption])[0]
            padded, lengths = dataset.tokenizer.pad_sequences([seq], Config.MAX_SEQ_LEN)
            txt_tensor = padded.to(device)
            length = lengths.to(device)
            
            txt_emb = model.get_text_features(txt_tensor, length)
            
            sims = (txt_emb.cpu() @ img_embs.t()).squeeze()
            top_indices = torch.topk(sims, top_k).indices.tolist()
            
            ax = axes[row, 0]
            ax.text(0.5, 0.5, f'Query:\n"{caption}"', ha='center', va='center',
                   fontsize=10, wrap=True, transform=ax.transAxes)
            ax.axis('off')
            ax.set_facecolor('#f0f0f0')
            
            for col, img_idx in enumerate(top_indices):
                ax = axes[row, col + 1]
                try:
                    img = Image.open(img_paths[img_idx])
                    ax.imshow(img)
                    
                    retrieved_fname = unique_fnames[img_idx]
                    is_correct = retrieved_fname == correct_fname
                    color = 'green' if is_correct else 'red'
                    ax.set_title(f'Rank {col + 1}\n{sims[img_idx]:.3f}', color=color, fontsize=9)
                except Exception:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved retrieval visualization to {save_path}")
    except PermissionError:
        fallback = os.path.basename(save_path)
        plt.savefig(fallback, dpi=150, bbox_inches='tight')
        print(f"Saved retrieval visualization to {fallback}")
    plt.close()


def plot_similarity_heatmap(model, dataset, device, num_samples=10, save_path="results/similarity_heatmap.png"):
    """Plot similarity heatmap between text and image embeddings."""
    model.eval()
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get a subset of samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    transform = get_val_transforms(Config.IMAGE_SIZE)
    
    img_embs = []
    txt_embs = []
    captions = []
    
    with torch.no_grad():
        for idx in indices:
            item = dataset.data[idx]
            caption = item["caption"]
            captions.append(caption[:25] + "..." if len(caption) > 25 else caption)
            
            # Image embedding
            img_path = os.path.join(dataset.image_root, item["file_name"])
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                img_emb = model.get_image_features(img_tensor)
                img_embs.append(img_emb.cpu())
            except:
                img_embs.append(torch.zeros(1, Config.EMBED_DIM))
            
            # Text embedding
            seq = dataset.tokenizer.texts_to_sequences([caption])[0]
            padded, lengths = dataset.tokenizer.pad_sequences([seq], Config.MAX_SEQ_LEN)
            txt_tensor = padded.to(device)
            length = lengths.to(device)
            txt_emb = model.get_text_features(txt_tensor, length)
            txt_embs.append(txt_emb.cpu())
    
    img_embs = torch.cat(img_embs, dim=0)
    txt_embs = torch.cat(txt_embs, dim=0)
    
    # Compute similarity matrix
    similarity = (txt_embs @ img_embs.t()).numpy()
    
    # Get model type from save path
    model_type = "LSTM" if "lstm" in save_path.lower() else "Transformer"
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=1.0)
    
    # Add text annotations in each cell
    for i in range(len(captions)):
        for j in range(len(captions)):
            text = ax.text(j, i, f'{similarity[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    ax.set_xticks(range(len(captions)))
    ax.set_yticks(range(len(captions)))
    ax.set_xticklabels([f"I{i+1}" for i in range(len(captions))], rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(captions, fontsize=9)
    
    ax.set_xlabel('Images', fontsize=12)
    ax.set_ylabel('Captions', fontsize=12)
    ax.set_title(f'Similarity Matrix ({model_type})\nDiagonal = matching pairs', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    # Highlight diagonal (matching pairs) with blue rectangles
    for i in range(len(captions)):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity heatmap to {save_path}")
    except PermissionError:
        fallback = os.path.basename(save_path)
        plt.savefig(fallback, dpi=150, bbox_inches='tight')
        print(f"Saved similarity heatmap to {fallback}")
    plt.close()


def plot_embedding_space(model, dataset, device, num_samples=30, save_path="results/embedding_space.png"):
    """Visualize embedding space with color-coded matching pairs.
    
    Uses t-SNE on CUDA/CPU, PCA on MPS (Apple Silicon) due to t-SNE segfault issues.
    """
    model.eval()
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get unique images first
    unique_images = list(dict.fromkeys([item["file_name"] for item in dataset.data]))
    selected_images = random.sample(unique_images, min(num_samples, len(unique_images)))
    
    transform = get_val_transforms(Config.IMAGE_SIZE)
    
    img_embs = []
    txt_embs = []
    img_ids = []  # Track which image each embedding belongs to
    
    with torch.no_grad():
        for img_idx, img_name in enumerate(selected_images):
            # Get image embedding
            img_path = os.path.join(dataset.image_root, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                img_emb = model.get_image_features(img_tensor)
                img_embs.append(img_emb.cpu())
            except:
                continue
            
            # Get all captions for this image
            captions_for_img = [item["caption"] for item in dataset.data if item["file_name"] == img_name]
            
            for caption in captions_for_img[:2]:  # Limit to 2 captions per image
                seq = dataset.tokenizer.texts_to_sequences([caption])[0]
                padded, lengths = dataset.tokenizer.pad_sequences([seq], Config.MAX_SEQ_LEN)
                txt_tensor = padded.to(device)
                length = lengths.to(device)
                txt_emb = model.get_text_features(txt_tensor, length)
                txt_embs.append(txt_emb.cpu())
                img_ids.append(img_idx)
    
    if not img_embs or not txt_embs:
        print("No embeddings to visualize")
        return
    
    img_embs = torch.cat(img_embs, dim=0).numpy()
    txt_embs = torch.cat(txt_embs, dim=0).numpy()
    
    # Combine embeddings
    all_embs = np.vstack([img_embs, txt_embs])
    
    # Use PCA for MPS (Apple Silicon) - t-SNE has segfault issues
    # Use t-SNE for CUDA/CPU
    device_type = str(device).split(':')[0] if isinstance(device, torch.device) else str(device)
    use_tsne = device_type not in ['mps']
    
    if use_tsne:
        try:
            from sklearn.manifold import TSNE
            n_samples = all_embs.shape[0]
            perplexity = min(30, n_samples - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                        max_iter=1000, learning_rate='auto', init='pca', n_jobs=1, method='exact')
            embs_2d = tsne.fit_transform(all_embs)
            method_name = "t-SNE"
        except Exception as e:
            print(f"t-SNE failed ({e}), using PCA instead")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            embs_2d = pca.fit_transform(all_embs)
            method_name = "PCA"
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embs_2d = pca.fit_transform(all_embs)
        method_name = "PCA"
    
    n_imgs = len(img_embs)
    img_2d = embs_2d[:n_imgs]
    txt_2d = embs_2d[n_imgs:]
    
    # Get model type from save path
    model_type = "LSTM" if "lstm" in save_path.lower() else "Transformer"
    
    # Create color map - one color per unique image
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(selected_images))))
    if len(selected_images) > 20:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_images)))
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot images as squares
    for i in range(n_imgs):
        color_idx = i % len(colors)
        ax.scatter(img_2d[i, 0], img_2d[i, 1], c=[colors[color_idx]], marker='s', s=150, 
                   edgecolors='black', linewidths=0.5, alpha=0.8)
    
    # Plot captions as circles with matching colors
    for i, img_idx in enumerate(img_ids):
        color_idx = img_idx % len(colors)
        ax.scatter(txt_2d[i, 0], txt_2d[i, 1], c=[colors[color_idx]], marker='o', s=80,
                   edgecolors='black', linewidths=0.5, alpha=0.7)
    
    # Draw lines connecting matching pairs
    txt_counter = [0]  # Use list to allow modification in loop
    for img_idx in range(n_imgs):
        while txt_counter[0] < len(img_ids) and img_ids[txt_counter[0]] == img_idx:
            ax.plot([img_2d[img_idx, 0], txt_2d[txt_counter[0], 0]], 
                    [img_2d[img_idx, 1], txt_2d[txt_counter[0], 1]], 
                    color='gray', alpha=0.3, linewidth=1)
            txt_counter[0] += 1
    
    ax.set_xlabel(f'{method_name} Dim 1', fontsize=12)
    ax.set_ylabel(f'{method_name} Dim 2', fontsize=12)
    ax.set_title(f'MiniCLIP Embedding Space ({model_type}) - {method_name}\nMatching pairs cluster together', 
                 fontsize=14, fontweight='bold')
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12, 
               markeredgecolor='black', label='Images'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10,
               markeredgecolor='black', label='Captions')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add note
    ax.text(0.02, 0.02, 'Same color = same image\nSquares=images, Circles=captions', 
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved embedding space to {save_path}")
    except PermissionError:
        fallback = os.path.basename(save_path)
        plt.savefig(fallback, dpi=150, bbox_inches='tight')
        print(f"Saved embedding space to {fallback}")
    finally:
        plt.close(fig)
        import gc
        gc.collect()


def plot_model_comparison(save_dir="results/"):
    """Compare LSTM vs Transformer metrics from saved JSON files."""
    os.makedirs(save_dir, exist_ok=True)
    
    lstm_path = os.path.join(save_dir, "metrics_lstm.json")
    trans_path = os.path.join(save_dir, "metrics_transformer.json")
    
    if not os.path.exists(lstm_path) or not os.path.exists(trans_path):
        print("Need both metrics_lstm.json and metrics_transformer.json to compare")
        return
    
    with open(lstm_path) as f:
        lstm_metrics = json.load(f)
    with open(trans_path) as f:
        trans_metrics = json.load(f)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Retrieval metrics bar chart
    ax1 = axes[0]
    labels = ["R@1", "R@5", "R@10"]
    x = np.arange(len(labels))
    width = 0.35
    
    lstm_vals = [lstm_metrics['test_metrics'][k] for k in labels]
    trans_vals = [trans_metrics['test_metrics'][k] for k in labels]
    
    bars1 = ax1.bar(x - width/2, lstm_vals, width, label='LSTM', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, trans_vals, width, label='Transformer', color='#3498db', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Add random baseline
    random_r1 = lstm_metrics['baselines']['random']['R@1']
    ax1.axhline(y=random_r1, color='gray', linestyle='--', alpha=0.7, label=f'Random ({random_r1:.1f}%)')
    
    ax1.set_ylabel('Score (%)', fontsize=11)
    ax1.set_title('Test Retrieval Performance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: MRR and MedR comparison
    ax2 = axes[1]
    
    metrics_names = ['MRR', 'MedR']
    lstm_vals2 = [lstm_metrics['test_metrics']['MRR'] * 100, lstm_metrics['test_metrics']['MedR']]
    trans_vals2 = [trans_metrics['test_metrics']['MRR'] * 100, trans_metrics['test_metrics']['MedR']]
    
    x2 = np.arange(len(metrics_names))
    bars3 = ax2.bar(x2 - width/2, lstm_vals2, width, label='LSTM', color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, trans_vals2, width, label='Transformer', color='#3498db', alpha=0.8)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('MRR (×100) and Median Rank', fontsize=12, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['MRR ×100', 'MedR (lower=better)'])
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "model_comparison.png")
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    except PermissionError:
        fallback = "model_comparison.png"
        plt.savefig(fallback, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison to {fallback}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\nLSTM:        R@1={lstm_vals[0]:.1f}%  R@5={lstm_vals[1]:.1f}%  MRR={lstm_metrics['test_metrics']['MRR']:.3f}")
    print(f"Transformer: R@1={trans_vals[0]:.1f}%  R@5={trans_vals[1]:.1f}%  MRR={trans_metrics['test_metrics']['MRR']:.3f}")
    
    if lstm_vals[0] > trans_vals[0]:
        print(f"\nWinner: LSTM (+{lstm_vals[0]-trans_vals[0]:.1f}% R@1)")
    else:
        print(f"\nWinner: Transformer (+{trans_vals[0]-lstm_vals[0]:.1f}% R@1)")


def plot_training_curves(train_losses, val_r1s, model_type, save_dir="results/"):
    """Plot and save training curves."""
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{model_type.upper()} - Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, val_r1s, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation R@1 (%)')
    ax2.set_title(f'{model_type.upper()} - Validation Recall@1')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_curves_{model_type}.png')
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    except PermissionError:
        # Try saving to current directory instead
        fallback_path = f'training_curves_{model_type}.png'
        plt.savefig(fallback_path, dpi=150)
        print(f"Saved training curves to {fallback_path} (permission issue with {save_dir})")
    plt.close()
