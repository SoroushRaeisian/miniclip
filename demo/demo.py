"""
MiniCLIP Interactive Demo
=========================
Usage:
    cd demo
    python demo.py --model_type lstm
    python demo.py --batch
"""

import os
import sys
import json
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import Config
from model import MiniCLIP, ImageEncoder, LSTMTextEncoder, TransformerTextEncoder
from utils import SimpleTokenizer, get_val_transforms


def load_model(model_type, tokenizer, device, checkpoint_dir="../checkpoints/"):
    weights_path = os.path.join(checkpoint_dir, f"best_model_{model_type}.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=device)
    config = checkpoint.get('config', {})
    
    embed_dim = config.get('embed_dim', Config.EMBED_DIM)
    vocab_size = config.get('vocab_size', tokenizer.vocab_size)
    
    img_enc = ImageEncoder(embed_dim=embed_dim, use_pretrained=True, dropout=0.0)
    
    if model_type == "lstm":
        txt_enc = LSTMTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim,
                                  word_embed_dim=Config.WORD_EMBED_DIM, hidden_dim=Config.LSTM_HIDDEN,
                                  num_layers=2, dropout=0.0)
    else:
        txt_enc = TransformerTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim,
                                         word_embed_dim=Config.WORD_EMBED_DIM, num_heads=Config.TRANSFORMER_HEADS,
                                         num_layers=Config.TRANSFORMER_LAYERS, dropout=0.0, max_seq_len=Config.MAX_SEQ_LEN)
    
    model = MiniCLIP(img_enc, txt_enc)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[Model] Loaded {model_type} from epoch {checkpoint.get('epoch', '?')}")
    print(f"[Model] Best R@1: {checkpoint.get('val_r1', '?'):.2f}%")
    
    return model


def build_image_index(model, captions_data, image_root, device):
    transform = get_val_transforms(Config.IMAGE_SIZE)
    unique_fnames = list(dict.fromkeys(item["file_name"] for item in captions_data))
    
    print(f"[Index] Building index for {len(unique_fnames)} images...")
    
    all_img_embs = []
    all_img_paths = []
    
    with torch.no_grad():
        for fname in unique_fnames:
            path = os.path.join(image_root, fname)
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                emb = model.get_image_features(img_tensor)
                all_img_embs.append(emb.cpu())
                all_img_paths.append(path)
            except Exception as e:
                print(f"  [Warning] Skipping {path}: {e}")
    
    if not all_img_embs:
        raise RuntimeError("No images indexed!")
    
    img_index = torch.cat(all_img_embs, dim=0)
    print(f"[Index] Indexed {img_index.size(0)} images")
    
    return img_index, all_img_paths, unique_fnames


def search(query, model, tokenizer, img_index, img_paths, device, top_k=5):
    tokens = tokenizer.encode(query, Config.MAX_SEQ_LEN).unsqueeze(0).to(device)
    length = (tokens != 0).sum(dim=1)
    
    with torch.no_grad():
        text_emb = model.get_text_features(tokens, length)
        sims = (text_emb.cpu() @ img_index.t()).squeeze(0)
        k = min(top_k, img_index.size(0))
        values, indices = torch.topk(sims, k)
    
    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        results.append({'path': img_paths[idx], 'score': score, 'rank': len(results) + 1})
    
    return results


def display_results(query, results, save_path=None):
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results, figsize=(4 * n_results, 4))
    
    if n_results == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        try:
            img = Image.open(result['path'])
            ax.imshow(img)
            ax.set_title(f"Rank {result['rank']}\nScore: {result['score']:.3f}", fontsize=10)
        except:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center')
        ax.axis('off')
    
    plt.suptitle(f'Query: "{query}"', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    
    plt.show()


def run_interactive(args):
    device = Config.get_device()
    print(f"\n{'='*60}")
    print("MiniCLIP Interactive Demo")
    print(f"{'='*60}")
    print(f"Device: {device}, Model: {args.model_type}")
    
    with open(args.captions_json, "r") as f:
        data = json.load(f)
    
    tokenizer = SimpleTokenizer(Config.MAX_VOCAB_SIZE)
    tokenizer.build_vocab([d["caption"] for d in data])
    
    try:
        model = load_model(args.model_type, tokenizer, device, args.checkpoint_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Train the model first: cd src && python main.py --model_type lstm")
        return
    
    img_index, img_paths, _ = build_image_index(model, data, args.image_root, device)
    
    print(f"\n{'='*60}")
    print("Ready! Type query or 'quit' to exit")
    print(f"{'='*60}\n")
    
    while True:
        try:
            query = input("🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not query or query.lower() in ['quit', 'exit', 'q']:
            print("Bye!")
            break
        
        results = search(query, model, tokenizer, img_index, img_paths, device, top_k=5)
        
        print(f"\n📸 Top {len(results)} results:")
        for r in results:
            print(f"   {r['rank']}. {os.path.basename(r['path'])} ({r['score']:.3f})")
        
        display_results(query, results)


def run_batch(args):
    device = Config.get_device()
    
    with open(args.captions_json, "r") as f:
        data = json.load(f)
    
    tokenizer = SimpleTokenizer(Config.MAX_VOCAB_SIZE)
    tokenizer.build_vocab([d["caption"] for d in data])
    
    model = load_model(args.model_type, tokenizer, device, args.checkpoint_dir)
    img_index, img_paths, _ = build_image_index(model, data, args.image_root, device)
    
    queries = ["christmas tree", "dog", "kitchen", "coffee", "white chair"]
    
    fig, axes = plt.subplots(len(queries), 6, figsize=(18, 3*len(queries)))
    
    for row, query in enumerate(queries):
        results = search(query, model, tokenizer, img_index, img_paths, device, top_k=5)
        
        axes[row, 0].text(0.5, 0.5, f'Query:\n"{query}"', ha='center', va='center', fontsize=11)
        axes[row, 0].axis('off')
        axes[row, 0].set_facecolor('#f5f5f5')
        
        for col, result in enumerate(results):
            ax = axes[row, col + 1]
            try:
                img = Image.open(result['path'])
                ax.imshow(img)
                ax.set_title(f"Rank {result['rank']}: {result['score']:.2f}", fontsize=9)
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(args.results_dir, f"batch_demo_{args.model_type}.png")
    os.makedirs(args.results_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_json", type=str, default="../data/captions.json")
    parser.add_argument("--image_root", type=str, default="../data/images/")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/")
    parser.add_argument("--results_dir", type=str, default="../results/")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--batch", action="store_true")
    
    args = parser.parse_args()
    
    if args.batch:
        run_batch(args)
    else:
        run_interactive(args)
