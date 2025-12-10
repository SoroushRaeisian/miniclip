"""
MiniCLIP Demo Script
====================
EE P 596 Deep Learning Final Project - Soroush Raeisian

This demo script:
1. Lets you choose between LSTM and Transformer models
2. Lets you choose Interactive or Automatic mode
3. Loads a pre-trained MiniCLIP model
4. Runs text-to-image retrieval
5. Saves results to the results/ folder

Usage:
    cd demo
    python demo.py
"""

import os
import sys
import json
import random
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from PIL import Image
import matplotlib.pyplot as plt

# Import from src/
from config import Config
from model import MiniCLIP, ImageEncoder, LSTMTextEncoder, TransformerTextEncoder
from utils import SimpleTokenizer, get_val_transforms


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if hasattr(Config, 'DEVICE'):
        return Config.DEVICE
    elif hasattr(Config, 'get_device'):
        return Config.get_device()
    else:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("       MiniCLIP Demo - Text-to-Image Retrieval")
    print("         EE P 596 Deep Learning Final Project")
    print("=" * 60)


def show_model_menu():
    """Display model selection menu and get user choice."""
    print("\n" + "-" * 40)
    print("  SELECT MODEL")
    print("-" * 40)
    print("  1. LSTM (Recommended - Better accuracy)")
    print("  2. Transformer")
    print("-" * 40)
    
    while True:
        try:
            choice = input("  Enter choice [1/2]: ").strip()
            if choice == "1":
                return "lstm"
            elif choice == "2":
                return "transformer"
            else:
                print("  Invalid choice. Please enter 1 or 2.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting...")
            sys.exit(0)


def show_mode_menu():
    """Display mode selection menu and get user choice."""
    print("\n" + "-" * 40)
    print("  SELECT MODE")
    print("-" * 40)
    print("  1. Automatic (5 random captions from dataset)")
    print("  2. Interactive (Type your own queries)")
    print("-" * 40)
    
    while True:
        try:
            choice = input("  Enter choice [1/2]: ").strip()
            if choice == "1":
                return "automatic"
            elif choice == "2":
                return "interactive"
            else:
                print("  Invalid choice. Please enter 1 or 2.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting...")
            sys.exit(0)


def load_model(model_type, tokenizer, device, checkpoint_dir):
    """Load pre-trained MiniCLIP model."""
    weights_path = os.path.join(checkpoint_dir, f"best_model_{model_type}.pth")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"\n  Model weights not found: {weights_path}\n"
            f"  Please either:\n"
            f"    1. Download pre-trained models from Google Drive link in README.md\n"
            f"    2. Train the model: cd src && python main.py --model_type {model_type}\n"
        )
    
    print(f"\n  Loading {model_type.upper()} model...")
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    embed_dim = config.get('embed_dim', Config.EMBED_DIM)
    vocab_size = config.get('vocab_size', tokenizer.vocab_size)
    
    img_enc = ImageEncoder(embed_dim=embed_dim, use_pretrained=True, dropout=0.0)
    
    if model_type == "lstm":
        txt_enc = LSTMTextEncoder(
            vocab_size=vocab_size, embed_dim=embed_dim,
            word_embed_dim=Config.WORD_EMBED_DIM, hidden_dim=Config.LSTM_HIDDEN,
            num_layers=2, dropout=0.0
        )
    else:
        txt_enc = TransformerTextEncoder(
            vocab_size=vocab_size, embed_dim=embed_dim,
            word_embed_dim=Config.WORD_EMBED_DIM, num_heads=Config.TRANSFORMER_HEADS,
            num_layers=Config.TRANSFORMER_LAYERS, dropout=0.0, max_seq_len=Config.MAX_SEQ_LEN
        )
    
    model = MiniCLIP(img_enc, txt_enc)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    val_r1 = checkpoint.get('val_r1', 0)
    print(f"  ✓ Loaded from epoch {epoch}, Val R@1: {val_r1:.2f}%")
    
    return model


def build_image_index(model, captions_data, image_root, device):
    """Build searchable index of all images."""
    transform = get_val_transforms(Config.IMAGE_SIZE)
    unique_fnames = list(dict.fromkeys(item["file_name"] for item in captions_data))
    
    print(f"\n  Building image index ({len(unique_fnames)} images)...")
    
    all_img_embs = []
    all_img_paths = []
    all_img_names = []
    
    with torch.no_grad():
        for fname in unique_fnames:
            path = os.path.join(image_root, fname)
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                emb = model.get_image_features(img_tensor)
                all_img_embs.append(emb.cpu())
                all_img_paths.append(path)
                all_img_names.append(fname)
            except Exception as e:
                pass  # Skip silently
    
    if not all_img_embs:
        raise RuntimeError("No images could be indexed!")
    
    img_index = torch.cat(all_img_embs, dim=0)
    print(f"  ✓ Indexed {img_index.size(0)} images")
    
    return img_index, all_img_paths, all_img_names


def search(query, model, tokenizer, img_index, img_paths, device, top_k=5):
    """Search for images matching a text query."""
    tokens = tokenizer.encode(query, Config.MAX_SEQ_LEN).unsqueeze(0).to(device)
    length = (tokens != 0).sum(dim=1)
    
    with torch.no_grad():
        text_emb = model.get_text_features(tokens, length)
        sims = (text_emb.cpu() @ img_index.t()).squeeze(0)
        k = min(top_k, img_index.size(0))
        values, indices = torch.topk(sims, k)
    
    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        results.append({
            'path': img_paths[idx],
            'filename': os.path.basename(img_paths[idx]),
            'score': score,
            'rank': len(results) + 1
        })
    
    return results


def get_random_captions(captions_data, n=5):
    """Get n random unique captions from the dataset."""
    all_captions = list(set(item["caption"] for item in captions_data))
    random.shuffle(all_captions)
    return all_captions[:n]


def run_automatic(model_type, model, tokenizer, img_index, img_paths, captions_data, results_dir, device):
    """Run automatic mode with random captions from dataset."""
    
    # Get 5 random captions
    queries = get_random_captions(captions_data, n=5)
    
    print("\n" + "=" * 60)
    print("  AUTOMATIC MODE - 5 Random Captions")
    print("=" * 60)
    print("\n  Selected queries:")
    for i, q in enumerate(queries, 1):
        print(f"    {i}. \"{q}\"")
    
    top_k = 5
    
    # Create figure
    fig, axes = plt.subplots(len(queries), top_k + 1, figsize=(3 * (top_k + 1), 3 * len(queries)))
    
    all_results = []
    
    print("\n" + "-" * 60)
    print("  RESULTS")
    print("-" * 60)
    
    for row, query in enumerate(queries):
        print(f"\n  🔍 Query: \"{query}\"")
        
        results = search(query, model, tokenizer, img_index, img_paths, device, top_k=top_k)
        all_results.append({'query': query, 'results': results})
        
        for r in results:
            marker = "✓" if r['rank'] == 1 else " "
            print(f"     {marker} {r['rank']}. {r['filename']} (score: {r['score']:.3f})")
        
        # Plot query text
        ax = axes[row, 0]
        ax.text(0.5, 0.5, f'Query:\n"{query}"', ha='center', va='center', fontsize=10, 
                fontweight='bold', wrap=True)
        ax.set_facecolor('#f0f0f0')
        ax.axis('off')
        
        # Plot images
        for col, result in enumerate(results):
            ax = axes[row, col + 1]
            try:
                img = Image.open(result['path'])
                ax.imshow(img)
                color = '#2e7d32' if result['rank'] == 1 else '#666666'
                ax.set_title(f"#{result['rank']} | {result['score']:.2f}", 
                           fontsize=10, color=color, fontweight='bold')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
    
    plt.suptitle(f'MiniCLIP Demo - {model_type.upper()} Model (Automatic Mode)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save outputs
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, f"demo_results_{model_type}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    json_path = os.path.join(results_dir, f"demo_results_{model_type}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'mode': 'automatic',
            'timestamp': datetime.now().isoformat(),
            'queries': all_results
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print("=" * 60)
    print(f"\n  📊 Results saved to:")
    print(f"     • {output_path}")
    print(f"     • {json_path}")
    print()


def run_interactive(model_type, model, tokenizer, img_index, img_paths, results_dir, device):
    """Run interactive mode."""
    
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE")
    print("  Type a query to search, or 'quit' to exit")
    print("=" * 60)
    
    os.makedirs(results_dir, exist_ok=True)
    query_count = 0
    
    while True:
        try:
            query = input("\n  🔍 Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break
        
        if not query:
            print("     Please enter a query.")
            continue
            
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n  Goodbye!")
            break
        
        results = search(query, model, tokenizer, img_index, img_paths, device, top_k=5)
        
        print(f"\n  📸 Top {len(results)} results:")
        for r in results:
            marker = "✓" if r['rank'] == 1 else " "
            print(f"     {marker} {r['rank']}. {r['filename']} (score: {r['score']:.3f})")
        
        query_count += 1
        
        # Create visualization
        fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for ax, result in zip(axes, results):
            try:
                img = Image.open(result['path'])
                ax.imshow(img)
                color = '#2e7d32' if result['rank'] == 1 else '#666666'
                ax.set_title(f"#{result['rank']} | {result['score']:.3f}", 
                           fontsize=11, color=color, fontweight='bold')
            except:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
            ax.axis('off')
        
        plt.suptitle(f'Query: "{query}"', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Save
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]
        output_path = os.path.join(results_dir, f"query_{query_count}_{safe_query}_{model_type}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n     💾 Saved: {output_path}")
        
        plt.show()


def main():
    print_banner()
    
    # Default paths
    captions_json = "../data/captions.json"
    image_root = "../data/images/"
    checkpoint_dir = "../checkpoints/"
    results_dir = "../results/"
    
    # Get device
    device = get_device()
    print(f"\n  Device: {device}")
    
    # Menu 1: Select model
    model_type = show_model_menu()
    
    # Menu 2: Select mode
    mode = show_mode_menu()
    
    # Load data
    print("\n" + "-" * 40)
    print("  LOADING...")
    print("-" * 40)
    
    if not os.path.exists(captions_json):
        print(f"\n  ❌ Error: Captions file not found: {captions_json}")
        return
    
    with open(captions_json, "r") as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data)} captions")
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(Config.MAX_VOCAB_SIZE)
    tokenizer.build_vocab([d["caption"] for d in data])
    
    # Load model
    try:
        model = load_model(model_type, tokenizer, device, checkpoint_dir)
    except FileNotFoundError as e:
        print(f"\n  ❌ Error: {e}")
        return
    
    # Build image index
    img_index, img_paths, img_names = build_image_index(model, data, image_root, device)
    
    # Run selected mode
    if mode == "automatic":
        run_automatic(model_type, model, tokenizer, img_index, img_paths, data, results_dir, device)
    else:
        run_interactive(model_type, model, tokenizer, img_index, img_paths, results_dir, device)


if __name__ == "__main__":
    main()
