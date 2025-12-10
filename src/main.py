"""
MiniCLIP Training Script
========================
Usage:
    cd src
    python main.py --model_type lstm
    python main.py --model_type transformer
"""

import os
import json
import math
import argparse

import torch
from torch.utils.data import DataLoader

from config import Config
from model import MiniCLIP, ImageEncoder, LSTMTextEncoder, TransformerTextEncoder
from utils import (
    set_seed, SimpleTokenizer, MiniCLIPDataset, collate_fn,
    get_train_transforms, get_val_transforms, load_data, split_data_by_images,
    compute_retrieval_metrics, evaluate_batch_metrics,
    compute_random_baseline, compute_bow_baseline,
    visualize_retrieval, plot_training_curves,
    plot_similarity_heatmap, plot_embedding_space, plot_model_comparison,
)


def train_one_epoch(model, loader, optimizer, scheduler, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for imgs, txts, lens, caps, fnames in loader:
        imgs = imgs.to(device)
        txts = txts.to(device)
        lens = lens.to(device)

        optimizer.zero_grad()
        _, _, loss = model(imgs, txts, lens)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train MiniCLIP model")
    parser.add_argument("--captions_json", type=str, default="../data/captions.json")
    parser.add_argument("--image_root", type=str, default="../data/images/")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/")
    parser.add_argument("--results_dir", type=str, default="../results/")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    set_seed(Config.SEED)
    device = Config.get_device()
    
    print("=" * 60)
    print("MiniCLIP: Text-Image Retrieval")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model_type}")
    print("=" * 60)
    
    # Load data
    data = load_data(args.captions_json)
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(Config.MAX_VOCAB_SIZE)
    tokenizer.build_vocab([d["caption"] for d in data])
    
    # Split data
    train_data, val_data, test_data = split_data_by_images(data)
    
    # Create datasets
    train_transform = get_train_transforms(Config.IMAGE_SIZE)
    val_transform = get_val_transforms(Config.IMAGE_SIZE)
    
    train_ds = MiniCLIPDataset(train_data, args.image_root, tokenizer, transform=train_transform)
    val_ds = MiniCLIPDataset(val_data, args.image_root, tokenizer, transform=val_transform)
    test_ds = MiniCLIPDataset(test_data, args.image_root, tokenizer, transform=val_transform)
    
    # Create data loaders
    use_pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    
    # Baselines
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)
    
    n_test_images = len(set(item["file_name"] for item in test_data))
    random_baseline = compute_random_baseline(n_test_images)
    print(f"Random baseline (N={n_test_images}): R@1={random_baseline['R@1']:.2f}%")
    
    bow_baseline = compute_bow_baseline(test_ds, tokenizer)
    print(f"BoW baseline: R@1={bow_baseline['R@1']:.2f}%")
    
    # Initialize model
    print("\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)
    
    img_enc = ImageEncoder(embed_dim=Config.EMBED_DIM, use_pretrained=Config.USE_PRETRAINED, dropout=Config.DROPOUT)
    
    if args.model_type == "lstm":
        txt_enc = LSTMTextEncoder(vocab_size=tokenizer.vocab_size, embed_dim=Config.EMBED_DIM,
                                  word_embed_dim=Config.WORD_EMBED_DIM, hidden_dim=Config.LSTM_HIDDEN,
                                  num_layers=Config.LSTM_LAYERS, dropout=Config.DROPOUT)
    else:
        txt_enc = TransformerTextEncoder(vocab_size=tokenizer.vocab_size, embed_dim=Config.EMBED_DIM,
                                         word_embed_dim=Config.WORD_EMBED_DIM, num_heads=Config.TRANSFORMER_HEADS,
                                         num_layers=Config.TRANSFORMER_LAYERS, dropout=Config.DROPOUT)
    
    model = MiniCLIP(img_enc, txt_enc, init_temperature=Config.INIT_TEMPERATURE)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY, betas=(0.9, 0.98))
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * Config.WARMUP_EPOCHS
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    print("\n" + "=" * 60)
    print(f"TRAINING ({args.model_type.upper()})")
    print("=" * 60)
    
    best_val_r1 = 0.0
    train_losses = []
    val_r1s = []
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, Config.MAX_GRAD_NORM)
        val_metrics = evaluate_batch_metrics(model, val_loader, device)
        val_r1 = val_metrics["R@1"]
        
        train_losses.append(train_loss)
        val_r1s.append(val_r1)
        
        current_lr = optimizer.param_groups[0]['lr']
        temp = model.logit_scale.exp().item()
        
        print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | Val R@1: {val_r1:.2f}% | LR: {current_lr:.2e} | Temp: {temp:.2f}")
        
        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_{args.model_type}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r1': val_r1,
                'config': {'embed_dim': Config.EMBED_DIM, 'vocab_size': tokenizer.vocab_size, 'model_type': args.model_type}
            }, checkpoint_path)
            print(f"  → Saved best model (R@1: {val_r1:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Plot training curves
    plot_training_curves(train_losses, val_r1s, args.model_type, args.results_dir)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_{args.model_type}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = compute_retrieval_metrics(model, test_loader, device)
    
    print(f"\nTest Results ({args.model_type.upper()}):")
    print(f"  R@1:  {test_metrics['R@1']:.2f}%")
    print(f"  R@5:  {test_metrics['R@5']:.2f}%")
    print(f"  R@10: {test_metrics['R@10']:.2f}%")
    print(f"  MRR:  {test_metrics['MRR']:.4f}")
    print(f"  MedR: {test_metrics['MedR']:.1f}")
    
    improvement = test_metrics['R@1'] / max(random_baseline['R@1'], 0.01)
    print(f"\n🎯 {improvement:.1f}x improvement over random!")
    
    # Save metrics
    results = {
        "model_type": args.model_type,
        "test_metrics": test_metrics,
        "baselines": {"random": random_baseline, "bow": bow_baseline},
        "training": {"best_epoch": checkpoint['epoch'], "best_val_r1": checkpoint['val_r1'], "final_train_loss": train_losses[-1]},
        "config": Config.to_dict(),
    }
    
    metrics_path = os.path.join(args.results_dir, f"metrics_{args.model_type}.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Metrics to {metrics_path}")
    
    # Visualize retrieval
    vis_path = os.path.join(args.results_dir, f"retrieval_results_{args.model_type}.png")
    try:
        visualize_retrieval(model, test_ds, device, num_queries=5, top_k=5, save_path=vis_path)
    except Exception as e:
        print(f"[Warning] Could not create retrieval visualization: {e}")
    
    # Similarity heatmap
    heatmap_path = os.path.join(args.results_dir, f"similarity_heatmap_{args.model_type}.png")
    try:
        plot_similarity_heatmap(model, test_ds, device, num_samples=10, save_path=heatmap_path)
    except Exception as e:
        print(f"[Warning] Could not create similarity heatmap: {e}")
    
    # Embedding space visualization
    embed_path = os.path.join(args.results_dir, f"embedding_space_{args.model_type}.png")
    try:
        plot_embedding_space(model, test_ds, device, num_samples=30, save_path=embed_path)
    except Exception as e:
        print(f"[Warning] Could not create embedding space plot: {e}")
    
    # Model comparison (only if both models exist)
    try:
        plot_model_comparison(save_dir=args.results_dir)
    except Exception as e:
        print(f"[Warning] Could not create model comparison: {e}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()