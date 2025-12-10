# Checkpoints Directory

Place trained model weights here:
- `best_model_lstm.pth` - LSTM text encoder model
- `best_model_transformer.pth` - Transformer text encoder model

## Download Pre-trained Models

Download from Google Drive: [Pre-trained Models](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)

## Checkpoint Format

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Training epoch
- `val_r1`: Best validation R@1 score
- `config`: Model configuration
