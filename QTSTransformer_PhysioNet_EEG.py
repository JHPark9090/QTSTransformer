import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
import time

# Import custom modules
from QTSTransformer_v2_5 import QuantumTSTransformer
from Load_PhysioNet_EEG import load_eeg_ts_revised
from logger import TrainingLogger

# Import metrics
from sklearn.metrics import accuracy_score, roc_auc_score


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quantum Time-series Transformer on PhysioNet EEG")

    # Model hyperparameters
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of ansatz layers")
    parser.add_argument("--degree", type=int, default=3, help="Degree of QSVT polynomial")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training hyperparameters
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    # Data hyperparameters
    parser.add_argument("--sampling-freq", type=int, default=16, help="EEG sampling frequency (Hz)")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of subjects to load (1-109)")

    # Experiment settings
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--job-id", type=str, default="QTS_PhysioNet", help="Job ID for logging")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")

    return parser.parse_args()


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate elapsed time in minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_loader, optimizer, criterion, device):
    """Train the model on the training set."""
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Store predictions and labels for metrics
        preds = torch.sigmoid(output).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)
    binary_preds = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, binary_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    avg_loss = epoch_loss / len(train_loader)

    return avg_loss, accuracy, auc


def validate(model, val_loader, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            epoch_loss += loss.item()

            # Store predictions and labels
            preds = torch.sigmoid(output).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)
    binary_preds = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, binary_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    avg_loss = epoch_loss / len(val_loader)

    return avg_loss, accuracy, auc


def test(model, test_loader, criterion, device):
    """Test the model on the test set."""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=False):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            epoch_loss += loss.item()

            # Store predictions and labels
            preds = torch.sigmoid(output).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)
    binary_preds = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, binary_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    avg_loss = epoch_loss / len(test_loader)

    return avg_loss, accuracy, auc


def main():
    # Parse arguments
    args = get_args()

    # Set random seeds
    set_all_seeds(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load PhysioNet EEG data
    print("\n" + "="*80)
    print("Loading PhysioNet EEG Dataset...")
    print("="*80)
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        sampling_freq=args.sampling_freq,
        sample_size=args.sample_size
    )

    n_trials, n_channels, n_timesteps = input_dim
    print(f"\nInput dimensions: {n_trials} trials, {n_channels} channels, {n_timesteps} timesteps")
    print(f"Feature dimension: {n_channels}")
    print(f"Sequence length: {n_timesteps}")

    # Initialize model
    print("\n" + "="*80)
    print("Initializing Quantum Time-series Transformer...")
    print("="*80)
    model = QuantumTSTransformer(
        n_qubits=args.n_qubits,
        n_timesteps=n_timesteps,
        degree=args.degree,
        n_ansatz_layers=args.n_layers,
        feature_dim=n_channels,
        output_dim=1,  # Binary classification
        dropout=args.dropout,
        device=device
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Qubits: {args.n_qubits}, Layers: {args.n_layers}, Degree: {args.degree}")

    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.BCEWithLogitsLoss()

    # Setup checkpoint directory
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"QTS_PhysioNet_{args.job_id}.pt")

    # Setup logger
    logger = TrainingLogger(save_dir=checkpoint_dir, job_id=args.job_id)

    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0

    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resuming from epoch {start_epoch}, Best Val AUC: {best_val_auc:.4f}")

    # Training loop
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)

    for epoch in range(start_epoch, args.n_epochs):
        start_time = time.time()

        # Train on training set
        train_loss, train_acc, train_auc = train(
            model, train_loader, optimizer, criterion, device
        )

        # Validate on validation set
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Print epoch results
        print(f"\nEpoch: {epoch+1:03d}/{args.n_epochs} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   AUC: {val_auc:.4f}")

        # Log to CSV (using accuracy as RMSE placeholder for compatibility)
        logger.logging_per_epochs(
            epoch=epoch,
            train_rmse=train_acc,  # Using accuracy instead of RMSE
            train_loss=train_loss,
            val_rmse=val_acc,
            val_loss=val_loss
        )

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'patience_counter': patience_counter,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'val_auc': val_auc,
            }, checkpoint_path)

            print(f"  *** New best model saved! Val AUC: {best_val_auc:.4f} ***")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs (patience: {args.patience})")
            break

    # Final evaluation on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set...")
    print("="*80)

    # Load best model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_auc = test(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
