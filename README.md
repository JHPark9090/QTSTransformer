# Quantum Time-series Transformer v1.5 - Toy Example

This is a simple toy example demonstrating how to run the Quantum Time-series Transformer v1.5 model on the PhysioNet EEG Motor Imagery dataset for binary classification.

## Overview

**Model**: Quantum Time-series Transformer v1.5 (QTSTransformer_v1_5)
- Genuine quantum QSVT via LCU with PREPARE-SELECT protocol and PCPhase signal-processing angles
- Sinusoidal positional encoding (Vaswani et al., 2017)
- 2π angle scaling for full rotation range
- Single coherent QNode (QSVT + QFF + measurement in one circuit)
- Implements the sim14 ansatz circuit from Sim et al. (2019)
- 8 qubits, 2 ansatz layers, degree 3 QSVT

**Task**: Binary classification of EEG motor imagery
- Left hand vs. Right hand motor imagery
- PhysioNet EEG Motor Imagery dataset
- 64 EEG channels, downsampled to 16 Hz

## Files

- `QTSTransformer_PhysioNet_EEG.py` - Main training script
- `QTSTransformer_v1_5.py` - Quantum Time-series Transformer v1.5 model
- `Load_PhysioNet_EEG.py` - Data loader for PhysioNet EEG dataset
- `run_QTSTransformer_example.sh` - Simple script to run the example

## Quick Start

### Method 1: Using the run script
```bash
bash run_QTSTransformer_example.sh
```

### Method 2: Direct Python command
```bash
# Activate environment
conda activate ./conda-envs/qml_eeg

# Run with default settings
python QTSTransformer_PhysioNet_EEG.py
```

### Method 3: Custom hyperparameters
```bash
python QTSTransformer_PhysioNet_EEG.py \
    --n-qubits=6 \
    --n-layers=2 \
    --degree=2 \
    --n-epochs=30 \
    --batch-size=16 \
    --sample-size=20
```

## Training Flow

The script follows a clear training loop:

```python
for each epoch:
    1. train(model, train_loader, ...)        # Train on training set
    2. validate(model, val_loader, ...)       # Validate on validation set
    3. Save best model based on validation AUC
    4. Check early stopping

After training:
    test(model, test_loader, ...)             # Final test on test set
```

## Arguments

### Model Hyperparameters
- `--n-qubits` (default: 8) - Number of qubits in quantum circuit
- `--n-layers` (default: 2) - Number of ansatz layers
- `--degree` (default: 3) - Degree of QSVT polynomial
- `--dropout` (default: 0.1) - Dropout rate

### Training Hyperparameters
- `--n-epochs` (default: 100) - Number of training epochs
- `--batch-size` (default: 32) - Batch size
- `--lr` (default: 1e-3) - Learning rate
- `--wd` (default: 1e-5) - Weight decay
- `--patience` (default: 20) - Early stopping patience

### Data Hyperparameters
- `--sampling-freq` (default: 16) - EEG sampling frequency in Hz
- `--sample-size` (default: 50) - Number of subjects to load (1-109)

### Experiment Settings
- `--seed` (default: 2025) - Random seed for reproducibility
- `--job-id` (default: "QTS_PhysioNet") - Job ID for saving checkpoints
- `--resume` - Resume from checkpoint (flag)

## Output

Results are saved in `./checkpoints/`:
- `QTS_PhysioNet_<job_id>.pt` - Best model checkpoint
- `training_logs_<job_id>.csv` - Training/validation metrics per epoch

### Checkpoint Contents
- Model state dict
- Optimizer state dict
- Best validation AUC
- Training/validation metrics

## Example Output

```
================================================================================
Loading PhysioNet EEG Dataset...
================================================================================
Subjects in Training Set: 35
Subjects in Validation Set: 7
Subjects in Test Set: 8

Training set shape: (630, 64, 497)
Validation set shape: (126, 64, 497)
Test set shape: (144, 64, 497)

Input dimensions: 630 trials, 64 channels, 497 timesteps
Feature dimension: 64
Sequence length: 497

================================================================================
Initializing Quantum Time-series Transformer...
================================================================================
Model parameters: 12,345
Qubits: 8, Layers: 2, Degree: 3

================================================================================
Starting Training...
================================================================================

Epoch: 001/100 | Time: 2m 15s
  Train Loss: 0.6543 | Train Acc: 0.6234 | Train AUC: 0.6789
  Val   Loss: 0.6234 | Val   Acc: 0.6543 | Val   AUC: 0.7012
  *** New best model saved! Val AUC: 0.7012 ***

...

================================================================================
Final Evaluation on Test Set...
================================================================================

Test Results:
  Test Loss: 0.5876
  Test Accuracy: 0.7153
  Test AUC: 0.7845

================================================================================
Training Complete!
Best Validation AUC: 0.7234
Final Test AUC: 0.7845
================================================================================
```

## Notes

- First run will download the PhysioNet dataset (~500 MB)
- GPU is automatically used if available
- Training time depends on number of qubits, layers, and dataset size
- For quick testing, use smaller `--sample-size` (e.g., 10 subjects)
