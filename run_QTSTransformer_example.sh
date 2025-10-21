#!/bin/bash
# Simple script to run the Quantum Time-series Transformer toy example
# This can be run directly on a login node or submitted as a batch job

# Activate the conda environment
source activate ./conda-envs/qml_eeg

# Run with default settings (8 qubits, 2 layers, degree 3)
echo "Running Quantum Time-series Transformer on PhysioNet EEG..."
echo "Model: 8 qubits, 2 layers, degree 3"
echo "Data: 50 subjects, sampling frequency 16 Hz"
echo ""

python QTSTransformer_PhysioNet_EEG.py \
    --n-qubits=8 \
    --n-layers=2 \
    --degree=3 \
    --dropout=0.1 \
    --n-epochs=50 \
    --batch-size=32 \
    --lr=1e-3 \
    --wd=1e-5 \
    --patience=10 \
    --sampling-freq=16 \
    --sample-size=50 \
    --seed=2025 \
    --job-id=toy_example

echo ""
echo "Done! Check ./checkpoints/ for results."
