# QTSTransformer — File Comparison

This directory contains multiple versions of the Quantum Time-Series Transformer model (Park et al., 2025), based on the Quixer architecture (Khatri et al., 2024). Each file represents a different implementation strategy or set of improvements.

## Quick Reference

| File | Framework | Quantum Approach | Pos. Encoding | Angle Range | Postselection | Temporal Chunking |
|------|-----------|-----------------|---------------|-------------|---------------|-------------------|
| `QuixerTSModel.py` | TorchQuantum | Classical simulation | No | [0, 1] | N/A | No |
| `QTSTransformer.py` | PennyLane | Classical simulation | No | [0, 1] | N/A | No |
| `QTSTransformer_ver1.2.py` | PennyLane | Actual quantum circuits | No | [0, 1] | Yes | No |
| `QTSTransformer_v1_5.py` | PennyLane | Actual quantum circuits | Yes | [0, 2pi] | No | No |
| `QTSTransformer_v2.py` | PennyLane | Actual quantum circuits | Yes | [0, 2pi] | No | Yes |
| `QTSTransformer_v2_5.py` | PennyLane | Classical simulation | Yes | [0, 2pi] | N/A | No |

> **"Classical simulation"** refers to the shortcut described in Khatri et al. Section 4.1: StatePrep + state extraction + einsum to compute M|0> directly, without ancilla qubits.
>
> **"Actual quantum circuits"** means the model uses a single coherent QNode with ancilla register, PREPARE/SELECT/PCPhase operators — a real LCU+QSVT circuit.
>
> **"Postselection"** is only relevant for actual quantum circuits. Classical simulation implicitly postselects because it directly computes the polynomial P(M)|0> without ancilla. Actual circuits produce |0>\_anc x P(M)|0> + |perp>\_anc x |garbage>, and postselection is needed to discard the garbage component.

---

## Model Files (Detailed)

### QuixerTSModel.py — TorchQuantum Original

The original implementation using the TorchQuantum framework. Serves as the baseline reference matching both papers' experimental methodology.

- **Class**: `QuixerTimeSeries`
- **Quantum framework**: TorchQuantum (`tq.QuantumDevice`, `tq.GeneralEncoder`)
- **LCU**: Classical simulation via `qdev.set_states()` + encoder + `get_states_1d()` + `einsum`
- **QSVT**: Classical polynomial accumulation (`evaluate_polynomial_state`)
- **QFF**: TorchQuantum encoder with `tq.MeasureMultipleTimes`
- **Trainable quantum params**: `poly_coeffs` (degree+1), `mix_coeffs` (complex, n_timesteps), `qff_params`
- **Return**: `(output, mean_postselection_prob)` — returns both prediction and LCU success probability
- **Known issues**:
  - `qff_params` sized for `n_ansatz_layers` but QFF uses 1 layer (oversized)
  - `qff_params.repeat(1, bsz)` shape is incorrect on a 1D tensor
  - No `x.permute()` — expects input as `(batch, n_timesteps, feature_dim)` directly

### QTSTransformer.py — PennyLane Classical Simulation (v1)

PennyLane port of `QuixerTSModel.py`. Same classical simulation approach, with some bug fixes.

- **Class**: `QuantumTSTransformer`
- **Quantum framework**: PennyLane (`default.qubit`, backprop)
- **QNodes**: Two separate QNodes:
  - `_timestep_state_qnode`: `StatePrep` → sim14 → `qml.state()` (returns statevector)
  - `_qff_qnode_expval`: `StatePrep` → sim14 → `qml.expval(PauliX/Y/Z)`
- **LCU**: Vectorized — single QNode call for all batch*timesteps, einsum for LCU coefficients
- **QSVT**: Same classical polynomial iteration as TorchQuantum version
- **Fixes over QuixerTSModel.py**: `qff_n_rots = 4 * n_qubits * 1` (correct sizing), PennyLane handles broadcast
- **Return**: `output` only (no postselection probability)
- **Known issues**:
  - Sigmoid [0, 1] angle range limits circuit expressivity
  - No positional encoding — model is order-agnostic
  - `StatePrep` with potentially unnormalized states in higher-order polynomial iterations

### QTSTransformer_ver1.2.py — Actual Quantum Circuits + Postselection

First version to implement actual LCU+QSVT quantum circuits instead of classical simulation. Includes ancilla postselection fix.

- **Class**: `QuantumTSTransformer`
- **Architecture**: Single coherent QNode with main + ancilla registers
- **Qubit layout**: `n_qubits` main + `ceil(log2(n_timesteps))` ancilla
- **PREPARE**: Learnable RY+RZ+CNOT ansatz on ancilla (`prepare_params`)
- **SELECT**: `qml.Select(build_select_ops(), control=anc_wires)` with sim14 unitaries per timestep
- **QSVT**: `PCPhase(phi_0)` → loop of `[PREPARE · SELECT/SELECT† · PREPARE† · PCPhase(phi_k)]`
- **Postselection**: `qml.measure(q, postselect=0)` on each ancilla qubit after QSVT, before QFF
- **Trainable quantum params**: `prepare_params`, `signal_angles` (degree+1), `qff_params`
- **Known issues**:
  - Sigmoid [0, 1] angle range
  - No positional encoding
  - `build_select_ops()` rebuilt inside each QSVT degree iteration (redundant)

### QTSTransformer_v1_5.py — Actual Circuits + PE + 2pi Scaling

Adds positional encoding and full angle range to the actual quantum circuit approach. Introduces SELECT caching and cleaner adjoint syntax.

- **Class**: `QuantumTSTransformer`
- **Improvements over v1.2**:
  1. **Sinusoidal positional encoding** (Vaswani et al., 2017) — injected before feature projection
  2. **2pi angle scaling** — `sigmoid * 2pi` for full [0, 2pi] rotation range
  3. **SELECT operator caching** — `build_select_ops()` called once, reused across degree iterations
  4. **Cleaner adjoint syntax** — `qml.adjoint(qml.Select)(ops, ...)` instead of `qml.adjoint(qml.Select(ops, ...))`
- **Known issues**:
  - **No postselection** on ancilla register — expectation values include garbage component (regression from v1.2)

### QTSTransformer_v2.py — Actual Circuits + Temporal Chunking

Extends v1.5 with temporal chunking to handle long sequences without requiring a large ancilla register.

- **Class**: `QuantumTSTransformer`
- **New parameter**: `chunk_size` (optional, defaults to `n_timesteps`)
- **Chunking mechanism**: Splits the time sequence into non-overlapping windows of `chunk_size`. Each chunk is processed by its own QSVT+LCU pass with `ceil(log2(chunk_size))` ancilla qubits. Chunk results are aggregated via mean pooling.
- **Benefit**: Ancilla count scales with chunk size, not total sequence length (e.g., chunk_size=16 → 4 ancilla regardless of T)
- **Padding**: Last chunk is zero-padded to `chunk_size` if needed
- **Inherits from v1.5**: PE, 2pi scaling, SELECT caching, cleaner adjoint
- **Known issues**:
  - **No postselection** (same as v1.5)
  - Zero-padding the last chunk may introduce artifacts (zeros → identity-like rotations after sigmoid)

### QTSTransformer_v2_5.py — Classical Simulation + PE + 2pi Scaling

Applies the v1.5 classical-side improvements (PE, 2pi scaling) to the original classical simulation approach. Useful for comparing whether PE and angle range improvements help independently of the quantum circuit architecture change.

- **Class**: `QuantumTSTransformer`
- **Base**: Same classical simulation as `QTSTransformer.py` (v1)
- **Improvements over v1**:
  1. **Sinusoidal positional encoding** (Vaswani et al., 2017)
  2. **2pi angle scaling** — `sigmoid * 2pi` for full [0, 2pi] rotation range
- **Same quantum mechanism as v1**: Two QNodes (StatePrep + state extraction + einsum), no ancilla

---

## Version Lineage

```
QuixerTSModel.py (TorchQuantum, classical sim)
  |
  v  PennyLane port
  |
QTSTransformer.py [v1] (PennyLane, classical sim)
  |
  |-----> QTSTransformer_v2_5.py [v2.5]
  |         + Positional encoding
  |         + 2pi angle scaling
  |         (still classical simulation)
  |
  v  Switch to actual quantum circuits
  |
QTSTransformer_ver1.2.py [v1.2]
  |   + Actual LCU (PREPARE/SELECT/PREPARE†)
  |   + Actual QSVT (PCPhase signal processing)
  |   + Ancilla postselection (qml.measure postselect=0)
  |
  v  Classical-side improvements
  |
QTSTransformer_v1_5.py [v1.5]
  |   + Positional encoding
  |   + 2pi angle scaling
  |   + SELECT caching
  |   - Lost postselection (regression)
  |
  v  Scalability
  |
QTSTransformer_v2.py [v2.0]
      + Temporal chunking (chunk_size parameter)
      + Mean pooling across chunks
      - Still no postselection
```

---

## Support Files

| File | Description |
|------|-------------|
| `QTSTransformer_PhysioNet_EEG.py` | Training script for PhysioNet EEG binary classification. Imports `QTSTransformer_v1_5`. Includes train/validate/test loops, early stopping, checkpointing, and metrics (accuracy, AUC). |
| `Load_PhysioNet_EEG.py` | Data loader for PhysioNet EEG Motor Imagery dataset. Subject-level train/val/test split (70/15/15). Loads left/right hand motor imagery trials, resamples to target frequency, returns DataLoaders. |
| `QTSTransformer_Model_Description.md` | Detailed model description for v1.5 with code walkthrough. |
| `README_QTSTransformer_Example.md` | Quick-start guide for running the PhysioNet EEG example. |

---

## Feature Comparison

### Quantum Architecture

| Feature | Classical Sim (v1, v2.5) | Actual Circuits (v1.2, v1.5, v2) |
|---------|-------------------------|----------------------------------|
| Ancilla qubits | None | ceil(log2(n_timesteps)) |
| LCU mechanism | einsum over evolved states | PREPARE · SELECT · PREPARE† |
| QSVT mechanism | Polynomial accumulation loop | PCPhase + alternating block encoding |
| QFF | Separate QNode with StatePrep | Same coherent QNode |
| Trainable LCU params | `mix_coeffs` (complex, T) | `prepare_params` (RY+RZ on ancilla) |
| Trainable QSVT params | `poly_coeffs` (real, degree+1) | `signal_angles` (real, degree+1) |
| Hardware compatible | No (uses StatePrep + state extraction) | Yes (single coherent circuit) |

### Classical Preprocessing

| Feature | v1 | v1.2 | v1.5 | v2 | v2.5 |
|---------|-----|------|------|-----|------|
| Positional encoding | - | - | Sinusoidal | Sinusoidal | Sinusoidal |
| Angle range | [0, 1] | [0, 1] | [0, 2pi] | [0, 2pi] | [0, 2pi] |
| Input permute | Yes | Yes | Yes | Yes | Yes |
| Dropout | Yes | Yes | Yes | Yes | Yes |

### Known Issues by Version

| Issue | v1 | v1.2 | v1.5 | v2 | v2.5 |
|-------|-----|------|------|-----|------|
| No positional encoding | X | X | - | - | - |
| Sigmoid [0,1] angles | X | X | - | - | - |
| Missing postselection | N/A | - | X | X | N/A |
| SELECT rebuilt per degree iter | N/A | X | - | - | N/A |
| StatePrep normalization risk | X | N/A | N/A | N/A | X |

> `-` = fixed/not present, `X` = present, `N/A` = not applicable

---

## References

- Park, J. J., Seo, J., Bae, S., Chen, S. Y.-C., Tseng, H.-H., Cha, J., & Yoo, S. (2025). Resting-state fMRI Analysis using Quantum Time-series Transformer. *2025 IEEE International Conference on Quantum Computing and Engineering (QCE)*. arXiv:2509.00711
- Khatri, N., Matos, G., Coopmans, L., & Clark, S. (2024). Quixer: A Quantum Transformer Model. arXiv:2406.04305
- Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019). Expressibility and Entangling Capability of Parameterized Quantum Circuits for Hybrid Quantum-Classical Algorithms. *Advanced Quantum Technologies*, 2(12), 1900070.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *NeurIPS*, 30.
