# Quantum Time-Series Transformer v1.5 (PennyLane Implementation)
This repository contains a PyTorch and PennyLane implementation of a quantum time-series transformer model developed by Park et al. (2025). The quantum time-series transformer is based on the Quixer model architecture proposed by Khatri et al. (2024). This model is designed to process time-series data by leveraging quantum computing principles for embedding, mixing, and non-linear transformation of temporal features.

## v1.5 Improvements over v1
1. **Sinusoidal Positional Encoding** (Vaswani et al., 2017) added to the input sequence
2. **2π angle scaling**: `sigmoid * 2π` for full `[0, 2π]` rotation range (v1 used sigmoid alone)
3. **Single coherent QNode**: The entire QSVT + QFF + measurement pipeline runs in one quantum circuit (v1 used multiple separate QNodes with classical state-vector manipulation in between)
4. **Genuine quantum LCU**: Uses `qml.Select` and a learnable PREPARE unitary on an ancilla register (v1 simulated LCU classically via `torch.einsum`)
5. **Genuine quantum QSVT**: Uses `qml.PCPhase` signal-processing angles with alternating SELECT/SELECT† blocks (v1 simulated QSVT via classical polynomial accumulation)
6. **SELECT operator caching**: `build_select_ops()` is called once per forward pass and reused across QSVT degree iterations

# Core Concepts
The model's architecture can be broken down into four key stages:

- **Unitary Temporal Embedding**: The feature vector at each point in a time sequence is mapped to a unique quantum circuit (a unitary matrix), creating a quantum representation of that specific moment's data.

- **Time Sequence Mixing (LCU)**: A Linear Combination of Unitaries (LCU) encodes all time-step unitaries ($U_j$) into a single block-encoded operator using a quantum PREPARE-SELECT protocol on an ancilla register. This serves as a quantum-native attention-like mechanism to mix information across the time sequence.

$$M = \sum^{n-1}_{j=0} |\alpha_j|^2 U_j$$

- **Non-linearity (QSVT)**: Quantum Singular Value Transformation (QSVT) applies a non-linear polynomial transformation to the block-encoded operator $M$ via alternating signal-processing phase rotations (`PCPhase`) and LCU block-encoding steps. This allows the model to capture richer, higher-order interactions between different time-steps.

$$P_d(M) \sim \prod_{k=0}^{d} \bigl[\text{PCPhase}(\varphi_k) \cdot \text{LCU-block}\bigr]$$

- **Readout and Classical Processing**: The final quantum state is measured to extract classical features (Pauli expectation values). These features are then processed by a classical feed-forward neural network to produce the final prediction.

# Code Implementation Breakdown
This section explains how each conceptual step is realized in the QTSTransformer_v1_5.py code.

## Qubit Layout

The circuit uses two registers:

- **Main register**: `n_qubits` wires for data processing
- **Ancilla register**: `ceil(log2(n_timesteps))` wires for LCU SELECT control

```
# QuantumTSTransformer -> __init__()
self.n_ancilla = ceil(log2(max(n_timesteps, 2)))
self.main_wires = list(range(n_qubits))
self.anc_wires = list(range(n_qubits, n_qubits + self.n_ancilla))
self.total_wires = n_qubits + self.n_ancilla
```

## 1) Unitary Temporal Embedding
**Concept**: Classical features for each time-step are mapped to a unique Parameterized Quantum Circuit (PQC), which defines a unitary transformation.

### Realization in the Code:

This process involves three main parts:

**1-1) Sinusoidal Positional Encoding (__init__ and forward methods)**

Before projection, sinusoidal positional encoding (Vaswani et al., 2017) is added to inject temporal position information into the input.
```
# QuantumTSTransformer -> __init__()
pe = torch.zeros(n_timesteps, feature_dim)
pos = torch.arange(n_timesteps).unsqueeze(1).float()
div = torch.exp(torch.arange(0, feature_dim, 2).float()
                * -(math.log(10000.0) / feature_dim))
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div[:feature_dim // 2])
self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, feature_dim)

# QuantumTSTransformer -> forward()
x = x.permute(0, 2, 1)              # (batch, T, C)
x = x + self.pe[:, :x.size(1)]      # positional encoding
```

**1-2) Classical Projection with 2π Scaling (forward method)**

The classical input feature vector for each time-step is passed through a `torch.nn.Linear` layer to generate rotation angles, then scaled to the full `[0, 2π]` range via `sigmoid * 2π`.
```
# QuantumTSTransformer -> forward()
x = self.feature_projection(self.dropout(x))  # (batch, T, n_rots)
ts_params = self.rot_sigm(x) * (2 * math.pi)  # sigmoid -> [0, 2pi]
```
- `self.feature_projection`: The learnable linear layer mapping `feature_dim → n_rots`.
- `ts_params`: The resulting tensor holding the gate parameters for every time-step in the batch, with values in `[0, 2π]`.

**1-3) Quantum Circuit Definition (sim14_circuit function)**

This function defines the fixed structure of the PQC, arranging RY and CRX gates. When parameterized by `ts_params`, it defines a unique unitary matrix $U_t$ for that time-step.
```
def sim14_circuit(params, wires, layers=1):
    # Gate sequence per layer: RY -> CRX(ring) -> RY -> CRX(counter-ring)
    # Parameters per layer: 4 * wires
```

**1-4) SELECT Operator Construction (_circuit → build_select_ops)**

Each time-step's sim14 unitary is wrapped as a `qml.prod` operator and collected into a list for `qml.Select`. The list is padded to `2^n_ancilla` with Identity operators.
```
# _circuit -> build_select_ops()
select_ops = []
for t in range(_n_timesteps):
    gates = []
    # ... build sim14 gates for timestep t using ts_params[..., t, :] ...
    select_ops.append(qml.prod(*reversed(gates)))
# Pad to 2^n_ancilla with Identity
while len(select_ops) < _n_select_ops:
    select_ops.append(qml.Identity(wires=_main_wires[0]))
return select_ops
```
This list is built **once** per forward pass and reused across all QSVT degree iterations.

## 2) Time Sequence Mixing via LCU
**Concept**: The model uses a quantum PREPARE-SELECT protocol to create a block-encoded superposition of all time-step unitaries on the main register, controlled by an ancilla register.

### Realization in the Code:

**2-1) Learnable PREPARE Unitary (_circuit → prepare)**

A trainable unitary $V$ is applied to the ancilla register, creating a superposition that determines the weighting of each time-step unitary in the LCU.
```
# _circuit -> prepare()
def prepare():
    for ly in range(_n_prep_layers):
        for qi, q in enumerate(_anc_wires):
            qml.RY(prep_p[ly, qi, 0], wires=q)
            qml.RZ(prep_p[ly, qi, 1], wires=q)
        for i in range(_n_ancilla - 1):
            qml.CNOT(wires=[_anc_wires[i], _anc_wires[i + 1]])
```
- `self.prepare_params`: Trainable parameter tensor of shape `(n_prep_layers, n_ancilla, 2)` for RY and RZ rotations.

**2-2) SELECT Application (_circuit)**

`qml.Select` applies the $t$-th time-step unitary $U_t$ on the main register, conditioned on the ancilla register being in state $|t\rangle$. Combined with the PREPARE unitary, this realizes the LCU block encoding.
```
# _circuit
qml.Select(select_ops, control=_anc_wires)
```

## 3) Non-linearity via QSVT

**Concept**: A polynomial transformation $P_d(M)$ is applied to the block-encoded LCU operator $M$ through alternating signal-processing phase rotations (`PCPhase`) and LCU block-encoding steps (PREPARE → SELECT → PREPARE†). The signal-processing angles $\varphi_k$ are learnable parameters that determine the polynomial being applied.

### Realization in the Code:

**3-1) Trainable Signal-Processing Angles (__init__ method)**

The QSVT signal-processing angles $(\varphi_0, \varphi_1, \ldots, \varphi_d)$ are defined as a learnable parameter.
```
# QuantumTSTransformer -> __init__()
self.signal_angles = torch.nn.Parameter(
    0.1 * torch.randn(degree + 1))
```

**3-2) QSVT Circuit (_circuit)**

The main QSVT loop alternates between `PCPhase` rotations and LCU block-encoding steps (PREPARE → SELECT/SELECT† → PREPARE†). Even-indexed iterations use SELECT, odd-indexed use adjoint(SELECT).
```
# _circuit
select_ops = build_select_ops()  # build once, reuse

qml.PCPhase(sig_ang[0], dim=_pcphase_dim, wires=_pcphase_wires)

for k in range(_degree):
    prepare()
    if k % 2 == 0:
        qml.Select(select_ops, control=_anc_wires)
    else:
        qml.adjoint(qml.Select)(select_ops, control=_anc_wires)
    qml.adjoint(prepare)()
    qml.PCPhase(sig_ang[k + 1], dim=_pcphase_dim,
                wires=_pcphase_wires)
```
- `PCPhase(φ, dim, wires)`: Applies a phase of $e^{i\varphi}$ to the first `dim` basis states and $e^{-i\varphi}$ to the rest, acting on the combined ancilla+main register.
- The alternating SELECT / adjoint(SELECT) pattern is standard in QSVT constructions for block-encoded matrices.

## 4) Readout and Classical Processing
**Concept**: After the QSVT polynomial transformation, a final trainable quantum feed-forward (QFF) layer is applied to the main register, followed by Pauli expectation value measurements. These classical features are then processed by a feed-forward network.

### Realization in the Code:

**4-1) QFF and Measurement (_circuit)**

Within the same coherent QNode, a final sim14 circuit is applied to the main register, followed by PauliX, PauliY, and PauliZ measurements on each main qubit.
```
# _circuit

# QFF on main register
sim14_circuit(qff_p, wires=_n_qubits, layers=1)

# Measure PauliX/Y/Z on main register
observables = (
    [qml.PauliX(i) for i in _main_wires] +
    [qml.PauliY(i) for i in _main_wires] +
    [qml.PauliZ(i) for i in _main_wires])
return [qml.expval(op) for op in observables]
```
- `self.qff_params`: Trainable parameters for the QFF sim14 circuit.
- The measurement yields `3 * n_qubits` real-valued expectation values.

**4-2) Classical Output Layer (forward method)**

The expectation values are stacked and passed through the final linear layer.
```
# QuantumTSTransformer -> forward()
exps = self._circuit(
    ts_params, self.prepare_params,
    self.signal_angles, self.qff_params)
exps = torch.stack(exps, dim=1).float()
return self.output_ff(exps).squeeze(1)
```
- `self.output_ff`: The final `torch.nn.Linear` layer that maps the `3 * n_qubits` measurement outcomes to the desired `output_dim`.

# Reference
- Park, J. J., Seo, J., Bae, S., Chen, S. Y.-C., Tseng, H.-H., Cha, J., & Yoo, S. (2025, Aug 31 - Sep 5). Resting-state fMRI Analysis using Quantum Time-series Transformer. _2025 IEEE International Conference on Quantum Computing and Engineering (QCE)_, Albuquerque, New Mexico, USA. https://doi.org/10.48550/arXiv.2509.00711
- Khatri, N., Matos, G., Coopmans, L., & Clark, S. (2024). Quixer: A Quantum Transformer Model. _arXiv:2406.04305_.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. _Advances in Neural Information Processing Systems (NeurIPS)_, 30.
