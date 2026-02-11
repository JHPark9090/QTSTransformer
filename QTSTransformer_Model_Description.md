# Quantum Time-Series Transformer v2.5 (PennyLane Implementation)
This repository contains a PyTorch and PennyLane implementation of a quantum time-series transformer model developed by Park et al. (2025). The quantum time-series transformer is based on the Quixer model architecture proposed by Khatri et al. (2024). This model is designed to process time-series data by leveraging quantum computing principles for embedding, mixing, and non-linear transformation of temporal features.

v2.5 uses the classical simulation approach described in Khatri et al. Section 4.1, where the LCU and QSVT are computed via direct statevector manipulation rather than actual quantum circuits with ancilla registers.

## v2.5 Improvements over v1
1. **Sinusoidal Positional Encoding** (Vaswani et al., 2017) added to the input sequence
2. **2π angle scaling**: `sigmoid * 2π` for full `[0, 2π]` rotation range (v1 used sigmoid alone, limiting angles to `[0, 1]`)

# Core Concepts
The model's architecture can be broken down into four key stages:

- **Unitary Temporal Embedding**: The feature vector at each point in a time sequence is mapped to a unique quantum circuit (a unitary matrix), creating a quantum representation of that specific moment's data.

- **Time Sequence Mixing (LCU)**: A Linear Combination of Unitaries (LCU) combines all time-step unitaries ($U_j$) with learnable complex coefficients into a single mixed state. This is computed via classical simulation: each timestep unitary is applied to the working state via a QNode, and the results are combined with `torch.einsum`. This serves as an attention-like mechanism to mix information across the time sequence.

$$M|\psi\rangle = \sum^{n-1}_{j=0} b_j \, U_j|\psi\rangle$$

- **Non-linearity (QSVT)**: The QSVT polynomial transformation is classically simulated by iteratively applying the LCU operator $M$ to the working state $d$ times (where $d$ is the polynomial degree), accumulating a weighted sum with learnable polynomial coefficients.

$$P_d(M)|0\rangle = \sum_{k=0}^{d} c_k \, M^k |0\rangle$$

- **Readout and Classical Processing**: The final quantum state is prepared via `StatePrep`, processed through a quantum feed-forward (QFF) circuit, and measured to extract Pauli expectation values. These features are then processed by a classical feed-forward neural network to produce the final prediction.

# Code Implementation Breakdown
This section explains how each conceptual step is realized in the QTSTransformer_v2_5.py code.

## 1) Unitary Temporal Embedding
**Concept**: Classical features for each time-step are mapped to a unique Parameterized Quantum Circuit (PQC), which defines a unitary transformation.

### Realization in the Code:

This process involves three main parts:

**1-1) Sinusoidal Positional Encoding (__init__ and forward methods)**

Before projection, sinusoidal positional encoding (Vaswani et al., 2017) is added to inject temporal position information into the input. Without positional encoding, the model would be order-agnostic and unable to distinguish timestep order.
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
x = self.feature_projection(self.dropout(x))       # (batch, T, n_rots)
timestep_params = self.rot_sigm(x) * (2 * math.pi) # sigmoid -> [0, 2pi]
```
- `self.feature_projection`: The learnable linear layer mapping `feature_dim → n_rots`.
- `timestep_params`: The resulting tensor holding the gate parameters for every time-step in the batch, with values in `[0, 2π]`.

**1-3) Quantum Circuit Definition (sim14_circuit function)**

This function defines the fixed structure of the PQC, arranging RY and CRX gates. When parameterized by `timestep_params`, it defines a unique unitary matrix $U_t$ for that time-step.
```
def sim14_circuit(params, wires, layers=1):
    # Gate sequence per layer: RY -> CRX(ring) -> RY -> CRX(counter-ring)
    # Parameters per layer: 4 * wires
```

## 2) Time Sequence Mixing via Classical LCU Simulation
**Concept**: The model computes a linear combination of unitaries classically by applying each timestep unitary to the working state via a QNode and combining the results with learnable complex coefficients.

### Realization in the Code:

**2-1) Timestep State QNode (_timestep_state_qnode)**

A QNode that prepares an initial state via `StatePrep`, applies the sim14 circuit, and returns the full statevector. This is called once per polynomial iteration with all batch*timestep combinations vectorized into a single call.
```
# QuantumTSTransformer -> __init__()
@qml.qnode(self.dev, interface="torch", diff_method="backprop")
def _timestep_state_qnode(initial_state, params):
    qml.StatePrep(initial_state, wires=range(self.n_qubits))
    sim14_circuit(params, wires=self.n_qubits, layers=self.n_ansatz_layers)
    return qml.state()
```

**2-2) Vectorized LCU Application (apply_unitaries_pl)**

Instead of looping over timesteps, all batch*timestep states and parameters are flattened into a single large batch and processed in one QNode call. The LCU combination is then performed via `torch.einsum`.
```
# apply_unitaries_pl()
flat_params = unitary_params.reshape(bsz * n_timesteps, n_rots)
repeated_base_states = base_states.repeat_interleave(n_timesteps, dim=0)
evolved_states = qnode_state(initial_state=repeated_base_states, params=flat_params)
evolved_states_reshaped = evolved_states.reshape(bsz, n_timesteps, 2**n_qbs)
lcs = torch.einsum('bti,bt->bi', evolved_states_reshaped, coeffs)
```
- `self.mix_coeffs`: Trainable complex-valued tensor of shape `(n_timesteps,)` — the LCU coefficients $b_j$.

## 3) Non-linearity via Classical QSVT Simulation

**Concept**: The QSVT polynomial $P_d(M) = \sum_{k=0}^{d} c_k M^k$ is computed by iteratively applying the LCU operator to the working state and accumulating a weighted sum with learnable polynomial coefficients.

### Realization in the Code:

**3-1) Trainable Polynomial Coefficients (__init__ method)**

The polynomial coefficients $(c_0, c_1, \ldots, c_d)$ are defined as a learnable parameter.
```
# QuantumTSTransformer -> __init__()
self.n_poly_coeffs = self.degree + 1
self.poly_coeffs = torch.nn.Parameter(torch.rand(self.n_poly_coeffs))
```

**3-2) Polynomial State Preparation (evaluate_polynomial_state_pl)**

The polynomial is evaluated by iteratively applying $M$ to the working register and accumulating the weighted sum. The result is normalized by the L1 norm of the polynomial coefficients.
```
# evaluate_polynomial_state_pl()
acc = poly_coeffs[0] * base_states            # c_0 * |0>
working_register = base_states
for c in poly_coeffs[1:]:
    working_register = apply_unitaries_pl(     # M^k |0>
        working_register, unitary_params,
        qnode_state, lcu_coeffs)
    acc = acc + c * working_register           # + c_k * M^k |0>
return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)
```
- Each iteration applies $M$ once more: `working_register` progresses through $|0\rangle \to M|0\rangle \to M^2|0\rangle \to \cdots$
- The final state is $P_d(M)|0\rangle / \|c\|_1$

## 4) Readout and Classical Processing
**Concept**: After the QSVT polynomial transformation, the resulting quantum state is prepared via `StatePrep`, processed through a quantum feed-forward (QFF) layer, and measured to extract Pauli expectation values. These classical features are then processed by a feed-forward network.

### Realization in the Code:

**4-1) State Normalization (forward method)**

The polynomial output state is normalized before being passed to the QFF QNode.
```
# QuantumTSTransformer -> forward()
norm = torch.linalg.vector_norm(mixed_timestep, dim=1, keepdim=True)
normalized_mixed_timestep = mixed_timestep / (norm + 1e-9)
```

**4-2) QFF and Measurement (_qff_qnode_expval)**

A separate QNode prepares the normalized state via `StatePrep`, applies a 1-layer sim14 circuit (QFF), and measures PauliX, PauliY, and PauliZ on each qubit.
```
# QuantumTSTransformer -> __init__()
@qml.qnode(self.dev, interface="torch", diff_method="backprop")
def _qff_qnode_expval(initial_state, params):
    qml.StatePrep(initial_state, wires=range(self.n_qubits))
    sim14_circuit(params, wires=self.n_qubits, layers=1)
    observables = [qml.PauliX(i) for i in range(self.n_qubits)] + \
                  [qml.PauliY(i) for i in range(self.n_qubits)] + \
                  [qml.PauliZ(i) for i in range(self.n_qubits)]
    return [qml.expval(op) for op in observables]
```
- `self.qff_params`: Trainable parameters for the QFF sim14 circuit.
- The measurement yields `3 * n_qubits` real-valued expectation values.

**4-3) Classical Output Layer (forward method)**

The expectation values are stacked and passed through the final linear layer.
```
# QuantumTSTransformer -> forward()
exps = self.qff_qnode_expval(
    initial_state=normalized_mixed_timestep,
    params=self.qff_params)
exps = torch.stack(exps, dim=1).float()
op = self.output_ff(exps)
return op.squeeze(1)
```
- `self.output_ff`: The final `torch.nn.Linear` layer that maps the `3 * n_qubits` measurement outcomes to the desired `output_dim`.

# Reference
- Park, J. J., Seo, J., Bae, S., Chen, S. Y.-C., Tseng, H.-H., Cha, J., & Yoo, S. (2025, Aug 31 - Sep 5). Resting-state fMRI Analysis using Quantum Time-series Transformer. _2025 IEEE International Conference on Quantum Computing and Engineering (QCE)_, Albuquerque, New Mexico, USA. https://doi.org/10.48550/arXiv.2509.00711
- Khatri, N., Matos, G., Coopmans, L., & Clark, S. (2024). Quixer: A Quantum Transformer Model. _arXiv:2406.04305_.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. _Advances in Neural Information Processing Systems (NeurIPS)_, 30.
