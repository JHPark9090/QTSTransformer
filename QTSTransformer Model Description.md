# Quantum Time-Series Transformer (PennyLane Implementation)
This repository contains a PyTorch and PennyLane implementation of a quantum time-series transformer model developed by Park et al. (2025). The quantum time-series transformer is based on the Quixer model architecture proposed by Khatri et al. (2024). This model is designed to process time-series data by leveraging quantum computing principles for embedding, mixing, and non-linear transformation of temporal features.

# Core Concepts
The model's architecture can be broken down into four key stages:

- **Unitary Temporal Embedding**: The feature vector at each point in a time sequence is mapped to a unique quantum circuit (a unitary matrix), creating a quantum representation of that specific moment's data.

- **Time Sequence Mixing (LCU)**: A Linear Combination of Unitaries (LCU) is used to create a weighted superposition ($M$) of all the time-step unitaries ($U_j$). This serves as a quantum-native attention-like mechanism to mix information across the time sequence.

$$M = \sum^{n-1}_{j=0}e^{i\gamma_j}{|a_j|}^2U_j$$

- **Non-linearity (QSVT)**: A Quantum Singular Value Transform (QSVT) is simulated to apply a non-linear polynomial transformation to the mixed state ($M$). This allows the model to capture richer, higher-order interactions between different time-steps.

$$P_c(M) = c_d(M^d) + c_{d-1}(M^{d-1}) + \cdots + c_1M+c_0I$$

- **Readout and Classical Processing**: The final quantum state is measured to extract classical features (Pauli expectation values). These features are then processed by a classical feed-forward neural network to produce the final prediction.

# Code Implementation Breakdown
This section explains how each conceptual step is realized in the QuixerTimeSeriesPennyLane.py code.

## 1) Unitary Temporal Embedding
**Concept**: Classical features for each time-step are mapped to a unique Parameterized Quantum Circuit (PQC), which defines a unitary transformation.

### Realization in the Code:

This process involves three main parts:

**1-1) Classical Projection (forward method)**

The classical input feature vector for each time-step is passed through a standard torch.nn.Linear layer to generate a set of rotation angles for the quantum gates.
```
# QuixerTimeSeriesPennyLane -> forward()
# Input x has shape (batch, n_timesteps, feature_dim) after permutation
# This linear layer maps feature_dim -> n_rots (number of rotation angles)
x = self.feature_projection(self.dropout(x))

#Sigmoid ensures angles are in a consistent range
timestep_params = self.rot_sigm(x)
```
- self.feature_projection: The learnable linear layer.
- timestep_params: The resulting tensor holding the gate parameters for every time-step in the batch.

**1-2) Quantum Circuit Definition (sim14_circuit function)**

This function defines the fixed structure of the PQC, arranging RY and CRX gates. When parameterized by timestep_params, it defines a unique unitary matrix U for that time-step.
```
def sim14_circuit(params, wires, layers=1):
    # ... circuit logic with qml.RY and qml.CRX gates ...
```

**1-3) QNode for State Evolution (__init__ method)**

The _timestep_state_qnode is the PennyLane object that simulates the quantum evolution. It takes an initial state and the timestep_params and applies the unitary transformation defined by sim14_circuit.
```
# QuixerTimeSeriesPennyLane -> __init__()
@qml.qnode(...)
def _timestep_state_qnode(initial_state, params):
    qml.StatePrep(initial_state, wires=range(self.n_qubits))
    sim14_circuit(params, ...)
    return qml.state()
```
## 2) Time Sequence Mixing via LCU
**Concept**: The model learns a complex-valued weight for each time-step and creates a weighted sum of the quantum states produced by each time-step's unitary.

### Realization in the Code:

**2-1) Trainable Coefficients (__init__ method)**

The complex weights for the LCU are defined as a learnable model parameter.
```
# QuixerTimeSeriesPennyLane -> __init__()
self.mix_coeffs = torch.nn.Parameter(torch.rand(self.n_timesteps, dtype=torch.complex64))
```
**2-2) LCU Application (apply_unitaries_pl function)**

This function simulates the LCU. It first calculates the evolved state for every time-step's unitary (U_t |ψ⟩) in a vectorized manner and then performs the weighted sum using torch.einsum.
```
apply_unitaries_pl()
# ... calculates evolved_states_reshaped ...

# This is the LCU:
# It multiplies each evolved state vector by its learned complex coefficient
# and sums them up, creating the superposition.
lcs = torch.einsum('bti,bt->bi', evolved_states_reshaped, coeffs)
return lcs
```

## 3) Non-linearity via QSVT

**Concept**: A polynomial transformation P(M) is applied to the LCU superposition M. This is simulated by iteratively applying the LCU operation and summing the results according to learnable polynomial coefficients.

### Realization in the Code:

**3-1) Trainable Coefficients (__init__ method)**

The coefficients of the polynomial (c₀, c₁, c₂, ...) are defined as a learnable parameter.
```
# QuixerTimeSeriesPennyLane -> __init__()
self.poly_coeffs = torch.nn.Parameter(torch.rand(self.n_poly_coeffs))
```
**3-2) Polynomial Application (evaluate_polynomial_state_pl function)**

This function orchestrates the QSVT simulation by looping through the polynomial coefficients and repeatedly calling apply_unitaries_pl to compute the higher-order terms (M|ψ⟩, M(M|ψ⟩), etc.).
```
# evaluate_polynomial_state_pl()

# Zeroth-order term (c₀ * I)|ψ⟩
acc = poly_coeffs[0] * base_states
working_register = base_states

# Loop for higher-order terms
for c in poly_coeffs[1:]:
    # Calculate the next power, e.g., M(working_register)
    working_register = apply_unitaries_pl(working_register, ...)
    # Add the weighted term to the accumulator: acc += c_k * M^k |ψ⟩
    acc = acc + c * working_register
```

## 4) Readout and Classical Processing
**Concept**: The final quantum state is measured to extract a classical feature vector. This vector is then processed by a classical feed-forward network to produce the final output.

### Realization in the Code:

**4-1) Measurement QNode (__init__ method)**

The _qff_qnode_expval is defined to perform the measurement. It applies a final trainable PQC and then computes the expectation values of Pauli X, Y, and Z operators for each qubit.
```
# QuixerTimeSeriesPennyLane -> __init__()

@qml.qnode(...)
def _qff_qnode_expval(initial_state, params):
    qml.StatePrep(initial_state, ...)
    sim14_circuit(params, ...) # Final trainable quantum layer
    # Define the Pauli measurements
    observables = [qml.PauliX(i) for i in range(n_qubits)] + \
                  [qml.PauliY(i) for i in range(n_qubits)] + \
                  [qml.PauliZ(i) for i in range(n_qubits)]
    return [qml.expval(op) for op in observables]
```
**4-2) Execution and Classical Layer (forward method)**

The forward pass executes the measurement and passes the classical results to the final linear layer.
```
# QuixerTimeSeriesPennyLane -> forward()

# ... calculates normalized_mixed_timestep ...

# Call the measurement QNode to get classical expectation values
exps = self.qff_qnode_expval(
    initial_state=normalized_mixed_timestep,
    params=self.qff_params
)

# Stack, format, and feed into the final linear layer for the prediction
exps = torch.stack(exps, dim=1).float()
op = self.output_ff(exps)
return op
```
- self.output_ff: The final torch.nn.Linear layer that maps the 3 * n_qubits measurement outcomes to the desired output_dim.

# Reference
- Park, J. J., Seo, J., Bae, S., Chen, S. Y.-C., Tseng, H.-H., Cha, J., & Yoo, S. (2025, Aug 31 - Sep 5). Resting-state fMRI Analysis using Quantum Time-series Transformer. _2025 IEEE International Conference on Quantum Computing and Engineering (QCE)_, Albuquerque, New Mexico, USA. https://doi.org/10.48550/arXiv.2509.00711
- Khatri, N., Matos, G., Coopmans, L., & Clark, S. (2024). Quixer: A Quantum Transformer Model. _arXiv:2406.04305_.
