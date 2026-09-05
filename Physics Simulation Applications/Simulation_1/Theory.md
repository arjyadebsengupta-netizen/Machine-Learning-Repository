# Physics-Based Neural Operator Learning for Noisy Superconducting Transmon Dynamics

## Overview

This project develops a physics-based neural-operator framework for learning the dynamics of a superconducting transmon system.

The central learning problem is the operator mapping

$$
\mathcal{G}: u(t) \longmapsto \mathbf{P}(t),
$$

where:

* $u(t)$ is a time-dependent vector of effective Hamiltonian coefficients.
* $\mathbf{P}(t)$ is the population vector of the lowest ten transmon energy levels.

The data are generated directly from a numerical open-quantum-system simulation.

The current implementation consists of two major components:

1. A physics-based ten-level transmon simulator.
2. A one-dimensional Fourier Neural Operator (FNO) trained to learn the resulting dynamical operator.

Additional neural-operator architectures will be added later for comparison.

---

# 1. Physical Model

## 1.1 Superconducting Transmon

The physical system is a superconducting transmon described by the full cosine Hamiltonian

$$
H_{\mathrm{charge}}
=
4E_C(\hat n-n_g)^2
-
E_J\cos\hat\phi.
$$

The simulation uses

$$
E_J = 20.0\ \mathrm{GHz},
$$

$$
E_C = 0.30\ \mathrm{GHz},
$$

and

$$
n_g = 0.
$$

The Hamiltonian is constructed directly in the charge basis.

The charge basis is truncated at

$$
n_{\mathrm{cut}}=50,
$$

giving

$$
2n_{\mathrm{cut}}+1=101
$$

charge states.

The resulting $101\times101$ Hamiltonian is numerically diagonalized.

The lowest ten eigenstates are retained for the subsequent dynamical simulation.

---

# 2. Energy Eigenbasis

The charge-basis Hamiltonian is diagonalized as

$$
H_{\mathrm{charge}}V
=
V\operatorname{diag}(E_0,E_1,\ldots).
$$

The retained energies are shifted such that

$$
E_0=0.
$$

The transition frequency between the first two levels is

$$
f_{01}=E_1-E_0.
$$

The next transition frequency is

$$
f_{12}=E_2-E_1.
$$

The anharmonicity is calculated as

$$
\alpha
=
f_{12}-f_{01}.
$$

The charge operator is transformed into the retained energy eigenbasis according to

$$
\hat n_{\mathrm{eig}}
=
V^\dagger \hat n V.
$$

All subsequent ten-level dynamics are performed in this eigenbasis.

---

# 3. Effective Hamiltonian Representation

The dynamical Hamiltonian is represented using a four-channel effective coefficient description.

The operator-learning input is

$$
u(t)
=
\begin{bmatrix}
u_{\mathrm{frequency}}(t)\\
u_I(t)\\
u_Q(t)\\
u_{\mathrm{charge}}(t)
\end{bmatrix}.
$$

The corresponding effective Hamiltonian is

$$
H_{\mathrm{eff}}(t)
=
H_0
+
u_{\mathrm{frequency}}(t)H_{\mathrm{frequency}}
+
u_I(t)H_I
+
u_Q(t)H_Q
+
u_{\mathrm{charge}}(t)H_{\mathrm{charge}}.
$$

Therefore, the machine-learning input contains

$$
\boxed{4}
$$

channels.

The output contains the populations of the ten retained levels,

$$
\mathbf{P}(t)
=
\begin{bmatrix}
P_0(t)\\
P_1(t)\\
\vdots\\
P_9(t)
\end{bmatrix}.
$$

Thus,

$$
\boxed{10}
$$

output channels are used.

---

# 4. Microwave I/Q Control

The microwave control is represented by in-phase and quadrature components,

$$
\Omega(t)=I(t)+iQ(t).
$$

The maximum I/Q amplitude is

$$
0.100\ \mathrm{GHz}.
$$

The control trajectories are generated from randomly sampled coarse control points and interpolated onto the simulation time grid.

This produces smooth time-dependent control functions rather than independently sampled values at every time step.

Amplitude noise and phase noise are subsequently applied to the complex microwave control.

The noisy control is

$$
\Omega_{\mathrm{noisy}}(t)
=
\Omega(t)
\left[1+\epsilon_A(t)\right]
e^{i\epsilon_\phi(t)}.
$$

The resulting real and imaginary components are used as the noisy I/Q controls.

---

# 5. Noise Model

The simulator includes several stochastic effects.

## 5.1 Frequency Noise

A slowly varying frequency-noise process is included with RMS scale

$$
10^{-4}\ \mathrm{GHz}.
$$

The configured frequency range is

$$
f_{\min}=1\ \mathrm{Hz},
$$

$$
f_{\mathrm{knee}}=10^7\ \mathrm{Hz},
$$

and

$$
f_{\max}=2\times10^9\ \mathrm{Hz}.
$$

## 5.2 Charge Noise

Charge noise is included with amplitude

$$
2\times10^{-3}.
$$

This enters the effective Hamiltonian through the charge-noise operator.

## 5.3 Microwave Amplitude Noise

The microwave amplitude noise has standard deviation

$$
\sigma_A=10^{-4}.
$$

## 5.4 Microwave Phase Noise

The microwave phase noise has standard deviation

$$
\sigma_\phi=10^{-3}\ \mathrm{rad}.
$$

## 5.5 Frequency Drift

A slowly varying frequency drift is included with scale

$$
\sigma_{\mathrm{drift}}
=
50\times10^{-6}\ \mathrm{GHz}.
$$

## 5.6 TLS Telegraph Switching

A two-level-system telegraph process is included.

The TLS switches between two states and produces a frequency shift with magnitude

$$
500\times10^{-6}\ \mathrm{GHz}.
$$

---

# 6. Open-System Quantum Dynamics

The transmon is treated as an open quantum system.

The density matrix evolves according to the Lindblad master equation

$$
\frac{d\rho}{dt}
=
-i[H(t),\rho]
+
\sum_k
\left(
L_k\rho L_k^\dagger
-
\frac{1}{2}
\left\{
L_k^\dagger L_k,\rho
\right\}
\right).
$$

The implementation uses the appropriate $2\pi$ conversion for frequencies expressed in GHz.

The initial state is the ground state,

$$
\rho(0)
=
|0\rangle\langle0|.
$$

---

# 7. Relaxation and Dephasing

The relaxation and coherence times are

$$
T_1=30\ \mu\mathrm{s},
$$

and

$$
T_2=20\ \mu\mathrm{s}.
$$

The corresponding rates are

$$
\Gamma_1=\frac{1}{T_1},
$$

$$
\Gamma_2=\frac{1}{T_2},
$$

and

$$
\Gamma_\phi
=
\Gamma_2-\frac{1}{2}\Gamma_1.
$$

The simulator includes relaxation channels connecting successive retained levels,

$$
|j\rangle\rightarrow|j-1\rangle,
$$

together with a dephasing channel.

---

# 8. Numerical Integration

The environment time step is

$$
\Delta t_{\mathrm{env}}
=
2.0\ \mathrm{ns}.
$$

The internal integration time step is

$$
\Delta t_{\mathrm{internal}}
=
0.05\ \mathrm{ns}.
$$

The total simulation time is

$$
T=200\ \mathrm{ns}.
$$

The neural-operator time step is

$$
\Delta t_{\mathrm{operator}}
=
2.0\ \mathrm{ns}.
$$

Therefore, the operator-learning time grid is

$$
t=0,2,4,\ldots,200\ \mathrm{ns},
$$

with

$$
\boxed{101}
$$

temporal points.

The density matrix is propagated using a fourth-order Runge--Kutta scheme.

---

# 9. Population Extraction

At each operator-learning time point, the ten populations are extracted from the density matrix,

$$
P_j(t)=\rho_{jj}(t).
$$

The resulting population vector is

$$
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)].
$$

The populations are constrained numerically to remain non-negative and are normalized so that

$$
\sum_{j=0}^{9}P_j(t)=1.
$$

---

# 10. Operator-Learning Dataset

The physics simulator generates pairs of input and output trajectories,

$$
\left(
u^{(i)}(t),
\mathbf{P}^{(i)}(t)
\right).
$$

The learned operator is therefore

$$
\boxed{
\mathcal{G}:u(t)\rightarrow\mathbf{P}(t).
}
$$

The input tensor has the form

$$
U\in\mathbb{R}^{B\times4\times101},
$$

while the output tensor has the form

$$
Y\in\mathbb{R}^{B\times10\times101}.
$$

The tensor convention is

```text
[B, C, N]
```

where:

* `B` is the batch dimension.
* `C` is the channel dimension.
* `N` is the temporal dimension.

---

# 11. Data Preprocessing

The input channels are standardized using statistics computed exclusively from the training dataset.

For each channel $c$,

$$
\mu_c
=
\operatorname{mean}(U_c),
$$

and

$$
\sigma_c
=
\operatorname{std}(U_c).
$$

The standardized input is

$$
\widetilde U_c
=
\frac{U_c-\mu_c}{\sigma_c}.
$$

The training-set statistics are also used to transform the test inputs.

No test-set statistics are used during preprocessing.

The population outputs are left unchanged.

---

# 12. PyTorch Data Pipeline

The processed data are converted to PyTorch tensors.

The current batch size is

$$
\boxed{8}.
$$

The resulting data representation is

```text
Input:
[B, 4, 101]

Output:
[B, 10, 101]
```

The training and test sets are provided through PyTorch `Dataset` and `DataLoader` objects.

---

# 13. Fourier Neural Operator

The first neural operator implemented in the project is a one-dimensional Fourier Neural Operator.

The FNO operates along the temporal dimension.

For an intermediate feature representation

$$
x\in\mathbb{R}^{B\times C\times N},
$$

the temporal dimension is transformed into Fourier space,

$$
\hat{x}
=
\mathcal{F}(x).
$$

A finite number of Fourier modes are retained.

The spectral convolution applies learned complex-valued weights,

$$
\hat{y}_k
=
W_k\hat{x}_k.
$$

The transformed representation is then returned to physical space using

$$
y
=
\mathcal{F}^{-1}(\hat{y}).
$$

---

# 14. FNO Architecture

The current FNO consists of:

* An input $1\times1$ convolution.
* Four spectral convolution layers.
* Four pointwise $1\times1$ convolution layers.
* GELU nonlinearities.
* A two-stage output projection.

The input dimension is

$$
4.
$$

The hidden width is

$$
32.
$$

The number of retained Fourier modes is

$$
16.
$$

The final output dimension is

$$
10.
$$

The complete mapping is therefore

$$
[B,4,101]
\longrightarrow
[B,10,101].
$$

---

# 15. FNO Training

The FNO is trained using mean-squared error.

The loss function is

$$
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{B10N}
\sum_{b,j,t}
\left(
\widehat{P}_{j}^{(b)}(t)
-
P_{j}^{(b)}(t)
\right)^2.
$$

The optimizer is Adam with learning rate

$$
\boxed{10^{-3}}.
$$

The batch size is

$$
\boxed{8}.
$$

The number of epochs is controlled by the active experiment configuration.

---

# 16. FNO Evaluation

The trained FNO is evaluated on the held-out test trajectories.

Two primary metrics are used.

## 16.1 Relative $L^2$ Error

For each test trajectory,

$$
\epsilon_{L^2}
=
\frac{
\left\|
\widehat{\mathbf{P}}
-
\mathbf{P}
\right\|_2
}{
\left\|
\mathbf{P}
\right\|_2
}.
$$

The reported Relative $L^2$ error is averaged over the test set.

## 16.2 Relative Temporal $H^1$ Error

Temporal derivatives are calculated using finite differences.

For interior points,

$$
\frac{dP}{dt}
\approx
\frac{
P(t+\Delta t)-P(t-\Delta t)
}{
2\Delta t
}.
$$

A forward difference is used at the initial point and a backward difference at the final point.

The temporal $H^1$ error combines the population error and the temporal-derivative error:

$$
\epsilon_{H^1}
=
\frac{
\sqrt{
\|\widehat{\mathbf{P}}-\mathbf{P}\|_2^2
+
\|\partial_t\widehat{\mathbf{P}}
-
\partial_t\mathbf{P}\|_2^2
}
}{
\sqrt{
\|\mathbf{P}\|_2^2
+
\|\partial_t\mathbf{P}\|_2^2
}
}.
$$

This metric evaluates both population accuracy and the ability of the neural operator to reproduce temporal dynamics.

---

# 17. Current Pipeline

The current implementation follows the pipeline

```text
Full cosine transmon Hamiltonian
              |
              v
       Charge-basis model
              |
              v
    Numerical diagonalization
              |
              v
      Lowest 10 eigenstates
              |
              v
   Effective Hamiltonian basis
              |
              v
       Control + noise
              |
              v
    Lindblad master equation
              |
              v
       RK4 time evolution
              |
              v
    10 population trajectories
              |
              v
       Dataset generation
              |
              v
   Training-only standardization
              |
              v
       PyTorch DataLoader
              |
              v
      Fourier Neural Operator
              |
              v
     Predicted population
          trajectories
```

---

# 18. Current Implementation Status

| Component                          | Status      |
| ---------------------------------- | ----------- |
| Full cosine transmon Hamiltonian   | Implemented |
| Charge-basis representation        | Implemented |
| Numerical diagonalization          | Implemented |
| Ten-level truncation               | Implemented |
| I/Q control generation             | Implemented |
| Frequency noise                    | Implemented |
| Charge noise                       | Implemented |
| Microwave amplitude noise          | Implemented |
| Microwave phase noise              | Implemented |
| Frequency drift                    | Implemented |
| TLS telegraph switching            | Implemented |
| Lindblad relaxation                | Implemented |
| Lindblad dephasing                 | Implemented |
| RK4 density-matrix evolution       | Implemented |
| Population trajectory generation   | Implemented |
| Operator-learning dataset          | Implemented |
| Channelwise input standardization  | Implemented |
| PyTorch dataset pipeline           | Implemented |
| Fourier spectral convolution       | Implemented |
| One-dimensional FNO                | Implemented |
| FNO training                       | Implemented |
| Relative $L^2$ evaluation          | Implemented |
| Relative temporal $H^1$ evaluation | Implemented |
| GNO                                | Planned     |
| CATO                               | Planned     |
| Final multi-model benchmark        | Planned     |

---

# 19. Current Notebook Structure

The current implementation is organized as follows:

```text
Cell 1   — Imports
Cell 2   — Global configuration
Cell 3   — Transmon Hamiltonian and eigenbasis
Cell 4   — Control operators
Cell 5   — Noise and control parameters
Cell 6   — Control and noise state
Cell 7   — Dissipation and Lindblad operators
Cell 8   — Effective Hamiltonian representation
Cell 9   — Hamiltonian and Lindblad evolution
Cell 10  — Single trajectory generation
Cell 11  — Dataset generation
Cell 12  — Channelwise standardization
Cell 13  — PyTorch datasets and dataloaders
Cell 14  — Fourier spectral convolution
Cell 15  — Complete FNO model
Cell 16  — FNO training setup
Cell 17  — FNO training
Cell 18  — FNO test evaluation
Cell 19  — Relative temporal H1 evaluation
Cell 20  — FNO training/evaluation summary
```

---

# 20. Reproducibility

The implementation uses:

* Python
* NumPy
* SciPy
* PyTorch
* Matplotlib

The current experiment uses the random seed

```python
1234
```

The principal simulation and training parameters are centralized in the `CONFIG` dictionary.

The physics simulation uses NumPy/SciPy, while the neural-operator model and training pipeline use PyTorch.

---

# 21. Future Work

The FNO currently provides the first neural-operator baseline for the physics-generated transmon dataset.

The next stage is to implement additional neural-operator architectures while keeping the underlying learning problem fixed.

The planned benchmark is

$$
\boxed{
\mathrm{FNO}
\quad\mathrm{vs.}\quad
\mathrm{GNO}
\quad\mathrm{vs.}\quad
\mathrm{CATO}
}
$$

using the same:

* Physics-generated dataset.
* Four-channel input representation.
* Ten-channel population output.
* Temporal discretization.
* Training/test split.
* Batch size.
* Optimizer.
* Learning rate.
* MSE objective.
* Relative $L^2$ metric.
* Relative temporal $H^1$ metric.

This will provide a controlled comparison of different neural-operator architectures for learning the same noisy open-quantum dynamical system.

---

# 22. Research Objective

The central research problem is to learn the operator

$$
\boxed{
\mathcal{G}:
\mathbb{R}^{4\times101}
\rightarrow
\mathbb{R}^{10\times101}
}
$$

that maps time-dependent effective Hamiltonian coefficients to ten-level transmon population dynamics.

The reference operator is generated by numerical solution of an open quantum system containing:

* Full cosine transmon physics.
* Finite anharmonicity.
* Higher-level dynamics and leakage.
* I/Q microwave control.
* Frequency noise.
* Charge noise.
* Microwave amplitude noise.
* Microwave phase noise.
* Slow frequency drift.
* TLS telegraph switching.
* Relaxation.
* Dephasing.

The neural operator therefore learns a functional mapping between physically meaningful time-dependent Hamiltonian inputs and the resulting quantum population trajectories.

The current stage establishes the complete physics-to-FNO pipeline. The subsequent stages will extend this framework to additional neural operators and a controlled architectural comparison.
