# Physics-Informed Neural Operators for Noisy Superconducting Transmon Dynamics

## Overview

This project develops a physics-based simulation and neural-operator learning framework for the dynamics of a single superconducting transmon qubit under realistic control and noise.

The workflow combines:

- Full-cosine transmon Hamiltonian modeling
- Numerical diagonalization in the charge basis
- Multilevel quantum dynamics
- Lindblad open-system evolution
- Stochastic control and noise processes
- Microwave $I/Q$ control
- Finite-bandwidth control construction
- Synthetic trajectory generation from the physics simulator
- Fourier Neural Operator (FNO) learning
- Quantitative operator-learning evaluation
- Qualitative population-trajectory comparison

The current implementation establishes the physics simulator and the first neural-operator benchmark using a one-dimensional Fourier Neural Operator.

---

## Physical Model

The transmon is modeled using the full cosine Hamiltonian

$$H_{\mathrm{tr}} = 4E_C(\hat n-n_g)^2-E_J\cos\hat\phi$$

rather than a truncated harmonic approximation.

The charge basis is

$$n=-50,\ldots,50$$

giving a Hilbert-space dimension of $101$.

The Hamiltonian is numerically diagonalized and the lowest $10$ eigenstates are retained for the dynamical simulation.

This provides a multilevel model capable of representing dynamics beyond an ideal two-level approximation.

---

## Transmon Parameters

The simulation uses

- $E_J = 20.0$ GHz
- $E_C = 0.30$ GHz
- $n_g=0$
- Charge-basis cutoff: $n_{\mathrm{cut}}=50$
- Retained energy levels: $10$

The numerical diagonalization provides the transmon energy spectrum and the operators required for the reduced multilevel dynamics.

The transition frequencies and anharmonicity are extracted directly from the numerically obtained spectrum.

---

## Reduced Multilevel Representation

After diagonalization, the lowest ten eigenstates are retained.

The reduced state is represented by a density matrix

$$\rho(t)\in\mathbb{C}^{10\times10}$$

and the model propagates the full density matrix rather than only level populations.

The final neural-operator output, however, consists of the ten level populations

$$P_j(t)=\rho_{jj}(t),\qquad j=0,\ldots,9$$

so that the learning problem maps the applied control/noise trajectory to the resulting population trajectories.

---

## Control Representation

The neural operator receives four input channels:

$$u(t)=\left[f(t),I(t),Q(t),q(t)\right]$$

where

- $f(t)$ is the frequency-related control/noise channel
- $I(t)$ is the in-phase microwave component
- $Q(t)$ is the quadrature microwave component
- $q(t)$ is the charge-related channel

The operator input therefore has the structure

$$U\in\mathbb{R}^{N_{\mathrm{samples}}\times4\times101}$$

with $101$ temporal grid points.

---

## Microwave $I/Q$ Control

The maximum microwave amplitude is set by

$$A_{\max}=0.100\ \mathrm{GHz}$$

The control trajectories are generated using smooth interpolation between randomly generated control values.

The resulting $I/Q$ signals provide temporally smooth control trajectories for the physics simulation.

The reduced control Hamiltonians are constructed from the numerically obtained transmon charge operator.

The in-phase and quadrature operators are

$$H_I=n_+ + n_+^\dagger$$

and

$$H_Q=i(n_+-n_+^\dagger)$$

where $n_+$ denotes the upper-triangular off-diagonal component of the charge operator in the retained eigenbasis.

---

## Charge Control

The charge-related Hamiltonian contribution is constructed from the transmon charge operator as

$$H_{\mathrm{charge}}=-8E_C(\hat n-n_{g0}I)$$

with the reference offset charge $n_{g0}=0$.

The four effective Hamiltonian channels are therefore associated with

- frequency
- $I$
- $Q$
- charge

and are assembled into the time-dependent effective Hamiltonian used by the simulator.

---

## Noise Model

The simulator includes several stochastic perturbations intended to produce non-ideal control trajectories.

The configured noise processes include:

- Frequency noise
- Charge noise
- Microwave amplitude noise
- Microwave phase noise
- Slow frequency drift
- TLS telegraph fluctuations

The configured parameters include

- Frequency-noise RMS: $10^{-4}$ GHz
- Charge-noise amplitude: $2\times10^{-3}$
- Microwave amplitude-noise standard deviation: $10^{-4}$
- Phase-noise standard deviation: $10^{-3}$ rad
- Slow frequency-drift standard deviation: $50\times10^{-6}$ GHz
- TLS frequency shift: $500\times10^{-6}$ GHz

The stochastic processes are incorporated directly into the trajectories supplied to the quantum simulator.

---

## Open-System Dynamics

The quantum dynamics are modeled using a Lindblad master equation

$$\frac{d\rho}{dt}=-i[H(t),\rho]+\sum_k\left(L_k\rho L_k^\dagger-\frac{1}{2}\{L_k^\dagger L_k,\rho\}\right)$$

with the Hamiltonian expressed in frequency units, giving the numerical evolution in the corresponding angular-frequency convention.

The relaxation time is

$$T_1=30\ \mu\mathrm{s}$$

and the transverse coherence time is

$$T_2=20\ \mu\mathrm{s}$$

The corresponding rates are constructed as

$$\gamma_1=\frac{1}{T_1}$$

and

$$\gamma_2=\frac{1}{T_2}$$

with the pure-dephasing contribution obtained from

$$\gamma_\phi=\gamma_2-\frac{\gamma_1}{2}$$

Relaxation operators connect adjacent retained transmon levels, while a diagonal dephasing operator provides the pure-dephasing channel.

---

## Numerical Time Integration

The operator-learning trajectories are sampled on

$$t=0,2,4,\ldots,200\ \mathrm{ns}$$

giving

$$N_{\mathrm{time}}=101$$

operator time points.

The internal quantum-dynamics integration uses

$$\Delta t_{\mathrm{internal}}=0.05\ \mathrm{ns}$$

while the operator-level temporal resolution is

$$\Delta t_{\mathrm{operator}}=2\ \mathrm{ns}$$

Each operator interval therefore contains multiple internal RK4 integration steps.

The density-matrix dynamics are propagated using a fourth-order Runge-Kutta scheme.

---

## Dataset Generation

Each trajectory is generated independently from the physics simulator.

For every trajectory, the simulator produces:

$$U\rightarrow P$$

where

$$U\in\mathbb{R}^{4\times101}$$

and

$$P\in\mathbb{R}^{10\times101}$$

The output populations satisfy the normalization condition

$$\sum_{j=0}^{9}P_j(t)=1$$

throughout the simulated trajectory, subject to the numerical population extraction and normalization procedure.

Independent random seeds are used for training and testing trajectories.

The random-number generation uses a fixed master seed for reproducibility.

---

## Dataset Standardization

The input channels are standardized using statistics calculated from the training dataset.

For each input channel,

$$\tilde U_c=\frac{U_c-\mu_c}{\sigma_c}$$

where $\mu_c$ and $\sigma_c$ are calculated from the training trajectories.

The same training statistics are applied to the test dataset.

The population targets are not standardized in this step.

---

## Neural-Operator Learning Problem

The learning problem is to approximate the operator

$$\mathcal{G}:U(t)\mapsto P(t)$$

where the input consists of the four control/noise channels and the output consists of the ten transmon-level population trajectories.

The learned operator therefore approximates

$$\mathcal{G}(f,I,Q,q)=\{P_0(t),P_1(t),\ldots,P_9(t)\}$$

over the complete temporal interval.

---

# Fourier Neural Operator

The first neural-operator model implemented in the project is a one-dimensional Fourier Neural Operator.

The FNO operates along the temporal dimension.

Its architecture consists of:

- Input projection
- Four spectral convolution layers
- Four pointwise convolution layers
- GELU nonlinearities
- Output projection

The model uses

- Input channels: $4$
- Output channels: $10$
- Hidden width: $32$
- Fourier modes: $16$
- Spectral layers: $4$
- Pointwise layers: $4$

The FNO input has shape

$$[B,4,101]$$

and the output has shape

$$[B,10,101]$$

where $B$ denotes the batch size.

---

## Spectral Convolution

The spectral convolution transforms the temporal signal into Fourier space using the real-valued fast Fourier transform.

For an input $x$, the Fourier representation is

$$\hat{x}=\mathrm{rFFT}(x)$$

Only a prescribed number of low-frequency Fourier modes are retained for the learned spectral transformation.

The learned Fourier weights operate on the retained modes before the inverse Fourier transform reconstructs the temporal representation.

The spectral layers are combined with pointwise convolutional layers to provide local nonlinear processing alongside global spectral interactions.

---

## FNO Architecture

The model begins by projecting the four input channels into a $32$-dimensional hidden representation.

The hidden representation is then processed through four spectral layers and four pointwise layers.

GELU nonlinearities are used between the transformations.

The final representation is projected from the hidden width to the ten population channels.

Conceptually,

$$4\rightarrow32\rightarrow32\rightarrow32\rightarrow32\rightarrow32\rightarrow10$$

with the intermediate transformations combining Fourier spectral convolution and pointwise convolution.

---

## Training

The FNO is trained using mean squared error:

$$\mathcal{L}_{\mathrm{MSE}}=\frac{1}{N}\sum_i\left(P_i^{\mathrm{pred}}-P_i^{\mathrm{true}}\right)^2$$

The optimizer is Adam with learning rate

$$\eta=10^{-3}$$

The batch size is $8$.

The number of training epochs is taken directly from the project configuration.

Training loss is recorded after every epoch.

The implementation checks that the training loss history contains finite, non-negative values and that the final training loss does not exceed the initial training loss, except in the single-epoch case.

---

## Evaluation Metrics

The trained FNO is evaluated on the independent test trajectories.

Three main quantitative quantities are calculated.

### Test MSE

The test mean squared error is calculated between the predicted and true population trajectories.

### Relative $L^2$ Error

For each test trajectory,

$$e_{L^2}=\frac{\|P_{\mathrm{pred}}-P_{\mathrm{true}}\|_2}{\max(\|P_{\mathrm{true}}\|_2,10^{-12})}$$

The reported Relative $L^2$ error is the mean over the test trajectories.

### Relative Temporal $H^1$ Error

Temporal derivatives are computed using second-order finite differences.

For interior points,

$$\frac{dP}{dt}(t_i)\approx\frac{P(t_{i+1})-P(t_{i-1})}{2\Delta t}$$

A forward difference is used at the initial point and a backward difference at the final point.

The temporal $H^1$ error combines both population and temporal-derivative errors:

$$e_{H^1}=\frac{\sqrt{\|P_{\mathrm{pred}}-P_{\mathrm{true}}\|_2^2+\|\partial_tP_{\mathrm{pred}}-\partial_tP_{\mathrm{true}}\|_2^2}}{\max\left(\sqrt{\|P_{\mathrm{true}}\|_2^2+\|\partial_tP_{\mathrm{true}}\|_2^2},10^{-12}\right)}$$

The reported value is averaged over the test trajectories.

---

## Physical Consistency Checks

The implementation performs numerical checks throughout the pipeline.

These include:

- Finite-valued Hamiltonian matrices
- Hermiticity of relevant Hamiltonian operators
- Nonzero microwave control operators
- Valid density-matrix traces
- Finite population trajectories
- Population normalization
- Valid tensor shapes
- Finite neural-network outputs
- Finite training losses
- Finite evaluation metrics
- Independent training and test trajectories

For the FNO predictions, the population sum is explicitly compared with the normalized target population sum.

---

## Qualitative Trajectory Comparison

A qualitative trajectory comparison is performed using one test trajectory.

The predicted and true population trajectories are plotted for all ten retained transmon levels.

For each level,

$$P_j(t),\qquad j=0,\ldots,9$$

the FNO prediction is compared directly against the physics-simulator trajectory.

This provides a visual assessment of whether the learned operator reproduces the temporal population dynamics across the complete multilevel system.

---

## Population Conservation Comparison

For the same test trajectory, the total population is calculated as

$$P_{\mathrm{total}}(t)=\sum_{j=0}^{9}P_j(t)$$

The true and predicted population sums are plotted together.

The maximum population-sum error is also calculated:

$$e_{\mathrm{pop}}=\max_t\left|P_{\mathrm{total}}^{\mathrm{pred}}(t)-P_{\mathrm{total}}^{\mathrm{true}}(t)\right|$$

This provides an additional diagnostic of how closely the neural operator preserves the population structure of the simulated trajectory.

---

## Reproducibility

The project uses a fixed random seed:

$$\mathrm{seed}=1234$$

Random-number generation is controlled for reproducible trajectory generation and model initialization.

The computational device is selected automatically according to availability:

1. CUDA
2. Apple MPS
3. CPU

The implementation therefore does not require a specific hardware accelerator.

---

## Current Workflow

The implemented workflow through Cell 21 is:

```text
Full-cosine transmon Hamiltonian
            ↓
Charge-basis numerical diagonalization
            ↓
Lowest 10 eigenstates
            ↓
Reduced multilevel operators
            ↓
Stochastic control/noise trajectories
            ↓
Lindblad master equation
            ↓
RK4 quantum evolution
            ↓
Population trajectories
            ↓
Input standardization
            ↓
FNO training
            ↓
Test evaluation
            ↓
Relative L2 + temporal H1
            ↓
Qualitative trajectory comparison
```

---

## Cell Organization

The current notebook is organized into the following computational stages.

### Cells 1–3 — Physical setup

- Imports
- Configuration
- Charge basis
- Full-cosine Hamiltonian
- Numerical diagonalization
- Ten-level truncation
- Spectral quantities

### Cells 4–8 — Control and noise construction

- Control operators
- Noise configuration
- Random-state initialization
- Lindblad parameters
- Effective Hamiltonian basis

### Cells 9–11 — Quantum simulation and dataset generation

- Effective Hamiltonian construction
- Lindblad dynamics
- RK4 propagation
- Population extraction
- Stochastic trajectory generation
- Training/test dataset generation

### Cells 12–13 — Dataset preparation

- Training-set input standardization
- Dataset objects
- Data loaders

### Cells 14–15 — FNO construction

- Spectral convolution
- FNO architecture
- Forward-pass diagnostics
- Parameter-count checks

### Cells 16–17 — FNO training

- MSE loss
- Adam optimizer
- Training configuration
- Training loop
- Training-history diagnostics

### Cells 18–19 — Quantitative evaluation

- Test-set evaluation
- Test MSE
- Relative $L^2$
- Temporal derivatives
- Relative temporal $H^1$

### Cell 20 — Training diagnostics

- Training-loss curve
- Training summary
- Evaluation metrics

### Cell 21 — Qualitative evaluation

- Single test-trajectory selection
- Ten-level population comparison
- Population-sum comparison
- Sample Relative $L^2$
- Maximum population-sum error
- Final numerical checks

---

## Current Model Scope

The current implementation establishes the first neural-operator benchmark using FNO.

The present model is a **physics-generated-data → FNO operator-learning pipeline**.

The physics simulator provides the reference trajectories, while the FNO learns the mapping from the four-channel input trajectory to the ten-channel population trajectory.

Future neural-operator comparisons can be built on the same dataset and evaluation framework.

Potential extensions include additional neural-operator architectures and systematic comparisons of their ability to learn the noisy multilevel transmon dynamics.

---

## Project Objective

The central objective is to investigate whether neural operators can efficiently learn the input-to-trajectory mapping generated by a physically detailed noisy superconducting-qubit simulator.

The resulting framework connects:

$$\boxed{\text{Quantum physics simulation}\rightarrow\text{operator learning}\rightarrow\text{trajectory prediction}}$$

while retaining the multilevel structure, open-system dynamics, stochastic perturbations, and temporal evolution of the underlying physical model.
