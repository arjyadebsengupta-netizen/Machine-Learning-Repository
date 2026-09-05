# Physics-Based Neural Operator Learning for Noisy Superconducting Transmon Dynamics

## Overview

This project develops a physics-based dataset and neural-operator framework for learning the dynamical response of a noisy superconducting transmon system.

The current implementation contains:

- a full cosine transmon Hamiltonian in the charge basis,
- numerical diagonalization into the lowest ten energy eigenstates,
- finite-dimensional microwave control,
- stochastic frequency, charge, amplitude, phase, drift, and TLS noise,
- open-system Lindblad dynamics,
- fourth-order Runge-Kutta time integration,
- population-trajectory dataset generation,
- training-only input standardization,
- a PyTorch data pipeline,
- a one-dimensional Fourier Neural Operator (FNO),
- supervised FNO training,
- relative $L^2$ evaluation,
- and a discrete temporal $H^1$-type Sobolev error.

The present repository state implements the **physics simulator and FNO baseline**.

Graph Neural Operator (GNO) and CATO are intended as subsequent models in the neural-operator benchmark and are **not yet implemented in the current notebook**.

---

## Physical System

The physical system is a single superconducting transmon.

The transmon is represented using the full cosine Hamiltonian

$$
H_{\mathrm{tr}}
=
4E_C(\hat n-n_g)^2
-
E_J\cos(\hat\phi),
$$

where:

- $E_J$ is the Josephson energy,
- $E_C$ is the charging energy,
- $\hat n$ is the charge-number operator,
- $\hat\phi$ is the superconducting phase operator,
- $n_g$ is the offset charge.

The simulation uses the numerical charge basis

$$
n \in \{-n_{\mathrm{cut}},\ldots,n_{\mathrm{cut}}\}.
$$

With the current configuration,

$$
n_{\mathrm{cut}} = 50,
$$

giving a charge-basis dimension of

$$
N_{\mathrm{charge}} = 101.
$$

The Hamiltonian is constructed directly from the full cosine potential rather than from a truncated two-level qubit approximation.

---

## Energy Eigenbasis

The charge-basis Hamiltonian is numerically diagonalized.

If

$$
H_{\mathrm{tr}} V = V E,
$$

then the eigenvectors contained in $V$ define the energy eigenbasis.

Only the lowest ten eigenstates are retained:

$$
N_{\mathrm{levels}} = 10.
$$

The resulting finite-dimensional model therefore contains

$$
\left\{
|0\rangle,
|1\rangle,
\ldots,
|9\rangle
\right\}.
$$

The ground-state energy is subtracted from all retained eigenenergies:

$$
E_j \leftarrow E_j-E_0.
$$

This produces the working ten-level Hamiltonian

$$
H_0
=
\operatorname{diag}
(E_0-E_0,E_1-E_0,\ldots,E_9-E_0).
$$

The retained ten-level representation allows population transfer outside the computational subspace to higher retained states to be represented by the simulator.

The lowest transition frequencies are also extracted from the diagonalized spectrum, including the $0\rightarrow1$ and $1\rightarrow2$ transitions and the corresponding anharmonicity.

---

## Effective Hamiltonian Representation

The dynamical model is written in terms of a fixed set of effective Hamiltonian operators.

The Hamiltonian used by the simulator has the form

$$
H(t)
=
H_0
+
u_f(t)H_f
+
u_I(t)H_I
+
u_Q(t)H_Q
+
u_c(t)H_c.
$$

The four time-dependent input channels are

$$
u(t)
=
\begin{bmatrix}
u_f(t)\\
u_I(t)\\
u_Q(t)\\
u_c(t)
\end{bmatrix}.
$$

They correspond to:

| Channel | Symbol | Description |
|---|---|---|
| Frequency | $u_f(t)$ | Frequency-noise, drift, and TLS contribution |
| In-phase | $u_I(t)$ | Microwave in-phase control |
| Quadrature | $u_Q(t)$ | Microwave quadrature control |
| Charge | $u_c(t)$ | Charge-noise contribution |

The effective operator basis is constructed numerically in the ten-level energy eigenbasis.

### Frequency operator

The current implementation uses

$$
H_f
=
\operatorname{diag}(0,1,\ldots,9)
$$

as the effective frequency-perturbation basis.

This is a parameterized operator basis for the frequency channel; it is distinct from the physical static Hamiltonian $H_0$ containing the numerically calculated transmon eigenenergies.

### Charge operator

The charge operator is transformed from the charge basis into the retained energy eigenbasis.

The charge-noise Hamiltonian is constructed from the corresponding transformed charge operator.

### Microwave operators

A raising-like component of the transformed charge operator is used to construct Hermitian in-phase and quadrature control operators.

The current implementation uses

$$
H_I
=
n_+ + n_+^\dagger
$$

and

$$
H_Q
=
i(n_+-n_+^\dagger).
$$

Both operators are Hermitian.

---

## Microwave I/Q Control

The microwave control consists of two channels:

$$
u_I(t)
$$

and

$$
u_Q(t).
$$

The controls are generated as smooth random trajectories.

The current trajectory construction uses randomly sampled coarse control points followed by interpolation onto the simulation time grid.

The configured maximum control scale is

$$
u_{\max}=0.100\ \mathrm{GHz}.
$$

A control-bandwidth parameter is also defined in the configuration.

The current implementation generates smooth control trajectories through interpolation; it does not explicitly apply a separate numerical low-pass filter.

The resulting microwave contribution to the Hamiltonian is

$$
H_{\mathrm{control}}(t)
=
u_I(t)H_I
+
u_Q(t)H_Q.
$$

---

## Stochastic Noise Model

The simulator contains several stochastic effects.

The implemented input trajectories include:

1. frequency noise,
2. slow frequency drift,
3. TLS telegraph fluctuations,
4. charge noise,
5. microwave amplitude noise,
6. microwave phase noise.

These effects enter the effective Hamiltonian through the four input channels.

### Frequency noise

The frequency channel contains a slowly varying stochastic component.

The implementation generates this component using an autoregressive-style update rather than explicitly synthesizing a frequency-domain $1/f$ spectrum.

The configuration contains parameters associated with the intended frequency-noise model, including low-frequency, knee, and high-frequency scales.

### Frequency drift

A slowly varying frequency drift contribution is included.

The configured drift scale is

$$
\sigma_{\mathrm{drift}}
=
50\times10^{-6}\ \mathrm{GHz}.
$$

### TLS telegraph fluctuations

A two-state telegraph process is used to represent slow TLS-induced frequency shifts.

The configured TLS shift is

$$
\Delta f_{\mathrm{TLS}}
=
500\times10^{-6}\ \mathrm{GHz}.
$$

### Charge noise

Charge fluctuations enter through the charge channel.

The configured charge-noise scale is

$$
\sigma_{\mathrm{charge}}
=
2\times10^{-3}.
$$

### Microwave amplitude noise

Amplitude noise is applied to the complex microwave control.

The configured amplitude-noise scale is

$$
\sigma_{\mathrm{amp}}
=
10^{-4}.
$$

### Microwave phase noise

Phase noise is also applied to the complex microwave control.

The configured phase-noise scale is

$$
\sigma_{\mathrm{phase}}
=
10^{-3}\ \mathrm{rad}.
$$

---

## Open-System Dynamics

The system is treated as an open quantum system.

The density matrix $\rho(t)$ evolves according to the Lindblad master equation

$$
\frac{d\rho}{dt}
=
-i[H(t),\rho]
+
\sum_k
\mathcal{D}[L_k]\rho,
$$

with dissipator

$$
\mathcal{D}[L]\rho
=
L\rho L^\dagger
-
\frac{1}{2}
\left(
L^\dagger L\rho
+
\rho L^\dagger L
\right).
$$

The Hamiltonian and dissipative contributions are evaluated directly in the retained ten-level eigenbasis.

Because the Hamiltonian coefficients are specified in GHz while the time axis is represented in ns, the Hamiltonian contribution in the implementation uses the corresponding $2\pi$ conversion.

---

## Relaxation and Dephasing

The simulator includes relaxation and dephasing channels.

The configured relaxation time is

$$
T_1=30\ \mu\mathrm{s},
$$

and the configured coherence time is

$$
T_2=20\ \mu\mathrm{s}.
$$

They are converted to ns internally:

$$
T_1=30000\ \mathrm{ns},
$$

$$
T_2=20000\ \mathrm{ns}.
$$

The corresponding rates are

$$
\gamma_1=\frac{1}{T_1},
$$

and

$$
\gamma_2=\frac{1}{T_2}.
$$

The pure-dephasing contribution is calculated as

$$
\gamma_\phi
=
\gamma_2-\frac{\gamma_1}{2}.
$$

### Relaxation operators

Successive retained levels are connected through relaxation operators

$$
L_j
=
\sqrt{\gamma_1}
|j-1\rangle\langle j|,
\qquad
j=1,\ldots,9.
$$

The current implementation therefore uses the configured relaxation rate for the successive-level relaxation channels.

### Dephasing operator

A diagonal dephasing operator based on the retained level index is also included.

The implementation uses the diagonal structure

$$
L_\phi
\propto
\operatorname{diag}(0,1,\ldots,9).
$$

---

## Numerical Time Evolution

The full physical simulation uses two time scales.

The operator-learning time grid is

$$
t\in[0,200]\ \mathrm{ns}
$$

with

$$
\Delta t_{\mathrm{operator}}=2\ \mathrm{ns}.
$$

Therefore,

$$
N_t=101.
$$

For each operator-level interval, the physical density-matrix evolution is resolved using an internal time step of

$$
\Delta t_{\mathrm{internal}}
=
0.05\ \mathrm{ns}.
$$

Thus each 2 ns operator interval contains

$$
\frac{2}{0.05}=40
$$

internal integration steps.

The density matrix is propagated using fourth-order Runge-Kutta integration.

For a state $\rho_n$,

$$
k_1=f(t_n,\rho_n),
$$

$$
k_2=f\left(t_n+\frac{\Delta t}{2},
\rho_n+\frac{\Delta t}{2}k_1\right),
$$

$$
k_3=f\left(t_n+\frac{\Delta t}{2},
\rho_n+\frac{\Delta t}{2}k_2\right),
$$

$$
k_4=f(t_n+\Delta t,\rho_n+\Delta t\,k_3),
$$

and

$$
\rho_{n+1}
=
\rho_n
+
\frac{\Delta t}{6}
(k_1+2k_2+2k_3+k_4).
$$

---

## Population Extraction

The machine-learning target is the population trajectory of the ten retained energy levels.

For each time point,

$$
P_j(t)
=
\langle j|\rho(t)|j\rangle.
$$

The resulting population vector is

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

The populations are clipped to remove numerical negative values and renormalized so that

$$
\sum_{j=0}^{9}P_j(t)=1.
$$

Therefore the dataset output has ten channels:

$$
\mathbf{P}(t)\in\mathbb{R}^{10}.
$$

---

## Operator-Learning Problem

The physical simulator defines a mapping from a time-dependent input function to a time-dependent output function.

The operator-learning problem is

$$
\mathcal{G}:u(t)\mapsto\mathbf{P}(t).
$$

The input is

$$
u(t)\in\mathbb{R}^{4},
$$

with four channels:

$$
u(t)
=
[u_f(t),u_I(t),u_Q(t),u_c(t)].
$$

The output is

$$
\mathbf{P}(t)\in\mathbb{R}^{10}.
$$

For the discretized dataset,

$$
U\in\mathbb{R}^{4\times101},
$$

and

$$
Y\in\mathbb{R}^{10\times101}.
$$

With a batch dimension, the tensors are represented as

$$
U\in\mathbb{R}^{B\times4\times101},
$$

and

$$
Y\in\mathbb{R}^{B\times10\times101}.
$$

The neural operator therefore learns

$$
\mathcal{G}_\theta:
\mathbb{R}^{4\times101}
\rightarrow
\mathbb{R}^{10\times101}.
$$

---

## Dataset Generation

Each trajectory is generated independently from a random seed.

For every sample:

1. smooth microwave control trajectories are generated,
2. stochastic noise trajectories are generated,
3. the four effective Hamiltonian coefficients are assembled,
4. the initial density matrix is prepared in the ground state,
5. the density matrix is propagated through the physical simulator,
6. the ten level populations are extracted,
7. the input and output trajectories are stored.

The resulting dataset has the structure

```text
U_train : [N_train, 4, 101]
Y_train : [N_train, 10, 101]

U_test  : [N_test, 4, 101]
Y_test  : [N_test, 10, 101]
```

The active sample counts are controlled through the project `CONFIG` dictionary.

The train and test trajectories are generated using independent random seeds.

The implementation also checks that:

- all generated values are finite,
- population sums remain normalized,
- the training and test trajectories are not identical.

---

## Input Standardization

Only the input channels are standardized.

The training-set mean and standard deviation are calculated independently for each of the four input channels.

For an input channel $c$,

$$
\mu_c
=
\operatorname{mean}(U_{\mathrm{train},c}),
$$

and

$$
\sigma_c
=
\operatorname{std}(U_{\mathrm{train},c}).
$$

The standardized input is

$$
\widetilde{U}_c
=
\frac{U_c-\mu_c}{\sigma_c}.
$$

A small numerical floor is applied to the standard deviation to prevent division by zero.

The test inputs are transformed using the **training-set** statistics:

$$
\widetilde{U}_{\mathrm{test},c}
=
\frac{U_{\mathrm{test},c}-\mu_c}{\sigma_c}.
$$

No test-set statistics are used during preprocessing.

The population targets are not standardized.

---

## PyTorch Dataset Representation

The standardized arrays are converted to PyTorch tensors using `float32`.

The dataset returns:

```text
input  -> [4, 101]
target -> [10, 101]
```

A batch produced by the `DataLoader` therefore has the shape

```text
input  -> [B, 4, 101]
target -> [B, 10, 101]
```

The training loader uses shuffling, while the test loader does not.

The configured batch size is

$$
B=8.
$$

---

## Fourier Neural Operator

The first neural operator implemented in the project is a one-dimensional Fourier Neural Operator.

The FNO operates directly on the temporal coordinate.

The input is

$$
U\in\mathbb{R}^{B\times4\times101},
$$

and the output is

$$
\widehat{Y}\in\mathbb{R}^{B\times10\times101}.
$$

No additional time-coordinate channel is appended to the four physical input channels.

---

## Spectral Convolution

The FNO uses a one-dimensional Fourier spectral convolution.

For an input $x(t)$, the discrete Fourier transform is computed as

$$
\widehat{x}(k)
=
\mathcal{F}[x](k).
$$

Only a fixed number of low-frequency Fourier modes are retained.

The spectral convolution applies a learned complex-valued weight to these modes:

$$
\widehat{y}(k)
=
W(k)\widehat{x}(k).
$$

The transformed representation is then mapped back to physical space using the inverse Fourier transform:

$$
y(t)
=
\mathcal{F}^{-1}[\widehat{y}](t).
$$

The current implementation uses

$$
N_{\mathrm{modes}}=16
$$

and a hidden width of

$$
W=32.
$$

---

## FNO Architecture

The current FNO consists of:

1. an input projection from four channels to 32 channels,
2. four spectral-convolution layers,
3. four pointwise convolution layers,
4. GELU nonlinearities,
5. an output projection from 32 hidden channels to ten population channels.

Each FNO block combines the spectral convolution and pointwise convolution contributions before applying the GELU activation.

The architecture can be summarized as

```text
Input
  [B, 4, 101]
       |
       v
Input projection
  [B, 32, 101]
       |
       v
+-------------------------+
| Spectral Conv 1d        |
| Pointwise Conv 1d       |
| GELU                    |
+-------------------------+
       |
       v
+-------------------------+
| Spectral Conv 1d        |
| Pointwise Conv 1d       |
| GELU                    |
+-------------------------+
       |
       v
+-------------------------+
| Spectral Conv 1d        |
| Pointwise Conv 1d       |
| GELU                    |
+-------------------------+
       |
       v
+-------------------------+
| Spectral Conv 1d        |
| Pointwise Conv 1d       |
| GELU                    |
+-------------------------+
       |
       v
Output projection
  [B, 10, 101]
```

The FNO is implemented directly in PyTorch.

---

## FNO Training

The training objective is the mean-squared error between predicted and simulated population trajectories.

The loss is

$$
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{N}
\sum
\left(
\widehat{Y}-Y
\right)^2.
$$

The optimizer is Adam with learning rate

$$
\eta=10^{-3}.
$$

The number of training epochs is controlled through

```python
CONFIG["epochs"]
```

The current implementation records the mean training loss for every epoch.

Training also records the elapsed time for each epoch and the total training time.

---

## Evaluation Metrics

The current implementation evaluates the FNO using:

1. test-set MSE,
2. relative $L^2$ error,
3. relative temporal $H^1$-type error.

### Test MSE

The test mean-squared error is

$$
\mathcal{L}_{\mathrm{test}}
=
\frac{1}{N}
\sum
\left(
\widehat{Y}-Y
\right)^2.
$$

### Relative $L^2$ Error

For each test trajectory,

$$
\epsilon_{L^2}
=
\frac{
\|\widehat{Y}-Y\|_2
}{
\|Y\|_2
}.
$$

The reported metric is the mean of this quantity over the test samples.

### Temporal $H^1$-Type Error

The temporal derivative is approximated numerically using finite differences on the operator time grid.

For a trajectory $Y(t)$,

$$
\partial_tY(t)
$$

is approximated using centered differences in the interior and one-sided differences at the endpoints.

The temporal $H^1$-type norm is then represented by

$$
\|Y\|_{H^1_t}^2
=
\|Y\|_2^2
+
\|\partial_tY\|_2^2.
$$

The relative temporal error is

$$
\epsilon_{H^1}
=
\frac{
\sqrt{
\|\widehat{Y}-Y\|_2^2
+
\|\partial_t\widehat{Y}
-
\partial_tY\|_2^2
}
}{
\sqrt{
\|Y\|_2^2
+
\|\partial_tY\|_2^2
}
}.
$$

The current implementation evaluates this quantity on the discretized temporal grid.

---

## Current Computational Pipeline

The implemented pipeline is:

```text
Full cosine transmon Hamiltonian
              |
              v
Charge-basis numerical diagonalization
              |
              v
Lowest 10 energy eigenstates
              |
              v
Effective Hamiltonian operator basis
              |
              v
Smooth I/Q controls + stochastic noise
              |
              v
Four-channel Hamiltonian trajectory
              |
              v
Lindblad master equation
              |
              v
RK4 density-matrix evolution
              |
              v
Ten-level population trajectory
              |
              v
Physics-generated dataset
              |
              v
Training-only input standardization
              |
              v
PyTorch DataLoader
              |
              v
Fourier Neural Operator
              |
              v
Predicted ten-level populations
              |
              v
MSE + Relative L2 + Temporal H1-type error
```

---

## Current Implementation Status

### Implemented

- [x] Full cosine transmon Hamiltonian
- [x] Charge-basis construction
- [x] Numerical diagonalization
- [x] Ten-level energy eigenbasis
- [x] Effective frequency operator
- [x] Microwave I/Q control operators
- [x] Charge-noise operator
- [x] Frequency noise
- [x] Charge noise
- [x] Amplitude noise
- [x] Phase noise
- [x] Slow frequency drift
- [x] TLS telegraph fluctuations
- [x] Lindblad relaxation
- [x] Lindblad dephasing
- [x] RK4 density-matrix evolution
- [x] Population extraction
- [x] Physics-based dataset generation
- [x] Training-only input standardization
- [x] PyTorch dataset and DataLoader
- [x] One-dimensional spectral convolution
- [x] Fourier Neural Operator
- [x] FNO training
- [x] Test-set evaluation
- [x] Relative $L^2$ metric
- [x] Temporal $H^1$-type metric

### Planned

- [ ] Graph Neural Operator (GNO)
- [ ] CATO
- [ ] Common benchmark across FNO, GNO, and CATO
- [ ] Multi-seed statistical evaluation
- [ ] Extended operator-learning experiments

---

## Notebook Structure

The current notebook is organized as a sequential physics-to-machine-learning pipeline.

| Cell | Component |
|---:|---|
| 1 | Imports |
| 2 | Configuration and simulation parameters |
| 3 | Charge-basis transmon Hamiltonian and numerical diagonalization |
| 4 | Initial control-operator construction |
| 5 | Noise, control, and open-system parameters |
| 6 | Random state and trajectory initialization |
| 7 | Relaxation and dephasing operators |
| 8 | Final effective Hamiltonian operator basis |
| 9 | Hamiltonian construction and Lindblad dynamics |
| 10 | Control/noise trajectory generation and single-trajectory simulation |
| 11 | Dataset generation |
| 12 | Training-only input standardization |
| 13 | PyTorch Dataset and DataLoader |
| 14 | One-dimensional spectral convolution |
| 15 | Fourier Neural Operator |
| 16 | Loss and optimizer |
| 17 | FNO training |
| 18 | FNO test evaluation |
| 19 | Temporal $H^1$-type evaluation |
| 20 | Training/evaluation visualization and summary |
| 21 | Current end-to-end FNO result state |

Cell 8 defines the effective operator basis used by the subsequent physical simulation and dataset generation.

---

## Reproducibility

The experiment uses explicit random seeds.

The current experiment seed is

```text
1234
```

The dataset-generation procedure uses independent spawned random seeds for individual trajectories.

The test dataset uses a separate seed offset from the training seed.

The main simulation parameters are centralized in the `CONFIG` dictionary.

Important parameters include:

```text
EJ_GHz
EC_GHz
ng0
n_cut
n_levels
dt_env_ns
dt_internal_ns
total_time_ns
operator_dt_ns
batch_size
learning_rate
epochs
```

The simulation automatically selects the available compute device according to the configured device-selection logic.

---

## Numerical Configuration

The main physical and numerical scales currently used by the simulator include:

| Parameter | Value |
|---|---:|
| $E_J$ | $20.0$ GHz |
| $E_C$ | $0.30$ GHz |
| $n_{\mathrm{cut}}$ | $50$ |
| Retained levels | $10$ |
| Operator time step | $2.0$ ns |
| Internal time step | $0.05$ ns |
| Total simulation time | $200$ ns |
| Operator-grid points | $101$ |
| FNO modes | $16$ |
| FNO width | $32$ |
| Batch size | $8$ |
| Learning rate | $10^{-3}$ |

The active train/test sample counts and epoch count are controlled by `CONFIG`.

---

## Research Objective

The objective is to determine how accurately neural operators can learn the input-output map of a physics-based, noisy superconducting-qubit simulator.

The learned operator is

$$
\mathcal{G}_\theta
:
\left[
u_f(t),
u_I(t),
u_Q(t),
u_c(t)
\right]
\mapsto
\left[
P_0(t),
P_1(t),
\ldots,
P_9(t)
\right].
$$

The important feature of the problem is that the training data are generated by an explicit physical simulator rather than by an arbitrary synthetic regression function.

The neural operator therefore approximates a dynamical map generated by:

- a nonlinear superconducting-qubit Hamiltonian,
- finite-dimensional energy-level structure,
- microwave control,
- stochastic environmental perturbations,
- relaxation,
- dephasing,
- and numerical open-system time evolution.

---

## Future Neural-Operator Benchmark

The current FNO provides the first neural-operator baseline.

The intended benchmark is:

$$
\boxed{
\mathrm{FNO}
\quad\text{vs.}\quad
\mathrm{GNO}
\quad\text{vs.}\quad
\mathrm{CATO}
}
$$

All models will ultimately be evaluated on the same physics-generated dataset and under the same train/test protocol.

The comparison will focus on:

- predictive accuracy,
- relative $L^2$ error,
- temporal $H^1$-type error,
- training cost,
- inference cost,
- parameter count,
- and robustness of the learned operator.

GNO and CATO are future components of the benchmark and are not part of the current implemented pipeline.

---

## Core Mathematical Problem

The complete problem can be summarized as follows.

A stochastic four-channel Hamiltonian coefficient function is generated:

$$
u(t)
=
\begin{bmatrix}
u_f(t)\\
u_I(t)\\
u_Q(t)\\
u_c(t)
\end{bmatrix}.
$$

It determines the time-dependent Hamiltonian

$$
H(t)
=
H_0
+
u_f(t)H_f
+
u_I(t)H_I
+
u_Q(t)H_Q
+
u_c(t)H_c.
$$

The density matrix evolves according to

$$
\frac{d\rho}{dt}
=
-i[H(t),\rho]
+
\sum_k
\mathcal{D}[L_k]\rho.
$$

The simulator produces the population trajectory

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

The machine-learning task is therefore

$$
\boxed{
\mathcal{G}:u(t)\longmapsto\mathbf{P}(t)
}
$$

with discretized representation

$$
\boxed{
\mathcal{G}:
\mathbb{R}^{4\times101}
\longrightarrow
\mathbb{R}^{10\times101}.
}
$$

The current implementation learns this operator using a Fourier Neural Operator.

---

## Project Status

The project currently represents the following stage:

```text
Physics Simulator
        |
        v
Physics-Generated Dataset
        |
        v
FNO Baseline
        |
        v
Quantitative Evaluation
        |
        v
Future GNO / CATO Benchmark
```

The present code therefore establishes the physics-based operator-learning benchmark infrastructure and its first neural-operator model.

