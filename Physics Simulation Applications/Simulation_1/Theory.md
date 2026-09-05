# Physics Theory

## 1. Overview

This project studies a superconducting transmon qubit using numerical simulation and machine learning.

The physical system is modeled as a quantum system with multiple energy levels rather than as an ideal two-level qubit. The simulator retains the lowest ten energy eigenstates and includes microwave control, noise, relaxation, and dephasing.

The main physical-to-machine-learning mapping is

```math
\mathcal{G}:u(t)\longrightarrow\mathbf{P}(t)
```

where:

- $u(t)$ represents the effective time-dependent Hamiltonian coefficients.
- $\mathbf{P}(t)$ is the vector of populations of the ten energy levels.

---

# 2. Superconducting Qubits

A superconducting qubit is an artificial quantum system fabricated from superconducting electrical circuits.

The transmon is one of the most widely used superconducting-qubit architectures.

A transmon is based on a Josephson junction combined with a superconducting capacitance.

The two important energy scales are:

- Josephson energy $E_J$
- charging energy $E_C$

The transmon regime is characterized by

```math
\frac{E_J}{E_C}\gg1
```

A large ratio reduces sensitivity to charge noise while retaining sufficient anharmonicity for quantum control.

---

# 3. Josephson Junction

A Josephson junction consists of two superconductors separated by a thin insulating barrier.

The junction supports a nonlinear current-phase relationship

```math
I = I_c\sin\phi
```

where:

- $I$ is the supercurrent.
- $I_c$ is the critical current.
- $\phi$ is the superconducting phase difference.

The Josephson potential energy is

```math
U_J(\phi)=-E_J\cos\phi
```

where $E_J$ is the Josephson energy.

The relation between critical current and Josephson energy is

```math
E_J=\frac{\Phi_0 I_c}{2\pi}
```

where $\Phi_0$ is the superconducting flux quantum.

---

# 4. Charge and Phase Variables

The two conjugate variables of the transmon are:

- superconducting phase $\hat{\phi}$
- Cooper-pair number $\hat n$

They satisfy the commutation relation

```math
[\hat{\phi},\hat n]=i
```

The operator $\hat n$ represents the number of excess Cooper pairs on the superconducting island.

The charging energy depends on the difference between $\hat n$ and the offset charge $n_g$.

---

# 5. Full Transmon Hamiltonian

The full transmon Hamiltonian is

```math
H = 4E_C(\hat n-n_g)^2-E_J\cos(\hat{\phi})
```

The first term is the charging energy.

The second term is the nonlinear Josephson potential.

This cosine term is important because it makes the system anharmonic.

---

# 6. Charge-Basis Representation

The Hamiltonian can be represented in the charge basis

```math
|n\rangle
```

where $n$ is an integer representing the number of Cooper pairs.

The charge operator acts as

```math
\hat n|n\rangle=n|n\rangle
```

The phase operator generates transitions between neighboring charge states.

Using

```math
\cos\hat{\phi}
=
\frac{1}{2}
\left(
e^{i\hat{\phi}}+e^{-i\hat{\phi}}
\right)
```

the Josephson term couples neighboring charge states.

The Hamiltonian matrix therefore contains diagonal charging-energy terms and off-diagonal Josephson couplings.

---

# 7. Numerical Diagonalization

The simulator constructs the Hamiltonian in a sufficiently large charge basis.

The resulting matrix is numerically diagonalized:

```math
H|\psi_j\rangle=E_j|\psi_j\rangle
```

where:

- $|\psi_j\rangle$ is an energy eigenstate.
- $E_j$ is the corresponding energy eigenvalue.

The eigenvalues are ordered as

```math
E_0<E_1<E_2<\cdots
```

The simulator retains the lowest ten eigenstates.

Therefore the working Hilbert space has dimension

```math
d=10
```

and the density matrix has dimensions

```math
\rho\in\mathbb{C}^{10\times10}
```

---

# 8. Energy Levels

The ten eigenstates are written as

```math
|0\rangle,|1\rangle,\ldots,|9\rangle
```

with corresponding energies

```math
E_0,E_1,\ldots,E_9
```

The energy difference between adjacent levels determines the transition frequency.

```math
f_{j,j+1}=\frac{E_{j+1}-E_j}{h}
```

In particular,

```math
f_{01}=\frac{E_1-E_0}{h}
```

is the fundamental transition frequency.

---

# 9. Anharmonicity

An ideal harmonic oscillator has equally spaced energy levels.

A transmon does not.

The difference between adjacent transition frequencies is described by the anharmonicity.

```math
\alpha=f_{12}-f_{01}
```

For a transmon,

```math
\alpha<0
```

The approximate transmon energy spectrum is

```math
E_m\approx
-h\frac{E_C}{2}m(m-1)
+
hf_{01}m
```

with the precise spectrum obtained numerically from the full cosine Hamiltonian.

The anharmonicity allows the lowest two levels to be used as a qubit while higher levels remain physically relevant.

---

# 10. Why Use Ten Levels?

An ideal qubit contains only

```math
|0\rangle,\quad |1\rangle
```

A real transmon is not restricted to these two states.

Microwave driving can populate higher levels such as

```math
|2\rangle,|3\rangle,\ldots
```

This phenomenon is called leakage.

A ten-level model therefore allows the simulator to represent:

- qubit dynamics
- leakage
- higher-level transitions
- population redistribution
- non-ideal control

The state space used by the simulator is therefore

```math
\mathcal{H}_{10}
=
\operatorname{span}
\{
|0\rangle,\ldots,|9\rangle
\}
```

---

# 11. Quantum State

A pure quantum state can be represented by a state vector

```math
|\psi\rangle
```

However, the simulator includes noise and dissipation.

Therefore the appropriate representation is a density matrix

```math
\rho
```

The density matrix satisfies

```math
\rho^\dagger=\rho
```

and

```math
\operatorname{Tr}(\rho)=1
```

For a physical state,

```math
\rho\succeq0
```

---

# 12. Populations

The population of energy level $j$ is the diagonal element of the density matrix:

```math
P_j(t)=\langle j|\rho(t)|j\rangle
```

The ten-level population vector is

```math
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)]
```

The populations satisfy

```math
\sum_{j=0}^{9}P_j(t)=1
```

The population vector is the primary output of the machine-learning model.

---

# 13. Quantum Coherence

The off-diagonal elements of the density matrix describe quantum coherence.

```math
\rho_{ij}
=
\langle i|\rho|j\rangle
```

for $i\neq j$.

The diagonal elements describe populations, while the off-diagonal elements contain phase-coherence information.

Therefore

```math
\rho
=
\begin{pmatrix}
P_0 & \rho_{01} & \cdots\\
\rho_{10} & P_1 & \cdots\\
\vdots & \vdots & \ddots
\end{pmatrix}
```

contains more information than the population vector alone.

---

# 14. Microwave Control

Microwave pulses are used to control the transmon.

The control field can be represented using two quadratures:

```math
u(t)=
\begin{bmatrix}
I(t)\\
Q(t)
\end{bmatrix}
```

where:

- $I(t)$ is the in-phase component.
- $Q(t)$ is the quadrature component.

Together they determine the amplitude and phase of the microwave drive.

The complex envelope can be written as

```math
\Omega(t)=I(t)+iQ(t)
```

and therefore

```math
|\Omega(t)|=\sqrt{I(t)^2+Q(t)^2}
```

The instantaneous phase is

```math
\theta(t)=\operatorname{atan2}(Q(t),I(t))
```

---

# 15. Driven Hamiltonian

The total Hamiltonian can be represented as

```math
H(t)=H_0+H_{\mathrm{drive}}(t)
```

A general linear representation of the driven Hamiltonian is

```math
H(t)
=
H_0+
\sum_k u_k(t)H_k
```

where:

- $H_0$ is the static Hamiltonian.
- $H_k$ are control Hamiltonians.
- $u_k(t)$ are time-dependent control coefficients.

This representation is particularly useful for operator learning.

---

# 16. Effective Hamiltonian Coefficients

The neural operator does not need to reproduce every internal simulator variable.

Instead, the input is defined using the effective Hamiltonian coefficients that determine the time-dependent Hamiltonian.

Thus

```math
u(t)
=
[u_1(t),u_2(t),\ldots,u_{C_{\mathrm{in}}}(t)]
```

and

```math
H(t)
=
H_0+
\sum_{k=1}^{C_{\mathrm{in}}}
u_k(t)H_k
```

The resulting physical evolution produces

```math
\mathbf{P}(t)
=
[P_0(t),\ldots,P_9(t)]
```

---

# 17. Control Bandwidth

Real microwave electronics cannot change the control field infinitely fast.

The applied control therefore has finite bandwidth.

A simple representation is a filtered control signal

```math
u_{\mathrm{eff}}(t)
=
\mathcal{F}[u_{\mathrm{raw}}(t)]
```

where $\mathcal{F}$ represents the control-bandwidth response.

This prevents the simulator from using unrealistically rapid control changes.

---

# 18. Frequency Noise

The transition frequency of a real superconducting qubit fluctuates with time.

The effective frequency can therefore be written as

```math
\omega_q(t)
=
\omega_q^{(0)}
+
\delta\omega(t)
```

where:

- $\omega_q^{(0)}$ is the nominal frequency.
- $\delta\omega(t)$ is the frequency fluctuation.

Frequency fluctuations cause errors in the accumulated quantum phase.

---

# 19. Low-Frequency Noise

Low-frequency noise changes slowly compared with the qubit dynamics.

A general representation is

```math
\delta\omega(t)
=
\delta\omega_{\mathrm{slow}}(t)
```

Such fluctuations can produce slowly varying detuning and frequency drift.

These effects are important because a control pulse designed for the nominal frequency may become detuned from the actual transition frequency.

---

# 20. Charge Noise

The charging term depends on the offset charge $n_g$:

```math
H_C=4E_C(\hat n-n_g)^2
```

If the offset charge fluctuates,

```math
n_g(t)=n_{g,0}+\delta n_g(t)
```

then the Hamiltonian becomes time dependent.

Charge noise can therefore modify the energy spectrum and transition frequencies.

---

# 21. Amplitude Noise

The microwave amplitude can fluctuate around its intended value.

The effective amplitude can be written as

```math
A(t)=A_0(t)+\delta A(t)
```

where $\delta A(t)$ represents amplitude noise.

Amplitude fluctuations change the strength of the applied control and can therefore change the resulting population dynamics.

---

# 22. Phase Noise

The phase of the microwave signal can also fluctuate.

The effective phase can be represented as

```math
\theta(t)=\theta_0(t)+\delta\theta(t)
```

Phase noise modifies the orientation of the control field in the rotating frame.

---

# 23. Frequency Drift

Slow drift can cause the qubit frequency to move gradually away from its nominal value.

A simple representation is

```math
\omega_q(t)
=
\omega_q^{(0)}
+
\delta\omega_{\mathrm{noise}}(t)
+
\delta\omega_{\mathrm{drift}}(t)
```

The resulting detuning is

```math
\Delta(t)=\omega_d(t)-\omega_q(t)
```

where $\omega_d(t)$ is the drive frequency.

---

# 24. Two-Level-System Noise

Two-level systems, often associated with microscopic defects in materials or interfaces, can produce random changes in the qubit environment.

A simplified telegraph-noise model switches between discrete states:

```math
\xi(t)\in\{-1,+1\}
```

The corresponding frequency shift can be written as

```math
\delta\omega_{\mathrm{TLS}}(t)
=
A_{\mathrm{TLS}}\xi(t)
```

where $A_{\mathrm{TLS}}$ is the coupling strength.

---

# 25. Complete Effective Hamiltonian

The time-dependent Hamiltonian can therefore be viewed schematically as

```math
H(t)
=
H_0
+
H_{\mathrm{control}}(t)
+
H_{\mathrm{noise}}(t)
```

or, more generally,

```math
H(t)
=
H_0+
\sum_k u_k(t)H_k
```

where the effective coefficients $u_k(t)$ contain the relevant control and time-dependent physical effects.

---

# 26. Open Quantum Systems

An isolated quantum system evolves according to the Schrödinger equation.

A real superconducting qubit interacts with its environment.

Therefore its evolution is modeled as an open quantum system.

The state is represented by the density matrix $\rho(t)$.

---

# 27. Unitary Evolution

For a closed system,

```math
i\hbar\frac{d}{dt}|\psi(t)\rangle
=
H(t)|\psi(t)\rangle
```

For a time-independent Hamiltonian,

```math
|\psi(t)\rangle
=
e^{-iHt/\hbar}|\psi(0)\rangle
```

For the density matrix,

```math
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H,\rho]
```

where

```math
[H,\rho]=H\rho-\rho H
```

is the commutator.

---

# 28. Lindblad Master Equation

Dissipation and decoherence can be modeled using the Lindblad master equation:

```math
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H(t),\rho]
+
\sum_k
\left(
L_k\rho L_k^\dagger
-
\frac{1}{2}
\{L_k^\dagger L_k,\rho\}
\right)
```

where $L_k$ are Lindblad operators.

The anticommutator is

```math
\{A,B\}=AB+BA
```

The first term describes coherent quantum evolution.

The Lindblad terms describe environmental effects.

---

# 29. Energy Relaxation

Energy relaxation causes an excited state to lose energy and move toward lower-energy states.

For a transition from level $j$ to level $i$ with $i<j$, a relaxation operator can be written as

```math
L_{i\leftarrow j}
=
\sqrt{\Gamma_{j\rightarrow i}}
|i\rangle\langle j|
```

where $\Gamma_{j\rightarrow i}$ is the corresponding transition rate.

---

# 30. Pure Dephasing

Pure dephasing reduces quantum coherence without necessarily changing populations.

A dephasing process can be represented by an appropriate diagonal Lindblad operator.

The important distinction is:

- relaxation changes populations
- dephasing primarily changes coherences

Both processes can influence the final population dynamics indirectly.

---

# 31. Coherence Times

Two commonly used timescales are:

- $T_1$: energy relaxation time
- $T_2$: coherence time

A common relation is

```math
\frac{1}{T_2}
=
\frac{1}{2T_1}
+
\frac{1}{T_\phi}
```

where $T_\phi$ is the pure-dephasing time.

---

# 32. Numerical Time Evolution

The simulator propagates the density matrix over small internal time steps.

The physical evolution can be represented schematically as

```math
\rho(t+\Delta t)
=
\mathcal{E}_{\Delta t}[\rho(t)]
```

where $\mathcal{E}_{\Delta t}$ represents the numerical evolution operator over one time step.

The simulator uses a small internal timestep to resolve the dynamics accurately.

---

# 33. Operator Time Grid

The machine-learning dataset uses a coarser operator time grid than the internal simulator integration.

The operator timestep is

```math
\Delta t_{\mathrm{operator}}=2\,\mathrm{ns}
```

The simulator may perform multiple internal integration steps between two consecutive operator samples.

This separates numerical integration resolution from the resolution presented to the neural operator.

---

# 34. Input Function

The input to the neural operator is the time-dependent effective Hamiltonian coefficient function:

```math
u(t)
=
[u_1(t),u_2(t),\ldots,u_{C_{\mathrm{in}}}(t)]
```

For a batch of samples, the input tensor has the shape

```math
U\in\mathbb{R}^{B\times C_{\mathrm{in}}\times N}
```

where:

- $B$ is the batch size.
- $C_{\mathrm{in}}$ is the number of input channels.
- $N$ is the number of temporal points.

---

# 35. Output Function

The output is the ten-level population trajectory:

```math
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)]
```

For a batch of samples,

```math
Y\in\mathbb{R}^{B\times10\times N}
```

The machine-learning problem is therefore a function-to-function mapping.

---

# 36. Neural Operator Formulation

The central learning problem is

```math
\mathcal{G}:u(t)\longrightarrow\mathbf{P}(t)
```

The neural network approximates the unknown operator:

```math
\mathcal{G}_\theta\approx\mathcal{G}
```

where $\theta$ denotes the trainable parameters.

Therefore,

```math
\hat{\mathbf{P}}(t)
=
\mathcal{G}_\theta[u](t)
```

---

# 37. Why an Operator?

A conventional neural network often learns a mapping between fixed-dimensional vectors.

An operator learns a mapping between functions.

Here the input is an entire time-dependent function:

```math
u(t)
```

and the output is another time-dependent function:

```math
\mathbf{P}(t)
```

Therefore the problem naturally has the structure

```math
\text{function}
\longrightarrow
\text{function}
```

---

# 38. Channelwise Standardization

The input channels are standardized using statistics calculated from the training set.

For input channel $k$,

```math
\tilde u_k(t)
=
\frac{u_k(t)-\mu_k}{\sigma_k}
```

where:

- $\mu_k$ is the training-set mean.
- $\sigma_k$ is the training-set standard deviation.

The same training statistics are applied to the test set.

This prevents information from the test set from entering preprocessing.

---

# 39. FNO

The Fourier Neural Operator represents functions using spectral transformations.

A Fourier layer can be written schematically as

```math
v_{l+1}
=
\sigma
\left(
Wv_l
+
\mathcal{F}^{-1}
\left(
K\cdot\mathcal{F}(v_l)
\right)
\right)
```

where:

- $\mathcal{F}$ is the Fourier transform.
- $\mathcal{F}^{-1}$ is the inverse Fourier transform.
- $K$ contains learned Fourier-mode weights.
- $W$ is a learned pointwise transformation.
- $\sigma$ is a nonlinear activation.

The Fourier representation allows the model to learn global temporal interactions.

---

# 40. GNO

The Graph Neural Operator represents the function using interactions between points.

A generic graph-based operator layer can be expressed as

```math
v_i^{(l+1)}
=
\Phi
\left(
v_i^{(l)},
\sum_{j}
K_\theta(x_i,x_j)
v_j^{(l)}
\right)
```

where $K_\theta$ is a learned kernel describing interactions between points.

For temporal data, the graph can represent relationships between different time locations.

---

# 41. CATO

CATO is another operator-learning architecture used in the comparison.

The important point for this project is that FNO, GNO, and CATO are trained to approximate the same physical operator:

```math
\mathcal{G}:u(t)\longrightarrow\mathbf{P}(t)
```

The architectures differ in how they represent and learn the operator.

---

# 42. Dataset

The dataset contains:

- 200 training trajectories
- 100 test trajectories
- 5 random seeds

Each trajectory contains a time-dependent effective Hamiltonian input and the corresponding simulated ten-level population trajectory.

The training set and test set are kept separate.

---

# 43. Training Objective

The models are trained using mean squared error.

For predicted populations $\hat P_j(t)$ and reference populations $P_j(t)$,

```math
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{10N}
\sum_{j=0}^{9}
\sum_{n=1}^{N}
\left(
\hat P_j(t_n)-P_j(t_n)
\right)^2
```

For a batch, the loss is averaged over the batch samples.

---

# 44. Relative $L^2$ Error

The relative temporal $L^2$ error for a trajectory is

```math
\epsilon_{L^2}
=
\frac{
\|\hat{\mathbf{P}}-\mathbf{P}\|_2
}{
\|\mathbf{P}\|_2
}
```

For discrete time samples,

```math
\|\mathbf{P}\|_2
=
\left(
\sum_n
\|\mathbf{P}(t_n)\|_2^2
\right)^{1/2}
```

This measures the overall magnitude of the prediction error relative to the reference trajectory.

---

# 45. Temporal Sobolev $H^1$ Error

The $H^1$ norm measures both the function and its temporal derivative.

For a function $f(t)$,

```math
\|f\|_{H^1}^2
=
\|f\|_{L^2}^2
+
\left\|
\frac{df}{dt}
\right\|_{L^2}^2
```

The relative $H^1$ error is

```math
\epsilon_{H^1}
=
\frac{
\|\hat{\mathbf{P}}-\mathbf{P}\|_{H^1}
}{
\|\mathbf{P}\|_{H^1}
}
```

This metric is sensitive not only to population values but also to the temporal structure of the dynamics.

---

# 46. Finite-Difference Derivative

The temporal derivative is estimated numerically.

For an interior point,

```math
\frac{df}{dt}(t_n)
\approx
\frac{
f(t_{n+1})-f(t_{n-1})
}{
2\Delta t
}
```

At the boundaries, one-sided finite differences can be used.

The resulting derivative is used to calculate the temporal $H^1$ metric.

---

# 47. Physical Interpretation of the Errors

A low relative $L^2$ error means that the predicted population trajectories are close to the simulator trajectories in overall magnitude.

A low $H^1$ error additionally indicates that the model reproduces the temporal variation of the dynamics.

This distinction is important for quantum-control trajectories because two predictions can have similar population values while differing significantly in their temporal derivatives.

---

# 48. Random Seeds

The experiment uses five random seeds.

For each seed, the training and evaluation procedure is repeated.

The final performance can therefore be reported using statistics across the five runs.

For a metric $\epsilon$,

```math
\bar{\epsilon}
=
\frac{1}{5}
\sum_{s=1}^{5}\epsilon_s
```

The standard deviation can be reported as

```math
\sigma_\epsilon
=
\sqrt{
\frac{1}{5}
\sum_{s=1}^{5}
(\epsilon_s-\bar{\epsilon})^2
}
```

---

# 49. Physical Quantities for Analysis

The population trajectories allow several physically meaningful quantities to be examined.

The ground-state population is

```math
P_0(t)
```

The first-excited-state population is

```math
P_1(t)
```

The leakage population outside the computational subspace is

```math
P_{\mathrm{leak}}(t)
=
\sum_{j=2}^{9}P_j(t)
```

Since the total population is normalized,

```math
P_0(t)+P_1(t)+P_{\mathrm{leak}}(t)=1
```

This provides a direct measure of leakage during control.

---

# 50. Computational-Basis Excitation

The probability of occupying the excited computational state is

```math
P_1(t)
```

A control pulse intended to perform an excitation should ideally increase $P_1$ while keeping the leakage population small.

The population trajectory therefore provides a direct physical measure of control performance.

---

# 51. Complete Physics-to-ML Pipeline

The complete pipeline can be summarized as

```math
\text{Transmon Hamiltonian}
\longrightarrow
\text{Numerical Diagonalization}
\longrightarrow
\text{10-Level Quantum Model}
```

followed by

```math
\text{Microwave Control}
+
\text{Noise}
+
\text{Dissipation}
\longrightarrow
\rho(t)
```

then

```math
\rho(t)
\longrightarrow
\mathbf{P}(t)
```

and finally

```math
u(t)
\longrightarrow
\mathbf{P}(t)
```

for the neural-operator learning problem.

---

# 52. Complete Mathematical Picture

The physical system can be summarized by

```math
H(t)
=
H_0+
\sum_k u_k(t)H_k
```

with open-system dynamics

```math
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H(t),\rho]
+
\sum_k
\left(
L_k\rho L_k^\dagger
-
\frac{1}{2}
\{L_k^\dagger L_k,\rho\}
\right)
```

and measured output

```math
P_j(t)=\langle j|\rho(t)|j\rangle
```

giving

```math
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)]
```

The machine-learning objective is therefore

```math
\boxed{
\mathcal{G}_\theta
:
u(t)
\longrightarrow
\mathbf{P}(t)
}
```

where $\mathcal{G}_\theta$ is learned from simulated physical trajectories.
