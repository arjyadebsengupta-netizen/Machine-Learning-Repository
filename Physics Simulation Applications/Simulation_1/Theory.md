# Physics Theory

## 1. Superconducting Qubit

A superconducting qubit is an artificial quantum system built from superconducting electrical circuits.

The transmon is based on a Josephson junction and a capacitance. Its two main energy scales are the Josephson energy $E_J$ and the charging energy $E_C$.

The transmon regime is

```math
\frac{E_J}{E_C}\gg1
```

A large ratio reduces sensitivity to charge fluctuations while preserving enough anharmonicity for quantum control.

---

## 2. Josephson Junction

A Josephson junction consists of two superconductors separated by a thin insulating barrier.

The current-phase relation is

```math
I=I_c\sin\phi
```

where $I_c$ is the critical current and $\phi$ is the superconducting phase difference.

The Josephson energy is

```math
E_J=\frac{\Phi_0 I_c}{2\pi}
```

The Josephson potential is

```math
U_J(\phi)=-E_J\cos\phi
```

The cosine potential introduces the nonlinearity required for an anharmonic qubit.

---

## 3. Charge and Phase

The transmon is described using the conjugate variables $\hat n$ and $\hat\phi$.

The charge operator $\hat n$ represents the number of excess Cooper pairs.

The phase operator $\hat\phi$ describes the superconducting phase difference.

Their commutation relation is

```math
[\hat\phi,\hat n]=i
```

---

## 4. Full Transmon Hamiltonian

The full transmon Hamiltonian is

```math
H=4E_C(\hat n-n_g)^2-E_J\cos(\hat\phi)
```

The charging contribution is

```math
H_C=4E_C(\hat n-n_g)^2
```

The Josephson contribution is

```math
H_J=-E_J\cos(\hat\phi)
```

Therefore

```math
H=H_C+H_J
```

---

## 5. Charge Basis

The Hamiltonian can be represented in the charge basis $|n\rangle$.

The charge operator acts as

```math
\hat n|n\rangle=n|n\rangle
```

The cosine term can be written as

```math
\cos(\hat\phi)
=
\frac{1}{2}
\left(
e^{i\hat\phi}+e^{-i\hat\phi}
\right)
```

The exponential phase operators connect neighboring charge states.

This produces a Hamiltonian matrix containing diagonal charging-energy terms and off-diagonal Josephson couplings.

---

## 6. Numerical Diagonalization

The Hamiltonian is constructed in a sufficiently large charge basis and numerically diagonalized.

The eigenvalue equation is

```math
H|j\rangle=E_j|j\rangle
```

The energies are ordered as

```math
E_0<E_1<E_2<\cdots
```

The simulator retains the lowest ten energy levels:

```math
|0\rangle,|1\rangle,\ldots,|9\rangle
```

The working Hilbert-space dimension is therefore

```math
d=10
```

and the density matrix has dimension

```math
\rho\in\mathbb{C}^{10\times10}
```

---

## 7. Transition Frequencies

The transition frequency between adjacent levels is

```math
f_{j,j+1}=\frac{E_{j+1}-E_j}{h}
```

The fundamental qubit transition is

```math
f_{01}=\frac{E_1-E_0}{h}
```

Higher transitions include

```math
f_{12}=\frac{E_2-E_1}{h}
```

and

```math
f_{23}=\frac{E_3-E_2}{h}
```

---

## 8. Anharmonicity

For a harmonic oscillator, the energy levels are equally spaced.

A transmon is anharmonic.

The anharmonicity is defined as

```math
\alpha=f_{12}-f_{01}
```

For a transmon,

```math
\alpha<0
```

The non-equally-spaced levels allow the lowest two states to serve as the computational states.

---

## 9. Leakage

The computational subspace consists of

```math
|0\rangle,\quad |1\rangle
```

Microwave driving can populate higher levels such as

```math
|2\rangle,|3\rangle,\ldots
```

This is called leakage.

For the ten-level model, the leakage population is

```math
P_{\mathrm{leak}}(t)
=
\sum_{j=2}^{9}P_j(t)
```

The total population satisfies

```math
\sum_{j=0}^{9}P_j(t)=1
```
# 10. Quantum State and Density Matrix

A quantum state can be represented by a state vector $|\psi\rangle$ when the system is in a pure state.

For an open quantum system affected by noise and dissipation, the density matrix $\rho$ is used.

The density matrix satisfies

```math
\rho^\dagger=\rho
```

and

```math
\mathrm{Tr}(\rho)=1
```

For a physical quantum state, $\rho$ must also be positive semidefinite.

---

## 11. Density Matrix in the Energy Basis

In the ten-level energy basis, the density matrix is a $10\times10$ matrix:

```math
\rho=
\begin{pmatrix}
\rho_{00} & \rho_{01} & \cdots & \rho_{09}\\
\rho_{10} & \rho_{11} & \cdots & \rho_{19}\\
\vdots & \vdots & \ddots & \vdots\\
\rho_{90} & \rho_{91} & \cdots & \rho_{99}
\end{pmatrix}
```

The diagonal elements describe level populations.

The off-diagonal elements describe quantum coherences.

---

## 12. Level Populations

The population of level $j$ is

```math
P_j(t)=\langle j|\rho(t)|j\rangle
```

For the ten-level system, the population vector is

```math
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)]
```

The populations satisfy

```math
\sum_{j=0}^{9}P_j(t)=1
```

The population vector is the primary output used in the neural-operator dataset.

---

## 13. Quantum Coherence

The matrix element between two different levels is

```math
\rho_{ij}(t)=\langle i|\rho(t)|j\rangle
```

for $i\neq j$.

The diagonal terms

```math
\rho_{jj}(t)
```

represent populations.

The off-diagonal terms

```math
\rho_{ij}(t),\quad i\neq j
```

represent coherences between energy levels.

---

# 14. Microwave Control

The transmon is controlled using microwave signals.

The microwave control is represented by two quadratures:

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

The complex microwave envelope can be written as

```math
\Omega(t)=I(t)+iQ(t)
```

The amplitude is

```math
|\Omega(t)|
=
\sqrt{I(t)^2+Q(t)^2}
```

and the phase is

```math
\theta(t)=\mathrm{atan2}(Q(t),I(t))
```

---

# 15. Driven Hamiltonian

The total Hamiltonian can be separated into a static part and a control-dependent part:

```math
H(t)=H_0+H_{\mathrm{drive}}(t)
```

A general effective representation is

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

This representation provides the connection between the physical simulator and the neural operator.

---

# 16. Effective Hamiltonian Input

The machine-learning input is the set of effective Hamiltonian coefficients:

```math
u(t)
=
[u_1(t),u_2(t),\ldots,u_{C_{\mathrm{in}}}(t)]
```

The corresponding Hamiltonian is

```math
H(t)
=
H_0+
\sum_{k=1}^{C_{\mathrm{in}}}
u_k(t)H_k
```

The quantum system evolves under this Hamiltonian and produces the population trajectory

```math
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)]
```

---

# 17. Control Bandwidth

A real microwave control system has finite bandwidth.

Therefore the applied control cannot change arbitrarily quickly.

The effective control can be represented as

```math
u_{\mathrm{eff}}(t)
=
\mathcal{F}
\left[
u_{\mathrm{raw}}(t)
\right]
```

where $\mathcal{F}$ represents the response of the control system.

Finite bandwidth smooths rapid changes in the control signal and affects the resulting quantum dynamics.

---

# 18. Frequency Noise

The qubit transition frequency can fluctuate around its nominal value.

The instantaneous frequency can be written as

```math
\omega_q(t)
=
\omega_q^{(0)}
+
\delta\omega(t)
```

where:

- $\omega_q^{(0)}$ is the nominal qubit frequency.
- $\delta\omega(t)$ is the frequency fluctuation.

The corresponding detuning from a drive at frequency $\omega_d(t)$ is

```math
\Delta(t)
=
\omega_d(t)-\omega_q(t)
```

Frequency fluctuations therefore modify the detuning during a control sequence.

---

# 19. Low-Frequency Noise

Low-frequency noise varies slowly compared with the characteristic timescale of the qubit dynamics.

It can be represented as a slowly varying frequency fluctuation:

```math
\delta\omega_{\mathrm{slow}}(t)
```

The resulting qubit frequency becomes

```math
\omega_q(t)
=
\omega_q^{(0)}
+
\delta\omega_{\mathrm{slow}}(t)
```

Such fluctuations can produce slowly varying detuning errors.

---

# 20. Charge Noise

The charging energy depends on the offset charge $n_g$:

```math
H_C=4E_C(\hat n-n_g)^2
```

If the offset charge fluctuates,

```math
n_g(t)=n_{g,0}+\delta n_g(t)
```

then the charging Hamiltonian becomes time dependent:

```math
H_C(t)
=
4E_C
\left(
\hat n-n_g(t)
\right)^2
```

Charge fluctuations can therefore modify the energy spectrum and transition frequencies.

---

# 21. Amplitude Noise

The amplitude of the microwave control can fluctuate around its intended value.

The effective amplitude can be written as

```math
A(t)=A_0(t)+\delta A(t)
```

where:

- $A_0(t)$ is the intended amplitude.
- $\delta A(t)$ is the amplitude fluctuation.

Amplitude noise changes the strength of the applied control.

---

# 22. Phase Noise

The phase of the microwave signal can also fluctuate.

The instantaneous phase can be written as

```math
\theta(t)=\theta_0(t)+\delta\theta(t)
```

where $\theta_0(t)$ is the intended phase and $\delta\theta(t)$ represents phase noise.

Phase fluctuations change the effective direction of the microwave control in the rotating frame.

---

# 23. Frequency Drift

Slow frequency drift can gradually move the qubit away from its nominal operating frequency.

The instantaneous frequency can be written as

```math
\omega_q(t)
=
\omega_q^{(0)}
+
\delta\omega_{\mathrm{noise}}(t)
+
\delta\omega_{\mathrm{drift}}(t)
```

The corresponding detuning is

```math
\Delta(t)
=
\omega_d(t)-\omega_q(t)
```

This detuning affects the effectiveness of microwave control.
# 24. Two-Level-System Noise

Two-level systems (TLS) are microscopic defects that can interact with a superconducting qubit.

A simple model represents the TLS state using a switching variable

```math
\xi(t)\in\{-1,+1\}
```

The resulting frequency fluctuation can be written as

```math
\delta\omega_{\mathrm{TLS}}(t)
=
A_{\mathrm{TLS}}\xi(t)
```

where $A_{\mathrm{TLS}}$ is the strength of the frequency shift.

TLS fluctuations can therefore produce time-dependent changes in the qubit frequency.

---

# 25. Open Quantum Systems

A real superconducting qubit interacts with its environment.

The state is therefore described by a density matrix $\rho(t)$ rather than only a state vector.

For a closed quantum system, the density matrix evolves according to

```math
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H,\rho]
```

where the commutator is

```math
[H,\rho]=H\rho-\rho H
```

Environmental interactions introduce additional terms into the evolution.

---

# 26. Lindblad Master Equation

The open-system dynamics are described by the Lindblad master equation

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

where $L_k$ are Lindblad operators describing different environmental processes.

The anticommutator is

```math
\{A,B\}=AB+BA
```

The first term describes coherent quantum evolution.

The remaining terms describe dissipation and decoherence.

---

# 27. Energy Relaxation

Energy relaxation causes an excited state to lose energy and transition toward a lower energy state.

For a transition from level $j$ to level $i$, where $i<j$, a relaxation operator can be written as

```math
L_{i\leftarrow j}
=
\sqrt{\Gamma_{j\rightarrow i}}
|i\rangle\langle j|
```

where $\Gamma_{j\rightarrow i}$ is the relaxation rate.

Relaxation changes the populations of the energy levels.

---

# 28. Pure Dephasing

Pure dephasing reduces the coherence between quantum states without directly transferring population between energy levels.

The off-diagonal density-matrix elements describe this coherence:

```math
\rho_{ij}(t)=\langle i|\rho(t)|j\rangle
```

for $i\neq j$.

Dephasing causes these off-diagonal terms to decay with time.

---

# 29. Relaxation and Coherence Times

The energy relaxation time is denoted by $T_1$.

The coherence time is denoted by $T_2$.

A common relation is

```math
\frac{1}{T_2}
=
\frac{1}{2T_1}
+
\frac{1}{T_\phi}
```

where $T_\phi$ is the pure-dephasing time.

Thus both energy relaxation and pure dephasing can contribute to loss of coherence.

---

# 30. Complete Time-Dependent Hamiltonian

The physical Hamiltonian can be viewed schematically as

```math
H(t)
=
H_0
+
H_{\mathrm{control}}(t)
+
H_{\mathrm{noise}}(t)
```

The effective representation used for the machine-learning problem is

```math
H(t)
=
H_0+
\sum_k u_k(t)H_k
```

The coefficients $u_k(t)$ contain the effective time-dependent information that determines the Hamiltonian.

---

# 31. Quantum Evolution

The density matrix evolves from an initial state

```math
\rho(0)
```

according to the time-dependent master equation.

The evolution can be written schematically as

```math
\rho(t+\Delta t)
=
\mathcal{E}_{\Delta t}
\left[
\rho(t)
\right]
```

where $\mathcal{E}_{\Delta t}$ represents the numerical evolution over a timestep $\Delta t$.

The simulator uses small internal timesteps to resolve the quantum dynamics.

---

# 32. Temporal Sampling

The simulator uses an internal timestep smaller than the timestep used to construct the machine-learning dataset.

The operator-learning timestep is

```math
\Delta t_{\mathrm{operator}}=2\,\mathrm{ns}
```

The simulated trajectory is sampled on this operator time grid to obtain the input and output functions.

---

# 33. Physics-to-Dataset Mapping

For each simulation, the effective Hamiltonian coefficients are recorded as time-dependent input functions:

```math
u(t)
=
[u_1(t),u_2(t),\ldots,u_{C_{\mathrm{in}}}(t)]
```

The quantum simulation produces the density matrix

```math
\rho(t)
```

The diagonal elements are extracted to obtain the ten-level populations:

```math
P_j(t)=\langle j|\rho(t)|j\rangle
```

The resulting output function is

```math
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)]
```

Therefore each dataset sample contains

```math
u(t)\longrightarrow\mathbf{P}(t)
```

---

# 34. Neural Operator Problem

The neural operator learns the physical mapping

```math
\mathcal{G}:u(t)\longrightarrow\mathbf{P}(t)
```

The learned approximation is

```math
\mathcal{G}_\theta\approx\mathcal{G}
```

and the predicted population trajectory is

```math
\hat{\mathbf{P}}(t)
=
\mathcal{G}_\theta[u](t)
```

The goal is to reproduce the simulator's population trajectories without explicitly solving the full quantum dynamics for every new input.

---

# 35. Input Tensor

For a batch of trajectories, the input is stored as

```math
U\in\mathbb{R}^{B\times C_{\mathrm{in}}\times N}
```

where:

- $B$ is the batch size.
- $C_{\mathrm{in}}$ is the number of effective Hamiltonian input channels.
- $N$ is the number of temporal samples.

---

# 36. Output Tensor

The output contains the ten population trajectories.

For a batch, the output tensor is

```math
Y\in\mathbb{R}^{B\times10\times N}
```

The ten channels correspond to

```math
P_0(t),P_1(t),\ldots,P_9(t)
```

---

# 37. Channelwise Standardization

Each input channel is standardized using statistics calculated only from the training set.

For channel $k$,

```math
\tilde u_k(t)
=
\frac{u_k(t)-\mu_k}{\sigma_k}
```

where $\mu_k$ and $\sigma_k$ are the training-set mean and standard deviation.

The same statistics are applied to the test set.

---

# 38. FNO

The Fourier Neural Operator (FNO) learns operators using Fourier-space representations.

A simplified Fourier layer can be written as

```math
v_{l+1}
=
\sigma
\left(
Wv_l
+
\mathcal{F}^{-1}
\left(
K_l\mathcal{F}(v_l)
\right)
\right)
```

where:

- $\mathcal{F}$ is the Fourier transform.
- $\mathcal{F}^{-1}$ is the inverse Fourier transform.
- $K_l$ contains learned Fourier-space weights.
- $W$ is a learned linear transformation.
- $\sigma$ is a nonlinear activation.

The Fourier representation allows the model to learn interactions across the temporal domain.

---

# 39. GNO

The Graph Neural Operator (GNO) represents the function using interactions between points.

A simplified graph operator can be written as

```math
v_i^{(l+1)}
=
\Phi
\left(
v_i^{(l)},
\sum_j
K_\theta(x_i,x_j)v_j^{(l)}
\right)
```

where:

- $x_i$ and $x_j$ are locations in the input domain.
- $K_\theta$ is a learned kernel.
- $\Phi$ is a learned transformation.

For this project, the domain is temporal.

---

# 40. CATO

CATO is evaluated as a third neural-operator architecture.

FNO, GNO, and CATO are trained on the same physical dataset.

The common learning problem is

```math
\mathcal{G}:u(t)\longrightarrow\mathbf{P}(t)
```

The comparison therefore focuses on how different operator architectures approximate the same physical mapping.

---

# 41. Training Dataset

The experiment uses exactly 200 training trajectories.

The training set contains different realizations of the effective Hamiltonian inputs and their corresponding simulated population trajectories.

The training samples are used to optimize the parameters of each neural operator.

---

# 42. Test Dataset

The experiment uses exactly 100 test trajectories.

The test trajectories are not used to optimize model parameters.

They are used only to evaluate how well the trained neural operators generalize to previously unseen physical trajectories.

---

# 43. Training

The models are trained using Adam.

The learning rate is

```math
\eta=10^{-3}
```

The batch size is

```math
B=8
```

The training objective is mean squared error.

---

# 44. Mean Squared Error

For predicted populations $\hat P_j(t_n)$ and reference populations $P_j(t_n)$, the MSE is

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

The optimizer adjusts the neural-operator parameters to minimize this loss.

---

# 45. Relative $L^2$ Error

The relative $L^2$ error compares the magnitude of the prediction error with the magnitude of the reference trajectory.

It is defined as

```math
\epsilon_{L^2}
=
\frac{
\|\hat{\mathbf{P}}-\mathbf{P}\|_2
}{
\|\mathbf{P}\|_2
}
```

For discrete temporal samples,

```math
\|\mathbf{P}\|_2
=
\left(
\sum_n
\|\mathbf{P}(t_n)\|_2^2
\right)^{1/2}
```

A smaller value indicates a more accurate prediction.

---

# 46. Temporal Sobolev $H^1$ Error

The temporal $H^1$ norm measures both the function and its temporal derivative.

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

The relative temporal $H^1$ error is

```math
\epsilon_{H^1}
=
\frac{
\|\hat{\mathbf{P}}-\mathbf{P}\|_{H^1}
}{
\|\mathbf{P}\|_{H^1}
}
```

This metric tests both population accuracy and temporal-dynamics accuracy.

---

# 47. Finite-Difference Derivative

The temporal derivative is calculated numerically.

For an interior temporal point,

```math
\frac{df}{dt}(t_n)
\approx
\frac{
f(t_{n+1})-f(t_{n-1})
}{
2\Delta t
}
```

The derivative of both the predicted and reference trajectories is used when calculating the temporal $H^1$ error.

---

# 48. Multiple Random Seeds

The complete training and evaluation procedure is repeated using five random seeds.

For a metric $\epsilon$, the mean across the five runs is

```math
\bar{\epsilon}
=
\frac{1}{5}
\sum_{s=1}^{5}
\epsilon_s
```

The standard deviation is

```math
\sigma_\epsilon
=
\sqrt{
\frac{1}{5}
\sum_{s=1}^{5}
(\epsilon_s-\bar{\epsilon})^2
}
```

The final results can therefore be reported as mean and standard deviation across the five seeds.

---

# 49. Physical Interpretation of Model Error

The relative $L^2$ error measures how closely the predicted population trajectories match the reference trajectories in overall magnitude.

The temporal $H^1$ error additionally measures differences in temporal variation.

A model can have a relatively small $L^2$ error while still producing incorrect sharp changes or oscillations in time.

The $H^1$ metric therefore provides additional information about whether the neural operator reproduces the temporal structure of the quantum dynamics.

---

# 50. Leakage Analysis

The population outside the computational subspace is

```math
P_{\mathrm{leak}}(t)
=
\sum_{j=2}^{9}P_j(t)
```

The computational-subspace population is

```math
P_{\mathrm{comp}}(t)
=
P_0(t)+P_1(t)
```

The two quantities satisfy

```math
P_{\mathrm{comp}}(t)+P_{\mathrm{leak}}(t)=1
```

Leakage provides a physically meaningful quantity for evaluating the effect of strong or imperfect microwave control.

---

# 51. Complete Physics-to-ML Pipeline

The complete process is

```math
\text{Transmon Hamiltonian}
\longrightarrow
\text{Numerical Diagonalization}
\longrightarrow
\text{10-Level Model}
```

followed by

```math
\text{Control}
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

for neural-operator learning.

---

# 52. Central Mathematical Model

The complete physical model can be summarized by

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

and population output

```math
P_j(t)=\langle j|\rho(t)|j\rangle
```

giving

```math
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)]
```

The neural operator learns

```math
\mathcal{G}:u(t)\longrightarrow\mathbf{P}(t)
```

using simulated trajectories generated by the physical transmon model.
