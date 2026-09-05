# Physics Theory for the Noisy Superconducting Transmon Neural-Operator Model

## 1. Overview

A superconducting transmon is a quantum circuit that behaves approximately like an artificial atom.

In this project, the transmon is modeled as a **10-level quantum system** rather than as an ideal two-level qubit. The system is driven by microwave control fields and affected by several realistic noise and dissipation mechanisms.

The physical simulator generates the true quantum dynamics. The machine-learning models then learn the mapping

\[
u(t) \longrightarrow \mathbf{P}(t),
\]

where

- \(u(t)\) represents the effective time-dependent Hamiltonian coefficients,
- \(\mathbf{P}(t)\) contains the populations of the ten energy levels.

The purpose of this section is to develop the physics required to understand this mapping.

---

# 2. Superconducting Qubits

A superconducting qubit is an electrical circuit whose quantum states arise from the quantization of electromagnetic degrees of freedom.

The basic building block is the **Josephson junction**.

A Josephson junction consists of two superconductors separated by a thin insulating barrier.

The junction is characterized by the Josephson energy

\[
E_J.
\]

A superconducting circuit also has charging energy

\[
E_C.
\]

The competition between these two energies determines the properties of the quantum circuit.

For a transmon,

\[
E_J \gg E_C.
\]

This regime makes the qubit relatively insensitive to charge noise while retaining sufficient anharmonicity for quantum control.

---

# 3. Josephson Junction

The Josephson junction has a superconducting phase difference

\[
\phi
\]

across the junction.

The Josephson potential energy is

\[
V_J(\phi)=-E_J\cos\phi.
\]

The cosine potential is important because the transmon is **not exactly a harmonic oscillator**.

The full Hamiltonian is

\[
\boxed{
H
=
4E_C(\hat n-n_g)^2
-
E_J\cos\hat\phi
}
\]

where

- \(\hat n\) is the Cooper-pair number operator,
- \(n_g\) is the offset charge,
- \(\hat\phi\) is the superconducting phase operator,
- \(E_C\) is the charging energy,
- \(E_J\) is the Josephson energy.

The canonical commutation relation is

\[
[\hat\phi,\hat n]=i.
\]

Thus \(\hat\phi\) and \(\hat n\) play roles analogous to position and momentum.

---

# 4. Charge Basis

The transmon Hamiltonian can be represented in the charge basis

\[
|n\rangle,
\]

where \(n\) represents the number of excess Cooper pairs.

The charge operator satisfies

\[
\hat n|n\rangle=n|n\rangle.
\]

The phase operators satisfy

\[
e^{\pm i\hat\phi}|n\rangle
=
|n\pm1\rangle.
\]

Therefore,

\[
\cos\hat\phi
=
\frac{1}{2}
\left(
e^{i\hat\phi}+e^{-i\hat\phi}
\right).
\]

Consequently,

\[
\langle n|\cos\hat\phi|m\rangle
=
\frac12
\left(
\delta_{n,m+1}
+
\delta_{n,m-1}
\right).
\]

The Hamiltonian can therefore be constructed as a matrix in a sufficiently large charge basis.

---

# 5. Full Cosine Hamiltonian

The simulator uses the full Hamiltonian

\[
\boxed{
H
=
4E_C(\hat n-n_g)^2
-
E_J\cos\hat\phi.
}
\]

This is preferable to replacing the cosine with a quadratic approximation because the nonlinearity of the cosine produces the transmon's finite anharmonicity.

For sufficiently small phase,

\[
\cos\phi
\approx
1-\frac{\phi^2}{2}
+\frac{\phi^4}{24}
-\cdots.
\]

Keeping only the quadratic term would produce an approximately harmonic oscillator.

The higher-order terms are responsible for the deviation from harmonic behavior.

---

# 6. Energy Eigenstates

The Hamiltonian is numerically diagonalized:

\[
H|j\rangle=E_j|j\rangle.
\]

The eigenstates are ordered as

\[
E_0<E_1<E_2<\cdots.
\]

The simulator retains the lowest ten states:

\[
\boxed{
|0\rangle,|1\rangle,\ldots,|9\rangle.
}
\]

These are the ten transmon energy levels used in the simulation.

The resulting Hilbert-space dimension is

\[
d=10.
\]

---

# 7. Transition Frequencies

The transition frequency between adjacent levels is

\[
\boxed{
f_{j,j+1}
=
\frac{E_{j+1}-E_j}{h}.
}
\]

The lowest transition is

\[
f_{01}
=
\frac{E_1-E_0}{h}.
\]

The next transition is

\[
f_{12}
=
\frac{E_2-E_1}{h}.
\]

For an ideal harmonic oscillator,

\[
f_{01}=f_{12}=f_{23}=\cdots.
\]

A transmon is anharmonic, so

\[
f_{01}\neq f_{12}.
\]

---

# 8. Anharmonicity

The transmon anharmonicity is commonly defined as

\[
\boxed{
\alpha
=
f_{12}-f_{01}.
}
\]

For a transmon,

\[
\alpha<0.
\]

The approximate transmon result is

\[
\alpha\approx-\frac{E_C}{h}.
\]

The finite anharmonicity is essential for selective qubit control.

If the system were perfectly harmonic, driving the \(0\rightarrow1\) transition would also strongly excite higher transitions.

---

# 9. Why a 10-Level Model Is Used

An ideal qubit has only

\[
|0\rangle,\quad |1\rangle.
\]

A real transmon has many energy levels.

When the microwave drive becomes sufficiently strong, population can leave the computational subspace:

\[
\{|0\rangle,|1\rangle\}.
\]

For this reason the simulator keeps ten levels:

\[
|0\rangle,\ldots,|9\rangle.
\]

This allows the model to capture **leakage** into higher levels.

The computational-subspace population is

\[
P_{\mathrm{comp}}(t)
=
P_0(t)+P_1(t).
\]

The leakage population is

\[
\boxed{
P_{\mathrm{leak}}(t)
=
\sum_{n=2}^{9}P_n(t).
}
\]

Since the density matrix is normalized,

\[
\sum_{n=0}^{9}P_n(t)=1,
\]

so

\[
P_{\mathrm{leak}}(t)
=
1-P_0(t)-P_1(t).
\]

---

# 10. Quantum State

The state of the ten-level transmon is represented by a density matrix

\[
\boxed{
\rho(t)\in\mathbb C^{10\times10}.
}
\]

For a pure state,

\[
\rho=|\psi\rangle\langle\psi|.
\]

For a mixed state,

\[
\rho
=
\sum_k p_k|\psi_k\rangle\langle\psi_k|.
\]

The density matrix satisfies

\[
\rho^\dagger=\rho,
\]

\[
\rho\succeq0,
\]

and

\[
\operatorname{Tr}(\rho)=1.
\]

---

# 11. Populations

The population of level \(n\) is

\[
\boxed{
P_n(t)
=
\langle n|\rho(t)|n\rangle
=
\rho_{nn}(t).
}
\]

Thus the output of the physical simulator can be written as

\[
\boxed{
\mathbf P(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)].
}
\]

This is the quantity predicted by the neural operators.

---

# 12. Expectation Values

For an observable \(A\), its expectation value is

\[
\boxed{
\langle A\rangle
=
\operatorname{Tr}(\rho A).
}
\]

For example,

\[
\langle \hat n\rangle
=
\operatorname{Tr}(\rho\hat n).
\]

Expectation values provide another way of extracting physical information from the quantum state.

---

# 13. Microwave Control

The transmon is controlled using a microwave field.

The control has two quadratures:

\[
I(t)
\]

and

\[
Q(t).
\]

These are the in-phase and quadrature components of the microwave signal.

A convenient complex representation is

\[
\Omega(t)
=
I(t)+iQ(t).
\]

The two quadratures determine the amplitude and phase of the applied microwave field.

The instantaneous amplitude is

\[
A(t)
=
\sqrt{I(t)^2+Q(t)^2},
\]

and its phase is

\[
\theta(t)
=
\operatorname{atan2}(Q(t),I(t)).
\]

---

# 14. Drive Hamiltonian

The microwave field couples to a transmon operator.

Depending on the chosen representation, the drive Hamiltonian can be written schematically as

\[
\boxed{
H_{\mathrm{drive}}(t)
=
u_1(t)H_1
+
u_2(t)H_2.
}
\]

The operators \(H_1\) and \(H_2\) describe the physical coupling of the two quadratures to the transmon.

The important point for this project is that the microwave control produces a **time-dependent Hamiltonian**.

---

# 15. Effective Hamiltonian Representation

For the neural-operator problem, the total Hamiltonian is represented as

\[
\boxed{
H(t)
=
H_0+
\sum_{k=1}^{C_{\mathrm{in}}}
u_k(t)H_k.
}
\]

Here,

- \(H_0\) is the static transmon Hamiltonian,
- \(H_k\) are fixed Hamiltonian operators,
- \(u_k(t)\) are time-dependent coefficients.

Therefore,

\[
\boxed{
u(t)
=
[u_1(t),u_2(t),\ldots,u_{C_{\mathrm{in}}}(t)]
}
\]

is the effective Hamiltonian input to the neural operator.

The neural network therefore learns

\[
\boxed{
u(t)
\longrightarrow
\mathbf P(t).
}
\]

---

# 16. Control Bandwidth

A real microwave control system cannot change its amplitude infinitely quickly.

The simulator therefore applies bandwidth limitations to the control signal.

A rapidly varying input is smoothed by the control system.

In general,

\[
I_{\mathrm{filtered}}(t)
=
G[I_{\mathrm{commanded}}(t)]
\]

and similarly

\[
Q_{\mathrm{filtered}}(t)
=
G[Q_{\mathrm{commanded}}(t)],
\]

where \(G\) represents the control-response filter.

This prevents the simulated control from containing unrealistically high-frequency components.

---

# 17. Frequency Noise

The transmon transition frequency is not perfectly constant.

Environmental fluctuations can cause

\[
\omega_{01}
\rightarrow
\omega_{01}+\delta\omega(t).
\]

This produces a time-dependent frequency shift.

The corresponding Hamiltonian contribution can be represented as

\[
\boxed{
H_{\mathrm{freq}}(t)
=
\delta\omega(t)A_{\mathrm{freq}},
}
\]

where \(A_{\mathrm{freq}}\) is the operator through which the frequency fluctuation acts.

---

# 18. Low-Frequency / \(1/f\)-Type Noise

Many physical noise sources have stronger low-frequency components.

A common idealized description is

\[
S(f)\propto\frac{1}{f^\beta},
\]

with

\[
\beta\approx1.
\]

Such noise is often called \(1/f\) noise.

The corresponding time-domain realization is a slowly varying stochastic process.

Different noise realizations therefore produce different transmon trajectories.

---

# 19. Charge Noise

The transmon Hamiltonian contains

\[
4E_C(\hat n-n_g)^2.
\]

Therefore fluctuations in offset charge,

\[
n_g(t)
=
n_{g,0}+\delta n_g(t),
\]

modify the Hamiltonian:

\[
H_C(t)
=
4E_C
\left[
\hat n-n_g(t)
\right]^2.
\]

Thus charge noise produces time-dependent energy shifts and changes the quantum dynamics.

---

# 20. Amplitude Noise

The microwave amplitude can fluctuate around its intended value.

If the ideal control amplitude is \(A(t)\), we can write

\[
A(t)
\rightarrow
A(t)+\delta A(t).
\]

This produces fluctuations in the effective drive strength.

Amplitude noise can therefore cause errors in population transfer and gate operations.

---

# 21. Phase Noise

The microwave phase can also fluctuate:

\[
\theta(t)
\rightarrow
\theta(t)+\delta\theta(t).
\]

Phase fluctuations modify the relative contribution of the two microwave quadratures.

Since

\[
I=A\cos\theta,
\]

\[
Q=A\sin\theta,
\]

a fluctuating phase changes the effective drive direction in the \(I-Q\) plane.

---

# 22. Frequency Drift

In addition to relatively fast fluctuations, the transition frequency can slowly drift:

\[
\omega_{01}(t)
=
\omega_{01}^{(0)}
+
\delta\omega_{\mathrm{drift}}(t).
\]

The drift occurs on a much longer timescale than the microwave pulse.

It represents slow changes in the operating point of the device.

---

# 23. TLS Noise

Two-level-system (TLS) defects can interact with superconducting circuits.

A simplified model represents a TLS-induced frequency shift as a stochastic switching process:

\[
s_{\mathrm{TLS}}(t)
\in
\{-1,+1\}.
\]

The corresponding frequency shift can be written as

\[
\delta\omega_{\mathrm{TLS}}(t)
=
\Delta_{\mathrm{TLS}}s_{\mathrm{TLS}}(t).
\]

The system therefore experiences random switching between different effective frequency offsets.

---

# 24. Open Quantum Systems

An isolated quantum system evolves according to the Schrödinger equation.

A realistic transmon interacts with its environment.

Therefore the evolution must include both

- coherent quantum evolution,
- irreversible dissipation.

The density matrix is used to represent this open-system evolution.

---

# 25. Hamiltonian Evolution

For a closed system,

\[
\boxed{
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H,\rho].
}
\]

The commutator is

\[
[H,\rho]
=
H\rho-\rho H.
\]

This term describes coherent quantum evolution.

---

# 26. Dissipation

Relaxation and dephasing are incorporated through dissipative terms.

The general Lindblad master equation is

\[
\boxed{
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H,\rho]
+
\sum_k
\left(
L_k\rho L_k^\dagger
-
\frac12
\{L_k^\dagger L_k,\rho\}
\right).
}
\]

Here \(L_k\) are collapse operators.

The anticommutator is

\[
\{A,B\}
=
AB+BA.
\]

---

# 27. Energy Relaxation

Energy relaxation causes an excited state to decay toward lower-energy states.

The characteristic timescale is

\[
T_1.
\]

For a simple two-level system,

\[
P_1(t)
\sim
e^{-t/T_1}
\]

in the absence of driving and thermal excitation.

For a multilevel transmon, relaxation can occur through several transitions:

\[
|j\rangle\rightarrow|j-1\rangle.
\]

---

# 28. Pure Dephasing

Dephasing destroys phase coherence without necessarily changing the populations.

It is characterized by a timescale associated with coherence decay.

The commonly measured coherence time \(T_2\) contains contributions from both relaxation and pure dephasing.

For a simple qubit,

\[
\boxed{
\frac{1}{T_2}
=
\frac{1}{2T_1}
+
\frac{1}{T_\phi}
}
\]

where \(T_\phi\) is the pure-dephasing time.

---

# 29. Coherence

The off-diagonal density-matrix elements

\[
\rho_{mn},
\qquad
m\neq n,
\]

describe quantum coherence between energy levels.

Populations are the diagonal terms:

\[
\rho_{nn}=P_n.
\]

Therefore:

\[
\boxed{
\text{diagonal elements}
\rightarrow
\text{populations}
}
\]

\[
\boxed{
\text{off-diagonal elements}
\rightarrow
\text{coherences}.
}
\]

Although the ML target in this project is the population field, the populations are generated by the full density-matrix evolution.

---

# 30. Thermal Effects

At finite temperature, the environment can produce both downward and upward transitions.

The thermal energy scale is

\[
k_BT.
\]

The corresponding frequency scale is

\[
\frac{k_BT}{h}.
\]

At sufficiently low temperature, the ground state dominates.

The simulator therefore includes a finite temperature parameter when constructing dissipative dynamics.

---

# 31. Complete Physical Model

The total Hamiltonian can be conceptually decomposed as

\[
\boxed{
H(t)
=
H_{\mathrm{transmon}}
+
H_{\mathrm{drive}}(t)
+
H_{\mathrm{frequency}}(t)
+
H_{\mathrm{charge}}(t)
+
H_{\mathrm{drift}}(t)
+
H_{\mathrm{TLS}}(t).
}
\]

The density matrix evolves according to

\[
\boxed{
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H(t),\rho]
+
\mathcal D(\rho).
}
\]

The simulator numerically integrates this equation.

---

# 32. From Physics to the Neural-Operator Dataset

For every physical realization, the simulator produces a time-dependent effective Hamiltonian

\[
H(t)
\]

and corresponding quantum dynamics

\[
\rho(t).
\]

We represent the Hamiltonian through its time-dependent coefficients:

\[
H(t)
=
H_0+
\sum_k u_k(t)H_k.
\]

Therefore the input is

\[
\boxed{
u(t)
=
[u_1(t),\ldots,u_{C_{\mathrm{in}}}(t)].
}
\]

The simulator produces

\[
\boxed{
\mathbf P(t)
=
[P_0(t),\ldots,P_9(t)].
}
\]

The supervised learning pair is therefore

\[
\boxed{
u^{(i)}(t)
\longrightarrow
\mathbf P^{(i)}(t).
}
\]

---

# 33. Temporal Discretization

The neural operator uses the environment time step

\[
\boxed{
\Delta t=2~\mathrm{ns}.
}
\]

The physical simulator may internally propagate the quantum state at a smaller timestep, but the operator dataset is sampled on the \(2\,\mathrm{ns}\) grid.

For a total simulation time of

\[
T=200~\mathrm{ns},
\]

the resulting operator grid contains approximately

\[
N=100
\]

post-step time samples.

Thus one input sample has the structure

\[
U\in\mathbb R^{C_{\mathrm{in}}\times N}
\]

and the corresponding target is

\[
Y\in\mathbb R^{10\times N}.
\]

---

# 34. Neural Operator Interpretation

A conventional neural network learns a finite-dimensional mapping

\[
\mathbf x\rightarrow\mathbf y.
\]

A neural operator instead learns a mapping between functions:

\[
\boxed{
\mathcal G:u(\cdot)\rightarrow y(\cdot).
}
\]

For this problem,

\[
\boxed{
\mathcal G:
u(t)\rightarrow\mathbf P(t).
}
\]

The objective is therefore to approximate the expensive quantum simulator with a learned operator.

---

# 35. FNO, GNO and CATO

Three neural-operator architectures are compared:

\[
\boxed{
\text{FNO}
}
\]

\[
\boxed{
\text{GNO}
}
\]

\[
\boxed{
\text{CATO}
}
\]

All three receive the same physical input function

\[
u(t)
\]

and predict the same output

\[
\mathbf P(t).
\]

The scientific question is therefore:

> How accurately can different neural-operator architectures learn the input-to-dynamics mapping of the noisy multilevel transmon?

---

# 36. Dataset

The benchmark contains

\[
\boxed{
200\text{ training samples}
}
\]

and

\[
\boxed{
100\text{ test samples}.
}
\]

Each sample corresponds to one complete transmon evolution.

The training and test sets must be generated independently so that the test trajectories represent unseen physical realizations.

---

# 37. Input Tensor

For a batch of \(B\) samples,

\[
\boxed{
U\in
\mathbb R^{B\times C_{\mathrm{in}}\times N}.
}
\]

Here:

- \(B\) = batch size,
- \(C_{\mathrm{in}}\) = number of effective Hamiltonian coefficient channels,
- \(N\) = number of temporal grid points.

---

# 38. Output Tensor

The population output is

\[
\boxed{
Y\in
\mathbb R^{B\times10\times N}.
}
\]

The ten channels correspond to

\[
P_0,P_1,\ldots,P_9.
\]

---

# 39. Channelwise Standardization

Each input channel is normalized independently.

For input channel \(k\),

\[
u_k'
=
\frac{u_k-\mu_k}{\sigma_k}.
\]

The statistics

\[
\mu_k,\sigma_k
\]

are calculated using the **200 training samples only**.

The same principle is applied to the output channels:

\[
P_n'
=
\frac{P_n-\mu_n}{\sigma_n}.
\]

The test set is never used to calculate normalization statistics.

---

# 40. Training Loss

The neural operator prediction is

\[
\hat{\mathbf P}(t)
=
\mathcal G_\theta[u](t).
\]

The training loss is mean squared error:

\[
\boxed{
\mathcal L_{\mathrm{MSE}}
=
\frac{1}{10N}
\sum_{n=0}^{9}
\sum_{j=1}^{N}
\left[
\hat P_n(t_j)-P_n(t_j)
\right]^2.
}
\]

The model parameters are optimized using Adam:

\[
\boxed{
\mathrm{Adam},
\qquad
LR=10^{-3},
\qquad
\mathrm{batch}=8.
}
\]

---

# 41. Relative \(L^2\) Error

The relative \(L^2\) error measures the overall population-field error.

For one test sample,

\[
\boxed{
E_{L^2}
=
\frac{
\|\hat Y-Y\|_2
}{
\|Y\|_2
}.
}
\]

For the discrete population field,

\[
\|Y\|_2
=
\left[
\sum_{j=1}^{N}
\sum_{n=0}^{9}
Y_{n,j}^2
\right]^{1/2}.
\]

Therefore,

\[
\boxed{
E_{L^2}
=
\frac{
\left[
\sum_{j,n}
(\hat P_{n,j}-P_{n,j})^2
\right]^{1/2}
}{
\left[
\sum_{j,n}
P_{n,j}^2
\right]^{1/2}
}.
}
\]

Lower values indicate better predictions.

---

# 42. Sobolev \(H^1\) Error

The \(H^1\) norm measures both the function and its derivative.

For a population trajectory,

\[
\|P\|_{H^1}^2
=
\int
\left[
P(t)^2+
\left(
\frac{\partial P}{\partial t}
\right)^2
\right]dt.
\]

For all ten population channels,

\[
\boxed{
\|Y\|_{H^1}^2
=
\int
\sum_{n=0}^{9}
\left[
P_n(t)^2+
\left(
\frac{\partial P_n}{\partial t}
\right)^2
\right]dt.
}
\]

The relative Sobolev error is

\[
\boxed{
E_{H^1}
=
\frac{
\|\hat Y-Y\|_{H^1}
}{
\|Y\|_{H^1}
}.
}
\]

---

# 43. Finite-Difference Derivative

The temporal derivative is calculated numerically.

For interior points,

\[
\boxed{
\frac{\partial P}{\partial t}
\bigg|_{t_j}
\approx
\frac{
P(t_{j+1})-P(t_{j-1})
}{
2\Delta t
}.
}
\]

At the boundaries, one-sided finite differences can be used.

For the prediction error,

\[
e(t)=\hat P(t)-P(t),
\]

we calculate

\[
\frac{\partial e}{\partial t}.
\]

The discrete \(H^1\) error therefore contains both

\[
e
\]

and

\[
\frac{\partial e}{\partial t}.
\]

This makes the metric sensitive not only to incorrect populations but also to incorrect temporal dynamics.

---

# 44. Five Random Seeds

Each architecture is trained using five independent seeds.

Therefore,

\[
3\text{ architectures}
\times
5\text{ seeds}
=
15\text{ training runs}.
\]

For every architecture, the final result is reported as

\[
\boxed{
\mathrm{mean}\pm\mathrm{standard\ deviation}.
}
\]

The two principal metrics are

\[
\boxed{
\text{Relative }L^2
}
\]

and

\[
\boxed{
\text{Relative }H^1.
}
\]

---

# 45. Physical Interpretation of the ML Error

A low Relative \(L^2\) error means that the predicted population trajectories are close to the simulator trajectories overall.

A low Relative \(H^1\) error additionally means that the temporal variation is accurately reproduced.

For example, a prediction might have reasonable population values but incorrect oscillation frequency or incorrect transition timing.

Such a prediction can have a relatively small \(L^2\) error but a larger \(H^1\) error.

Therefore the two metrics provide complementary information.

---

# 46. Important Physical Quantities for Analysis

From the predicted populations we can calculate

\[
P_{\mathrm{comp}}(t)
=
P_0(t)+P_1(t),
\]

and

\[
P_{\mathrm{leak}}(t)
=
\sum_{n=2}^{9}P_n(t).
\]

These quantities allow us to determine whether the neural operator reproduces important physical behavior such as

- population transfer,
- relaxation,
- excitation,
- leakage,
- temporal oscillations.

---

# 47. Complete Physics-to-ML Pipeline

The complete physical and machine-learning process is

\[
\boxed{
\text{Transmon Hamiltonian}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{Microwave drive + noise + dissipation}
}
\]

\[
\downarrow
\]

\[
\boxed{
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H(t),\rho]
+\mathcal D(\rho)
}
\]

\[
\downarrow
\]

\[
\boxed{
\rho(t)
}
\]

\[
\downarrow
\]

\[
\boxed{
P_0(t),\ldots,P_9(t)
}
\]

while the Hamiltonian is represented by

\[
\boxed{
H(t)
=
H_0+\sum_k u_k(t)H_k.
}
\]

The neural-operator learning problem is therefore

\[
\boxed{
u(t)
\overset{\mathrm{FNO/GNO/CATO}}{\longrightarrow}
[P_0(t),\ldots,P_9(t)].
}
\]

The learned operators are evaluated using

\[
\boxed{
\text{Relative }L^2
\quad\text{and}\quad
\text{Relative }H^1.
}
\]
