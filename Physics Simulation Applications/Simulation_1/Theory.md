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
