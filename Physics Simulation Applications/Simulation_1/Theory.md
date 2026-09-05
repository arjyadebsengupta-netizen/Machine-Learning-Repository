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

