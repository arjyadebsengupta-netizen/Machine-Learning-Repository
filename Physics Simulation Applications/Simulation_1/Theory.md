# Physics Theory Test

## 1. Transmon Hamiltonian

The transmon Hamiltonian is

$$
H =
4E_C(\hat n-n_g)^2
-
E_J\cos\hat\phi.
$$

where:

- $E_J$ is the Josephson energy.
- $E_C$ is the charging energy.
- $\hat n$ is the Cooper-pair number operator.
- $n_g$ is the offset charge.
- $\hat\phi$ is the superconducting phase operator.

---

## 2. Energy Eigenstates

The energy eigenstates satisfy

$$
H|j\rangle = E_j|j\rangle.
$$

The transition frequency between two adjacent levels is

$$
f_{j,j+1}
=
\frac{E_{j+1}-E_j}{h}.
$$

For the lowest transition,

$$
f_{01}
=
\frac{E_1-E_0}{h}.
$$

---

## 3. Anharmonicity

The transmon anharmonicity is

$$
\alpha
=
f_{12}-f_{01}.
$$

For a transmon,

$$
\alpha < 0.
$$

---

## 4. Quantum State

The quantum state is represented by a density matrix

$$
\rho(t)\in\mathbb{C}^{10\times10}.
$$

The population of level $n$ is

$$
P_n(t)
=
\langle n|\rho(t)|n\rangle.
$$

Therefore the population vector is

$$
\mathbf{P}(t)
=
[P_0(t),P_1(t),\ldots,P_9(t)].
$$

---

## 5. Open-System Dynamics

The density matrix evolves according to the Lindblad master equation

$$
\frac{d\rho}{dt}
=
-\frac{i}{\hbar}[H(t),\rho]
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

---

## 6. Neural-Operator Mapping

The effective Hamiltonian is represented as

$$
H(t)
=
H_0+
\sum_k u_k(t)H_k.
$$

The neural operator therefore learns

$$
\mathcal{G}:u(t)\longrightarrow\mathbf{P}(t).
$$

In this project,

$$
\boxed{
u(t)
\longrightarrow
[P_0(t),P_1(t),\ldots,P_9(t)]
}
$$

is the central physics-to-machine-learning mapping.
