# Physics-Based Neural Operator Learning for Noisy Superconducting Transmon Dynamics

## Physical Model

The transmon is modeled using the full cosine Hamiltonian:

$$
H_{\mathrm{tr}}
=
4E_C(\hat n-n_g)^2
-
E_J\cos(\hat\phi)
$$

The operator-learning problem is

$$
\mathcal{G}:u(t)\mapsto\mathbf{P}(t).
$$

The four input channels are

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

The output contains the populations of the ten retained energy levels:

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

## Hamiltonian

The effective Hamiltonian used by the simulator is

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

## Python Implementation

```python
H = build_effective_hamiltonian(
    u_frequency,
    u_I,
    u_Q,
    u_charge,
)
```

## Data Representation

The neural operator receives

```text
Input:  [B, 4, 101]
Output: [B, 10, 101]
```

The learned mapping is therefore

$$
\mathcal{G}_\theta:
\mathbb{R}^{4\times101}
\rightarrow
\mathbb{R}^{10\times101}.
$$
