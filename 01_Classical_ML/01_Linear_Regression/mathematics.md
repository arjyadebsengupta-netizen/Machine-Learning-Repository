# Linear Regression: Mathematical Foundation

Suppose we observe \(N\) data points.

```math
\mathcal{D}
=
\{(x_i,y_i)\}_{i=1}^{N}
```

The basic linear regression model is

```math
y_i
=
\beta_0+\beta_1x_i+\epsilon_i
```

where

```math
\epsilon_i
```

represents measurement noise.

The prediction is

```math
\hat{y}_i
=
\beta_0+\beta_1x_i
```

More generally, linear regression is linear in the parameters, not necessarily in \(x\).

Using basis functions

```math
\phi_0(x),\phi_1(x),\ldots,\phi_p(x)
```

we write

```math
f(x)
=
\sum_{j=0}^{p}
\beta_j\phi_j(x)
```

For ordinary linear regression, the basis functions are

```math
\phi_0(x)=1
```

and

```math
\phi_1(x)=x
```

so that

```math
f(x)
=
\beta_0+\beta_1x
```

For a polynomial basis,

```math
\phi_j(x)=x^j
```

and therefore

```math
f(x)
=
\sum_{j=0}^{p}
\beta_jx^j
```

Define the basis vector

```math
\boldsymbol{\phi}(x)
=
\begin{bmatrix}
\phi_0(x)\\
\phi_1(x)\\
\vdots\\
\phi_p(x)
\end{bmatrix}
```

Then the model can be written compactly as

```math
f(x)
=
\boldsymbol{\beta}^T
\boldsymbol{\phi}(x)
```

For \(N\) observations, the design matrix is

```math
X
=
\begin{bmatrix}
\phi_0(x_1) & \phi_1(x_1) & \cdots & \phi_p(x_1)\\
\phi_0(x_2) & \phi_1(x_2) & \cdots & \phi_p(x_2)\\
\vdots & \vdots & \ddots & \vdots\\
\phi_0(x_N) & \phi_1(x_N) & \cdots & \phi_p(x_N)
\end{bmatrix}
```

The matrix form of the regression model is

```math
\mathbf{y}
=
X\boldsymbol{\beta}
+
\boldsymbol{\epsilon}
```

The least-squares objective is

```math
\mathcal{L}(\boldsymbol{\beta})
=
\left\|
\mathbf{y}
-
X\boldsymbol{\beta}
\right\|_2^2
```

Taking the gradient gives

```math
\nabla_{\boldsymbol{\beta}}\mathcal{L}
=
-2X^T\mathbf{y}
+
2X^TX\boldsymbol{\beta}
```

At the minimum,

```math
\nabla_{\boldsymbol{\beta}}\mathcal{L}
=
0
```

which gives the normal equation

```math
X^TX\hat{\boldsymbol{\beta}}
=
X^T\mathbf{y}
```

and, when \(X^TX\) is invertible,

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf{y}
```

The column space is

```math
\mathrm{Col}(X)
=
\mathrm{span}
\left\{
\begin{bmatrix}
1\\
1\\
\vdots\\
1
\end{bmatrix},
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_N
\end{bmatrix}
\right\}
```
