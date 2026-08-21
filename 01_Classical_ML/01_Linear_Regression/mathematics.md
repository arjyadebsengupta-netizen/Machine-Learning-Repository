# Formatting Test

The model is

```math
y_i=\beta_0+\beta_1x_i+\epsilon_i
```

where

```math
\epsilon_i
```

represents measurement noise.

The loss function is

```math
\mathcal{L}(\boldsymbol{\beta})
=
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
```

The design matrix is

```math
X=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_N
\end{bmatrix}
```

The gradient is

```math
\nabla_{\boldsymbol{\beta}}\mathcal{L}
=
-2X^T\mathbf y
+
2X^TX\boldsymbol{\beta}
```

The column space is

```math
\operatorname{Col}(X)
=
\operatorname{span}
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

The fitted parameters are

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y
```

And the prediction is

```math
\hat y
=
\hat{\boldsymbol{\beta}}^T\boldsymbol{\phi}(x)
```
