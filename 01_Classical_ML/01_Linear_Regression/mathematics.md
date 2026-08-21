# Linear Regression: Mathematical Theory

## 1. What Do We Have?

Suppose we have measured two quantities for \(N\) physical systems.

```math
(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)
```

Here:

- ```math
  x_i
  ```
  is the input or explanatory variable.

- ```math
  y_i
  ```
  is the observed output.

- ```math
  N
  ```
  is the number of observations.

For a physics example, imagine measuring the velocity of an object at different times.

Under constant acceleration,

```math
v(t)=v_0+at
```

We can therefore identify

```math
x=t
```

and

```math
y=v.
```

Our experimental data are therefore

```math
(t_1,v_1),(t_2,v_2),\ldots,(t_N,v_N).
```

The measurements will generally not lie perfectly on a straight line because of experimental noise.

---

## 2. What Is Our Goal?

We want to find a mathematical function that describes the relationship between the input and output.

For the simplest model,

```math
\hat{y}=\beta_0+\beta_1x
```

where:

- ```math
  \beta_0
  ```
  is the intercept.

- ```math
  \beta_1
  ```
  is the slope.

- ```math
  \hat{y}
  ```
  is the predicted value.

For the velocity example,

```math
\hat{v}=\beta_0+\beta_1t
```

where

```math
\beta_0\approx v_0
```

and

```math
\beta_1\approx a.
```

Therefore, linear regression is not merely "drawing a line through points."

It is a method for **estimating unknown parameters of a mathematical model from noisy observations**.

---

## 3. The Statistical Model

We distinguish between the actual physical quantity and the measured value.

We assume

```math
y_i=\beta_0+\beta_1x_i+\epsilon_i
```

where

```math
\epsilon_i
```

represents unexplained variation or measurement noise.

Therefore,

```math
y_i
=
\underbrace{\beta_0+\beta_1x_i}_{\text{systematic component}}
+
\underbrace{\epsilon_i}_{\text{noise}}
```

The regression function is the conditional expectation:

```math
E[Y\mid X=x]=\beta_0+\beta_1x
```

provided that

```math
E[\epsilon\mid X=x]=0.
```

This distinction is fundamental.

Linear regression does **not** necessarily claim that every observation lies exactly on a line.

It claims that the **conditional mean** is linear.

---

## 4. From Individual Observations to the Dataset

For observation \(i\),

```math
y_i=\beta_0+\beta_1x_i+\epsilon_i
```

For all \(N\) observations,

```math
\begin{aligned}
y_1 &= \beta_0+\beta_1x_1+\epsilon_1\\
y_2 &= \beta_0+\beta_1x_2+\epsilon_2\\
&\vdots\\
y_N &= \beta_0+\beta_1x_N+\epsilon_N
\end{aligned}
```

We now introduce matrix notation.

Define the observation vector

```math
\mathbf y=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_N
\end{bmatrix}
```

and the parameter vector

```math
\boldsymbol{\beta}
=
\begin{bmatrix}
\beta_0\\
\beta_1
\end{bmatrix}
```

Construct the design matrix

```math
X=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_N
\end{bmatrix}
```

Then the entire regression model becomes

```math
\boxed{
\mathbf y=X\boldsymbol{\beta}+\boldsymbol{\epsilon}
}
```

This is the fundamental equation of linear regression.

---

## 5. What Exactly Are We Learning?

The data \(X\) and \(\mathbf y\) are known.

The unknown quantity is the parameter vector:

```math
\boldsymbol{\beta}
```

Therefore, the learning problem is:

```math
\boxed{
\text{Given }X\text{ and }\mathbf y,
\text{ estimate }\boldsymbol{\beta}.
}
```

For simple linear regression,

```math
\boldsymbol{\beta}
=
\begin{bmatrix}
\beta_0\\
\beta_1
\end{bmatrix}
```

Once we know the parameters, we know the entire function:

```math
f(x)=\beta_0+\beta_1x
```

---

## 6. Why Can't We Simply Solve the Equations?

If there were exactly two observations and no noise,

```math
y_1=\beta_0+\beta_1x_1
```

and

```math
y_2=\beta_0+\beta_1x_2
```

we could solve exactly for the two unknown parameters.

But in an experiment we generally have many measurements,

```math
N\gg2
```

and the observations do not lie perfectly on a line.

Therefore, the system

```math
X\boldsymbol{\beta}=\mathbf y
```

usually has no exact solution.

Instead, we search for the parameter vector that produces predictions as close as possible to the observations.

This leads to **least squares**.

---

## 7. Residuals

For observation \(i\), the prediction is

```math
\hat{y}_i=\beta_0+\beta_1x_i
```

The residual is

```math
r_i=y_i-\hat{y}_i
```

Therefore,

```math
r_i=y_i-\beta_0-\beta_1x_i
```

In vector form,

```math
\mathbf r
=
\mathbf y-X\boldsymbol{\beta}
```

We now need a way to measure the total size of the residuals.

---

## 8. The Least-Squares Objective

We choose the squared Euclidean norm:

```math
\|\mathbf r\|_2^2
=
\mathbf r^T\mathbf r
```

Therefore,

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
```

Explicitly,

```math
\boxed{
\mathcal L(\boldsymbol{\beta})
=
\sum_{i=1}^{N}
(y_i-\beta_0-\beta_1x_i)^2
}
```

The least-squares estimator is therefore

```math
\boxed{
\hat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
}
```

This is the mathematical core of ordinary least-squares linear regression.

---

## 9. Why Square the Errors?

There are several mathematical and statistical reasons.

### 9.1 Positive and Negative Errors Should Not Cancel

If we minimized

```math
\sum_i r_i
```

positive and negative residuals could cancel.

Instead,

```math
r_i^2\geq0
```

so every residual contributes positively.

### 9.2 Large Errors Are Penalized More Strongly

For example,

```math
1^2=1
```

while

```math
5^2=25.
```

Thus large deviations receive disproportionately greater weight.

### 9.3 Differentiability

The squared loss is smooth:

```math
\frac{d}{dr}r^2=2r
```

which makes analytical and numerical optimization convenient.

### 9.4 Connection With Gaussian Noise

Suppose the measurement errors satisfy

```math
\epsilon_i\sim\mathcal N(0,\sigma^2)
```

Then

```math
p(y_i\mid x_i,\boldsymbol{\beta})
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left[
-\frac{(y_i-\beta_0-\beta_1x_i)^2}
{2\sigma^2}
\right]
```

For independent observations, the likelihood is

```math
L(\boldsymbol{\beta})
=
\prod_{i=1}^{N}
p(y_i\mid x_i,\boldsymbol{\beta})
```

Taking the logarithm gives

```math
\log L(\boldsymbol{\beta})
=
C
-
\frac{1}{2\sigma^2}
\sum_{i=1}^{N}
(y_i-\beta_0-\beta_1x_i)^2
```

where \(C\) does not depend on the parameters.

Therefore, maximizing the likelihood is equivalent to minimizing the sum of squared residuals.

Hence,

```math
\boxed{
\text{Least Squares}
\equiv
\text{Maximum Likelihood under Gaussian errors}
}
```

---

## 10. Deriving the Normal Equation

We want to minimize

```math
\mathcal L(\boldsymbol{\beta})
=
(\mathbf y-X\boldsymbol{\beta})^T
(\mathbf y-X\boldsymbol{\beta})
```

Expand the expression:

```math
\mathcal L
=
\mathbf y^T\mathbf y
-
2\boldsymbol{\beta}^TX^T\mathbf y
+
\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}
```

Take the gradient with respect to ```math \boldsymbol{\beta} ```:

```math
\nabla_{\boldsymbol{\beta}}\mathcal L
=
-2X^T\mathbf y
+
2X^TX\boldsymbol{\beta}
```

At the minimum,

```math
\nabla_{\boldsymbol{\beta}}\mathcal L=0
```

Therefore,

```math
-2X^T\mathbf y
+
2X^TX\hat{\boldsymbol{\beta}}
=0
```

Hence,

```math
X^TX\hat{\boldsymbol{\beta}}
=
X^T\mathbf y
```

This is the **normal equation**:

```math
\boxed{
X^TX\hat{\boldsymbol{\beta}}
=
X^T\mathbf y
}
```

If \(X^TX\) is invertible,

```math
\boxed{
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y
}
```

---

## 11. What Is Happening Geometrically?

The observation vector is

```math
\mathbf y\in\mathbb R^N
```

The columns of \(X\) span a subspace of ```math \mathbb R^N ```.

For simple linear regression,

```math
X=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_N
\end{bmatrix}
```

so the column space is

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

Every prediction

```math
X\boldsymbol{\beta}
```

lies inside this subspace.

But generally,

```math
\mathbf y\notin\operatorname{Col}(X)
```

because the observations contain noise.

Therefore, we find the point in the column space closest to ```math \mathbf y ```.

That point is

```math
\hat{\mathbf y}
=
X\hat{\boldsymbol{\beta}}
```

Thus linear regression is a **projection problem**.

---

## 12. Orthogonality of the Residual

At the optimum,

```math
X^T
\left(
\mathbf y-X\hat{\boldsymbol{\beta}}
\right)
=
0
```

The residual vector is

```math
\mathbf r
=
\mathbf y-X\hat{\boldsymbol{\beta}}
```

Therefore,

```math
\boxed{
X^T\mathbf r=0
}
```

The residual vector is orthogonal to every column of \(X\).

Geometrically,

```math
\mathbf y
=
\hat{\mathbf y}
+
\mathbf r
```

where

```math
\hat{\mathbf y}\in\operatorname{Col}(X)
```

and

```math
\mathbf r\perp\operatorname{Col}(X).
```

This is the geometric meaning of the normal equations.

---

## 13. Projection Matrix

The fitted values are

```math
\hat{\mathbf y}
=
X\hat{\boldsymbol{\beta}}
```

Substituting the least-squares solution,

```math
\hat{\mathbf y}
=
X(X^TX)^{-1}X^T\mathbf y
```

Define the matrix

```math
P
=
X(X^TX)^{-1}X^T
```

Then

```math
\boxed{
\hat{\mathbf y}=P\mathbf y
}
```

The matrix \(P\) is the orthogonal projection matrix onto the column space of \(X\).

It satisfies

```math
P^2=P
```

and

```math
P^T=P.
```

The residual-maker matrix is

```math
M=I-P
```

and therefore

```math
\mathbf r
=
(I-P)\mathbf y.
```

---

## 14. Simple Linear Regression in Closed Form

For

```math
y_i=\beta_0+\beta_1x_i+\epsilon_i
```

the slope estimator can be written as

```math
\boxed{
\hat\beta_1
=
\frac{
\sum_{i=1}^{N}
(x_i-\bar x)(y_i-\bar y)
}{
\sum_{i=1}^{N}
(x_i-\bar x)^2
}
}
```

where

```math
\bar x
=
\frac{1}{N}
\sum_{i=1}^{N}x_i
```

and

```math
\bar y
=
\frac{1}{N}
\sum_{i=1}^{N}y_i.
```

The intercept is

```math
\boxed{
\hat\beta_0
=
\bar y-\hat\beta_1\bar x
}
```

Therefore,

```math
\boxed{
\hat y
=
\hat\beta_0+\hat\beta_1x
}
```

---

## 15. Basis Functions

A crucial point in linear regression is that **"linear" refers to linearity in the parameters, not necessarily linearity in the input variable**.

The general basis-function model is

```math
f(x)
=
\sum_{j=0}^{p}
\beta_j\phi_j(x)
```

where

```math
\phi_j(x)
```

are known basis functions and

```math
\beta_j
```

are the parameters learned from data.

The model is linear in the parameters even if the basis functions are nonlinear in \(x\).

Therefore,

```math
\boxed{
\text{Linear in parameters}
\neq
\text{linear in }x
}
```

This distinction is fundamental.

---

## 16. Ordinary Linear Regression as a Basis-Function Model

Ordinary linear regression uses the basis functions

```math
\phi_0(x)=1
```

and

```math
\phi_1(x)=x.
```

Therefore,

```math
f(x)
=
\beta_0\phi_0(x)
+
\beta_1\phi_1(x)
```

becomes

```math
f(x)
=
\beta_0+\beta_1x.
```

This particular model is linear both in the parameters and in \(x\).

But this is only one possible choice of basis.

---

## 17. Polynomial Basis Functions

Consider the basis

```math
\phi_0(x)=1,
\qquad
\phi_1(x)=x,
\qquad
\phi_2(x)=x^2,
\qquad
\ldots,
\qquad
\phi_p(x)=x^p.
```

The resulting model is

```math
f(x)
=
\beta_0
+
\beta_1x
+
\beta_2x^2
+
\cdots
+
\beta_px^p.
```

This is nonlinear in \(x\).

However, it is still linear in the parameters:

```math
f(x)
=
\beta_0\phi_0(x)
+
\beta_1\phi_1(x)
+
\cdots
+
\beta_p\phi_p(x).
```

Hence polynomial regression is a linear regression model in the parameters.

---

## 18. General Basis-Function Representation

Define the feature vector

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

and the parameter vector

```math
\boldsymbol{\beta}
=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_p
\end{bmatrix}.
```

Then the model becomes

```math
f(x)
=
\boldsymbol{\beta}^{T}
\boldsymbol{\phi}(x).
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
\end{bmatrix}.
```

The model remains

```math
\mathbf y
=
X\boldsymbol{\beta}
+
\boldsymbol{\epsilon}.
```

The least-squares problem remains

```math
\hat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\left\|
\mathbf y-X\boldsymbol{\beta}
\right\|_2^2.
```

If \(X^TX\) is invertible,

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y.
```

The optimization machinery is therefore unchanged.

Only the **representation of the input** has changed.

---

## 19. Examples of Basis Functions

Different choices of basis functions produce different model classes.

### 19.1 Polynomial Basis

```math
\phi_j(x)=x^j
```

giving

```math
f(x)
=
\sum_{j=0}^{p}
\beta_jx^j.
```

### 19.2 Fourier Basis

For periodic phenomena, we may use

```math
\phi_k^{(1)}(x)=\sin(kx)
```

and

```math
\phi_k^{(2)}(x)=\cos(kx).
```

The resulting model can represent periodic behaviour:

```math
f(x)
=
a_0
+
\sum_{k=1}^{K}
\left[
a_k\cos(kx)
+
b_k\sin(kx)
\right].
```

### 19.3 Gaussian Radial Basis Functions

We may define

```math
\phi_j(x)
=
\exp
\left(
-\frac{(x-\mu_j)^2}{2\sigma_j^2}
\right).
```

The resulting model is

```math
f(x)
=
\sum_{j=1}^{M}
\beta_j\phi_j(x).
```

This can represent localized structures in the data.

Other choices include:

- Wavelets
- Splines
- Orthogonal polynomials
- Radial basis functions
- Eigenfunctions of differential operators
- Fourier modes

---

## 20. Basis Functions and Physics

Basis functions are particularly important in physics because many physical systems have natural mathematical representations.

For example, a periodic physical field can be represented using Fourier modes:

```math
u(x)
=
\sum_k
\left[
a_k\cos(kx)
+
b_k\sin(kx)
\right].
```

A solution to a differential equation can similarly be approximated as

```math
u(x)
=
\sum_{j=1}^{M}
c_j\phi_j(x).
```

The choice of basis can therefore encode knowledge about the structure of the physical problem.

This gives a useful scientific interpretation:

```math
\boxed{
\text{Basis choice}
\rightarrow
\text{Representation of the physical system}
}
```

---

## 21. Fixed Basis vs Learned Representation

In classical basis-function regression, the basis functions are chosen beforehand.

```math
x
\rightarrow
\{\phi_1(x),\phi_2(x),\ldots,\phi_M(x)\}.
```

The learning algorithm then determines the coefficients:

```math
\{\phi_j\}
\quad\text{fixed}
\qquad
\beta_j
\quad\text{learned}.
```

In neural networks, the situation changes.

Instead of manually specifying all useful features, the network can learn internal representations from data.

Conceptually:

```math
\text{Classical ML}
:
\text{Choose basis}
\rightarrow
\text{Learn coefficients}
```

whereas

```math
\text{Deep Learning}
:
\text{Learn representation}
\rightarrow
\text{Learn coefficients}
```

This transition from **hand-designed features** to **learned representations** is one of the central ideas connecting classical Machine Learning to neural networks.

---

## 22. Statistical Assumptions

The least-squares coefficients can be calculated without requiring every classical statistical assumption.

However, statistical inference requires assumptions about the data-generating process.

A classical linear regression model commonly assumes:

### 22.1 Linearity

The conditional mean is linear in the parameters:

```math
E[Y\mid X=x]
=
\beta_0+\beta_1x.
```

More generally, with basis functions:

```math
E[Y\mid X=x]
=
\sum_{j=0}^{p}
\beta_j\phi_j(x).
```

### 22.2 Zero Conditional Mean

```math
E[\epsilon\mid X]=0.
```

This is crucial for unbiased estimation.

### 22.3 Constant Variance

```math
\operatorname{Var}(\epsilon\mid X)=\sigma^2.
```

This is called **homoscedasticity**.

### 22.4 Independence

The errors are assumed independent in the classical setting.

### 22.5 Gaussian Errors

For exact small-sample hypothesis tests and likelihood-based inference, one often assumes

```math
\epsilon_i\sim\mathcal N(0,\sigma^2).
```

Gaussianity is **not required simply to calculate the ordinary least-squares coefficients**.

---

## 23. Bias of the Estimator

Suppose

```math
\mathbf y
=
X\boldsymbol{\beta}
+
\boldsymbol{\epsilon}
```

with

```math
E[\boldsymbol{\epsilon}\mid X]=0.
```

The estimator is

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y.
```

Substitute the model:

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}
X^T
(X\boldsymbol{\beta}+\boldsymbol{\epsilon}).
```

Therefore,

```math
\hat{\boldsymbol{\beta}}
=
\boldsymbol{\beta}
+
(X^TX)^{-1}X^T\boldsymbol{\epsilon}.
```

Taking the conditional expectation,

```math
E[\hat{\boldsymbol{\beta}}\mid X]
=
\boldsymbol{\beta}.
```

Thus the ordinary least-squares estimator is unbiased under the zero-conditional-mean assumption.

---

## 24. Variance of the Estimator

Assume

```math
\operatorname{Var}(\boldsymbol{\epsilon}\mid X)
=
\sigma^2I.
```

Then

```math
\boxed{
\operatorname{Var}
(\hat{\boldsymbol{\beta}}\mid X)
=
\sigma^2(X^TX)^{-1}
}
```

This equation tells us how uncertainty in the measurements propagates into uncertainty in the estimated parameters.

For example, if the \(x_i\) values have very little spread, estimating the slope becomes difficult.

This is why **experimental design** matters.

---

## 25. What Happens If the Assumptions Fail?

The model can still produce a fitted function.

But its interpretation and statistical properties can change.

### Nonlinearity

If the conditional mean cannot be adequately represented by the chosen basis functions, the model may systematically miss the underlying relationship.

### Heteroscedasticity

If

```math
\operatorname{Var}(\epsilon\mid X)
```

depends on \(X\), ordinary standard errors may be unreliable.

### Correlated Errors

This is particularly important in physics.

Measurements taken sequentially in time may satisfy

```math
\operatorname{Cov}(\epsilon_i,\epsilon_j)\neq0.
```

The observations are then not independent.

### Outliers

Because the loss contains

```math
r_i^2
```

large residuals can have enormous influence on the fitted model.

This motivates robust regression methods such as:

- Huber regression
- RANSAC
- Least absolute deviations

---

## 26. Multiple Linear Regression

Linear regression does not have to involve only one input.

Suppose we want to predict a physical quantity using several measurements:

```math
x_1,x_2,\ldots,x_p.
```

The model becomes

```math
y
=
\beta_0
+
\beta_1x_1
+
\beta_2x_2
+\cdots+
\beta_px_p
+
\epsilon.
```

In matrix notation,

```math
\boxed{
\mathbf y=X\boldsymbol{\beta}+\boldsymbol{\epsilon}
}
```

where

```math
X\in\mathbb R^{N\times(p+1)}.
```

The least-squares estimator remains

```math
\boxed{
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y
}
```

when \(X^TX\) is invertible.

Again, the crucial point is that the model is **linear in the unknown coefficients**.

---

## 27. Polynomial Regression

Consider

```math
y
=
\beta_0
+
\beta_1x
+
\beta_2x^2
+
\beta_3x^3
+
\epsilon.
```

This is nonlinear in \(x\), but it is still linear in the parameters.

The design matrix is

```math
X=
\begin{bmatrix}
1 & x_1 & x_1^2 & x_1^3\\
1 & x_2 & x_2^2 & x_2^3\\
\vdots & \vdots & \vdots & \vdots\\
1 & x_N & x_N^2 & x_N^3
\end{bmatrix}.
```

The same least-squares machinery therefore applies.

---

## 28. Identifiability and Rank

The expression

```math
(X^TX)^{-1}
```

requires \(X^TX\) to be invertible.

This is connected to the rank of the design matrix.

If the columns of \(X\) are linearly independent, then

```math
\operatorname{rank}(X)=p+1.
```

and \(X^TX\) is invertible.

If the columns are linearly dependent, the parameters cannot be uniquely determined by ordinary least squares.

For example, suppose two features satisfy

```math
x_2=2x_1.
```

Then the corresponding columns contain redundant information.

This is the mathematical foundation of **multicollinearity**.

---

## 29. Numerical Stability

Although the normal equation gives the elegant expression

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y
```

one should generally avoid explicitly computing the inverse in numerical implementations.

Instead, numerical linear algebra methods such as:

- QR decomposition
- Singular Value Decomposition
- Cholesky decomposition

can be used.

The Singular Value Decomposition is particularly useful when \(X\) is poorly conditioned or rank-deficient.

If

```math
X=U\Sigma V^T
```

then the least-squares solution can be expressed using the pseudoinverse:

```math
\hat{\boldsymbol{\beta}}
=
X^+\mathbf y
```

where \(X^+\) is the Moore-Penrose pseudoinverse.

This is the more general linear-algebraic viewpoint.

---

## 30. Training by Optimization

The closed-form solution is not the only way to minimize the loss.

We can instead use an iterative optimization algorithm.

The loss is

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2.
```

Its gradient is

```math
\nabla_{\boldsymbol{\beta}}\mathcal L
=
2X^T(X\boldsymbol{\beta}-\mathbf y).
```

Gradient descent updates the parameters according to

```math
\boldsymbol{\beta}_{k+1}
=
\boldsymbol{\beta}_k
-
\eta
\nabla_{\boldsymbol{\beta}}
\mathcal L(\boldsymbol{\beta}_k)
```

where \(\eta\) is the learning rate.

Therefore,

```math
\boldsymbol{\beta}_{k+1}
=
\boldsymbol{\beta}_k
-
2\eta
X^T
(X\boldsymbol{\beta}_k-\mathbf y).
```

This is conceptually important because the same optimization idea appears throughout Machine Learning.

For linear regression, however, the problem has a closed-form solution, so iterative optimization is usually unnecessary for small and moderate problems.

---

## 31. Convexity

The least-squares objective is a convex quadratic function.

Its Hessian is

```math
\nabla^2_{\boldsymbol{\beta}}\mathcal L
=
2X^TX.
```

Since

```math
X^TX
```

is positive semidefinite,

```math
\nabla^2_{\boldsymbol{\beta}}\mathcal L
\succeq0.
```

Therefore, the loss has no spurious local minima.

If \(X\) has full column rank, then \(X^TX\) is positive definite and the objective has a unique global minimum.

This makes ordinary least squares mathematically much simpler than many modern neural-network optimization problems.

---

## 32. Training Error and Generalization

The least-squares objective measures how well the model fits the training data:

```math
\mathcal L_{\text{train}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2.
```

But our real objective is not simply to memorize the training observations.

We want the model to perform well on unseen observations.

Therefore, we distinguish between training error and test error.

A model that fits the training data extremely well can still perform poorly on new data.

This introduces the central ML concept of **generalization**.

---

## 33. Bias-Variance Perspective

Prediction error can be understood through the bias-variance decomposition.

Conceptually,

```math
\text{Expected Error}
=
\text{Bias}^2
+
\text{Variance}
+
\text{Irreducible Noise}.
```

A model with too little flexibility may have high bias.

A model with excessive flexibility may have high variance.

The choice of basis functions therefore affects the bias-variance trade-off.

For example, increasing the degree of a polynomial can make the model more flexible:

```math
\text{Linear}
\rightarrow
\text{Quadratic}
\rightarrow
\text{Cubic}
\rightarrow
\cdots
```

But excessive flexibility can lead to overfitting.

---

## 34. Regularization

If the number of basis functions becomes large, the model can become unstable or overfit.

We can modify the optimization problem by adding a penalty.

### Ridge Regression

Ridge regression minimizes

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
+
\lambda\|\boldsymbol{\beta}\|_2^2.
```

where

```math
\|\boldsymbol{\beta}\|_2^2
=
\sum_j\beta_j^2.
```

The solution is

```math
\boxed{
\hat{\boldsymbol{\beta}}
=
(X^TX+\lambda I)^{-1}X^T\mathbf y
}
```

### Lasso Regression

Lasso instead uses an \(L^1\) penalty:

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
+
\lambda\|\boldsymbol{\beta}\|_1.
```

where

```math
\|\boldsymbol{\beta}\|_1
=
\sum_j|\beta_j|.
```

The \(L^1\) penalty can encourage sparse parameter vectors.

These methods introduce an important ML principle:

```math
\boxed{
\text{Fit the data}
+
\text{Control model complexity}
}
```

---

## 35. Connection to Physics: Constant Acceleration

Consider an experiment in which the velocity of an object is measured at different times.

The physical model is

```math
v(t)=v_0+at.
```

Suppose our measurements contain noise:

```math
v_i=v_0+at_i+\epsilon_i.
```

Compare this with the regression model:

```math
y_i=\beta_0+\beta_1x_i+\epsilon_i.
```

The correspondence is:

```math
y_i\leftrightarrow v_i
```

```math
x_i\leftrightarrow t_i
```

```math
\beta_0\leftrightarrow v_0
```

```math
\beta_1\leftrightarrow a.
```

Therefore, fitting linear regression to the experimental data estimates the physical parameters.

The scientific workflow is

```math
\boxed{
\text{Physical Law}
\rightarrow
\text{Mathematical Model}
\rightarrow
\text{Experimental Data}
\rightarrow
\text{Parameter Estimation}
}
```

This illustrates why Machine Learning and scientific modelling are closely related.

---

## 36. Connection to Physics: Hubble's Law

A particularly useful scientific example is the relationship between recession velocity and distance.

In its simplest form,

```math
v=H_0d
```

where \(H_0\) is the Hubble constant.

Real observations contain uncertainties:

```math
v_i=H_0d_i+\epsilon_i.
```

This is a linear regression model with zero intercept.

The least-squares estimate is obtained by minimizing

```math
\mathcal L(H_0)
=
\sum_{i=1}^{N}
(v_i-H_0d_i)^2.
```

Taking the derivative,

```math
\frac{d\mathcal L}{dH_0}
=
-2
\sum_{i=1}^{N}
d_i(v_i-H_0d_i).
```

Setting it to zero gives

```math
\sum_i d_iv_i
=
H_0\sum_i d_i^2.
```

Therefore,

```math
\boxed{
\hat H_0
=
\frac{
\sum_i d_iv_i
}{
\sum_i d_i^2
}
}
```

Here the regression coefficient has a direct physical interpretation.

This illustrates an important principle:

```math
\boxed{
\text{Machine Learning parameter}
\neq
\text{necessarily abstract number}
}
```

A learned parameter can correspond directly to a physical constant.

---

## 37. Linear Regression as Function Approximation

At a more general level, Machine Learning can be viewed as function approximation.

We assume that there exists some unknown relationship

```math
y=f(x)+\epsilon.
```

We do not know \(f\).

Instead, we choose a function class.

For linear regression with basis functions,

```math
f_\beta(x)
=
\sum_{j=0}^{p}
\beta_j\phi_j(x).
```

The learning problem is therefore

```math
\boxed{
\text{Find the function }f_\beta
\text{ that best explains the observed data.}
}
```

The basis functions determine the space of functions that our model can represent.

The parameters determine the particular function selected from that space.

---

## 38. Model Class, Parameters, and Data

It is useful to separate three concepts.

### Model Class

The model class specifies the functions that are allowed.

For example,

```math
\mathcal F
=
\left\{
f(x)=\beta_0+\beta_1x
\right\}.
```

### Parameters

The parameters specify one particular member of that class.

```math
\boldsymbol{\beta}
=
(\beta_0,\beta_1).
```

### Data

The data tell us which member of the model class is most appropriate.

```math
\mathcal D
=
\{(x_i,y_i)\}_{i=1}^{N}.
```

Therefore,

```math
\boxed{
\text{Model class}
+
\text{Data}
\rightarrow
\text{Learned parameters}
}
```

This distinction becomes extremely important when moving from classical ML to neural networks.

---

## 39. What Does "Learning" Actually Mean?

The word **learning** can sound mysterious.

Mathematically, in ordinary linear regression, nothing mysterious is happening.

We have:

```math
\text{unknown parameters}
\rightarrow
\text{define an objective}
\rightarrow
\text{minimize the objective}
```

Specifically,

```math
\hat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\mathcal L(\boldsymbol{\beta})
```

where

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2.
```

Thus "training" the model means finding the parameter values that minimize the chosen objective.

---

## 40. Why Does the Model Generalize?

The training data are only a finite sample from some underlying process.

We want the learned function to work on new observations from the same process.

Generalization depends on several factors:

- The amount of training data
- The noise level
- The choice of model class
- The choice of basis functions
- The complexity of the model
- The assumptions made about the data
- Regularization
- The relationship between training and test distributions

A simple model can generalize well when its assumptions are appropriate.

A highly complicated model can fail when it learns patterns specific to the training sample.

Thus, good Machine Learning is not simply about minimizing training loss.

It is about finding a useful approximation to the underlying data-generating relationship.

---

## 41. What Assumptions Does the Model Make?

This question is especially important in scientific Machine Learning.

A linear regression model assumes a particular functional structure:

```math
E[Y\mid X=x]
=
\sum_j\beta_j\phi_j(x).
```

The basis functions therefore encode assumptions about what kinds of relationships are possible.

For ordinary linear regression,

```math
\phi_0(x)=1
```

and

```math
\phi_1(x)=x.
```

The model therefore assumes the conditional mean can be represented by an affine function of \(x\).

If we use a polynomial basis,

```math
\phi_j(x)=x^j,
```

we allow more complex relationships.

If we use Fourier basis functions, we assume periodic structures may be useful.

Therefore:

```math
\boxed{
\text{Model architecture}
=
\text{Set of assumptions about representable functions}
}
```

This idea becomes central in physics-informed and scientific Machine Learning.

---

## 42. Why Can Training Become Unstable?

For ordinary least squares, the optimization problem is convex and therefore relatively well behaved.

However, numerical difficulties can still occur.

One important source is ill-conditioning.

If two basis functions are nearly linearly dependent, then the columns of \(X\) become nearly dependent.

Consequently,

```math
X^TX
```

can become poorly conditioned.

Small changes in the data can then produce large changes in the estimated parameters.

This can be especially severe for high-degree polynomial bases.

For example,

```math
1,x,x^2,x^3,\ldots,x^p
```

can become highly correlated over certain ranges of \(x\).

This is one reason why numerical methods, scaling, orthogonal bases, and regularization are important.

---

## 43. Connection Between Classical ML and Scientific Modelling

Classical scientific modelling often begins with a known physical law.

For example,

```math
F=ma
```

or

```math
v=v_0+at.
```

The model parameters may have direct physical meaning.

Machine Learning reverses part of this process.

Instead of starting with a completely known relationship, we may start with observations:

```math
\{(x_i,y_i)\}_{i=1}^{N}
```

and ask the model to infer a useful relationship.

Linear regression is therefore a bridge between the two viewpoints.

```math
\boxed{
\text{Physics}
\leftrightarrow
\text{Mathematical modelling}
\leftrightarrow
\text{Machine Learning}
}
```

The difference is not that one uses mathematics and the other does not.

Both rely on assumptions, mathematical representations, parameters, objectives, and inference.

---

## 44. The Full Linear Regression Pipeline

The complete mathematical workflow can be summarized as follows.

### Step 1: Obtain measurements

```math
\mathcal D
=
\{(x_i,y_i)\}_{i=1}^{N}
```

### Step 2: Choose a model representation

```math
f(x)
=
\sum_{j=0}^{p}
\beta_j\phi_j(x)
```

### Step 3: Construct the design matrix

```math
X_{ij}
=
\phi_j(x_i)
```

### Step 4: Define the prediction

```math
\hat{\mathbf y}
=
X\boldsymbol{\beta}
```

### Step 5: Define the loss

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
```

### Step 6: Optimize

```math
\hat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\mathcal L(\boldsymbol{\beta})
```

### Step 7: Obtain the fitted model

```math
\hat f(x)
=
\sum_{j=0}^{p}
\hat\beta_j\phi_j(x)
```

### Step 8: Evaluate generalization

Use previously unseen data to estimate how well

```math
\hat f(x)
```

approximates the underlying relationship.

---

## 45. The Central Mathematical Picture

Linear regression can be understood simultaneously from several perspectives.

### Function Approximation

```math
f_\theta(x)\approx y
```

### Statistical Model

```math
\mathbf y=X\boldsymbol{\beta}+\boldsymbol{\epsilon}
```

### Optimization Problem

```math
\hat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
```

### Linear Algebra Problem

```math
X^TX\hat{\boldsymbol{\beta}}
=
X^T\mathbf y
```

### Geometric Projection

```math
\hat{\mathbf y}
=
\operatorname{Proj}_{\operatorname{Col}(X)}
(\mathbf y)
```

### Statistical Estimation

```math
\hat{\boldsymbol{\beta}}
```

estimates unknown parameters of a data-generating process.

These are not different algorithms.

They are different mathematical perspectives on the same method.

---

## 46. What the Algorithm Actually Does

At the most fundamental level:

1. We have observations \(X\) and \(\mathbf y\).
2. We choose a model class.
3. We choose a representation or basis.
4. We define a loss function.
5. We minimize that loss.
6. We obtain parameter estimates.
7. We use the resulting function to make predictions.
8. We evaluate whether the model generalizes to unseen data.

For ordinary least squares,

```math
\boxed{
(X,\mathbf y)
\rightarrow
\min_{\boldsymbol{\beta}}
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
\rightarrow
\hat{\boldsymbol{\beta}}
\rightarrow
\hat{\mathbf y}
}
```

This simple pipeline is the foundation upon which much more complicated Machine Learning is built.

The progression is therefore:

```math
\boxed{
\text{Choose a function class}
\rightarrow
\text{Represent the input}
\rightarrow
\text{Define a loss}
\rightarrow
\text{Optimize parameters}
\rightarrow
\text{Generalize}
}
```

Understanding these ideas in linear regression provides the mathematical foundation for the models that follow, from logistic regression and support vector machines to neural networks, transformers, and ultimately neural operators.
