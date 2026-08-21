# Linear Regression: Mathematical Theory

## 1. What Do We Have?

Suppose we have measured two quantities for \(N\) physical systems:

```math
(x_1,y_1),(x_2,y_2),\ldots,(x_N,y_N)
```

Here:

- \(x_i\) is the input or explanatory variable.
- \(y_i\) is the observed output.
- \(N\) is the number of observations.

For a physics example, imagine measuring the velocity of an object at different times. Under constant acceleration,

```math
v(t)=v_0+at
```

We can therefore identify

```math
x=t
```

and

```math
y=v
```

Our experimental data might therefore be

```math
(t_1,v_1),(t_2,v_2),\ldots,(t_N,v_N)
```

The measurements will generally not lie perfectly on a straight line because of experimental noise.

---

## 2. What Is Our Goal?

We want to find a mathematical function that describes the relationship between the input and output.

For the simplest model,

```math
\hat y=\beta_0+\beta_1x
```

where:

- \(\beta_0\) is the intercept.
- \(\beta_1\) is the slope.
- \(\hat y\) is the predicted value.

For the velocity example,

```math
\hat v=\beta_0+\beta_1t
```

Physically, we would interpret

```math
\beta_0\approx v_0
```

and

```math
\beta_1\approx a
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

where \(\epsilon_i\) represents unexplained variation or measurement noise.

Therefore,

```math
\boxed{
y_i
=
\underbrace{\beta_0+\beta_1x_i}_{\text{systematic component}}
+
\underbrace{\epsilon_i}_{\text{noise}}
}
```

The regression function is the conditional expectation:

```math
E[Y\mid X=x]=\beta_0+\beta_1x
```

provided that

```math
E[\epsilon\mid X=x]=0
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

we could solve exactly for \(\beta_0\) and \(\beta_1\).

But in an experiment we generally have

```math
N\gg2
```

measurements, and the observations do not lie perfectly on a line.

Therefore, the system

```math
X\boldsymbol{\beta}=\mathbf y
```

usually has **no exact solution**.

Instead, we search for the parameter vector that produces predictions as close as possible to the observations.

This leads to **least squares**.

---

## 7. Residuals

For observation \(i\), the prediction is

```math
\hat y_i=\beta_0+\beta_1x_i
```

The residual is

```math
r_i=y_i-\hat y_i
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
5^2=25
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
p(y_i\mid x_i,\beta)
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

Hence:

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

Take the gradient with respect to \(\boldsymbol{\beta}\):

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

This is one of the most important interpretations of linear regression.

The observation vector is

```math
\mathbf y\in\mathbb R^N
```

The columns of \(X\) span a subspace of \(\mathbb R^N\).

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

Therefore, we find the point in the column space closest to \(\mathbf y\).

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
\mathbf r\perp\operatorname{Col}(X)
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
P^T=P
```

The residual-maker matrix is

```math
M=I-P
```

and therefore

```math
\mathbf r
=
(I-P)\mathbf y
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
\sum_{i=1}^{N}y_i
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

## 15. Connection to Covariance and Variance

The slope is

```math
\hat\beta_1
=
\frac{
\sum_i(x_i-\bar x)(y_i-\bar y)
}{
\sum_i(x_i-\bar x)^2
}
```

The numerator measures the joint variation of \(X\) and \(Y\).

Therefore, using the usual definitions of sample covariance and sample variance,

```math
\boxed{
\hat\beta_1
=
\frac{\operatorname{Cov}(X,Y)}
{\operatorname{Var}(X)}
}
```

This gives an important interpretation:

> The regression slope measures the change in the predicted response associated with a unit change in the predictor, determined by their joint variation relative to the variation in the predictor.

---

## 16. Basis Functions

A crucial point in linear regression is that **"linear" refers to linearity in the parameters, not necessarily linearity in the input variables**.

The general model is

```math
f(x)
=
\sum_{j=0}^{p}
\beta_j\phi_j(x)
```

where:

- \(\phi_j(x)\) are known **basis functions**.
- \(\beta_j\) are the parameters learned from data.

The model is linear in the parameters \(\beta_j\), even when the basis functions are nonlinear in \(x\).

Therefore, a model can be highly nonlinear in \(x\) and still be a linear regression model.

```math
\boxed{
\text{Linear regression}
\neq
\text{necessarily linear in }x
}
```

Rather,

```math
\boxed{
\text{Linear regression}
=
\text{linear in the unknown parameters}
}
```

---

## 17. Ordinary Linear Regression as a Basis-Function Model

Ordinary linear regression uses

```math
\phi_0(x)=1
```

and

```math
\phi_1(x)=x
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
\beta_0+\beta_1x
```

This particular model happens to be linear in both the parameters and \(x\).

But this is only one possible choice of basis.

---

## 18. Polynomial Basis Functions

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
\phi_p(x)=x^p
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
\beta_px^p
```

This is **nonlinear in \(x\)**.

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
\beta_p\phi_p(x)
```

Hence polynomial regression is a linear regression model in the parameters.

---

## 19. General Basis-Function Representation

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
\end{bmatrix}
```

Then

```math
f(x)
=
\boldsymbol{\beta}^{T}
\boldsymbol{\phi}(x)
```

For \(N\) observations, the design matrix becomes

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

The model remains

```math
\mathbf y
=
X\boldsymbol{\beta}
+
\boldsymbol{\epsilon}
```

Therefore, the least-squares problem remains

```math
\hat{\boldsymbol{\beta}}
=
\arg\min_{\boldsymbol{\beta}}
\left\|
\mathbf y-X\boldsymbol{\beta}
\right\|_2^2
```

and, when the inverse exists,

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y
```

The optimization machinery is therefore unchanged.

Only the **representation of the input** has changed.

---

## 20. Examples of Basis Functions

Different choices of basis functions produce different model classes.

### Polynomial Basis

```math
\phi_j(x)=x^j
```

giving

```math
f(x)
=
\sum_{j=0}^{p}
\beta_jx^j
```

### Fourier Basis

For periodic phenomena, we may use

```math
\phi_k^{(1)}(x)=\sin(kx)
```

and

```math
\phi_k^{(2)}(x)=\cos(kx)
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
\right]
```

### Gaussian Radial Basis Functions

We may define

```math
\phi_j(x)
=
\exp
\left(
-\frac{(x-\mu_j)^2}{2\sigma_j^2}
\right)
```

The resulting model is

```math
f(x)
=
\sum_{j=1}^{M}
\beta_j\phi_j(x)
```

This can represent localized structures in the data.

Other important choices include:

- Wavelets
- Splines
- Orthogonal polynomials
- Radial basis functions
- Eigenfunctions of differential operators
- Fourier modes

---

## 21. Basis Functions and Physics

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
\right]
```

A solution to a differential equation can similarly be approximated as

```math
u(x)
=
\sum_{j=1}^{M}
c_j\phi_j(x)
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

## 22. Fixed Basis vs Learned Representation

In classical basis-function regression, the basis functions are chosen beforehand.

```math
x
\rightarrow
\{
\phi_1(x),
\phi_2(x),
\ldots,
\phi_M(x)
\}
```

The learning algorithm then determines the coefficients.

```math
\{\phi_j\}
\quad\text{fixed}
\qquad
\beta_j
\quad\text{learned}
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

## 23. Statistical Assumptions

The least-squares coefficients can be calculated without requiring every classical statistical assumption.

However, **statistical inference** requires assumptions about the data-generating process.

A classical linear regression model commonly assumes:

### 23.1 Linearity

The conditional mean is linear in the parameters:

```math
E[Y\mid X=x]
=
\beta_0+\beta_1x
```

More generally, with basis functions:

```math
E[Y\mid X=x]
=
\sum_{j=0}^{p}
\beta_j\phi_j(x)
```

### 23.2 Zero Conditional Mean

```math
E[\epsilon\mid X]=0
```

This is crucial for unbiased estimation.

### 23.3 Constant Variance

```math
\operatorname{Var}(\epsilon\mid X)=\sigma^2
```

This is called **homoscedasticity**.

### 23.4 Independence

The errors are assumed independent in the classical setting.

### 23.5 Gaussian Errors

For exact small-sample hypothesis tests and likelihood-based inference, one often assumes

```math
\epsilon_i\sim\mathcal N(0,\sigma^2)
```

Gaussianity is **not required simply to calculate the ordinary least-squares coefficients**.

---

## 24. Bias of the Estimator

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
E[\boldsymbol{\epsilon}\mid X]=0
```

The estimator is

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}X^T\mathbf y
```

Substitute the model:

```math
\hat{\boldsymbol{\beta}}
=
(X^TX)^{-1}
X^T
(X\boldsymbol{\beta}+\boldsymbol{\epsilon})
```

Therefore,

```math
\hat{\boldsymbol{\beta}}
=
\boldsymbol{\beta}
+
(X^TX)^{-1}X^T\boldsymbol{\epsilon}
```

Taking the conditional expectation,

```math
E[\hat{\boldsymbol{\beta}}\mid X]
=
\boldsymbol{\beta}
```

Thus the ordinary least-squares estimator is unbiased under the zero-conditional-mean assumption.

---

## 25. Variance of the Estimator

Assume

```math
\operatorname{Var}(\boldsymbol{\epsilon}\mid X)
=
\sigma^2I
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

## 26. What Happens If the Assumptions Fail?

The model can still produce a fitted function.

But its interpretation and statistical properties can change.

### Nonlinearity

If

```math
E[Y\mid X=x]
```

cannot be adequately represented by the chosen basis functions, the model may systematically miss the underlying relationship.

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
\operatorname{Cov}(\epsilon_i,\epsilon_j)\neq0
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

## 27. Multiple Linear Regression

Linear regression does not have to involve only one input.

Suppose we want to predict a physical quantity using several measurements:

```math
x_1,x_2,\ldots,x_p
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
\epsilon
```

In matrix notation,

```math
\boxed{
\mathbf y=X\boldsymbol{\beta}+\boldsymbol{\epsilon}
}
```

where

```math
X\in\mathbb R^{N\times(p+1)}
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

## 28. Polynomial Regression

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
\epsilon
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
\end{bmatrix}
```

The same least-squares machinery therefore applies.

---

## 29. Identifiability and Rank

The expression

```math
(X^TX)^{-1}
```

requires \(X^TX\) to be invertible.

This is connected to the rank of the design matrix.

If the columns of \(X\) are linearly independent, then

```math
\operatorname{rank}(X)=p+1
```

and \(X^TX\) is invertible.

If the columns are linearly dependent, the parameters cannot be uniquely determined by ordinary least squares.

For example, suppose two features satisfy

```math
x_2=2x_1
```

Then the corresponding columns contain redundant information.

This is the mathematical foundation of **multicollinearity**.

---

## 30. Numerical Stability

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

## 31. Training by Optimization

The closed-form solution is not the only way to minimize the loss.

We can instead use an iterative optimization algorithm.

The loss is

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
```

Its gradient is

```math
\nabla_{\boldsymbol{\beta}}\mathcal L
=
2X^T(X\boldsymbol{\beta}-\mathbf y)
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
(X\boldsymbol{\beta}_k-\mathbf y)
```

This is conceptually important because the same optimization idea appears throughout Machine Learning.

For linear regression, however, the problem has a closed-form solution, so iterative optimization is usually unnecessary for small and moderate problems.

---

## 32. Convexity

The least-squares objective is a convex quadratic function.

Its Hessian is

```math
\nabla^2_{\boldsymbol{\beta}}\mathcal L
=
2X^TX
```

Since

```math
X^TX
```

is positive semidefinite,

```math
\nabla^2_{\boldsymbol{\beta}}\mathcal L
\succeq0
```

Therefore, the loss has no spurious local minima.

If \(X\) has full column rank, then \(X^TX\) is positive definite and the objective has a unique global minimum.

This makes ordinary least squares mathematically much simpler than many modern neural-network optimization problems.

---

## 33. Training Error and Generalization

The least-squares objective measures how well the model fits the training data:

```math
\mathcal L_{\text{train}}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
```

But our real objective is not simply to memorize the training observations.

We want the model to perform well on unseen observations.

Therefore, we distinguish between:

```math
\text{Training Error}
```

and

```math
\text{Test Error}
```

A model that fits the training data extremely well can still perform poorly on new data.

This introduces the central ML concept of **generalization**.

---

## 34. Bias-Variance Perspective

Prediction error can be understood through the bias-variance decomposition.

For an estimator of a target quantity, the expected squared prediction error can be decomposed conceptually into:

```math
\text{Expected Error}
=
\text{Bias}^2
+
\text{Variance}
+
\text{Irreducible Noise}
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

## 35. Regularization

If the number of basis functions becomes large, the model can become unstable or overfit.

We can modify the optimization problem by adding a penalty.

### Ridge Regression

Ridge regression minimizes

```math
\mathcal L(\boldsymbol{\beta})
=
\|\mathbf y-X\boldsymbol{\beta}\|_2^2
+
\lambda\|\boldsymbol{\beta}\|_2^2
```

where

```math
\|\boldsymbol{\beta}\|_2^2
=
\sum_j\beta_j^2
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
\lambda\|\boldsymbol{\beta}\|_1
```

where

```math
\|\boldsymbol{\beta}\|_1
=
\sum_j|\beta_j|
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

## 36. Connection to Physics: Constant Acceleration

Consider an experiment in which the velocity of an object is measured at different times.

The physical model is

```math
v(t)=v_0+at
```

Suppose our measurements contain noise:

```math
v_i=v_0+at_i+\epsilon_i
```

Compare this with the regression model:

```math
y_i=\beta_0+\beta_1x_i+\epsilon_i
```

The correspondence is

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
\beta_1\leftrightarrow a
```

Therefore, fitting linear regression to the experimental data estimates the physical parameters.

The scientific workflow is:

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

This is a useful way to understand why Machine Learning and scientific modelling are closely related.

---

## 37. A Deeper Physics Example: Hubble's Law

A particularly useful scientific example is the relationship between recession velocity and distance.

In its simplest form,

```math
v=H_0d
```

where \(H_0\) is the Hubble constant.

Real observations contain uncertainties:

```math
v_i=H_0d_i+\epsilon_i
```

This is a linear regression model with zero intercept.

The least-squares estimate is obtained by minimizing

```math
\mathcal L(H_0)
=
\sum_{i=1}^{N}
(v_i-H_0d_i)^2
```

Taking the derivative,

```math
\frac{d\mathcal L}{dH_0}
=
-2
\sum_{i=1}^{N}
d_i(v_i-H_0d_i)
```

Setting it to zero gives

```math
\sum_i d_iv_i
=
H_0\sum_i d_i^2
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

## 38. The Central Mathematical Picture

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

These are **not different algorithms**.

They are different mathematical perspectives on the same method.

---

## 39. What the Algorithm Actually Does

At the most fundamental level:

1. We have observations \(X\) and \(\mathbf y\).
2. We choose a model class.
3. We choose a representation or basis.
4. We define a loss function.
5. We minimize that loss.
6. We obtain parameter estimates.
7. We use the resulting function to make predictions.
8. We evaluate whether the model generalizes to unseen data.

For ordinary least squares:

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
