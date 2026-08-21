# Linear Regression: Foundations

## 1. What Do We Have?

Suppose we perform an experiment and record measurements.

Each observation consists of an input and an output.

```math
(x_i,y_i)
```

If we collect a total of $N$ observations, the dataset is

```math
\mathcal{D}
=
\{(x_i,y_i)\}_{i=1}^{N}
```

The input is the quantity that we use to explain or predict the output.

The output is the quantity that we want to understand or predict.

For example, the input could be time and the output could be velocity.

In a more general setting, the input could represent temperature, distance, pressure, energy, or another measurable physical quantity.

The output could represent velocity, displacement, intensity, temperature, or another physical response.

The fundamental question is:

**Can we construct a mathematical relationship between the input and the output from the observed data?**

---

## 2. A Physics Scenario

Consider an object moving with approximately constant acceleration.

Classical mechanics gives the relationship

```math
v(t)
=
v_0+at
```

where the initial velocity is represented by

```math
v_0
```

the acceleration is represented by

```math
a
```

and the velocity at time \(t\) is represented by

```math
v(t)
```

Suppose we perform an experiment and measure the velocity at several different times.

Our observations can be represented as

```math
(t_1,v_1),
(t_2,v_2),
\ldots,
(t_N,v_N)
```

If the experiment were perfectly accurate, every observation would lie exactly on the theoretical relationship.

Real measurements are not perfectly accurate.

There may be uncertainty in the measuring instrument, environmental effects, fluctuations in the experimental system, or physical effects that have not been included in the simplified model.

We therefore introduce an error term.

```math
v_i
=
v_0+at_i+\epsilon_i
```

The term

```math
\epsilon_i
```

represents measurement noise or other unexplained variation.

The important point is that the observed velocity is not necessarily equal to the ideal theoretical prediction.

Instead, it consists of two components:

```math
v_i
=
\text{systematic physical component}
+
\text{unexplained variation}
```

For this particular problem, the systematic component is

```math
v_0+at_i
```

and the unexplained component is

```math
\epsilon_i
```

---

## 3. Connection to the Regression Model

The physical equation

```math
v_i
=
v_0+at_i+\epsilon_i
```

has exactly the same mathematical structure as the general regression model

```math
y_i
=
\beta_0+\beta_1x_i+\epsilon_i
```

The correspondence is

```math
x_i
=
t_i
```

```math
y_i
=
v_i
```

```math
\beta_0
=
v_0
```

```math
\beta_1
=
a
```

Therefore, if the acceleration is unknown, we can use the measured data to estimate it.

The regression problem becomes a parameter-estimation problem.

Instead of knowing

```math
v_0
```

and

```math
a
```

in advance, we estimate them from the observations.

The fitted model becomes

```math
\hat{v}(t)
=
\hat{\beta}_0+\hat{\beta}_1t
```

The estimated acceleration is therefore

```math
\hat{a}
=
\hat{\beta}_1
```

---

## 4. What Is the Goal?

The goal of regression is to construct a function that describes the systematic relationship between the input and output.

For the simplest regression model, we write

```math
\hat{y}
=
\beta_0+\beta_1x
```

The two parameters are

```math
\beta_0
```

and

```math
\beta_1
```

The first parameter controls the intercept.

The second parameter controls the rate at which the prediction changes with the input.

However, these parameters are initially unknown.

Therefore, the central problem is to determine which values of the parameters provide the best description of the observed data.

After fitting the model, we obtain estimated parameters.

```math
\hat{\beta}_0
```

and

```math
\hat{\beta}_1
```

The fitted prediction function is then

```math
\hat{y}
=
\hat{\beta}_0+\hat{\beta}_1x
```

The hat indicates that the quantity has been estimated from data.

---

## 5. Observation Versus Prediction

It is important to distinguish between an observed value and a predicted value.

The observed value is

```math
y_i
```

The model prediction for the same input is

```math
\hat{y}_i
```

These two quantities are generally not identical.

Their difference is called the residual.

```math
r_i
=
y_i-\hat{y}_i
```

A good model should generally produce predictions that are close to the observed values.

Therefore, the collection of residuals provides information about how well the model fits the data.

---

## 6. The Statistical Model

The basic statistical formulation is

```math
y_i
=
\beta_0+\beta_1x_i+\epsilon_i
```

This equation separates the observed quantity into a systematic component and an error component.

```math
y_i
=
\underbrace{\beta_0+\beta_1x_i}_{\text{systematic component}}
+
\underbrace{\epsilon_i}_{\text{noise}}
```

The systematic component represents the relationship that we are trying to learn.

The error component represents variation that is not captured by that relationship.

A common assumption is that the average error, conditional on the input, is zero.

```math
E[\epsilon_i\mid x_i]
=
0
```

Under this assumption, the conditional mean of the output is

```math
E[y_i\mid x_i]
=
\beta_0+\beta_1x_i
```

Thus, the regression function can be interpreted as the conditional mean of the output.

---

## 7. The Physical Interpretation

Return to the velocity example.

The model is

```math
v_i
=
v_0+at_i+\epsilon_i
```

Suppose the measured velocities fluctuate around the theoretical relationship.

One observation might lie above the theoretical prediction.

Another might lie below it.

The regression model does not require every observation to lie exactly on the same line.

Instead, it attempts to identify the underlying systematic relationship.

In this example, that relationship is

```math
v(t)
=
v_0+at
```

The slope has a direct physical interpretation.

```math
\frac{dv}{dt}
=
a
```

Therefore, estimating the slope of the velocity-time relationship allows us to estimate the acceleration.

This illustrates an important scientific use of regression:

```math
\boxed{
\text{observations}
\rightarrow
\text{mathematical model}
\rightarrow
\text{physical parameter}
}
```

The regression parameters need not merely be abstract mathematical quantities.

They can correspond directly to physical constants or measurable physical properties.

---

## 8. From a Straight Line to a General Model

The model

```math
\hat{y}
=
\beta_0+\beta_1x
```

is only the simplest possible regression model.

A more general model can contain several functions of the input.

For example,

```math
f(x)
=
\beta_0+\beta_1x+\beta_2x^2
```

or

```math
f(x)
=
\beta_0+\beta_1x+\beta_2x^2+\beta_3x^3
```

The important idea is that the unknown quantities are still the coefficients.

This leads to the broader mathematical framework of basis functions, which will be developed later.

For now, the central idea is:

```math
\boxed{
\text{Regression estimates unknown parameters from observed data.}
}
```

The parameters determine the mathematical relationship between the inputs and outputs.

The observations provide the information needed to estimate those parameters.

---

## 9. The Core Problem

We therefore begin with

```math
\mathcal{D}
=
\{(x_i,y_i)\}_{i=1}^{N}
```

and assume a model of the form

```math
y_i
=
f(x_i;\boldsymbol{\beta})
+
\epsilon_i
```

The unknown parameter vector is

```math
\boldsymbol{\beta}
```

Our objective is to determine an estimate of that vector.

```math
\hat{\boldsymbol{\beta}}
```

Once the parameters have been estimated, we obtain a fitted function.

```math
\hat{f}(x)
=
f(x;\hat{\boldsymbol{\beta}})
```

The entire theory of linear regression is fundamentally concerned with how to obtain this parameter estimate, why the resulting solution has desirable mathematical properties, and how the fitted function should be interpreted.

The next step is to understand precisely what **linear** means in linear regression.
## 10. Basis Functions

The previous sections introduced the idea that a regression model does not have to be a straight line in the input.

The more general idea is to construct the model from a collection of known functions of the input.

These functions are called **basis functions**.

We denote them by

```math
\phi_0(x),
\phi_1(x),
\ldots,
\phi_p(x)
```

The regression model is then written as

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

or, more compactly,

```math
f(x)
=
\sum_{j=0}^{p}
\beta_j\phi_j(x)
```

The basis functions are chosen in advance.

The coefficients are the unknown quantities that must be estimated from the data.

This is the central idea behind the basis-function representation.

---

## 11. The Simplest Basis

For ordinary linear regression, we choose two basis functions.

```math
\phi_0(x)
=
1
```

and

```math
\phi_1(x)
=
x
```

Substituting these into the general model gives

```math
f(x)
=
\beta_0\phi_0(x)
+
\beta_1\phi_1(x)
```

Therefore,

```math
f(x)
=
\beta_0+\beta_1x
```

So the ordinary straight-line model is simply a special case of the basis-function framework.

The intercept exists because of the constant basis function.

```math
\phi_0(x)
=
1
```

The slope term exists because of the basis function

```math
\phi_1(x)
=
x
```

Thus, even the familiar straight-line model can be understood in terms of basis functions.

---

## 12. Polynomial Basis Functions

Suppose we want a model that can represent curvature.

We can choose the basis functions

```math
\phi_0(x)
=
1
```

```math
\phi_1(x)
=
x
```

```math
\phi_2(x)
=
x^2
```

```math
\phi_3(x)
=
x^3
```

and so on.

For a polynomial of degree p, we choose

```math
\phi_j(x)
=
x^j
```

The resulting model is

```math
f(x)
=
\sum_{j=0}^{p}
\beta_jx^j
```

For example, a quadratic model is

```math
f(x)
=
\beta_0+\beta_1x+\beta_2x^2
```

A cubic model is

```math
f(x)
=
\beta_0+\beta_1x+\beta_2x^2+\beta_3x^3
```

These models are nonlinear functions of x.

However, they remain linear in the coefficients.

---

## 13. Why the Word "Linear" Still Applies

Consider

```math
f(x)
=
\beta_0+\beta_1x+\beta_2x^2
```

The function is clearly not a straight line in x.

However, the parameters occur as

```math
\beta_0
```

```math
\beta_1
```

and

```math
\beta_2
```

with no powers of the parameters and no products between unknown parameters.

Therefore, the model is linear in the parameters.

This gives the important distinction.

```math
\boxed{
\text{Linear regression means linear in the parameters,}
\newline
\text{not necessarily linear in the input.}
}
```

This distinction must be kept in mind throughout the derivation.

---

## 14. Other Possible Basis Functions

There is no requirement that the basis functions be powers of x.

For example, we can use trigonometric functions.

```math
\phi_0(x)
=
1
```

```math
\phi_1(x)
=
\sin(x)
```

```math
\phi_2(x)
=
\cos(x)
```

This gives

```math
f(x)
=
\beta_0
+
\beta_1\sin(x)
+
\beta_2\cos(x)
```

We could also use higher-frequency functions.

```math
\phi_k(x)
=
\sin(kx)
```

or

```math
\phi_k(x)
=
\cos(kx)
```

Another possibility is a Gaussian basis function.

```math
\phi_j(x)
=
\exp
\left(
-\frac{(x-\mu_j)^2}{2\sigma_j^2}
\right)
```

The resulting model may have a complicated nonlinear shape.

Nevertheless, if the coefficients multiply known basis functions linearly, the model remains linear in the parameters.

---

## 15. Basis Functions Define the Function Space

The choice of basis functions determines which functions the model can represent.

Suppose we use

```math
\phi_0(x)
=
1
```

and

```math
\phi_1(x)
=
x
```

Then the model can represent functions of the form

```math
f(x)
=
\beta_0+\beta_1x
```

The model therefore represents a two-dimensional family of functions.

If we add

```math
\phi_2(x)
=
x^2
```

then we obtain

```math
f(x)
=
\beta_0+\beta_1x+\beta_2x^2
```

The set of representable functions becomes larger.

Adding more basis functions gives the model more freedom.

Thus, the basis functions determine the space in which the regression function is searched for.

---

## 16. Parameter Vector and Basis Vector

The basis-function representation can be written using vectors.

Define the parameter vector as

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

Define the basis-function vector as

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

Then the model becomes

```math
f(x)
=
\boldsymbol{\beta}^{T}
\boldsymbol{\phi}(x)
```

Expanding the matrix multiplication gives

```math
f(x)
=
\begin{bmatrix}
\beta_0&
\beta_1&
\cdots&
\beta_p
\end{bmatrix}
\begin{bmatrix}
\phi_0(x)\\
\phi_1(x)\\
\vdots\\
\phi_p(x)
\end{bmatrix}
```

which is exactly

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

This vector form will allow us to move from individual observations to the full dataset.

---

## 17. Applying the Basis Functions to the Dataset

Suppose we have N observations.

```math
(x_1,y_1),
(x_2,y_2),
\ldots,
(x_N,y_N)
```

For the first input, the basis vector is

```math
\boldsymbol{\phi}(x_1)
=
\begin{bmatrix}
\phi_0(x_1)\\
\phi_1(x_1)\\
\vdots\\
\phi_p(x_1)
\end{bmatrix}
```

For the second input, it is

```math
\boldsymbol{\phi}(x_2)
=
\begin{bmatrix}
\phi_0(x_2)\\
\phi_1(x_2)\\
\vdots\\
\phi_p(x_2)
\end{bmatrix}
```

Continuing in this way gives one basis vector for every observation.

The prediction for the i-th observation is

```math
\hat{y}_i
=
\boldsymbol{\beta}^{T}
\boldsymbol{\phi}(x_i)
```

This is the key step that allows the entire regression problem to be expressed using matrix notation.

---

## 18. Constructing the Design Matrix

We now place the basis-function evaluations into a matrix.

The resulting matrix is called the **design matrix**.

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

Each row corresponds to one observation.

Each column corresponds to one basis function.

This interpretation is extremely important.

```math
\boxed{
\text{Rows}
=
\text{observations}
}
```

```math
\boxed{
\text{Columns}
=
\text{basis functions}
}
```

---

## 19. The Ordinary Linear Regression Design Matrix

For ordinary linear regression, the basis functions are

```math
\phi_0(x)
=
1
```

and

```math
\phi_1(x)
=
x
```

Therefore, the design matrix becomes

```math
X
=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_N
\end{bmatrix}
```

The first column contains the constant basis function.

The second column contains the input itself.

This is why the intercept appears in the matrix formulation.

It is not an arbitrary extra term.

It is the coefficient of the constant basis function.

---

## 20. Polynomial Regression Design Matrix

For a quadratic model, the basis functions are

```math
\phi_0(x)
=
1
```

```math
\phi_1(x)
=
x
```

```math
\phi_2(x)
=
x^2
```

Therefore,

```math
X
=
\begin{bmatrix}
1 & x_1 & x_1^2\\
1 & x_2 & x_2^2\\
\vdots & \vdots & \vdots\\
1 & x_N & x_N^2
\end{bmatrix}
```

For a cubic model,

```math
X
=
\begin{bmatrix}
1 & x_1 & x_1^2 & x_1^3\\
1 & x_2 & x_2^2 & x_2^3\\
\vdots & \vdots & \vdots & \vdots\\
1 & x_N & x_N^2 & x_N^3
\end{bmatrix}
```

The structure is exactly the same.

Only the selected basis functions have changed.

---

## 21. The Prediction Vector

Collect all observed outputs into a vector.

```math
\mathbf{y}
=
\begin{bmatrix}
y_1\\
y_2\\
\vdots\\
y_N
\end{bmatrix}
```

The corresponding predictions are

```math
\hat{\mathbf{y}}
=
\begin{bmatrix}
\hat{y}_1\\
\hat{y}_2\\
\vdots\\
\hat{y}_N
\end{bmatrix}
```

The parameter vector is

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

Using the design matrix, all predictions can be written simultaneously as

```math
\hat{\mathbf{y}}
=
X\boldsymbol{\beta}
```

This is one of the most important equations in linear regression.

It converts the regression model into a matrix equation.

---

## 22. The Full Statistical Model in Matrix Form

The individual observation model is

```math
y_i
=
\sum_{j=0}^{p}
\beta_j\phi_j(x_i)
+
\epsilon_i
```

Collecting all observations gives

```math
\mathbf{y}
=
X\boldsymbol{\beta}
+
\boldsymbol{\epsilon}
```

where the noise vector is

```math
\boldsymbol{\epsilon}
=
\begin{bmatrix}
\epsilon_1\\
\epsilon_2\\
\vdots\\
\epsilon_N
\end{bmatrix}
```

Thus, the entire regression problem can be represented by

```math
\boxed{
\mathbf{y}
=
X\boldsymbol{\beta}
+
\boldsymbol{\epsilon}
}
```

The observed data are represented by

```math
\mathbf{y}
```

The basis functions are represented through

```math
X
```

The unknown parameters are represented by

```math
\boldsymbol{\beta}
```

and the unexplained variation is represented by

```math
\boldsymbol{\epsilon}
```

The next step is to determine the parameter vector that makes the model fit the observed data as closely as possible.
