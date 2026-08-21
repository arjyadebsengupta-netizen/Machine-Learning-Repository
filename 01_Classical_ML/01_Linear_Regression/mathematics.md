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
