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
