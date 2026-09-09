# JAX vs NumPy and PyTorch

## 1. What is JAX?

[JAX](https://docs.jax.dev/) is a Python library for high-performance numerical computing and machine learning.

Its array API is intentionally very similar to NumPy:

```python
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
```

However, JAX is designed around **composable program transformations**.

The most important transformations are:

* `jax.grad` — automatic differentiation
* `jax.jit` — just-in-time compilation
* `jax.vmap` — automatic vectorization
* `jax.jacfwd` / `jax.jacrev` — Jacobians
* `jax.hessian` — Hessians
* `jax.random` — explicit, functional pseudorandom number generation

This makes JAX particularly useful for scientific machine learning, differentiable simulations, optimization, PDEs, physics-informed neural networks, and large numerical workloads.

---

# 2. JAX vs NumPy

JAX is **not simply a faster version of NumPy**.

The basic array syntax is deliberately similar:

| Operation             | NumPy             | JAX                |
| --------------------- | ----------------- | ------------------ |
| Array                 | `np.array(...)`   | `jnp.array(...)`   |
| Mean                  | `np.mean(x)`      | `jnp.mean(x)`      |
| Matrix multiplication | `np.matmul(A, B)` | `jnp.matmul(A, B)` |
| Sine                  | `np.sin(x)`       | `jnp.sin(x)`       |
| Exponential           | `np.exp(x)`       | `jnp.exp(x)`       |

For example:

```python
import numpy as np
import jax.numpy as jnp

x_np = np.array([1.0, 2.0, 3.0])
x_jax = jnp.array([1.0, 2.0, 3.0])

print(np.mean(x_np))
print(jnp.mean(x_jax))
```

The important difference appears when we apply JAX transformations.

NumPy:

```python
def f(x):
    return x**3 + 2*x**2 + x
```

NumPy does not automatically construct a differentiable version of `f`.

JAX:

```python
import jax

def f(x):
    return x**3 + 2*x**2 + x

df = jax.grad(f)

print(df(2.0))
```

Output:

```text
21.0
```

Mathematically,

$$
f(x)=x^3+2x^2+x
$$

so

$$
f'(x)=3x^2+4x+1.
$$

At \(x=2\),

$$
f'(2)=21.
$$

The important point is that we wrote **the original function**, not its derivative.

JAX transformed the function into a function that computes its derivative.

---

# 3. JAX vs PyTorch

JAX and PyTorch can both perform:

* tensor/array computation
* automatic differentiation
* GPU/accelerator computation
* neural-network training
* scientific machine learning

But they have different design philosophies.

## PyTorch

PyTorch is primarily a **deep-learning framework** built around tensors, automatic differentiation, neural-network modules, optimizers, and a dynamic execution model.

Typical PyTorch code looks like:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**3 + 2*x**2 + x

y.backward()

print(x.grad)
```

The computation creates an autograd graph, and:

```python
y.backward()
```

propagates gradients through that graph.

---

## JAX

JAX separates numerical computation from transformations.

The function itself can remain an ordinary Python function:

```python
import jax
import jax.numpy as jnp

def f(x):
    return x**3 + 2*x**2 + x

df = jax.grad(f)

print(df(2.0))
```

JAX then allows us to compose transformations:

```python
df = jax.grad(f)
df_jit = jax.jit(df)
```

or:

```python
batched_df = jax.vmap(jax.grad(f))
```

or:

```python
fast_batched_df = jax.jit(
    jax.vmap(jax.grad(f))
)
```

This composability is one of the defining characteristics of JAX.

---

# 4. Main conceptual difference

A useful way to think about the difference is:

### PyTorch

```text
Tensor
   ↓
Operations
   ↓
Autograd graph
   ↓
Backward pass
```

### JAX

```text
Python function
      ↓
JAX transformation
      ↓
Transformed function
      ↓
Execution / compilation
```

For example:

```python
jax.grad(f)
```

means:

> Transform `f` into a function that computes its gradient.

Similarly:

```python
jax.jit(f)
```

means:

> Transform `f` into a compiled version.

And:

```python
jax.vmap(f)
```

means:

> Transform `f` so that it automatically operates over a batch.

These transformations can be composed.

---

# 5. JAX's main advantage: composable transformations

Consider a function:

```python
def loss(w, x, y):
    prediction = w * x
    return (prediction - y)**2
```

We can differentiate it:

```python
grad_loss = jax.grad(loss)
```

We can vectorize the gradient over samples:

```python
batched_grad = jax.vmap(
    grad_loss,
    in_axes=(None, 0, 0)
)
```

And compile the result:

```python
fast_batched_grad = jax.jit(batched_grad)
```

Conceptually:

```text
loss
 │
 ├── grad
 │
 ├── vmap
 │
 └── jit
```

The transformations are composable rather than being tied to one particular execution mechanism.

This becomes especially powerful in scientific machine learning.

---

# 6. Why JAX is useful for scientific machine learning

JAX is particularly attractive when mathematical computation is central to the model.

Important applications include:

* Physics-Informed Neural Networks (PINNs)
* Neural Operators
* differentiable numerical solvers
* PDE optimization
* inverse problems
* differentiable simulations
* scientific optimization
* uncertainty and sensitivity analysis
* reinforcement learning
* meta-learning
* large-scale vectorized computation

For example, a PINN may require:

$$
u(x),
\qquad
\frac{du}{dx},
\qquad
\frac{d^2u}{dx^2}.
$$

With JAX:

```python
u = lambda x: model(x)

du_dx = jax.grad(u)
d2u_dx2 = jax.grad(du_dx)
```

The second derivative can then be incorporated into a PDE residual.

This is particularly convenient for problems where derivatives of neural-network outputs are part of the mathematical objective.

---

# 7. JAX vs PyTorch: practical comparison

| Feature                         |                   NumPy |                                     PyTorch |                   JAX |
| ------------------------------- | ----------------------: | ------------------------------------------: | --------------------: |
| Array computation               |                       ✓ |                                           ✓ |                     ✓ |
| Automatic differentiation       |                       ✗ |                                           ✓ |                     ✓ |
| GPU/accelerator support         | Limited/basic ecosystem |                                           ✓ |                     ✓ |
| Neural-network framework        |                       ✗ |                                           ✓ | Via Flax/Equinox/etc. |
| JIT compilation                 |                       ✗ | Available through PyTorch compilation tools |          Core feature |
| Automatic vectorization         |                       ✗ |                     Limited/manual patterns |                `vmap` |
| Higher-order differentiation    |                       ✗ |                                           ✓ |                     ✓ |
| Functional programming emphasis |                     Low |                                    Moderate |                  High |
| Explicit functional RNG         |                       ✗ |                                           ✗ |                     ✓ |
| Scientific computing            |               Excellent |                                   Excellent |             Excellent |
| PDE/PINN workflows              |                Possible |                                 Very common |       Very convenient |
| Composable transformations      |                       ✗ |                     More framework-specific | Core design principle |

---

# 8. Apples-to-apples numerical comparison

The following example performs the **same mathematical operations** using NumPy, PyTorch, and JAX.

We calculate:

$$
y = \sin(x)^2 + e^x
$$

for

$$
x = [0,1,2,3].
$$

---

## NumPy

```python
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0])

y = np.sin(x)**2 + np.exp(x)

print(y)
```

NumPy performs the numerical computation efficiently, but it does not automatically provide derivatives of arbitrary Python functions.

---

## PyTorch

```python
import torch

x = torch.tensor(
    [0.0, 1.0, 2.0, 3.0],
    requires_grad=True
)

y = torch.sin(x)**2 + torch.exp(x)

print(y)
```

PyTorch can also differentiate the computation:

```python
loss = y.sum()

loss.backward()

print(x.grad)
```

Here PyTorch constructs an autograd computation graph and performs a backward pass.

---

## JAX

```python
import jax
import jax.numpy as jnp

x = jnp.array([0.0, 1.0, 2.0, 3.0])

def f(x):
    return jnp.sin(x)**2 + jnp.exp(x)

y = f(x)

print(y)
```

The same function can then be differentiated:

```python
df = jax.grad(lambda x: jnp.sum(f(x)))

print(df(x))
```

JAX therefore allows the numerical function to remain clean and separate from the differentiation transformation.

---

# 9. A more meaningful comparison: derivatives

Consider:

$$
f(x)=\sin(x^2)e^x.
$$

## NumPy

```python
import numpy as np

def f_numpy(x):
    return np.sin(x**2) * np.exp(x)

x = 1.0

print(f_numpy(x))
```

To obtain the derivative with NumPy, we would have to derive and implement it ourselves or use a separate numerical/symbolic differentiation approach.

---

## PyTorch

```python
import torch

def f_torch(x):
    return torch.sin(x**2) * torch.exp(x)

x = torch.tensor(1.0, requires_grad=True)

y = f_torch(x)

y.backward()

print(x.grad)
```

PyTorch calculates the derivative using its autograd system.

---

## JAX

```python
import jax
import jax.numpy as jnp

def f_jax(x):
    return jnp.sin(x**2) * jnp.exp(x)

df = jax.grad(f_jax)

x = 1.0

print(df(x))
```

The mathematical function remains unchanged, and differentiation is applied as a transformation.

---

# 10. Where JAX becomes substantially different

The real strength of JAX appears when its transformations are composed.

Consider:

$$
L(w,x,y)=(wx-y)^2.
$$

First define the loss for **one sample**:

```python
def loss(w, x, y):
    prediction = w * x
    return (prediction - y)**2
```

Differentiate it:

```python
grad_loss = jax.grad(loss)
```

Now suppose we have many samples:

```python
x = jnp.array([1.0, 2.0, 3.0, 4.0])
y = jnp.array([2.0, 4.0, 6.0, 8.0])

w = 1.5
```

We can automatically vectorize the per-sample gradient:

```python
batched_grad = jax.vmap(
    grad_loss,
    in_axes=(None, 0, 0)
)
```

Then compile the entire operation:

```python
batched_grad_jit = jax.jit(batched_grad)
```

Finally:

```python
print(batched_grad_jit(w, x, y))
```

The important sequence is:

```text
single-sample loss
       ↓
      grad
       ↓
per-sample gradient
       ↓
      vmap
       ↓
batched gradient
       ↓
      jit
       ↓
compiled batched gradient
```

This is a central JAX programming pattern.

---

# 11. JAX's functional programming model

JAX strongly encourages functions that behave like:

```python
output = f(input)
```

rather than relying heavily on hidden mutable state.

For example:

```python
def square(x):
    return x**2
```

This makes transformations such as:

```python
jax.grad(...)
jax.jit(...)
jax.vmap(...)
```

easier to apply and compose.

JAX's random-number generation follows the same philosophy.

Instead of relying on a global random state:

```python
key = jax.random.key(0)

key1, key2 = jax.random.split(key)

x = jax.random.normal(key1, (5,))
y = jax.random.normal(key2, (5,))
```

The random state is explicitly passed through the computation.

---

# 12. JAX is not a replacement for PyTorch

JAX and PyTorch overlap substantially, but they are optimized around somewhat different workflows.

PyTorch is an excellent choice when:

* building conventional deep-learning models
* using the PyTorch ecosystem
* working with standard computer-vision or NLP architectures
* using dynamic, imperative tensor programming
* taking advantage of mature PyTorch tooling

JAX is particularly attractive when:

* mathematical transformations are central
* automatic differentiation is heavily used
* higher-order derivatives are required
* vectorization is important
* compilation is important
* functional programming is desirable
* differentiable scientific computation is required

Neither is universally "better."

The choice depends on the problem.

---

# 13. The key takeaway

A useful summary is:

```text
NumPy
  ↓
Numerical array computing

PyTorch
  ↓
Tensor computing
+ automatic differentiation
+ deep learning ecosystem
+ GPU acceleration

JAX
  ↓
NumPy-like numerical computing
+ automatic differentiation
+ JIT compilation
+ automatic vectorization
+ functional transformations
+ accelerator-oriented execution
```

The defining idea of JAX is therefore not simply:

> "JAX is faster NumPy."

Nor is it simply:

> "JAX is another PyTorch."

A better description is:

> **JAX provides NumPy-like numerical programming together with composable transformations such as automatic differentiation, compilation, and automatic vectorization.**

For scientific machine learning, this combination is particularly powerful because the mathematical structure of the problem can often be expressed directly as differentiable functions.
