# Machine Learning for Scientific Applications

## 1. Why Machine Learning?

Machine Learning (ML) provides a framework for learning patterns, relationships, and representations from data rather than explicitly specifying every rule governing a system.

For physics and chemistry, this does **not** mean replacing physical theory. Instead, ML can be used to:

- Approximate expensive calculations
- Discover patterns in experimental data
- Construct surrogate models
- Solve inverse problems
- Accelerate simulations
- Predict molecular and material properties
- Identify phases and transitions
- Learn complex dynamical systems

The central idea is:

```math
\boxed{
\text{Data}
\rightarrow
\text{Model}
\rightarrow
\text{Prediction}
}
```

But understanding ML properly requires going further:

```math
\boxed{
\text{Data}
\rightarrow
\text{Model}
\rightarrow
\text{Loss}
\rightarrow
\text{Optimization}
\rightarrow
\text{Generalization}
}
```

---

## 2. Traditional Scientific Modelling vs Machine Learning

In traditional modelling, we generally begin with a physical or chemical theory.

For example,

```math
F = ma
```

or a differential equation

```math
\frac{\partial u}{\partial t}
=
\mathcal{F}(u)
```

We then solve the equations, analytically or numerically.

In Machine Learning, we instead attempt to learn a mapping from observations:

```math
f_\theta(x) \approx y
```

The parameters $\theta$ are determined from data by minimizing an objective function.

This distinction is important:

> **Machine Learning learns from examples; scientific modelling usually starts from governing principles.**

Modern scientific ML increasingly combines both approaches.

---

## 3. The Mathematical View

A machine-learning problem can often be formulated as finding a function from some hypothesis class:

```math
f_\theta : \mathcal{X} \rightarrow \mathcal{Y}
```

such that its predictions agree with observed data.

Given training data

```math
\mathcal{D}
=
\{(x_i,y_i)\}_{i=1}^{N}
```

we define a loss

```math
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell(f_\theta(x_i),y_i)
```

Training then becomes an optimization problem:

```math
\theta^*
=
\arg\min_\theta
\mathcal{L}(\theta)
```

This connects Machine Learning directly to:

- Linear algebra
- Probability
- Statistics
- Calculus
- Optimization
- Numerical methods
- Differential equations

---

## 4. Major Learning Paradigms

### Supervised Learning

We have input-output pairs:

```math
(x_i,y_i)
```

The objective is to learn:

```math
x \rightarrow y
```

Examples:

- Predicting molecular energy
- Predicting material properties
- Estimating physical parameters
- Classifying particles

### Unsupervised Learning

Only the observations are provided:

```math
\{x_1,x_2,\ldots,x_N\}
```

The objective is to discover structure in the data.

Examples:

- Clustering molecular configurations
- Finding phases of matter
- Dimensionality reduction
- Discovering latent representations

### Reinforcement Learning

An agent interacts with an environment:

```math
s_t
\rightarrow
a_t
\rightarrow
r_t
\rightarrow
s_{t+1}
```

The objective is to learn a policy that maximizes expected cumulative reward.

Applications include:

- Control of physical systems
- Molecular design
- Experimental optimization
- Autonomous scientific discovery

### Transfer Learning

A model trained on one task or dataset is adapted to a related task.

```math
\text{Source Task}
\rightarrow
\text{Learned Representation}
\rightarrow
\text{Target Task}
```

This is useful when the target dataset is small but a related, larger dataset is available.

Examples:

- Adapting pretrained models to scientific imaging
- Transferring molecular representations between related tasks
- Fine-tuning pretrained scientific models

### Zero-Shot Learning

A model performs a task or makes predictions for classes or situations that were **not explicitly represented during training**.

The model relies on additional information or a learned relationship connecting known and unseen cases.

Conceptually:

```math
\text{Training Classes}
+
\text{Semantic/Structural Information}
\rightarrow
\text{Unseen Classes}
```

Examples include:

- Classifying previously unseen categories
- Predicting properties for previously unseen molecular classes
- Applying pretrained models without task-specific examples

---

## 5. Where Physics and Chemistry Enter

Scientific problems often contain information that ordinary ML does not explicitly know.

For example, a model predicting a physical quantity may need to respect:

- Conservation laws
- Symmetries
- Dimensional consistency
- Boundary conditions
- Initial conditions
- Thermodynamic constraints
- Quantum-mechanical structure

This leads to **Scientific Machine Learning (SciML)**.

Instead of simply learning

```math
f_\theta(x) \approx y
```

we may construct a model that incorporates known scientific structure.

For example:

```math
\mathcal{L}
=
\mathcal{L}_{\text{data}}
+
\lambda
\mathcal{L}_{\text{physics}}
```

This idea eventually leads to Physics-Informed Neural Networks, physics-informed operators, equivariant networks, and other forms of scientific ML.

---

## 6. Examples in Physics

### Statistical Physics

Given configurations of a system,

```math
x_i=(s_1,s_2,\ldots,s_N)
```

ML can help classify phases or identify phase transitions.

### Computational Physics

A neural network can approximate an expensive numerical calculation:

```math
x \rightarrow f(x)
```

The trained model can then act as a surrogate.

### Astrophysics

ML can be used for:

- Object classification
- Parameter estimation
- Gravitational-wave analysis
- Cosmological inference
- Image analysis

### Differential Equations

Instead of solving a PDE independently for every parameter configuration, one can attempt to learn

```math
u(x;\mu)
```

or even the operator

```math
\mathcal{G}:u\rightarrow v
```

The latter idea leads to **Neural Operators**, which will be discussed later in this repository.

---

## 7. Examples in Chemistry

Chemistry provides naturally structured datasets involving molecules, atoms, reactions, and materials.

ML can be used for:

- Molecular property prediction
- Reaction prediction
- Drug discovery
- Materials discovery
- Molecular generation
- Potential-energy surfaces
- Spectroscopy
- Quantum-chemistry approximation

For example:

```math
\text{Molecular Structure}
\rightarrow
\text{Predicted Property}
```

The representation of the molecule itself becomes an important mathematical and ML problem.

---

## 8. Why Learn the Mathematics?

Modern ML libraries make it possible to train models with only a few lines of code.

However, this does not explain:

- What function is being learned?
- Why is the loss function appropriate?
- What is the optimization algorithm doing?
- Why does the model generalize?
- Why does training become unstable?
- What assumptions does the model make?

For scientific applications, these questions are especially important.

Therefore, this repository follows a **mathematics-first approach**.

The code is treated as an exercise in translating mathematical concepts into computation.

---

## 9. The Roadmap

The material progresses from relatively simple mathematical models toward modern scientific ML:

```math
\boxed{
\text{Classical ML}
\rightarrow
\text{Neural Networks}
\rightarrow
\text{CNNs}
\rightarrow
\text{RNNs}
\rightarrow
\text{Transformers}
\rightarrow
\text{Neural Operators}
}
```

Later, these ideas can be combined with scientific knowledge:

```math
\boxed{
\text{Machine Learning}
+
\text{Physical/Chemical Knowledge}
\rightarrow
\text{Scientific Machine Learning}
}
```

The goal is not simply to learn how to **use** ML libraries, but to understand the mathematical structures underlying the algorithms and eventually apply them to scientific problems.
