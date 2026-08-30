# Physics Theory + ML Simulations

This section of the repository is dedicated to the integration of **physics theory, mathematical formulation, computational simulation, and machine learning**.

The purpose of this section is to take physical theories and represent them in a **functional and computationally usable form**, followed by numerical simulation and, where appropriate, machine-learning-based modelling.

The emphasis is on establishing the physical and mathematical structure of a problem before applying machine learning.

---

## Objectives

This section aims to:

- Study the relevant **physical theories and mathematical foundations** behind each problem.
- Represent physical theories through their **fundamental equations, functions, operators, and mathematical relations**.
- Translate theoretical descriptions of physical systems into computationally usable formulations.
- Implement numerical simulations based on the resulting mathematical models.
- Generate or utilise physically meaningful data for machine-learning applications.
- Develop and investigate machine-learning models for physical systems.
- Compare ML-based predictions and simulations with theoretical and numerical results.
- Investigate how physical laws and mathematical structure can be incorporated into machine-learning models.
- Study the relationship between **physical theory, mathematical formulation, simulation, and data-driven modelling**.

---

## General Framework

The projects in this section will generally follow the progression:

    Physics Theory
          ↓
    Functional / Mathematical Formulation
          ↓
    Numerical Implementation
          ↓
    Physical Simulation
          ↓
    Dataset / Physical Data
          ↓
    Machine Learning Model
          ↓
    ML Simulation / Prediction
          ↓
    Comparison with Theory
          ↓
    Physical Analysis

The exact workflow may vary depending on the physical system and the objective of the individual project.

---

## Functional Representation of Physics

The physical theory associated with each problem will be represented in a **functional and mathematically explicit form** suitable for analytical study, numerical implementation, and machine-learning applications.

This may include:

- Governing equations
- Mathematical functions
- Differential equations
- Partial differential equations
- Operators
- Initial conditions
- Boundary conditions
- Conservation laws
- Constitutive relations
- State-space representations
- Probability distributions
- Physical parameters
- Constraints
- Analytical solutions
- Approximation schemes

The functional representation will provide the foundation for subsequent numerical simulations and machine-learning models.

For example, a physical system may be represented schematically as:

    Physical System
          ↓
    Governing Functional / Equation
          ↓
    Parameters + Initial / Boundary Conditions
          ↓
    Numerical Solver
          ↓
    Physical Solution
          ↓
    Data
          ↓
    Machine Learning Model

The mathematical formulation is therefore treated as the connection between the physical theory and its computational representation.

---

## Topics

The folder will be expanded progressively and may include topics from areas such as:

- Classical Mechanics
- Electromagnetism
- Thermodynamics
- Statistical Mechanics
- Quantum Mechanics
- Fluid Mechanics
- Computational Physics
- Astrophysics and Cosmology
- Condensed Matter Physics
- Dynamical Systems
- Partial Differential Equations
- Mathematical Physics
- Quantum Information
- Quantum Computing
- Other areas of theoretical and computational physics

---

## Machine Learning Methods

Depending on the structure of the physical problem, different machine-learning methods may be investigated, including:

- Linear and nonlinear regression
- Classification
- Clustering
- Neural Networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformers
- Graph Neural Networks (GNNs)
- Autoencoders
- Generative Models
- Reinforcement Learning
- Physics-Informed Neural Networks (PINNs)
- Neural Operators
- Physics-Informed Neural Operators
- Physics-aware machine learning
- Physics-guided machine learning
- Scientific machine learning
- Other data-driven modelling approaches

The choice of machine-learning method will depend on the mathematical structure and physical properties of the problem.

---

## Methodology

Each project will attempt to establish a connection between physical theory and machine learning through the following stages.

### 1. Physical Theory

The fundamental physical principles governing the system are identified and described.

This may include:

- Physical laws
- Fundamental principles
- Assumptions
- Approximations
- Physical parameters
- Relevant scales
- Conservation principles
- Symmetries

### 2. Functional / Mathematical Formulation

The physical theory is expressed mathematically in a functional or computationally usable form.

This may involve:

- Functions
- Differential equations
- Partial differential equations
- Integral equations
- Operators
- Algebraic relations
- Probability distributions
- Dynamical systems
- Variational formulations
- Constitutive equations

The formulation should provide the mathematical basis from which the physical system can be evaluated, simulated, or used to construct a machine-learning problem.

### 3. Analytical Analysis

Whenever possible, analytical solutions or theoretical predictions are obtained.

This stage may include:

- Exact solutions
- Approximate solutions
- Perturbative solutions
- Limiting cases
- Scaling relations
- Asymptotic behaviour
- Stability analysis

### 4. Numerical Implementation

The mathematical formulation is implemented computationally.

Depending on the problem, this may involve:

- Numerical integration
- Ordinary differential equation solvers
- Partial differential equation solvers
- Finite Difference Methods
- Finite Element Methods
- Spectral Methods
- Monte Carlo Methods
- Numerical optimisation
- Other appropriate numerical methods

### 5. Physical Simulation

The implemented mathematical model is used to simulate the physical system.

The simulation may produce:

- Time series
- Spatial fields
- Trajectories
- Probability distributions
- Physical observables
- Parameter-dependent solutions
- Other physically meaningful quantities

### 6. Dataset Formation

Where machine learning is applied, the simulation results or available physical data may be converted into a suitable dataset.

The origin of the data will be documented clearly.

Data may originate from:

- Analytical solutions
- Numerical simulations
- Experimental measurements
- Observational measurements
- Public scientific datasets
- Scientific databases

When data are generated numerically, the corresponding physical model and numerical procedure will be documented.

### 7. Machine Learning

A machine-learning model is then formulated according to the physical problem.

This includes defining:

- Inputs
- Outputs
- Features
- Targets
- Architecture
- Loss function
- Physical constraints
- Training procedure
- Evaluation metrics

### 8. Comparison and Validation

The ML results will be compared against the physical theory and, where available:

- Analytical solutions
- Numerical simulations
- Experimental measurements
- Observational data
- Established theoretical results

### 9. Physical Analysis

The final results will be interpreted from both the physics and machine-learning perspectives.

The objective is not only to measure predictive performance but also to determine whether the learned behaviour is consistent with the underlying physical system.

---

## Physics Before Machine Learning

A central principle of this section is:

> **The physical theory and mathematical formulation come before the machine-learning model.**

Machine learning is treated as a modelling, prediction, or simulation tool rather than as a substitute for understanding the underlying physical system.

Consequently, whenever possible, ML results will be examined with respect to:

- Governing equations
- Conservation laws
- Initial conditions
- Boundary conditions
- Symmetries
- Dimensional consistency
- Physical constraints
- Stability
- Parameter dependence
- Limiting behaviour
- Analytical solutions
- Numerical solutions
- Experimental or observational evidence

---

## Simulation and Data

The simulations in this section may serve different purposes.

They may be used to:

- Demonstrate a physical theory computationally.
- Investigate the behaviour of a physical system.
- Generate datasets for machine learning.
- Provide reference solutions for ML models.
- Test physics-informed architectures.
- Compare conventional numerical methods with ML-based methods.
- Investigate surrogate models for computationally expensive simulations.

The distinction between the **physical model**, **numerical solver**, **generated data**, and **machine-learning model** will be maintained wherever relevant.

---

## Reproducibility

Projects will be organised with reproducibility in mind.

Where appropriate, an individual project may contain:

    Project
    │
    ├── Theory
    │   └── Physical theory and derivations
    │
    ├── Functional_Formulation
    │   └── Mathematical / functional representation
    │
    ├── Analytical_Solution
    │   └── Exact or approximate solutions
    │
    ├── Numerical_Simulation
    │   └── Numerical implementation
    │
    ├── Data
    │   └── Experimental / observational / simulated data
    │
    ├── Machine_Learning
    │   └── ML implementation
    │
    ├── Training
    │   └── Training configuration and procedures
    │
    ├── Evaluation
    │   └── Evaluation and validation
    │
    ├── Results
    │   └── Figures, tables, and analysis
    │
    └── README.md

The source code, mathematical formulation, data-generation procedures, model configurations, and relevant results will be documented as the section develops.

---

## Software and Computational Tools

The implementations may use scientific-computing and machine-learning tools such as:

- Python
- NumPy
- SciPy
- SymPy
- Matplotlib
- Pandas
- scikit-learn
- PyTorch
- Other specialised scientific-computing libraries where appropriate

The choice of computational tools will depend on the requirements of each physical problem.

---

## Research Direction

This section is intended to progressively develop from fundamental physical theory and computational simulation toward more advanced forms of scientific machine learning.

Possible directions include:

- Physics-Informed Machine Learning
- Physics-Aware Machine Learning
- Neural Differential Equations
- Neural Operators
- Scientific Machine Learning
- Data-Driven Discovery of Physical Laws
- Surrogate Modelling
- Reduced-Order Modelling
- ML-Based Simulation
- Hybrid Physics-ML Models
- Data Assimilation
- Physical System Identification
- Computational Physics with Machine Learning

The long-term objective is to investigate how machine learning can complement established physical and mathematical methods while maintaining physical consistency and reproducibility.

---

## Purpose

This section serves as a bridge between:

**Physics Theory → Functional / Mathematical Representation → Computational Simulation → Machine Learning**

The physical theory provides the foundation, the functional and mathematical formulation provides the computational representation, numerical simulation provides a reference for the physical behaviour, and machine learning is subsequently used for modelling, prediction, simulation, or scientific investigation.

This folder will be continuously expanded as new physical theories, mathematical formulations, computational simulations, datasets, and machine-learning approaches are studied and implemented.

