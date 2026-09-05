# Mathematical Foundations of Machine Learning

This document develops the mathematical foundations required to understand machine learning from a rigorous mathematical perspective.

The emphasis is on the underlying mathematics rather than on specific machine-learning architectures or named models. The main areas are linear algebra, multivariable calculus, probability, statistics, information theory, optimization, functional analysis, approximation theory, numerical mathematics, Fourier analysis, differential equations, and learning theory.

---

# 1. Mathematical Notation

## 1.1 Scalars

A scalar is a single numerical quantity.

Scalars are typically represented by lowercase letters:

$$x\in\mathbb{R}$$

A scalar may also belong to the complex numbers:

$$z\in\mathbb{C}$$

The set of real numbers is denoted by $\mathbb{R}$, while the set of complex numbers is denoted by $\mathbb{C}$.

---

## 1.2 Vectors

A vector is an ordered collection of scalars.

A vector in an $n$-dimensional real vector space can be written as

$$\mathbf{x}=\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}\in\mathbb{R}^n$$

The transpose of a column vector is

$$\mathbf{x}^{\mathsf{T}}=\begin{bmatrix}x_1&x_2&\cdots&x_n\end{bmatrix}$$

---

## 1.3 Matrices

A matrix is a rectangular arrangement of numbers.

An $m\times n$ matrix is written as

$$A\in\mathbb{R}^{m\times n}$$

with entries

$$A=\begin{bmatrix}a_{11}&a_{12}&\cdots&a_{1n}\\a_{21}&a_{22}&\cdots&a_{2n}\\\vdots&\vdots&\ddots&\vdots\\a_{m1}&a_{m2}&\cdots&a_{mn}\end{bmatrix}$$

Matrix multiplication is defined by

$$(AB)_{ij}=\sum_{k}A_{ik}B_{kj}$$

provided that the dimensions are compatible.

---

## 1.4 Tensors

A tensor generalizes scalars, vectors, and matrices to arbitrary numbers of dimensions.

A scalar is a zero-order tensor.

A vector is a first-order tensor.

A matrix is a second-order tensor.

A general tensor may be represented as

$$\mathcal{X}\in\mathbb{R}^{n_1\times n_2\times\cdots\times n_k}$$

---

## 1.5 Sets

A set is a collection of distinct mathematical objects.

For example,

$$A=\{1,2,3,4\}$$

The cardinality of a finite set $A$ is denoted by

$$|A|$$

The Cartesian product of two sets is

$$A\times B=\{(a,b):a\in A,\ b\in B\}$$

---

## 1.6 Functions

A function maps elements from one set to another.

$$f:X\rightarrow Y$$

The input space is called the domain and the output space is called the codomain.

For a vector-valued function,

$$f:\mathbb{R}^n\rightarrow\mathbb{R}^m$$

we may write

$$f(\mathbf{x})=\begin{bmatrix}f_1(\mathbf{x})\\\vdots\\f_m(\mathbf{x})\end{bmatrix}$$

---

# 2. Linear Algebra

Linear algebra provides the mathematical language for representing data, transformations, projections, and high-dimensional spaces.

## 2.1 Vector Spaces

A vector space $V$ over a field $\mathbb{F}$ is a set equipped with vector addition and scalar multiplication satisfying the vector-space axioms.

For $\mathbf{u},\mathbf{v}\in V$ and $a,b\in\mathbb{F}$,

$$a\mathbf{u}+b\mathbf{v}\in V$$

The space $\mathbb{R}^n$ is the most common finite-dimensional real vector space.

---

## 2.2 Subspaces

A subset $W\subseteq V$ is a subspace if it is closed under linear combinations.

Thus, for all $\mathbf{u},\mathbf{v}\in W$ and scalars $a,b$,

$$a\mathbf{u}+b\mathbf{v}\in W$$

A subspace must also contain the zero vector.

---

## 2.3 Linear Combinations

A linear combination of vectors $\mathbf{v}_1,\ldots,\mathbf{v}_k$ is

$$\mathbf{x}=\sum_{i=1}^{k}c_i\mathbf{v}_i$$

where $c_i$ are scalars.

Linear combinations are fundamental because many mathematical representations can be expressed as combinations of basis elements.

---

## 2.4 Span

The span of vectors $\mathbf{v}_1,\ldots,\mathbf{v}_k$ is

$$\operatorname{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}=\left\{\sum_{i=1}^{k}c_i\mathbf{v}_i:c_i\in\mathbb{R}\right\}$$

The span is the smallest subspace containing the given vectors.

---

## 2.5 Linear Independence

Vectors $\mathbf{v}_1,\ldots,\mathbf{v}_k$ are linearly independent if

$$\sum_{i=1}^{k}c_i\mathbf{v}_i=\mathbf{0}$$

implies

$$c_1=c_2=\cdots=c_k=0$$

Linear independence means that no vector in the collection can be represented as a linear combination of the others.

---

## 2.6 Basis and Dimension

A basis of a vector space is a linearly independent spanning set.

If

$$\mathcal{B}=\{\mathbf{v}_1,\ldots,\mathbf{v}_n\}$$

is a basis of $V$, then every vector $\mathbf{x}\in V$ has a unique representation

$$\mathbf{x}=\sum_{i=1}^{n}c_i\mathbf{v}_i$$

The number of basis vectors is the dimension:

$$\dim(V)=n$$

---

## 2.7 Inner Products

The standard Euclidean inner product is

$$\langle\mathbf{x},\mathbf{y}\rangle=\mathbf{x}^{\mathsf{T}}\mathbf{y}=\sum_{i=1}^{n}x_iy_i$$

Two vectors are orthogonal when

$$\langle\mathbf{x},\mathbf{y}\rangle=0$$

The inner product induces the Euclidean norm.

---

## 2.8 Norms

A norm measures the magnitude of a vector.

The $L^2$ norm is

$$\|\mathbf{x}\|_2=\sqrt{\sum_{i=1}^{n}x_i^2}$$

The $L^1$ norm is

$$\|\mathbf{x}\|_1=\sum_{i=1}^{n}|x_i|$$

The $L^\infty$ norm is

$$\|\mathbf{x}\|_\infty=\max_i|x_i|$$

More generally,

$$\|\mathbf{x}\|_p=\left(\sum_{i=1}^{n}|x_i|^p\right)^{1/p}$$

for $p\geq1$.

---

## 2.9 Euclidean Distance

The distance between two vectors is

$$d(\mathbf{x},\mathbf{y})=\|\mathbf{x}-\mathbf{y}\|_2$$

More generally, a metric $d$ satisfies non-negativity, identity of indiscernibles, symmetry, and the triangle inequality.

---

## 2.10 Orthogonal Projection

The projection of $\mathbf{x}$ onto a nonzero vector $\mathbf{u}$ is

$$\operatorname{proj}_{\mathbf{u}}\mathbf{x}=\frac{\mathbf{x}^{\mathsf{T}}\mathbf{u}}{\mathbf{u}^{\mathsf{T}}\mathbf{u}}\mathbf{u}$$

If $\mathbf{u}$ is normalized,

$$\operatorname{proj}_{\mathbf{u}}\mathbf{x}=(\mathbf{x}^{\mathsf{T}}\mathbf{u})\mathbf{u}$$

Projection is closely connected to least-squares approximation.

---

## 2.11 Linear Transformations

A function $T:V\rightarrow W$ is linear if

$$T(a\mathbf{x}+b\mathbf{y})=aT(\mathbf{x})+bT(\mathbf{y})$$

for all vectors $\mathbf{x},\mathbf{y}$ and scalars $a,b$.

A matrix represents a linear transformation between finite-dimensional vector spaces.

---

## 2.12 Rank

The rank of a matrix is the dimension of its column space.

$$\operatorname{rank}(A)=\dim(\operatorname{Col}(A))$$

It is also equal to the dimension of the row space.

The rank measures the number of linearly independent directions represented by the matrix.

---

## 2.13 Null Space

The null space of $A$ is

$$\operatorname{Null}(A)=\{\mathbf{x}:A\mathbf{x}=\mathbf{0}\}$$

The rank-nullity theorem states

$$\operatorname{rank}(A)+\operatorname{nullity}(A)=n$$

for $A\in\mathbb{R}^{m\times n}$.

---

## 2.14 Determinants

For a square matrix $A$, the determinant is denoted by

$$\det(A)$$

A matrix is invertible if and only if

$$\det(A)\neq0$$

The determinant measures the signed volume scaling produced by a linear transformation.

---

## 2.15 Eigenvalues and Eigenvectors

A nonzero vector $\mathbf{v}$ is an eigenvector of $A$ if

$$A\mathbf{v}=\lambda\mathbf{v}$$

where $\lambda$ is the corresponding eigenvalue.

The eigenvalues satisfy

$$\det(A-\lambda I)=0$$

Eigenvectors identify directions that remain unchanged in direction under the linear transformation.

---

## 2.16 Diagonalization

A matrix is diagonalizable if it can be written as

$$A=PDP^{-1}$$

where $D$ is diagonal and the columns of $P$ are eigenvectors of $A$.

For a symmetric matrix,

$$A=Q\Lambda Q^{\mathsf{T}}$$

where $Q$ is orthogonal and $\Lambda$ is diagonal.

---

## 2.17 Positive-Definite Matrices

A symmetric matrix $A$ is positive definite if

$$\mathbf{x}^{\mathsf{T}}A\mathbf{x}>0$$

for every nonzero $\mathbf{x}$.

It is positive semidefinite if

$$\mathbf{x}^{\mathsf{T}}A\mathbf{x}\geq0$$

Positive-definite matrices play an important role in quadratic optimization and covariance analysis.

---

## 2.18 Singular Value Decomposition

Every real matrix $A\in\mathbb{R}^{m\times n}$ can be decomposed as

$$A=U\Sigma V^{\mathsf{T}}$$

where $U$ and $V$ are orthogonal matrices and $\Sigma$ contains the singular values.

The singular values are nonnegative and are related to the eigenvalues of $A^{\mathsf{T}}A$:

$$\sigma_i=\sqrt{\lambda_i(A^{\mathsf{T}}A)}$$

SVD provides a fundamental decomposition for dimensionality reduction, least-squares problems, and numerical linear algebra.

---

# 3. Multivariable Calculus

Calculus provides the mathematical foundation for understanding how functions change and how objective functions can be optimized.

## 3.1 Functions of Several Variables

A scalar-valued function of $n$ variables is

$$f:\mathbb{R}^n\rightarrow\mathbb{R}$$

with

$$f(\mathbf{x})=f(x_1,\ldots,x_n)$$

---

## 3.2 Partial Derivatives

The partial derivative with respect to $x_i$ is

$$\frac{\partial f}{\partial x_i}=\lim_{h\rightarrow0}\frac{f(x_1,\ldots,x_i+h,\ldots,x_n)-f(\mathbf{x})}{h}$$

It measures the local change in $f$ when only $x_i$ changes.

---

## 3.3 Gradient

The gradient of a scalar function is

$$\nabla f(\mathbf{x})=\begin{bmatrix}\frac{\partial f}{\partial x_1}\\\vdots\\\frac{\partial f}{\partial x_n}\end{bmatrix}$$

The gradient points in the direction of steepest local increase.

The directional derivative in direction $\mathbf{v}$ is

$$D_{\mathbf{v}}f(\mathbf{x})=\nabla f(\mathbf{x})^{\mathsf{T}}\mathbf{v}$$

for a normalized direction $\mathbf{v}$.

---

## 3.4 Jacobian

For

$$f:\mathbb{R}^n\rightarrow\mathbb{R}^m$$

the Jacobian is the matrix

$$J_f(\mathbf{x})=\begin{bmatrix}\frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_n}\\\vdots&\ddots&\vdots\\\frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}\end{bmatrix}$$

The Jacobian describes the local linear transformation induced by $f$.

---

## 3.5 Hessian

For a scalar function $f:\mathbb{R}^n\rightarrow\mathbb{R}$, the Hessian is

$$H_f(\mathbf{x})=\nabla^2f(\mathbf{x})$$

with entries

$$[H_f(\mathbf{x})]_{ij}=\frac{\partial^2f}{\partial x_i\partial x_j}$$

The Hessian describes local curvature.

---

## 3.6 Taylor Expansion

The first-order Taylor approximation around $\mathbf{x}$ is

$$f(\mathbf{x}+\Delta\mathbf{x})\approx f(\mathbf{x})+\nabla f(\mathbf{x})^{\mathsf{T}}\Delta\mathbf{x}$$

The second-order approximation is

$$f(\mathbf{x}+\Delta\mathbf{x})\approx f(\mathbf{x})+\nabla f(\mathbf{x})^{\mathsf{T}}\Delta\mathbf{x}+\frac{1}{2}\Delta\mathbf{x}^{\mathsf{T}}H_f(\mathbf{x})\Delta\mathbf{x}$$

---

## 3.7 Chain Rule

For a composition

$$h(\mathbf{x})=f(g(\mathbf{x}))$$

the Jacobian satisfies

$$J_h(\mathbf{x})=J_f(g(\mathbf{x}))J_g(\mathbf{x})$$

For scalar functions, this becomes

$$\frac{dh}{dx}=\frac{df}{dg}\frac{dg}{dx}$$

The chain rule is fundamental to differentiation through compositions of functions.

---

## 3.8 Stationary Points

A stationary point of a differentiable scalar function satisfies

$$\nabla f(\mathbf{x}^*)=\mathbf{0}$$

A stationary point may be a local minimum, local maximum, or saddle point.

---

# 4. Probability Theory

Probability provides the mathematical framework for uncertainty and random phenomena.

## 4.1 Sample Spaces

A sample space $\Omega$ contains all possible outcomes of an experiment.

An event $A$ is a subset of $\Omega$:

$$A\subseteq\Omega$$

A probability measure satisfies

$$0\leq P(A)\leq1$$

and

$$P(\Omega)=1$$

---

## 4.2 Conditional Probability

The conditional probability of $A$ given $B$ is

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}$$

provided that $P(B)>0$.

---

## 4.3 Independence

Two events $A$ and $B$ are independent if

$$P(A\cap B)=P(A)P(B)$$

Equivalently,

$$P(A\mid B)=P(A)$$

when $P(B)>0$.

---

## 4.4 Random Variables

A random variable is a function

$$X:\Omega\rightarrow\mathbb{R}$$

that assigns a numerical value to each outcome.

Random variables may be discrete or continuous.

---

## 4.5 Probability Mass Function

For a discrete random variable,

$$p_X(x)=P(X=x)$$

and

$$\sum_xp_X(x)=1$$

---

## 4.6 Probability Density Function

For a continuous random variable, a probability density function $p_X(x)$ satisfies

$$P(a\leq X\leq b)=\int_a^b p_X(x)\,dx$$

and

$$\int_{-\infty}^{\infty}p_X(x)\,dx=1$$

---

## 4.7 Cumulative Distribution Function

The cumulative distribution function is

$$F_X(x)=P(X\leq x)$$

For a continuous random variable,

$$F_X(x)=\int_{-\infty}^{x}p_X(t)\,dt$$

---

## 4.8 Joint Distributions

For two random variables $X$ and $Y$, the joint density is

$$p_{X,Y}(x,y)$$

The marginal distribution is obtained by integration:

$$p_X(x)=\int p_{X,Y}(x,y)\,dy$$

---

## 4.9 Conditional Distributions

The conditional distribution satisfies

$$p_{X\mid Y}(x\mid y)=\frac{p_{X,Y}(x,y)}{p_Y(y)}$$

---

## 4.10 Bayes' Theorem

Bayes' theorem follows from the definition of conditional probability:

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}$$

For random variables,

$$p(\theta\mid x)=\frac{p(x\mid\theta)p(\theta)}{p(x)}$$

where

$$p(x)=\int p(x\mid\theta)p(\theta)\,d\theta$$

---

## 4.11 Expectation

The expectation of a discrete random variable is

$$\mathbb{E}[X]=\sum_xxp_X(x)$$

For a continuous random variable,

$$\mathbb{E}[X]=\int_{-\infty}^{\infty}xp_X(x)\,dx$$

More generally,

$$\mathbb{E}[g(X)]=\int g(x)p_X(x)\,dx$$

---

## 4.12 Variance

The variance is

$$\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]$$

Equivalently,

$$\operatorname{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$$

The standard deviation is

$$\sigma_X=\sqrt{\operatorname{Var}(X)}$$

---

## 4.13 Covariance

The covariance between $X$ and $Y$ is

$$\operatorname{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$$

Equivalently,

$$\operatorname{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$$

---

## 4.14 Covariance Matrix

For a random vector $\mathbf{X}$,

$$\Sigma=\mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^{\mathsf{T}}]$$

where

$$\boldsymbol{\mu}=\mathbb{E}[\mathbf{X}]$$

The covariance matrix is positive semidefinite.

---

## 4.15 Law of Large Numbers

For independent identically distributed random variables with finite expectation,

$$\frac{1}{n}\sum_{i=1}^{n}X_i\rightarrow\mathbb{E}[X]$$

as $n\rightarrow\infty$ under the appropriate form of the law.

The law explains why empirical averages become representative of population quantities as the sample size grows.

---

## 4.16 Central Limit Theorem

Under standard conditions,

$$\frac{\sqrt{n}(\bar X-\mu)}{\sigma}\xrightarrow{d}\mathcal{N}(0,1)$$

as $n\rightarrow\infty$.

The theorem explains the emergence of Gaussian behavior in normalized sums of random variables.

---

# 5. Important Probability Distributions

## 5.1 Bernoulli Distribution

A Bernoulli random variable takes values in $\{0,1\}$.

$$P(X=x)=p^x(1-p)^{1-x}$$

Its expectation and variance are

$$\mathbb{E}[X]=p$$

and

$$\operatorname{Var}(X)=p(1-p)$$

---

## 5.2 Binomial Distribution

For $n$ independent Bernoulli trials,

$$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$$

with

$$\mathbb{E}[X]=np$$

and

$$\operatorname{Var}(X)=np(1-p)$$

---

## 5.3 Gaussian Distribution

The one-dimensional Gaussian density is

$$p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the mean and $\sigma^2$ is the variance.

---

## 5.4 Multivariate Gaussian Distribution

For $\mathbf{x}\in\mathbb{R}^d$,

$$p(\mathbf{x})=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^{\mathsf{T}}\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

where $\boldsymbol{\mu}$ is the mean vector and $\Sigma$ is the covariance matrix.

---

## 5.5 Uniform Distribution

A continuous uniform distribution on $[a,b]$ has density

$$p(x)=\frac{1}{b-a},\qquad a\leq x\leq b$$

---

## 5.6 Exponential Distribution

The exponential density is

$$p(x)=\lambda e^{-\lambda x},\qquad x\geq0$$

where $\lambda>0$.

---

## 5.7 Poisson Distribution

The Poisson distribution is

$$P(X=k)=\frac{\lambda^ke^{-\lambda}}{k!}$$

for $k=0,1,2,\ldots$.

Its expectation and variance are both $\lambda$.

---

# 6. Mathematical Statistics

## 6.1 Population and Sample

A population represents the complete distribution of interest.

A sample consists of observations drawn from the population:

$$X_1,\ldots,X_n$$

A common assumption is that the observations are independent and identically distributed:

$$X_i\overset{\mathrm{iid}}{\sim}P$$

---

## 6.2 Parameters and Statistics

A parameter describes a population distribution.

A statistic is a function of observed data:

$$T=T(X_1,\ldots,X_n)$$

An estimator is a statistic used to estimate an unknown parameter.

---

## 6.3 Bias

For an estimator $\hat\theta$ of $\theta$,

$$\operatorname{Bias}(\hat\theta)=\mathbb{E}[\hat\theta]-\theta$$

An estimator is unbiased when

$$\mathbb{E}[\hat\theta]=\theta$$

---

## 6.4 Mean Squared Error

The mean squared error is

$$\operatorname{MSE}(\hat\theta)=\mathbb{E}[(\hat\theta-\theta)^2]$$

It decomposes as

$$\operatorname{MSE}(\hat\theta)=\operatorname{Var}(\hat\theta)+\operatorname{Bias}(\hat\theta)^2$$

---

## 6.5 Consistency

An estimator $\hat\theta_n$ is consistent for $\theta$ if

$$\hat\theta_n\xrightarrow{P}\theta$$

as $n\rightarrow\infty$.

Consistency means that the estimator approaches the true parameter as the amount of data increases.

---

# 7. Estimation Theory

## 7.1 Likelihood

Given observations $x_1,\ldots,x_n$ and parameter $\theta$, the likelihood is

$$L(\theta)=p(x_1,\ldots,x_n\mid\theta)$$

For independent observations,

$$L(\theta)=\prod_{i=1}^{n}p(x_i\mid\theta)$$

---

## 7.2 Log-Likelihood

The log-likelihood is

$$\ell(\theta)=\log L(\theta)$$

For independent observations,

$$\ell(\theta)=\sum_{i=1}^{n}\log p(x_i\mid\theta)$$

Because the logarithm is monotonic, maximizing likelihood is equivalent to maximizing log-likelihood.

---

## 7.3 Maximum Likelihood Estimation

The maximum likelihood estimator is

$$\hat\theta_{\mathrm{MLE}}=\arg\max_\theta L(\theta)$$

or equivalently,

$$\hat\theta_{\mathrm{MLE}}=\arg\max_\theta\ell(\theta)$$

---

## 7.4 Maximum A Posteriori Estimation

Bayesian inference gives

$$p(\theta\mid x)\propto p(x\mid\theta)p(\theta)$$

The MAP estimator is

$$\hat\theta_{\mathrm{MAP}}=\arg\max_\theta p(\theta\mid x)$$

Therefore,

$$\hat\theta_{\mathrm{MAP}}=\arg\max_\theta\left[\log p(x\mid\theta)+\log p(\theta)\right]$$

---

# 8. Information Theory

Information theory quantifies uncertainty and the information carried by random variables.

## 8.1 Entropy

For a discrete random variable,

$$H(X)=-\sum_xp(x)\log p(x)$$

Entropy measures uncertainty in the distribution.

---

## 8.2 Joint Entropy

For two random variables,

$$H(X,Y)=-\sum_{x,y}p(x,y)\log p(x,y)$$

---

## 8.3 Conditional Entropy

The conditional entropy is

$$H(Y\mid X)=-\sum_{x,y}p(x,y)\log p(y\mid x)$$

It measures the remaining uncertainty in $Y$ after observing $X$.

---

## 8.4 Cross-Entropy

The cross-entropy between distributions $p$ and $q$ is

$$H(p,q)=-\sum_xp(x)\log q(x)$$

It measures the expected negative log-probability assigned by $q$ when the data follow $p$.

---

## 8.5 Kullback-Leibler Divergence

The KL divergence is

$$D_{\mathrm{KL}}(p\|q)=\sum_xp(x)\log\frac{p(x)}{q(x)}$$

For continuous distributions,

$$D_{\mathrm{KL}}(p\|q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx$$

KL divergence satisfies

$$D_{\mathrm{KL}}(p\|q)\geq0$$

but generally

$$D_{\mathrm{KL}}(p\|q)\neq D_{\mathrm{KL}}(q\|p)$$

---

## 8.6 Mutual Information

The mutual information between $X$ and $Y$ is

$$I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$

Equivalently,

$$I(X;Y)=H(X)-H(X\mid Y)$$

It measures statistical dependence between random variables.

---

# 9. Optimization

Machine learning frequently involves finding parameters that minimize or maximize an objective function.

## 9.1 Optimization Problem

A general unconstrained optimization problem is

$$\min_{\mathbf{x}\in\mathbb{R}^n}f(\mathbf{x})$$

The function $f$ is the objective function.

---

## 9.2 Local and Global Minima

A point $\mathbf{x}^*$ is a global minimum if

$$f(\mathbf{x}^*)\leq f(\mathbf{x})$$

for every $\mathbf{x}$ in the domain.

It is a local minimum if

$$f(\mathbf{x}^*)\leq f(\mathbf{x})$$

for all $\mathbf{x}$ in some neighborhood of $\mathbf{x}^*$.

---

## 9.3 First-Order Optimality

For an unconstrained differentiable problem, an interior local minimum satisfies

$$\nabla f(\mathbf{x}^*)=\mathbf{0}$$

This condition is necessary but not sufficient for a local minimum.

---

## 9.4 Second-Order Conditions

At a stationary point, if

$$\nabla^2f(\mathbf{x}^*)\succ0$$

then $\mathbf{x}^*$ is a strict local minimum.

If

$$\nabla^2f(\mathbf{x}^*)\prec0$$

then it is a strict local maximum.

If the Hessian is indefinite, the point is a saddle point.

---

## 9.5 Gradient Descent

Gradient descent updates parameters according to

$$\mathbf{x}_{k+1}=\mathbf{x}_k-\eta\nabla f(\mathbf{x}_k)$$

where $\eta>0$ is the step size.

The update moves opposite to the gradient because the negative gradient is the direction of steepest local decrease.

---

## 9.6 Stochastic Optimization

When the objective is an expectation,

$$F(\theta)=\mathbb{E}_{X}[f(\theta;X)]$$

it can be approximated using samples:

$$\nabla F(\theta)\approx\frac{1}{m}\sum_{i=1}^{m}\nabla_\theta f(\theta;X_i)$$

This produces a stochastic estimate of the gradient.

---

# 10. Convex Analysis

## 10.1 Convex Sets

A set $C$ is convex if for any $\mathbf{x},\mathbf{y}\in C$ and $\lambda\in[0,1]$,

$$\lambda\mathbf{x}+(1-\lambda)\mathbf{y}\in C$$

---

## 10.2 Convex Functions

A function $f$ is convex if

$$f(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})\leq\lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})$$

for all $\lambda\in[0,1]$.

---

## 10.3 First-Order Characterization

For a differentiable convex function,

$$f(\mathbf{y})\geq f(\mathbf{x})+\nabla f(\mathbf{x})^{\mathsf{T}}(\mathbf{y}-\mathbf{x})$$

This means the tangent plane lies below the graph of the function.

---

## 10.4 Second-Order Characterization

For a twice-differentiable function,

$$f\text{ is convex}\iff\nabla^2f(\mathbf{x})\succeq0$$

throughout the domain.

---

# 11. Constrained Optimization

## 11.1 Equality Constraints

A constrained optimization problem may be written as

$$\min_{\mathbf{x}}f(\mathbf{x})$$

subject to

$$g(\mathbf{x})=0$$

---

## 11.2 Lagrangian

The Lagrangian is

$$\mathcal{L}(\mathbf{x},\lambda)=f(\mathbf{x})+\lambda g(\mathbf{x})$$

For multiple equality constraints,

$$\mathcal{L}(\mathbf{x},\boldsymbol{\lambda})=f(\mathbf{x})+\sum_i\lambda_i g_i(\mathbf{x})$$

---

## 11.3 Lagrange Multiplier Conditions

At an appropriate constrained optimum,

$$\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x}^*,\lambda^*)=\mathbf{0}$$

and

$$g(\mathbf{x}^*)=0$$

---

## 11.4 Inequality Constraints

Consider

$$\min_{\mathbf{x}}f(\mathbf{x})$$

subject to

$$g_i(\mathbf{x})\leq0$$

and

$$h_j(\mathbf{x})=0$$

The associated Lagrangian is

$$\mathcal{L}(\mathbf{x},\boldsymbol{\lambda},\boldsymbol{\nu})=f(\mathbf{x})+\sum_i\lambda_i g_i(\mathbf{x})+\sum_j\nu_jh_j(\mathbf{x})$$

---

## 11.5 Karush-Kuhn-Tucker Conditions

Under appropriate regularity conditions, an optimum satisfies:

### Stationarity

$$\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x}^*,\boldsymbol{\lambda}^*,\boldsymbol{\nu}^*)=\mathbf{0}$$

### Primal feasibility

$$g_i(\mathbf{x}^*)\leq0$$

and

$$h_j(\mathbf{x}^*)=0$$

### Dual feasibility

$$\lambda_i^*\geq0$$

### Complementary slackness

$$\lambda_i^*g_i(\mathbf{x}^*)=0$$

---

# 12. Norms and Function Spaces

Machine learning can involve finite-dimensional vectors as well as functions.

## 12.1 Metric Spaces

A metric space is a set $X$ equipped with a distance function

$$d:X\times X\rightarrow\mathbb{R}$$

satisfying

$$d(x,y)\geq0$$

$$d(x,y)=0\iff x=y$$

$$d(x,y)=d(y,x)$$

and

$$d(x,z)\leq d(x,y)+d(y,z)$$

---

## 12.2 Normed Vector Spaces

A normed vector space is a vector space equipped with a norm

$$\|\cdot\|:V\rightarrow\mathbb{R}$$

satisfying

$$\|\alpha x\|=|\alpha|\|x\|$$

and

$$\|x+y\|\leq\|x\|+\|y\|$$

---

## 12.3 $L^p$ Spaces

For a measurable function $f$,

$$\|f\|_p=\left(\int |f(x)|^p\,dx\right)^{1/p}$$

for $1\leq p<\infty$.

The essential supremum defines

$$\|f\|_\infty=\operatorname*{ess\,sup}_x|f(x)|$$

---

## 12.4 Hilbert Spaces

A Hilbert space is a complete inner-product space.

For functions, an inner product may be defined as

$$\langle f,g\rangle=\int f(x)g(x)\,dx$$

The corresponding norm is

$$\|f\|=\sqrt{\langle f,f\rangle}$$

---

## 12.5 Banach Spaces

A Banach space is a complete normed vector space.

Completeness means that every Cauchy sequence converges to an element within the space.

---

# 13. Approximation Theory

## 13.1 Function Approximation

The general approximation problem is to find an approximating function $g$ such that

$$g\approx f$$

according to an appropriate measure of error.

A common objective is

$$\|f-g\|_2^2$$

---

## 13.2 Projection

Given a subspace $V$, the best approximation to $f$ in $V$ is

$$g^*=\arg\min_{g\in V}\|f-g\|^2$$

In an inner-product space, the projection error satisfies

$$\langle f-g^*,v\rangle=0$$

for every $v\in V$.

---

## 13.3 Basis Expansion

If $\{\phi_1,\ldots,\phi_n\}$ is a basis, an approximation may be written as

$$f_n(x)=\sum_{i=1}^{n}c_i\phi_i(x)$$

The coefficients $c_i$ determine the representation.

---

## 13.4 Approximation Error

The approximation error can be measured by

$$E=\|f-f_n\|$$

Different norms produce different notions of approximation quality.

---

# 14. Numerical Mathematics

## 14.1 Floating-Point Representation

Computers represent real numbers using finite-precision floating-point arithmetic.

A floating-point number has the general form

$$x=\pm m\times b^e$$

where $m$ is a finite-precision significand and $e$ is an exponent.

---

## 14.2 Absolute Error

If $x$ is the exact value and $\hat{x}$ is the numerical approximation,

$$E_{\mathrm{abs}}=|\hat{x}-x|$$

---

## 14.3 Relative Error

The relative error is

$$E_{\mathrm{rel}}=\frac{|\hat{x}-x|}{|x|}$$

when $x\neq0$.

---

## 14.4 Conditioning

The condition number measures sensitivity of a mathematical problem to perturbations in its input.

For a scalar function,

$$\kappa(x)=\left|\frac{x f'(x)}{f(x)}\right|$$

when the expression is defined.

A large condition number indicates that small input perturbations can produce large relative output changes.

---

## 14.5 Numerical Stability

An algorithm is numerically stable when computational errors do not become disproportionately large relative to the underlying mathematical problem.

Stability is distinct from conditioning:

- Conditioning concerns the mathematical problem.
- Stability concerns the numerical algorithm.

---

## 14.6 Numerical Differentiation

A forward finite difference is

$$f'(x)\approx\frac{f(x+h)-f(x)}{h}$$

A backward finite difference is

$$f'(x)\approx\frac{f(x)-f(x-h)}{h}$$

A centered finite difference is

$$f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}$$

---

## 14.7 Numerical Integration

The integral

$$\int_a^bf(x)\,dx$$

can be approximated numerically.

For example, the trapezoidal rule gives

$$\int_a^bf(x)\,dx\approx\frac{h}{2}\left[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\right]$$

---

# 15. Fourier Analysis

Fourier analysis provides a representation of functions in terms of oscillatory basis functions.

## 15.1 Fourier Series

A periodic function may be represented as

$$f(x)=\sum_{k=-\infty}^{\infty}c_ke^{ikx}$$

where

$$c_k=\frac{1}{2\pi}\int_{-\pi}^{\pi}f(x)e^{-ikx}\,dx$$

---

## 15.2 Fourier Transform

The Fourier transform is

$$\hat f(\omega)=\int_{-\infty}^{\infty}f(x)e^{-i\omega x}\,dx$$

The inverse transform is

$$f(x)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\hat f(\omega)e^{i\omega x}\,d\omega$$

---

## 15.3 Discrete Fourier Transform

For a sequence $x_0,\ldots,x_{N-1}$,

$$X_k=\sum_{n=0}^{N-1}x_ne^{-2\pi ikn/N}$$

The inverse transform is

$$x_n=\frac{1}{N}\sum_{k=0}^{N-1}X_ke^{2\pi ikn/N}$$

---

## 15.4 Convolution

The continuous convolution of $f$ and $g$ is

$$(f*g)(x)=\int_{-\infty}^{\infty}f(\tau)g(x-\tau)\,d\tau$$

The Fourier transform converts convolution into multiplication:

$$\mathcal{F}\{f*g\}=\hat f\,\hat g$$

---

## 15.5 Frequency-Domain Representation

A function can be analyzed either in its original domain or through its frequency components.

The Fourier representation provides information about how different frequencies contribute to a signal.

---

# 16. Differential Equations

## 16.1 Ordinary Differential Equations

An ordinary differential equation involves derivatives with respect to one independent variable.

A general first-order ODE is

$$\frac{dy}{dt}=f(t,y)$$

---

## 16.2 Initial-Value Problems

An initial-value problem specifies

$$\frac{dy}{dt}=f(t,y)$$

together with

$$y(t_0)=y_0$$

The solution is a function satisfying both conditions.

---

## 16.3 Partial Differential Equations

A partial differential equation contains derivatives with respect to multiple independent variables.

A general PDE may be written as

$$\mathcal{F}(u,\nabla u,\nabla^2u,\ldots)=0$$

where $u$ is an unknown function.

---

## 16.4 Boundary-Value Problems

A boundary-value problem specifies conditions on the boundary of a domain.

For example,

$$\mathcal{L}u=f$$

with

$$u|_{\partial\Omega}=g$$

---

## 16.5 Well-Posedness

A mathematical problem is well posed when:

1. A solution exists.
2. The solution is unique.
3. The solution depends continuously on the input data.

These conditions are fundamental for understanding the stability of mathematical problems.

---

# 17. Functional Analysis

Functional analysis extends linear algebra and analysis to infinite-dimensional spaces.

## 17.1 Functions as Vectors

Functions can themselves be treated as elements of vector spaces.

For example,

$$f,g\in L^2(\Omega)$$

and their sum is

$$f+g$$

while scalar multiplication is

$$\alpha f$$

---

## 17.2 Linear Operators

An operator maps one function space into another:

$$\mathcal{G}:X\rightarrow Y$$

An operator is linear if

$$\mathcal{G}(af+bg)=a\mathcal{G}(f)+b\mathcal{G}(g)$$

---

## 17.3 Operator Norm

The operator norm is

$$\|\mathcal{G}\|=\sup_{\|f\|_X\neq0}\frac{\|\mathcal{G}f\|_Y}{\|f\|_X}$$

Equivalently,

$$\|\mathcal{G}\|=\sup_{\|f\|_X=1}\|\mathcal{G}f\|_Y$$

when the norm is finite.

---

## 17.4 Bounded Operators

An operator is bounded if there exists $C>0$ such that

$$\|\mathcal{G}f\|_Y\leq C\|f\|_X$$

for every $f\in X$.

For linear operators between normed spaces, boundedness is equivalent to continuity.

---

## 17.5 Functionals

A functional maps a vector space to the scalar field:

$$F:X\rightarrow\mathbb{R}$$

A linear functional satisfies

$$F(af+bg)=aF(f)+bF(g)$$

---

## 17.6 Dual Spaces

The dual space $X^*$ consists of continuous linear functionals on $X$.

Thus,

$$X^*=\{F:X\rightarrow\mathbb{R}:F\text{ is continuous and linear}\}$$

---

## 17.7 Weak Convergence

A sequence $x_n$ converges weakly to $x$ if

$$F(x_n)\rightarrow F(x)$$

for every continuous linear functional $F\in X^*$.

Weak convergence is generally weaker than convergence in norm.

---

# 18. Mathematical Foundations of Learning

## 18.1 Data as Random Variables

A dataset can be viewed as observations from an underlying probability distribution.

Let

$$Z=(X,Y)\sim P$$

where $X$ represents an input and $Y$ represents an associated target.

A dataset of $n$ observations is

$$D=\{(X_i,Y_i)\}_{i=1}^{n}$$

---

## 18.2 Hypothesis Space

A hypothesis space is a collection of candidate functions:

$$\mathcal{H}=\{h:X\rightarrow Y\}$$

Learning can be interpreted mathematically as selecting a suitable function from $\mathcal{H}$.

---

## 18.3 Loss Functions

A loss function measures the discrepancy between a prediction and the target:

$$\ell(y,\hat y)$$

A squared loss is

$$\ell(y,\hat y)=(y-\hat y)^2$$

An absolute loss is

$$\ell(y,\hat y)=|y-\hat y|$$

---

## 18.4 Expected Risk

The expected risk of a hypothesis $h$ is

$$R(h)=\mathbb{E}_{(X,Y)\sim P}[\ell(Y,h(X))]$$

The expected risk measures performance with respect to the underlying data distribution.

---

## 18.5 Empirical Risk

Given a dataset $D$,

$$\hat R(h)=\frac{1}{n}\sum_{i=1}^{n}\ell(Y_i,h(X_i))$$

Empirical risk is the sample approximation to expected risk.

---

## 18.6 Empirical Risk Minimization

The empirical-risk minimization principle selects

$$\hat h=\arg\min_{h\in\mathcal{H}}\hat R(h)$$

The mathematical challenge is that minimizing empirical risk does not automatically guarantee minimum expected risk.

---

## 18.7 Generalization Error

The difference between expected and empirical risk can be expressed as

$$R(h)-\hat R(h)$$

A central objective of learning theory is to understand when this difference becomes small.

---

## 18.8 Approximation and Estimation Error

The total learning error can conceptually be decomposed into components associated with:

- Approximation error
- Estimation error
- Optimization error
- Noise or irreducible error

The exact decomposition depends on the mathematical setting and loss function.

---

# 19. Bias-Variance Decomposition

Consider a regression problem with target

$$Y=f(X)+\epsilon$$

where

$$\mathbb{E}[\epsilon\mid X]=0$$

For an estimator $\hat f(x)$, the expected squared prediction error can be decomposed into

$$\mathbb{E}[(Y-\hat f(x))^2]=\operatorname{Bias}[\hat f(x)]^2+\operatorname{Var}[\hat f(x)]+\operatorname{Var}(\epsilon)$$

The three terms represent systematic error, estimator variability, and irreducible noise respectively.

---

# 20. Concentration Inequalities

Concentration inequalities quantify the probability that a random quantity deviates from its expectation.

## 20.1 Markov's Inequality

For a nonnegative random variable $X$ and $a>0$,

$$P(X\geq a)\leq\frac{\mathbb{E}[X]}{a}$$

---

## 20.2 Chebyshev's Inequality

For a random variable with finite variance,

$$P(|X-\mu|\geq k\sigma)\leq\frac{1}{k^2}$$

---

## 20.3 Hoeffding's Inequality

For independent random variables $X_i\in[a_i,b_i]$,

$$P\left(\frac{1}{n}\sum_{i=1}^{n}X_i-\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^{n}X_i\right]\geq t\right)\leq\exp\left(-\frac{2n^2t^2}{\sum_{i=1}^{n}(b_i-a_i)^2}\right)$$

Concentration inequalities provide mathematical bounds on deviations between empirical and population quantities.

---

# 21. Measure-Theoretic Foundations

## 21.1 Sigma-Algebras

A sigma-algebra $\mathcal{F}$ over a set $\Omega$ is a collection of subsets satisfying:

$$\Omega\in\mathcal{F}$$

If $A\in\mathcal{F}$, then

$$A^c\in\mathcal{F}$$

and for a countable collection $A_1,A_2,\ldots$,

$$\bigcup_{i=1}^{\infty}A_i\in\mathcal{F}$$

---

## 21.2 Measures

A measure $\mu$ assigns a nonnegative quantity to measurable sets.

It satisfies

$$\mu(\emptyset)=0$$

and countable additivity:

$$\mu\left(\bigcup_{i=1}^{\infty}A_i\right)=\sum_{i=1}^{\infty}\mu(A_i)$$

for pairwise disjoint measurable sets.

---

## 21.3 Probability as a Measure

Probability theory can be formulated using a probability space

$$(\Omega,\mathcal{F},P)$$

where $P$ is a measure satisfying

$$P(\Omega)=1$$

---

## 21.4 Measurable Functions

A function

$$X:\Omega\rightarrow\mathbb{R}$$

is measurable if inverse images of measurable sets are measurable.

This provides the formal mathematical foundation for random variables.

---

## 21.5 Lebesgue Integration

The Lebesgue integral generalizes the ordinary Riemann integral and provides the mathematical basis for modern probability theory and $L^p$ spaces.

For an integrable function $f$,

$$\int_\Omega f\,d\mu$$

denotes its Lebesgue integral with respect to the measure $\mu$.

---

# 22. Connections Between the Mathematical Foundations

The mathematical foundations of machine learning are not independent subjects. They form an interconnected system.

## 22.1 Linear Algebra and Optimization

Quadratic objectives have the form

$$f(\mathbf{x})=\frac{1}{2}\mathbf{x}^{\mathsf{T}}A\mathbf{x}-\mathbf{b}^{\mathsf{T}}\mathbf{x}$$

Their gradient is

$$\nabla f(\mathbf{x})=A\mathbf{x}-\mathbf{b}$$

and their Hessian is

$$\nabla^2f(\mathbf{x})=A$$

Thus, optimization of quadratic functions directly depends on matrix properties such as positive definiteness and eigenvalues.

---

## 22.2 Probability and Statistics

Statistics uses probability to describe uncertainty in data and estimators.

The likelihood

$$p(D\mid\theta)$$

describes how probable the observed data are for a parameter value, while the posterior

$$p(\theta\mid D)$$

combines the likelihood with prior information.

---

## 22.3 Calculus and Optimization

Optimization relies on derivatives.

The first-order condition

$$\nabla f(\mathbf{x}^*)=0$$

identifies stationary points, while the Hessian determines local curvature.

---

## 22.4 Probability and Information Theory

Entropy measures uncertainty:

$$H(X)=-\mathbb{E}[\log p(X)]$$

KL divergence compares probability distributions:

$$D_{\mathrm{KL}}(p\|q)=\mathbb{E}_p\left[\log\frac{p(X)}{q(X)}\right]$$

These quantities connect probability distributions with quantitative measures of information.

---

## 22.5 Functional Analysis and Approximation

When the objects being approximated are functions rather than finite-dimensional vectors, the relevant mathematical setting becomes a function space.

The approximation problem becomes

$$\min_{g\in V}\|f-g\|$$

where $V$ may be a finite- or infinite-dimensional subspace.

---

## 22.6 Fourier Analysis and Function Representation

Fourier analysis represents functions using frequency components:

$$f(x)=\sum_kc_k\phi_k(x)$$

This provides an alternative basis for representing and analyzing functions.

---

## 22.7 Numerical Mathematics and Computational Learning

Continuous mathematical objects must often be discretized before numerical computation.

For example, a function

$$f:[0,T]\rightarrow\mathbb{R}$$

may be represented at discrete points

$$t_0,t_1,\ldots,t_N$$

producing the vector

$$\mathbf{f}=[f(t_0),f(t_1),\ldots,f(t_N)]^{\mathsf{T}}$$

Thus, numerical mathematics provides the bridge between continuous mathematics and finite computational representations.

---

# 23. Core Mathematical Perspective

The mathematical foundations can be viewed as a hierarchy:

```text
Set Theory and Mathematical Logic
                ↓
       Linear Algebra
                ↓
          Multivariable Calculus
                ↓
        Probability Theory
                ↓
      Mathematical Statistics
                ↓
       Information Theory
                ↓
          Optimization
                ↓
       Approximation Theory
                ↓
       Numerical Mathematics
                ↓
      Function Spaces
                ↓
      Functional Analysis
                ↓
        Learning Theory
```

These areas collectively provide the mathematical framework required to formulate, analyze, and understand machine learning as a problem of approximation, inference, optimization, and generalization.

---

# 24. Summary of Essential Mathematical Objects

| Mathematical Object | Fundamental Role |
|---|---|
| Vector | Representation of elements in finite-dimensional spaces |
| Matrix | Representation of linear transformations |
| Tensor | Generalization of multidimensional numerical arrays |
| Inner product | Geometry and orthogonality |
| Norm | Magnitude and distance |
| Eigenvalue | Spectral characterization |
| Singular value | Matrix scaling and decomposition |
| Gradient | First-order variation |
| Jacobian | Local linearization of vector-valued functions |
| Hessian | Local curvature |
| Probability distribution | Mathematical representation of uncertainty |
| Expectation | Population average |
| Variance | Measure of dispersion |
| Covariance | Dependence between variables |
| Entropy | Quantification of uncertainty |
| KL divergence | Comparison of probability distributions |
| Objective function | Quantity to be optimized |
| Convex function | Function with favorable global optimization structure |
| Normed space | Space equipped with a notion of magnitude |
| Hilbert space | Complete inner-product space |
| Operator | Mapping between function spaces |
| Fourier transform | Frequency-domain representation |
| Measure | Generalized notion of size |
| Risk | Expected loss under a probability distribution |

---

# 25. Final Perspective

Machine learning can be viewed mathematically as the study of constructing functions that infer or approximate relationships from finite observations.

At its foundation are several interconnected mathematical ideas:

$$\boxed{\text{Representation}+\text{Probability}+\text{Calculus}+\text{Optimization}+\text{Approximation}+\text{Generalization}}$$

Linear algebra provides the language for finite-dimensional representation.

Calculus describes variation and sensitivity.

Probability provides a framework for uncertainty.

Statistics provides methods for inference from finite samples.

Information theory quantifies uncertainty and distributional differences.

Optimization provides methods for selecting parameters or functions according to an objective.

Approximation theory studies how accurately mathematical objects can be represented.

Functional analysis extends these ideas to infinite-dimensional spaces and operators.

Numerical mathematics provides the connection between continuous mathematical problems and finite computational procedures.

Learning theory studies why and when a function inferred from finite data can generalize beyond the observations used to construct it.
