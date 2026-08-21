# Test

The model learns a function \(f_\theta : \mathcal{X} \rightarrow \mathcal{Y}\).

The loss function is

\[
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell(f_\theta(x_i),y_i).
\]

Training solves

\[
\theta^*
=
\arg\min_\theta \mathcal{L}(\theta).
\]

