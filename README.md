# Machine Learning, Deep Learning, and Neural Operators

A comprehensive repository covering classical machine learning algorithms, deep learning architectures, and neural operators.
The best order for reading is:

<ol type="I">
  <li>Overview of Machine Learning</li>
  <li>Dataset_types_in_Machine_learning</li>
  <li>01_Classical_ML</li>
  <li>02_Deep_Learning</li>
  <li>03_Neural Operators</li>
</ol>

## Repository Structure

```text
.
├── 01_Classical_ML/
│   ├── 01_Linear_Regression/
│   ├── 02_Logistic_Regression/
│   ├── 03_LDA/
│   ├── 04_QDA/
│   ├── 05_KNN/
│   ├── 07_Decision_Trees/
│   ├── 08_Bagging/
│   ├── 09_Boosting/
│   ├── 10_Stacking/
│   ├── 11_Random_Forest/
│   ├── 12_SVM/
│   ├── 13_K_Means/
│   ├── 14_PCA/
│   └── 15_GMM_EM/
├── 02_Deep_Learning/
│   ├── 01_NN/
│   ├── 02_CNN/
│   ├── 03_RNN/
│   ├── 04_PINN/
│   ├── 05_Autoencoder/
│   └── 06_Transformer/
|   |___07_GNN/
└── 03_Neural_Operators/
    └── 00_Introduction to Neural Operators
    └── CATO
    └── DeepONet
    └── FNO
    └── Factformer
    └── GNO
    └── GNOT
    └── LinearNO
    └── MAGNO
    └── MPNOT
    └── PINO
    └── Transolver

```

---

## Overview

### Part 1: Classical Machine Learning (`01_Classical_ML`)

| Directory | Topic | Description |
| :--- | :--- | :--- |
| `01_Linear_Regression` | **Linear Regression** | Ordinary Least Squares, Ridge, Lasso, and closed-form normal equations. |
| `02_Logistic_Regression` | **Logistic Regression** | Binary and multinomial cross-entropy classification, link functions, and optimization. |
| `03_LDA` | **Linear Discriminant Analysis** | Homoscedastic Gaussian generative modeling and Fisher's linear discriminant. |
| `04_QDA` | **Quadratic Discriminant Analysis** | Heteroscedastic Gaussian decision hyper-quadrics and quadratic boundaries. |
| `05_KNN` | **K-Nearest Neighbors** | Non-parametric instance-based learning, metric spaces, and Voronoi partitioning. |
| `07_Decision_Trees` | **Decision Trees** | CART framework, Information Gain, Gini Impurity, and cost-complexity pruning. |
| `08_Bagging` | **Bagging** | Non-parametric bootstrap aggregating, variance decay bounds, and Out-of-Bag (OOB) scoring. |
| `09_Boosting` | **Boosting** | Forward stagewise additive modeling, AdaBoost, and Functional Gradient Descent. |
| `10_Stacking` | **Stacking** | Multi-tier meta-learning and Out-of-Fold (OOF) cross-validation prediction matrices. |
| `11_Random_Forest` | **Random Forests** | Random subspace sub-selection ($m_{\text{try}}$), tree de-correlation, and feature importance. |
| `12_SVM` | **Support Vector Machines** | Hard/Soft margin optimization, Lagrange duality, KKT conditions, and Kernel trick. |
| `13_K_Means` | **K-Means & K-Medoids** | Lloyd's algorithm, $K$-Means++ initialization, PAM medoid optimization, and vector quantization. |
| `14_PCA` | **Principal Component Analysis** | Maximum variance projections, spectral covariance eigen-decomposition, SVD, and Kernel PCA. |
| `15_GMM_EM` | **GMM & EM** | Soft clustering, latent variables, Expectation-Maximization (EM) algorithm, and Log-Sum-Exp stability. |

---

### Part 2: Deep Learning (`02_Deep_Learning`)

| Directory | Topic | Description |
| :--- | :--- | :--- |
| `01_NN` | **Neural Networks** | Multilayer Perceptrons (MLP), forward/backward propagation, activation functions, and gradient optimizers. |
| `02_CNN` | **Convolutional Neural Networks** | Spatial convolutions, pooling layers, feature map extraction, and translation invariance. |
| `03_RNN` | **Recurrent Neural Networks** | Recurrent hidden state dynamics, Backpropagation Through Time (BPTT), LSTM, and GRU units. |
| `04_PINN` | **Physics-Informed Neural Networks** | Solving differential equations by embedding PDE constraints and boundary conditions into loss functions. |
| `05_Autoencoder` | **Autoencoders** | Bottleneck feature compression, reconstruction loss, and latent space representations. |
| `06_Transformer` | **Transformers** | Self-attention mechanisms, Scaled Dot-Product Attention, Multi-Head Attention, and positional encodings. |
|`07_GNN` | **Graph Neural Networks** | Preliminary graph theory, graph construction and it's usage|
---

### Part 3: Neural Operators (`03_Neural_Operators`)

| Directory | Topic | Description |
| :--- | :--- | :--- |
| `00_Introduction to Neural Operators` | **Introduction to Neural Operators** | Foundations of operator learning, function spaces, operator mappings, discretization invariance, and applications to parametric PDEs. |
| `CATO` | **Charted Axial Transformer Operator** | Transformer-based neural operator using axial attention for learning mappings between function spaces. |
| `DeepONet` | **Deep Operator Network** | Branch-trunk neural architecture for learning nonlinear operators from input functions to output functions. |
| `FNO` | **Fourier Neural Operator** | Learns operators through Fourier-domain transformations for efficient PDE modeling. |
| `Factformer` | **FactFormer** | Factorized-attention Transformer architecture for efficient neural operator learning. |
| `GNO` | **Graph Neural Operator** | Graph-based neural operator for learning operator mappings on irregular domains and meshes. |
| `GNOT` | **General Neural Operator Transformer** | Transformer-based neural operator using attention mechanisms to learn complex operator mappings. |
| `LinearNO` | **Linear Attention Neural Operator** | Neural operator based on linear attention for efficient operator learning. |
| `MAGNO` | **Multiscale Attentional Graph Neural Operator** | Graph neural operator incorporating multiscale representations and attentional spatial interactions. |
| `MPNOT` | **Multi-particle Neural Operator Transformer** | Transformer-based neural operator designed for learning multi-particle system dynamics. |
| `PINO` | **Physics-Informed Neural Operator** | Neural operator that incorporates governing physical laws and PDE constraints into training. |
| `Transolver` | **Transolver: A Fast Transformer Solver for PDEs on General Geometries** | Transformer-based PDE solver designed to efficiently learn physical fields on general geometries. |

---

## Environment & Requirements

```bash
# Clone the repository
git clone <your-repository-url>
cd <your-repository-name>

# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy torch matplotlib neuraloperator

# Neural Operator imports
from neuralop.models import FNO
from neuralop.layers import SpectralConv
from neuralop.losses import LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
```
