# Mathematical Foundations of AI & Deep Learning: Master Handbook

> A rigorous university-level reference course on the mathematical engines powering modern Machine Learning, Deep Neural Networks, Transformers, Generative Models, and Frontier Reasoning systems. Features first-principles formula derivations, geometric interpretations, and step-by-step solved numerical problem typologies across all topics.

---

## 🏛️ Pedagogical Framework (Applied to Every Topic)
Every topic is authored in an uncompromised 8-part pedagogical framework:
1. **Intuition & 101 Motivation** — Why does this mathematical object exist? What machine learning failure does it prevent?
2. **Rigorous Mathematical Formulation** — Formal definitions, theorems, vector/matrix spaces, and LaTeX equations.
3. **First-Principles Formula Derivations** — Complete step-by-step proofs from foundational axioms with zero skipped steps.
4. **Geometric & Physical Interpretation** — Subspace projections, manifolds, hyperplanes, force balances, and curvature.
5. **Real-World Mental Models** — Concrete physical analogies grounding abstract mathematics.
6. **"AI by Hand" Visual Grids** — Cell-by-cell matrix/tensor arithmetic walkthroughs with concrete numbers.
7. **Multiple Solved Numerical Problem Typologies** — Standard cases, boundary conditions, edge cases, and applied deep learning scenarios.
8. **Deep Learning Architectural Connection & Code Verification** — Real-world application in PyTorch/NumPy.

---

## 📚 Curriculum Roadmap & Master Index

| Module | Core Mathematical Domain | Chapter Count | Master Coverage |
| :--- | :--- | :---: | :--- |
| **Module 01** | [Linear Algebra for Machine Learning](./01_linear_algebra) | 9 Chapters | Vector Spaces, Norms, 4 Subspaces, Projections, SVD, PSD, Tensors |
| **Module 02** | [Multivariable Calculus & Matrix Calculus](./02_multivariable_calculus) | 7 Chapters | Taylor Series, Gradients, Jacobians, Hessians, Matrix Calculus, Autodiff |
| **Module 03** | [Probability Theory & Information Theory](./03_probability_theory) | 7 Chapters | Bayes' Rule, Random Variables, Distributions, Inequalities, Entropy, KL |
| **Module 04** | [Mathematical Statistics & Estimation](./04_mathematical_statistics) | 6 Chapters | Sampling, Estimator Properties, MLE, MAP, Bias-Variance, Monte Carlo |
| **Module 05** | [Optimization & Convex Analysis](./05_optimization) | 7 Chapters | Convexity, KKT Conditions, SGD, Momentum, AdamW, Natural Gradients |
| **Module 06** | [Deep Learning Foundations](./06_deep_learning_foundations) | 8 Chapters | Perceptron, Backpropagation, Loss Functions, Init Schemes, Normalization |
| **Module 07** | [Convolutional Networks & Computer Vision](./07_convolutional_networks) | 4 Chapters | Spatial Convolutions, Backprop through Convs, Modern ConvNets, ViTs |
| **Module 08** | [Recurrent Networks & Sequence Modeling](./08_recurrent_networks) | 4 Chapters | BPTT, Exploding/Vanishing Gradients, LSTM, GRU, Classical Attention |
| **Module 09** | [Transformers, LLMs & Alignment](./09_transformers_and_llms) | 8 Chapters | Scaled Dot-Product, RoPE, FlashAttention, Scaling Laws, RLHF, DPO |
| **Module 10** | [Generative Modeling](./10_generative_models) | 4 Chapters | VAEs & ELBO, GANs & WGAN, Diffusion Models (DDPM & Score SDEs) |
| **Module 11** | [Reinforcement Learning](./11_reinforcement_learning) | 28 Chapters | Bellman Equations, Policy Gradients, TRPO, PPO, SAC, MCTS, GRPO |
| **Module 12** | [Modern LLM Architectures](./12_modern_llm_architectures) | 12 Chapters | Tokenization, SwiGLU, GQA, MoE, LoRA/QLoRA, DiTs, Vision-Language |
| **Module 13** | [Frontier Reasoning & Inference-Time Compute](./13_reasoning_and_test_time_compute) | 9 Chapters | CoT, Test-Time Scaling, PRMs, Search (MCTS), DeepSeek-R1, Lean 4 |

---

# SECTION 1: LINEAR ALGEBRA FOR MACHINE LEARNING

## 1.1 Core Formula Derivations

### Derivation 1.1.1: The Normal Equations & Orthogonal Projection Matrix
In supervised learning, we fit a linear model $y = X w$. When $X \in \mathbb{R}^{n \times d}$ with $n > d$, the system $X w = y$ is overdetermined and has no exact solution because $y \notin C(X)$. We seek $\hat{w}$ minimizing the squared residual error:
$$E(w) = \|y - X w\|_2^2 = (y - X w)^T (y - X w)$$

**Step-by-step derivation:**
1. Expand the quadratic objective:
   $$E(w) = y^T y - y^T X w - w^T X^T y + w^T X^T X w$$
2. Since $y^T X w$ is a scalar, $y^T X w = (y^T X w)^T = w^T X^T y$. Combining terms:
   $$E(w) = y^T y - 2 w^T X^T y + w^T (X^T X) w$$
3. Take the matrix derivative with respect to vector $w$ using standard vector calculus identities ($\nabla_w (w^T a) = a$, $\nabla_w (w^T A w) = 2 A w$ for symmetric $A$):
   $$\nabla_w E(w) = -2 X^T y + 2 X^T X w$$
4. Set the gradient to zero to find the critical stationary point:
   $$-2 X^T y + 2 X^T X \hat{w} = \mathbf{0} \implies \mathbf{(X^T X) \hat{w} = X^T y}$$
   *(These are the classical **Normal Equations**).*
5. When the feature columns of $X$ are linearly independent, the Gram matrix $X^T X \in \mathbb{R}^{d \times d}$ is strictly non-singular and invertible:
   $$\mathbf{\hat{w} = (X^T X)^{-1} X^T y}$$
6. The orthogonal projection of $y$ onto the column space $C(X)$ is $\hat{y} = X \hat{w}$:
   $$\hat{y} = X (X^T X)^{-1} X^T y = P_X y$$
   where $\mathbf{P_X = X (X^T X)^{-1} X^T}$ is the **Orthogonal Projection Matrix**, satisfying idempotency ($P^2 = P$) and symmetry ($P^T = P$). $\blacksquare$

---

### Derivation 1.1.2: Singular Value Decomposition (SVD) from Symmetric Eigendecomposition
For any real matrix $A \in \mathbb{R}^{m \times n}$, we derive the factorization $A = U \Sigma V^T$.

**Step-by-step derivation:**
1. Form the symmetric, positive semi-definite matrix $S = A^T A \in \mathbb{R}^{n \times n}$.
   By the **Spectral Theorem**, every real symmetric matrix has an orthonormal basis of eigenvectors $v_1, v_2, \dots, v_n \in \mathbb{R}^n$ with real, non-negative eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$:
   $$(A^T A) v_i = \lambda_i v_i, \quad v_i^T v_j = \delta_{ij}$$
2. Define the **singular values** as $\sigma_i = \sqrt{\lambda_i}$ for $i = 1, \dots, r$, where $r = \text{rank}(A)$.
3. For each non-zero singular value $\sigma_i > 0$, construct the vector $u_i \in \mathbb{R}^m$:
   $$u_i = \frac{1}{\sigma_i} A v_i$$
4. Verify that the vectors $\{u_i\}_{i=1}^r$ form an orthonormal set in $\mathbb{R}^m$:
   $$u_i^T u_j = \left(\frac{1}{\sigma_i} A v_i\right)^T \left(\frac{1}{\sigma_j} A v_j\right) = \frac{1}{\sigma_i \sigma_j} v_i^T (A^T A v_j) = \frac{1}{\sigma_i \sigma_j} v_i^T (\lambda_j v_j) = \frac{\lambda_j}{\sigma_i \sigma_j} v_i^T v_j$$
   - When $i \ne j$, $v_i^T v_j = 0 \implies u_i^T u_j = 0$.
   - When $i = j$, $\frac{\lambda_i}{\sigma_i^2} v_i^T v_i = \frac{\sigma_i^2}{\sigma_i^2} (1) = 1$.
5. Extend $\{u_1, \dots, u_r\}$ to an orthonormal basis $\{u_1, \dots, u_m\}$ for $\mathbb{R}^m$ via Gram-Schmidt.
6. Assemble matrices $U = [u_1, \dots, u_m] \in \mathbb{R}^{m \times m}$, $V = [v_1, \dots, v_n] \in \mathbb{R}^{n \times n}$, and $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_r, 0, \dots, 0) \in \mathbb{R}^{m \times n}$.
7. Since $A v_i = \sigma_i u_i$, in matrix form:
   $$A V = U \Sigma \implies \mathbf{A = U \Sigma V^T} = \sum_{i=1}^r \sigma_i u_i v_i^T \quad \blacksquare$$

---

## 1.2 Solved Numerical Problem Typologies

### Problem Type 1: Least Squares Linear Regression via Normal Equations
**Problem:** Given a design matrix with $n=3$ samples and $d=2$ features (including bias column) and target labels $y$:
$$X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad y = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}$$
1. Compute the Gram matrix $X^T X$ and moment vector $X^T y$.
2. Solve for optimal parameter weights $\hat{w} = [w_0, w_1]^T$.
3. Compute the fitted vector $\hat{y}$ and residual error $\|y - \hat{y}\|_2^2$.

**Step-by-step Numerical Solution:**
1. Compute $X^T X$:
   $$X^T X = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 1+1+1 & 1+2+3 \\ 1+2+3 & 1+4+9 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}$$
2. Compute $X^T y$:
   $$X^T y = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} = \begin{bmatrix} 2+3+5 \\ 2(1) + 3(2) + 5(3) \end{bmatrix} = \begin{bmatrix} 10 \\ 2 + 6 + 15 \end{bmatrix} = \begin{bmatrix} 10 \\ 23 \end{bmatrix}$$
3. Compute $(X^T X)^{-1}$ using the $2 \times 2$ matrix inverse formula $\frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$:
   $$\det(X^T X) = (3)(14) - (6)(6) = 42 - 36 = 6$$
   $$(X^T X)^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} = \begin{bmatrix} 7/3 & -1 \\ -1 & 1/2 \end{bmatrix}$$
4. Solve for $\hat{w}$:
   $$\hat{w} = (X^T X)^{-1} (X^T y) = \begin{bmatrix} 7/3 & -1 \\ -1 & 1/2 \end{bmatrix} \begin{bmatrix} 10 \\ 23 \end{bmatrix} = \begin{bmatrix} \frac{70}{3} - 23 \\ -10 + \frac{23}{2} \end{bmatrix} = \begin{bmatrix} \frac{70 - 69}{3} \\ \frac{-20 + 23}{2} \end{bmatrix} = \begin{bmatrix} 1/3 \\ 3/2 \end{bmatrix} \approx \begin{bmatrix} 0.3333 \\ 1.5000 \end{bmatrix}$$
5. Compute fitted predictions $\hat{y} = X \hat{w}$:
   $$\hat{y} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1/3 \\ 3/2 \end{bmatrix} = \begin{bmatrix} 1/3 + 3/2 \\ 1/3 + 3 \\ 1/3 + 9/2 \end{bmatrix} = \begin{bmatrix} 11/6 \\ 20/6 \\ 29/6 \end{bmatrix} \approx \begin{bmatrix} 1.8333 \\ 3.3333 \\ 4.8333 \end{bmatrix}$$
6. Residual vector $e = y - \hat{y} = \begin{bmatrix} 2 - 11/6 \\ 3 - 20/6 \\ 5 - 29/6 \end{bmatrix} = \begin{bmatrix} 1/6 \\ -2/6 \\ 1/6 \end{bmatrix}$.
   Check orthogonality $X^T e$:
   $$X^T e = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1/6 \\ -2/6 \\ 1/6 \end{bmatrix} = \begin{bmatrix} \frac{1 - 2 + 1}{6} \\ \frac{1 - 4 + 3}{6} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \quad \checkmark$$
   Residual squared norm $\|e\|_2^2 = (1/6)^2 + (-2/6)^2 + (1/6)^2 = \frac{1 + 4 + 1}{36} = \frac{6}{36} = \mathbf{0.1667}$.

---

### Problem Type 2: Eigendecomposition & Principal Component Analysis (PCA)
**Problem:** Given a centered data covariance matrix $C \in \mathbb{R}^{2 \times 2}$:
$$C = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix}$$
1. Find all eigenvalues $\lambda_1, \lambda_2$ and normalized eigenvectors $v_1, v_2$.
2. Calculate the proportion of total variance captured by the 1st principal component.

**Step-by-step Numerical Solution:**
1. Characteristic equation $\det(C - \lambda I) = 0$:
   $$\det\begin{bmatrix} 4 - \lambda & 2 \\ 2 & 1 - \lambda \end{bmatrix} = (4 - \lambda)(1 - \lambda) - 4 = \lambda^2 - 5\lambda + 4 - 4 = \lambda^2 - 5\lambda = 0$$
   $$\lambda(\lambda - 5) = 0 \implies \mathbf{\lambda_1 = 5, \quad \lambda_2 = 0}$$
2. Find eigenvector $v_1$ for $\lambda_1 = 5$:
   $$(C - 5I) v_1 = \begin{bmatrix} 4-5 & 2 \\ 2 & 1-5 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} -1 & 2 \\ 2 & -4 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
   $$-x + 2y = 0 \implies x = 2y \implies v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$
   Normalize to unit length $\|v_1\|_2 = \sqrt{2^2 + 1^2} = \sqrt{5}$:
   $$\mathbf{v_1 = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 \\ 1 \end{bmatrix} \approx \begin{bmatrix} 0.8944 \\ 0.4472 \end{bmatrix}}$$
3. Find eigenvector $v_2$ for $\lambda_2 = 0$:
   $$(C - 0I) v_2 = \begin{bmatrix} 4 & 2 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies 2x + y = 0 \implies y = -2x \implies v_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$
   Normalized: $\mathbf{v_2 = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ -2 \end{bmatrix}}$.
   Check orthogonality: $v_1^T v_2 = \frac{1}{5}(2(1) + 1(-2)) = 0 \quad \checkmark$.
4. Total variance = $\text{Tr}(C) = 4 + 1 = 5$.
   Proportion explained by PC 1: $\frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{5}{5 + 0} = \mathbf{1.0000}$ (100% of variance captured in 1D!).

---

### Problem Type 3: SVD & Low-Rank LoRA Truncation
**Problem:** Given a rank-2 weight matrix $W \in \mathbb{R}^{2 \times 2}$:
$$W = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}$$
1. Determine the exact SVD components $U, \Sigma, V^T$.
2. Compute the rank-1 Eckart-Young approximation $W_1$.
3. Compute the reconstruction error $\|W - W_1\|_F$.

**Step-by-step Numerical Solution:**
1. Compute $W^T W$:
   $$W^T W = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix} = \begin{bmatrix} 9 & 0 \\ 0 & 4 \end{bmatrix}$$
   Eigenvalues are $\lambda_1 = 9, \lambda_2 = 4 \implies$ Singular values $\sigma_1 = \sqrt{9} = \mathbf{3}, \,\, \sigma_2 = \sqrt{4} = \mathbf{2}$.
   Eigenvectors of $W^T W$: $v_1 = [1, 0]^T$, $v_2 = [0, 1]^T \implies V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$.
2. Compute left singular vectors $u_i = \frac{1}{\sigma_i} W v_i$:
   $$u_1 = \frac{1}{3} \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
   $$u_2 = \frac{1}{2} \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$
   $$U = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}, \quad V^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$
3. Rank-1 truncation:
   $$W_1 = \sigma_1 u_1 v_1^T = 3 \begin{bmatrix} 1 \\ 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 0 & 0 \end{bmatrix}$$
4. Frobenius error by Eckart-Young Theorem:
   $$\|W - W_1\|_F = \sigma_2 = \mathbf{2.0000}$$
   Direct verification: $W - W_1 = \begin{bmatrix} 0 & 0 \\ 0 & -2 \end{bmatrix} \implies \sqrt{0^2 + 0^2 + 0^2 + (-2)^2} = \sqrt{4} = 2 \quad \checkmark$.

---

# SECTION 2: MULTIVARIABLE & MATRIX CALCULUS

## 2.1 Core Formula Derivations

### Derivation 2.1.1: Multivariate Taylor Expansion from 1D Parameterization
Let $f: \mathbb{R}^n \to \mathbb{R}$ be $C^2$ smooth. We derive the 2nd-order expansion around $x_0$:
$$f(x_0 + h) = f(x_0) + \nabla f(x_0)^T h + \frac{1}{2} h^T H(x_0) h + \mathcal{O}(\|h\|^3)$$

**Step-by-step derivation:**
1. Define an auxiliary scalar function $g: [0, 1] \to \mathbb{R}$ along the straight line segment between $x_0$ and $x_0 + h$:
   $$g(t) = f(x_0 + t h)$$
   Notice $g(0) = f(x_0)$ and $g(1) = f(x_0 + h)$.
2. Apply the single-variable Taylor's theorem to $g(t)$ around $t=0$ evaluated at $t=1$:
   $$g(1) = g(0) + g'(0) \cdot (1) + \frac{1}{2} g''(0) \cdot (1)^2 + \mathcal{O}(1)$$
3. Compute $g'(t)$ using the multivariate chain rule:
   Let $z(t) = x_0 + t h$. Then $\frac{d z_i}{d t} = h_i$.
   $$g'(t) = \frac{d}{dt} f(z(t)) = \sum_{i=1}^n \frac{\partial f}{\partial z_i} \frac{d z_i}{dt} = \sum_{i=1}^n \frac{\partial f}{\partial z_i} h_i = \nabla f(z(t))^T h$$
   At $t=0$:
   $$g'(0) = \nabla f(x_0)^T h$$
4. Differentiate again to compute $g''(t)$:
   $$g''(t) = \frac{d}{dt} [g'(t)] = \frac{d}{dt} \left( \sum_{i=1}^n \frac{\partial f}{\partial z_i} h_i \right) = \sum_{i=1}^n \left( \frac{d}{dt} \frac{\partial f}{\partial z_i} \right) h_i$$
   Apply chain rule to each partial derivative:
   $$\frac{d}{dt} \left( \frac{\partial f}{\partial z_i} \right) = \sum_{j=1}^n \frac{\partial^2 f}{\partial z_j \partial z_i} \frac{d z_j}{dt} = \sum_{j=1}^n H_{ij}(z(t)) h_j$$
   Substituting into $g''(t)$:
   $$g''(t) = \sum_{i=1}^n \sum_{j=1}^n h_i H_{ij}(z(t)) h_j = h^T H(z(t)) h$$
   At $t=0$:
   $$g''(0) = h^T H(x_0) h$$
5. Substitute $g'(0)$ and $g''(0)$ back into the expansion of $g(1)$:
   $$\mathbf{f(x_0 + h) = f(x_0) + \nabla f(x_0)^T h + \frac{1}{2} h^T H(x_0) h + \mathcal{O}(\|h\|_2^3)} \quad \blacksquare$$

---

### Derivation 2.1.2: Matrix Backpropagation Gradients of a Dense Layer
Consider the standard deep learning linear transformation $Z = X W + \mathbf{1} b^T$ where $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, $b \in \mathbb{R}^{D_{\text{out}}}$, and $Z \in \mathbb{R}^{B \times D_{\text{out}}}$. Given upstream loss gradient $\frac{\partial \mathcal{L}}{\partial Z} \in \mathbb{R}^{B \times D_{\text{out}}}$, we derive parameter and activation gradients.

**Step-by-step derivation:**
1. Express total differential of scalar loss $\mathcal{L}$ using Frobenius inner products $\langle A, B \rangle = \text{Tr}(A^T B)$:
   $$d\mathcal{L} = \text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T dZ \right)$$
2. Compute the differential of the forward mapping $Z = X W + \mathbf{1} b^T$:
   $$dZ = (dX) W + X (dW) + \mathbf{1} (db)^T$$
3. Substitute $dZ$ into $d\mathcal{L}$ and use trace linearity $\text{Tr}(A + B) = \text{Tr}(A) + \text{Tr}(B)$:
   $$d\mathcal{L} = \text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T (dX) W \right) + \text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T X (dW) \right) + \text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T \mathbf{1} (db)^T \right)$$
4. Isolate $dW$ using the cyclic property of the trace $\text{Tr}(A B C) = \text{Tr}(C A B)$:
   $$\text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T X \, dW \right) = \text{Tr}\left( \left( X^T \frac{\partial \mathcal{L}}{\partial Z} \right)^T dW \right) \implies \mathbf{\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Z}}$$
5. Isolate $dX$:
   $$\text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T dX \, W \right) = \text{Tr}\left( W \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T dX \right) = \text{Tr}\left( \left( \frac{\partial \mathcal{L}}{\partial Z} W^T \right)^T dX \right) \implies \mathbf{\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Z} W^T}$$
6. Isolate $db$:
   $$\text{Tr}\left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T \mathbf{1} (db)^T \right) = \text{Tr}\left( (db)^T \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T \mathbf{1} \right) = db^T \left( \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T \mathbf{1} \right) \implies \mathbf{\frac{\partial \mathcal{L}}{\partial b} = \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^T \mathbf{1} = \sum_{i=1}^B \left(\frac{\partial \mathcal{L}}{\partial Z}\right)_{i, :}} \quad \blacksquare$$

---

## 2.2 Solved Numerical Problem Typologies

### Problem Type 1: Second-Order Multivariate Taylor Approximation
**Problem:** Given non-linear loss surface $f(x, y) = x^2 y + 2 x e^y$:
1. Compute the analytical gradient vector $\nabla f(x, y)$ and Hessian matrix $H(x, y)$ at point $(x_0, y_0) = (1.0, 0.0)$.
2. Construct the 2nd-order Taylor approximation $\hat{f}(1.0 + \Delta x, 0.0 + \Delta y)$.
3. Predict the loss at perturbation $(\Delta x, \Delta y) = (0.1, -0.2)$ and compare with exact function evaluation.

**Step-by-step Numerical Solution:**
1. Partial derivatives:
   - $f(1, 0) = (1)^2(0) + 2(1)e^0 = 0 + 2 = \mathbf{2.0000}$.
   - $\frac{\partial f}{\partial x} = 2xy + 2e^y \implies \frac{\partial f}{\partial x}(1, 0) = 2(1)(0) + 2e^0 = \mathbf{2.0000}$.
   - $\frac{\partial f}{\partial y} = x^2 + 2xe^y \implies \frac{\partial f}{\partial y}(1, 0) = 1^2 + 2(1)e^0 = 1 + 2 = \mathbf{3.0000}$.
   $$\nabla f(1, 0) = \begin{bmatrix} 2.0 \\ 3.0 \end{bmatrix}$$
2. Second partial derivatives for Hessian:
   - $\frac{\partial^2 f}{\partial x^2} = 2y \implies H_{11}(1, 0) = 2(0) = \mathbf{0.0}$.
   - $\frac{\partial^2 f}{\partial x \partial y} = 2x + 2e^y \implies H_{12}(1, 0) = 2(1) + 2(1) = \mathbf{4.0}$.
   - $\frac{\partial^2 f}{\partial y^2} = 2xe^y \implies H_{22}(1, 0) = 2(1)e^0 = \mathbf{2.0}$.
   $$H(1, 0) = \begin{bmatrix} 0.0 & 4.0 \\ 4.0 & 2.0 \end{bmatrix}$$
3. Evaluate 2nd-order Taylor formula for $h = [\Delta x, \Delta y]^T = [0.1, -0.2]^T$:
   - Linear term: $\nabla f^T h = 2(0.1) + 3(-0.2) = 0.2 - 0.6 = \mathbf{-0.4000}$.
   - Quadratic term $\frac{1}{2} h^T H h$:
     $$H h = \begin{bmatrix} 0 & 4 \\ 4 & 2 \end{bmatrix} \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix} = \begin{bmatrix} 0(0.1) + 4(-0.2) \\ 4(0.1) + 2(-0.2) \end{bmatrix} = \begin{bmatrix} -0.8 \\ 0.0 \end{bmatrix}$$
     $$\frac{1}{2} h^T (H h) = \frac{1}{2} [0.1(-0.8) + (-0.2)(0.0)] = \frac{1}{2}(-0.08) = \mathbf{-0.0400}$$
4. Total Taylor approximation:
   $$\hat{f}(1.1, -0.2) = 2.0000 - 0.4000 - 0.0400 = \mathbf{1.5600}$$
5. Exact evaluation:
   $$f(1.1, -0.2) = (1.1)^2(-0.2) + 2(1.1)e^{-0.2} = 1.21(-0.2) + 2.2(0.81873) = -0.2420 + 1.8012 = \mathbf{1.5592}$$
   Absolute error $|1.5600 - 1.5592| = \mathbf{0.0008}$ ($\le 0.05\%$ relative error).

---

### Problem Type 2: Softmax Jacobian Matrix Calculation
**Problem:** Given raw logit vector $z = [z_1, z_2, z_3]^T = [2.0, 1.0, 0.1]^T$:
1. Compute the softmax output probability vector $p = \text{softmax}(z)$.
2. Calculate the complete $3 \times 3$ Jacobian matrix $J = \frac{\partial p}{\partial z}$ where $J_{ij} = p_i(\delta_{ij} - p_j)$.

**Step-by-step Numerical Solution:**
1. Compute exponentials:
   - $e^{2.0} \approx 7.3891$
   - $e^{1.0} \approx 2.7183$
   - $e^{0.1} \approx 1.1052$
   - Denominator $\sum e^{z_k} = 7.3891 + 2.7183 + 1.1052 = 11.2126$
   Probabilities:
   $$p_1 = \frac{7.3891}{11.2126} \approx \mathbf{0.6590}, \quad p_2 = \frac{2.7183}{11.2126} \approx \mathbf{0.2424}, \quad p_3 = \frac{1.1052}{11.2126} \approx \mathbf{0.0986}$$
   Check: $0.6590 + 0.2424 + 0.0986 = 1.0000 \quad \checkmark$.
2. Compute diagonal elements $J_{ii} = p_i(1 - p_i)$:
   - $J_{11} = 0.6590(1 - 0.6590) = 0.6590(0.3410) = \mathbf{0.2247}$
   - $J_{22} = 0.2424(1 - 0.2424) = 0.2424(0.7576) = \mathbf{0.1836}$
   - $J_{33} = 0.0986(1 - 0.0986) = 0.0986(0.9014) = \mathbf{0.0889}$
3. Compute off-diagonal elements $J_{ij} = -p_i p_j$:
   - $J_{12} = J_{21} = -(0.6590)(0.2424) = \mathbf{-0.1597}$
   - $J_{13} = J_{31} = -(0.6590)(0.0986) = \mathbf{-0.0650}$
   - $J_{23} = J_{32} = -(0.2424)(0.0986) = \mathbf{-0.0239}$
4. Complete symmetric Jacobian matrix:
   $$\mathbf{J = \begin{bmatrix} 0.2247 & -0.1597 & -0.0650 \\ -0.1597 & 0.1836 & -0.0239 \\ -0.0650 & -0.0239 & 0.0889 \end{bmatrix}}$$
   Check row sums (must sum to zero since $\sum p_i = 1$):
   Row 1 sum: $0.2247 - 0.1597 - 0.0650 = 0.0000 \quad \checkmark$.

---

# SECTION 3: PROBABILITY THEORY & INFORMATION THEORY

## 3.1 Core Formula Derivations

### Derivation 3.1.1: Non-Negativity of KL Divergence ($D_{\text{KL}}(P \parallel Q) \ge 0$) via Jensen's Inequality
We prove Gibbs' Inequality: for any two discrete probability mass functions $P$ and $Q$:
$$D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} \ge 0$$
with equality if and only if $P(x) = Q(x)$ for all $x$.

**Step-by-step derivation:**
1. Express the negative KL divergence:
   $$-D_{\text{KL}}(P \parallel Q) = -\sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \sum_{x} P(x) \log \frac{Q(x)}{P(x)} = \mathbb{E}_{X \sim P}\left[ \log \frac{Q(X)}{P(X)} \right]$$
2. The logarithm function $f(t) = \log(t)$ is strictly concave on $(0, \infty)$ because $f''(t) = -1/t^2 < 0$.
3. By **Jensen's Inequality**, for any concave function $f$ and random variable $Y$:
   $$\mathbb{E}[f(Y)] \le f(\mathbb{E}[Y])$$
4. Set $Y = \frac{Q(X)}{P(X)}$ with expectation taken under $P$:
   $$-D_{\text{KL}}(P \parallel Q) = \mathbb{E}_P\left[ \log \frac{Q(X)}{P(X)} \right] \le \log \left( \mathbb{E}_P\left[ \frac{Q(X)}{P(X)} \right] \right)$$
5. Evaluate the inner expectation:
   $$\mathbb{E}_P\left[ \frac{Q(X)}{P(X)} \right] = \sum_{x \in \mathcal{X}} P(x) \frac{Q(x)}{P(x)} = \sum_{x \in \mathcal{X}} Q(x) = 1.0$$
6. Therefore:
   $$-D_{\text{KL}}(P \parallel Q) \le \log(1.0) = 0$$
7. Multiplying by $-1$ reverses the inequality:
   $$\mathbf{D_{\text{KL}}(P \parallel Q) \ge 0} \quad \blacksquare$$
   Because $\log$ is strictly concave, equality holds if and only if $Y = \frac{Q(X)}{P(X)}$ is constant almost surely $\implies P(x) = Q(x)$ everywhere.

---

### Derivation 3.1.2: Cross-Entropy Decomposition
For true distribution $P$ and predicted distribution $Q$:
$$H(P, Q) = -\sum_x P(x) \log Q(x)$$
$$H(P, Q) = -\sum_x P(x) \log \left( \frac{Q(x)}{P(x)} P(x) \right) = -\sum_x P(x) \log P(x) - \sum_x P(x) \log \frac{Q(x)}{P(x)}$$
$$H(P, Q) = H(P) + \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbf{H(P) + D_{\text{KL}}(P \parallel Q)} \quad \blacksquare$$
*Deep Learning Consequence:* Since the true labels $P$ are fixed during training ($H(P)$ is constant), minimizing cross-entropy loss is mathematically equivalent to minimizing the KL divergence between the true distribution and the neural network's predictions!

---

## 3.2 Solved Numerical Problem Typologies

### Problem Type 1: Bayesian Inference & Base-Rate Fallacy
**Problem:** In an AI fraud detection system:
- Prior probability of a fraudulent transaction: $P(\text{Fraud}) = 0.01$ (1%).
- True positive detection rate (sensitivity): $P(\text{Alert} \mid \text{Fraud}) = 0.95$ (95%).
- False positive rate: $P(\text{Alert} \mid \text{Legit}) = 0.05$ (5%).
Calculate the posterior probability that a transaction is actually fraudulent given an alert: $P(\text{Fraud} \mid \text{Alert})$.

**Step-by-step Numerical Solution:**
1. Compute total probability of an alert $P(\text{Alert})$ using the Law of Total Probability:
   $$P(\text{Alert}) = P(\text{Alert} \mid \text{Fraud}) P(\text{Fraud}) + P(\text{Alert} \mid \text{Legit}) P(\text{Legit})$$
   $$P(\text{Legit}) = 1 - 0.01 = 0.99$$
   $$P(\text{Alert}) = (0.95)(0.01) + (0.05)(0.99) = 0.0095 + 0.0495 = \mathbf{0.0590}$$
2. Apply Bayes' Theorem:
   $$P(\text{Fraud} \mid \text{Alert}) = \frac{P(\text{Alert} \mid \text{Fraud}) P(\text{Fraud})}{P(\text{Alert})} = \frac{0.0095}{0.0590} = \frac{95}{590} \approx \mathbf{0.1610}$$
   **Result:** Even with 95% detection accuracy, an alert has only a **16.1% probability** of being true fraud due to the rare base rate!

---

### Problem Type 2: Entropy, Cross-Entropy & KL Divergence Numerical Calculation
**Problem:** Given ground truth one-hot distribution $P = [1.0, 0.0, 0.0]$ and two competing model probability distributions:
- Model A: $Q_A = [0.70, 0.20, 0.10]$
- Model B: $Q_B = [0.40, 0.30, 0.30]$
(Use natural log $\ln$).
1. Compute Shannon entropy $H(P)$.
2. Compute Cross-Entropy $H(P, Q_A)$ and $H(P, Q_B)$.
3. Compute $D_{\text{KL}}(P \parallel Q_A)$ and $D_{\text{KL}}(P \parallel Q_B)$.

**Step-by-step Numerical Solution:**
1. Shannon entropy of pure one-hot state:
   $$H(P) = -(1.0 \ln 1.0 + 0 + 0) = -1.0(0) = \mathbf{0.0000 \text{ nats}}$$
2. Cross-Entropy:
   - Model A: $H(P, Q_A) = -\sum P_i \ln Q_{A, i} = -(1.0 \ln 0.70) = -(-0.35667) = \mathbf{0.3567 \text{ nats}}$.
   - Model B: $H(P, Q_B) = -(1.0 \ln 0.40) = -(-0.91629) = \mathbf{0.9163 \text{ nats}}$.
3. KL Divergence:
   Using $D_{\text{KL}}(P \parallel Q) = H(P, Q) - H(P)$:
   - Model A: $D_{\text{KL}}(P \parallel Q_A) = 0.3567 - 0 = \mathbf{0.3567 \text{ nats}}$.
   - Model B: $D_{\text{KL}}(P \parallel Q_B) = 0.9163 - 0 = \mathbf{0.9163 \text{ nats}}$.
   Model A is substantially closer to the true distribution.

---

# SECTION 4: MATHEMATICAL STATISTICS & ESTIMATION

## 4.1 Core Formula Derivations

### Derivation 4.1.1: Bessel's Correction for Sample Variance
Let $X_1, X_2, \dots, X_N$ be $i.i.d.$ random variables with mean $\mu$ and variance $\sigma^2$.
We prove that the naive plug-in estimator $S_n^2 = \frac{1}{N}\sum_{i=1}^N (X_i - \bar{X})^2$ is biased with expectation $\frac{N-1}{N}\sigma^2$, and $S^2 = \frac{1}{N-1}\sum_{i=1}^N (X_i - \bar{X})^2$ is unbiased.

**Step-by-step derivation:**
1. Expand the deviation term $(X_i - \bar{X})$ around the true population mean $\mu$:
   $$X_i - \bar{X} = (X_i - \mu) - (\bar{X} - \mu)$$
2. Square both sides:
   $$(X_i - \bar{X})^2 = (X_i - \mu)^2 - 2(X_i - \mu)(\bar{X} - \mu) + (\bar{X} - \mu)^2$$
3. Sum over all $i = 1, \dots, N$:
   $$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - 2 (\bar{X} - \mu) \sum_{i=1}^N (X_i - \mu) + N (\bar{X} - \mu)^2$$
4. Note that $\sum_{i=1}^N (X_i - \mu) = \sum X_i - N\mu = N\bar{X} - N\mu = N(\bar{X} - \mu)$.
   Substitute this into the middle term:
   $$\sum_{i=1}^N (X_i - \bar{X})^2 = \sum_{i=1}^N (X_i - \mu)^2 - 2 N (\bar{X} - \mu)^2 + N (\bar{X} - \mu)^2 = \sum_{i=1}^N (X_i - \mu)^2 - N (\bar{X} - \mu)^2$$
5. Take expectations on both sides:
   $$\mathbb{E}\left[ \sum_{i=1}^N (X_i - \bar{X})^2 \right] = \sum_{i=1}^N \mathbb{E}[(X_i - \mu)^2] - N \mathbb{E}[(\bar{X} - \mu)^2]$$
6. Evaluate both expectation terms:
   - By definition of variance: $\mathbb{E}[(X_i - \mu)^2] = \text{Var}(X_i) = \sigma^2$. Thus $\sum_{i=1}^N \sigma^2 = N \sigma^2$.
   - For sample mean: $\mathbb{E}[(\bar{X} - \mu)^2] = \text{Var}(\bar{X}) = \frac{\sigma^2}{N}$.
7. Combine the terms:
   $$\mathbb{E}\left[ \sum_{i=1}^N (X_i - \bar{X})^2 \right] = N \sigma^2 - N \left(\frac{\sigma^2}{N}\right) = N \sigma^2 - \sigma^2 = \mathbf{(N - 1) \sigma^2}$$
8. Dividing by $N$:
   $$\mathbb{E}[S_n^2] = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N (X_i - \bar{X})^2\right] = \mathbf{\frac{N - 1}{N} \sigma^2 \ne \sigma^2} \quad (\text{Biased!})$$
9. Dividing by $N - 1$:
   $$\mathbb{E}[S^2] = \mathbb{E}\left[\frac{1}{N - 1}\sum_{i=1}^N (X_i - \bar{X})^2\right] = \frac{(N - 1)\sigma^2}{N - 1} = \mathbf{\sigma^2} \quad (\text{Unbiased!}) \quad \blacksquare$$

---

### Derivation 4.1.2: Bias-Variance Decomposition of Mean Squared Error (MSE)
For a model $\hat{f}(x)$ trained on dataset $\mathcal{D}$ predicting target $y = f(x) + \epsilon$ with noise $\mathbb{E}[\epsilon] = 0, \text{Var}(\epsilon) = \sigma^2$:

**Step-by-step derivation:**
1. The expected squared test error at point $x$ is:
   $$\mathbb{E}_\mathcal{D, \epsilon}[(y - \hat{f}(x))^2] = \mathbb{E}[(f(x) + \epsilon - \hat{f}(x))^2]$$
2. Add and subtract $\mathbb{E}[\hat{f}(x)]$:
   $$y - \hat{f}(x) = (f(x) - \mathbb{E}[\hat{f}(x)]) + (\mathbb{E}[\hat{f}(x)] - \hat{f}(x)) + \epsilon$$
3. Let $A = f(x) - \mathbb{E}[\hat{f}(x)] = \text{Bias}(\hat{f})$, $B = \mathbb{E}[\hat{f}(x)] - \hat{f}(x)$, and $C = \epsilon$.
   $$(A + B + C)^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC$$
4. Take expectations:
   - $\mathbb{E}[A^2] = \text{Bias}^2(\hat{f}(x))$ (since $A$ is deterministic).
   - $\mathbb{E}[B^2] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2] = \text{Var}(\hat{f}(x))$.
   - $\mathbb{E}[C^2] = \mathbb{E}[\epsilon^2] = \sigma_\epsilon^2$ (irreducible error).
   - Cross-terms vanish: $\mathbb{E}[2AB] = 2A \mathbb{E}[B] = 2A(0) = 0$, $\mathbb{E}[2AC] = 0$, $\mathbb{E}[2BC] = 0$ due to independence of noise $\epsilon$.
5. Combining terms:
   $$\mathbf{\text{MSE}(\hat{f}(x)) = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma_\epsilon^2} \quad \blacksquare$$

---

## 4.2 Solved Numerical Problem Typologies

### Problem Type 1: Maximum Likelihood Estimation (MLE) for Exponential Distribution
**Problem:** Given an $i.i.d.$ sample of response latencies (in milliseconds) from an LLM server: $\mathbf{x} = [20, 50, 80, 10, 40]$. Assume $X_i \sim \text{Exponential}(\lambda)$ with PDF $p(x \mid \lambda) = \lambda e^{-\lambda x}$.
1. Derive the analytical formula for $\hat{\lambda}_{\text{MLE}}$.
2. Compute the concrete numerical estimate for the given sample.

**Step-by-step Numerical Solution:**
1. Formulate the likelihood function:
   $$L(\lambda) = \prod_{i=1}^N \lambda e^{-\lambda x_i} = \lambda^N \exp\left( -\lambda \sum_{i=1}^N x_i \right)$$
2. Log-likelihood $\ell(\lambda) = \ln L(\lambda)$:
   $$\ell(\lambda) = N \ln \lambda - \lambda \sum_{i=1}^N x_i$$
3. Take derivative w.r.t. $\lambda$ and set to zero:
   $$\frac{d \ell}{d \lambda} = \frac{N}{\lambda} - \sum_{i=1}^N x_i = 0 \implies \mathbf{\hat{\lambda}_{\text{MLE}} = \frac{N}{\sum_{i=1}^N x_i} = \frac{1}{\bar{x}}}$$
4. Compute numerical estimate:
   $$N = 5, \quad \sum x_i = 20 + 50 + 80 + 10 + 40 = 200 \text{ ms}$$
   $$\bar{x} = \frac{200}{5} = 40 \text{ ms}$$
   $$\mathbf{\hat{\lambda}_{\text{MLE}} = \frac{1}{40} = 0.0250 \text{ ms}^{-1}}$$

---

### Problem Type 2: Maximum A Posteriori (MAP) with Gaussian Prior ($L_2$ Ridge)
**Problem:** Estimating an unknown scalar parameter $\theta$ with Gaussian likelihood $X \mid \theta \sim \mathcal{N}(\theta, \sigma^2 = 4)$ and Gaussian prior $\theta \sim \mathcal{N}(\mu_0 = 0, \sigma_0^2 = 1)$.
Observed sample of $N = 4$ measurements has sample mean $\bar{X} = 3.0$.
1. Compute the analytical MAP estimate $\hat{\theta}_{\text{MAP}}$.
2. Compare with the MLE estimate $\hat{\theta}_{\text{MLE}}$.

**Step-by-step Numerical Solution:**
1. The analytical MAP estimator for a Gaussian conjugate model is:
   $$\hat{\theta}_{\text{MAP}} = \frac{\frac{N}{\sigma^2}}{\frac{N}{\sigma^2} + \frac{1}{\sigma_0^2}} \bar{X} + \frac{\frac{1}{\sigma_0^2}}{\frac{N}{\sigma^2} + \frac{1}{\sigma_0^2}} \mu_0$$
2. Compute precisions:
   - Data precision: $\frac{N}{\sigma^2} = \frac{4}{4} = 1.0$.
   - Prior precision: $\frac{1}{\sigma_0^2} = \frac{1}{1.0} = 1.0$.
   - Total posterior precision: $1.0 + 1.0 = 2.0$.
3. Compute estimate:
   $$\hat{\theta}_{\text{MAP}} = \frac{1.0}{2.0}(3.0) + \frac{1.0}{2.0}(0.0) = 0.5(3.0) + 0 = \mathbf{1.5000}$$
4. Comparison:
   $$\hat{\theta}_{\text{MLE}} = \bar{X} = \mathbf{3.0000}$$
   **Insight:** The prior pulled (regularized) the estimate towards $0$ by exactly $50\%$, directly demonstrating how $L_2$ regularization penalizes weight magnitude!

---

# SECTION 5: OPTIMIZATION & CONVEX ANALYSIS

## 5.1 Core Formula Derivations

### Derivation 5.1.1: Karush-Kuhn-Tucker (KKT) Necessary Conditions
For constrained problem: $\min f(x)$ subject to $g_i(x) \le 0$ ($i=1..m$) and $h_j(x) = 0$ ($j=1..p$).
The Lagrangian is $\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)$.

**Step-by-step derivation:**
1. Define the primal problem infimum: $p^* = \inf_{x} \sup_{\lambda \ge 0, \nu} \mathcal{L}(x, \lambda, \nu)$.
   - If any $g_i(x) > 0$, $\sup_{\lambda_i \ge 0} \lambda_i g_i(x) = +\infty$.
   - If all $g_i(x) \le 0$ and $h_j(x) = 0$, the supremum is achieved at $\lambda_i g_i(x) = 0$, recovering original objective $f(x)$.
2. The Lagrange Dual function is $g(\lambda, \nu) = \inf_x \mathcal{L}(x, \lambda, \nu)$.
   Dual problem: $d^* = \sup_{\lambda \ge 0, \nu} g(\lambda, \nu)$.
3. When Slater's condition holds, Strong Duality holds: $p^* = d^* = f(x^*) = g(\lambda^*, \nu^*)$.
4. Expanding the equality:
   $$f(x^*) = g(\lambda^*, \nu^*) = \inf_x \mathcal{L}(x, \lambda^*, \nu^*) \le \mathcal{L}(x^*, \lambda^*, \nu^*) = f(x^*) + \sum_{i=1}^m \lambda_i^* g_i(x^*) + \sum_{j=1}^p \nu_j^* h_j(x^*)$$
5. Since $h_j(x^*) = 0$ and $\lambda_i^* \ge 0, g_i(x^*) \le 0$, the sum $\sum \lambda_i^* g_i(x^*) \le 0$.
   For equality to hold across $f(x^*) \le f(x^*) + \sum \lambda_i^* g_i(x^*)$, we must have:
   $$\mathbf{\sum_{i=1}^m \lambda_i^* g_i(x^*) = 0 \implies \lambda_i^* g_i(x^*) = 0 \quad \forall i} \quad (\text{Complementary Slackness})$$
6. Furthermore, $x^*$ minimizes $\mathcal{L}(x, \lambda^*, \nu^*)$ over all $x$. Therefore, its gradient must vanish:
   $$\mathbf{\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = \nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = \mathbf{0}} \quad (\text{Stationarity}) \quad \blacksquare$$

---

### Derivation 5.1.2: Adam Bias Correction Proof
In Adam, moving averages of gradient $g_t$ are tracked:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad m_0 = 0$$

**Step-by-step derivation:**
1. Unroll the recursion backwards to $t=0$:
   $$m_t = (1 - \beta_1) g_t + \beta_1 [(1 - \beta_1) g_{t-1} + \beta_1 m_{t-2}] = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t - i} g_i$$
2. Take expectations on both sides, assuming gradients $g_i$ come from a stationary distribution with expectation $\mathbb{E}[g_i] = \mathbb{E}[g_t]$:
   $$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t - i} g_i \right] = (1 - \beta_1) \left( \sum_{i=1}^t \beta_1^{t - i} \right) \mathbb{E}[g_t]$$
3. The sum is a finite geometric series with ratio $\beta_1$ and $t$ terms:
   $$\sum_{i=1}^t \beta_1^{t - i} = \beta_1^{t-1} + \beta_1^{t-2} + \dots + 1 = \frac{1 - \beta_1^t}{1 - \beta_1}$$
4. Substitute the sum back:
   $$\mathbb{E}[m_t] = (1 - \beta_1) \left( \frac{1 - \beta_1^t}{1 - \beta_1} \right) \mathbb{E}[g_t] = \mathbf{(1 - \beta_1^t) \mathbb{E}[g_t]}$$
5. To make the estimator unbiased ($\mathbb{E}[\hat{m}_t] = \mathbb{E}[g_t]$), we must divide by $(1 - \beta_1^t)$:
   $$\mathbf{\hat{m}_t = \frac{m_t}{1 - \beta_1^t}} \quad \blacksquare$$
   The exact same algebraic derivation applies to the second moment $v_t$, yielding $\mathbf{\hat{v}_t = \frac{v_t}{1 - \beta_2^t}}$.

---

## 5.2 Solved Numerical Problem Typologies

### Problem Type 1: Constrained Optimization via KKT System
**Problem:** Solve by hand:
$$\min_{x, y} f(x, y) = x^2 + 2y^2 \quad \text{subject to } g(x, y) = 2 - x - y \le 0$$

**Step-by-step Numerical Solution:**
1. Lagrangian: $\mathcal{L}(x, y, \lambda) = x^2 + 2y^2 + \lambda(2 - x - y)$.
2. KKT Stationarity:
   - $\frac{\partial \mathcal{L}}{\partial x} = 2x - \lambda = 0 \implies x = \lambda / 2$
   - $\frac{\partial \mathcal{L}}{\partial y} = 4y - \lambda = 0 \implies y = \lambda / 4$
3. Test Case A ($\lambda = 0$):
   $x = 0, y = 0 \implies g(0, 0) = 2 - 0 - 0 = 2 > 0$ (Violates primal feasibility!).
4. Test Case B ($\lambda > 0$, active constraint $x + y = 2$):
   $$\frac{\lambda}{2} + \frac{\lambda}{4} = 2 \implies \frac{3\lambda}{4} = 2 \implies \mathbf{\lambda = \frac{8}{3} \approx 2.6667}$$
5. Compute optimal coordinates:
   $$\mathbf{x^* = \frac{8/3}{2} = \frac{4}{3} \approx 1.3333}, \quad \mathbf{y^* = \frac{8/3}{4} = \frac{2}{3} \approx 0.6667}$$
   Check constraint: $x^* + y^* = 4/3 + 2/3 = 6/3 = 2.0 \quad \checkmark$.
6. Minimum loss value:
   $$f(x^*, y^*) = (4/3)^2 + 2(2/3)^2 = 16/9 + 2(4/9) = \frac{16 + 8}{9} = \frac{24}{9} = \mathbf{\frac{8}{3} \approx 2.6667}$$

---

### Problem Type 2: Complete Single Step of Adam Optimizer by Hand
**Problem:** A scalar parameter weight is currently $w_0 = 1.0000$. At step $t=1$, the evaluated gradient is $g_1 = 0.4000$.
Parameters: $\alpha = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, initial states $m_0 = 0, v_0 = 0$.
Compute the updated weight $w_1$ by hand to 6 decimal places.

**Step-by-step Numerical Solution:**
1. First moment update ($m_1$):
   $$m_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = 0.9(0) + (1 - 0.9)(0.4000) = 0.1(0.4000) = \mathbf{0.040000}$$
2. Second moment update ($v_1$):
   $$v_1 = \beta_2 v_0 + (1 - \beta_2) g_1^2 = 0.999(0) + (0.001)(0.4000)^2 = 0.001(0.160000) = \mathbf{0.000160}$$
3. Bias correction at $t = 1$:
   $$\hat{m}_1 = \frac{m_1}{1 - \beta_1^1} = \frac{0.040000}{1 - 0.9} = \frac{0.040000}{0.1} = \mathbf{0.400000}$$
   $$\hat{v}_1 = \frac{v_1}{1 - \beta_2^1} = \frac{0.000160}{1 - 0.999} = \frac{0.000160}{0.001} = \mathbf{0.160000}$$
4. Parameter update:
   $$\Delta w = \frac{\alpha \hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = \frac{0.01 \times 0.400000}{\sqrt{0.160000} + 10^{-8}} = \frac{0.004000}{0.400000} = \mathbf{0.010000}$$
   $$w_1 = w_0 - \Delta w = 1.000000 - 0.010000 = \mathbf{0.990000}$$

---

# SECTION 6: DEEP LEARNING & FRONTIER ARCHITECTURE MATHEMATICS

## 6.1 Core Formula Derivations

### Derivation 6.1.1: Softmax Combined with Cross-Entropy Loss Gradient
Let $z \in \mathbb{R}^C$ be logits, $p = \text{softmax}(z) = \frac{e^{z_i}}{\sum_k e^{z_k}}$, and true one-hot target $y \in \{0, 1\}^C$ where $y_c = 1$.
The cross-entropy loss is $\mathcal{L} = -\sum_{k=1}^C y_k \ln p_k = -\ln p_c$.

**Step-by-step derivation:**
1. Express loss directly in terms of logits $z$:
   $$\mathcal{L} = -\ln \left( \frac{e^{z_c}}{\sum_{k=1}^C e^{z_k}} \right) = -z_c + \ln\left( \sum_{k=1}^C e^{z_k} \right)$$
2. Differentiate with respect to an arbitrary logit $z_i$:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = -\frac{\partial z_c}{\partial z_i} + \frac{\partial}{\partial z_i} \ln\left( \sum_{k=1}^C e^{z_k} \right)$$
3. The derivative of the first term is $-\delta_{ic}$ where $\delta_{ic} = 1$ if $i = c$, else $0$ (which is $y_i$).
4. The derivative of the log-sum-exp term:
   $$\frac{\partial}{\partial z_i} \ln\left( \sum_{k=1}^C e^{z_k} \right) = \frac{1}{\sum_{k=1}^C e^{z_k}} \frac{\partial}{\partial z_i}\left( \sum_{k=1}^C e^{z_k} \right) = \frac{e^{z_i}}{\sum_{k=1}^C e^{z_k}} = p_i$$
5. Combine both terms:
   $$\mathbf{\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i} \quad \implies \quad \mathbf{\nabla_z \mathcal{L} = p - y} \quad \blacksquare$$
   *Pedagogical Insight:* The complex Softmax Jacobian and Cross-Entropy derivatives magically collapse into the simplest possible subtraction: predicted probability minus ground truth label!

---

### Derivation 6.1.2: Scaled Dot-Product Attention & Online Softmax (FlashAttention)
Given query, key, value matrices $Q, K, V \in \mathbb{R}^{N \times d}$:
$$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d}} \right) V$$

In hardware-accelerated attention (FlashAttention-2), computing and materializing the $N \times N$ attention matrix $S = Q K^T / \sqrt{d}$ in High-Bandwidth Memory (HBM) is an $\mathcal{O}(N^2)$ memory bottleneck.
FlashAttention uses the **Online Softmax trick**:
For vectors $x^{(1)}$ and $x^{(2)}$:
$$m_{\text{new}} = \max(m_1, m_2)$$
$$d_{\text{new}} = d_1 e^{m_1 - m_{\text{new}}} + d_2 e^{m_2 - m_{\text{new}}}$$
$$O_{\text{new}} = O_1 \frac{d_1 e^{m_1 - m_{\text{new}}}}{d_{\text{new}}} + O_2 \frac{d_2 e^{m_2 - m_{\text{new}}}}{d_{\text{new}}}$$
This allows calculating exact attention outputs in tiles entirely inside Fast SRAM with $\mathcal{O}(N)$ memory footprint! $\blacksquare$

---

## 6.2 Solved Numerical Problem Typologies

### Problem Type 1: Scaled Dot-Product Attention Hand Pass
**Problem:** Given query vector $q = [1.0, 0.0]$, key matrix $K = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$, value matrix $V = \begin{bmatrix} 10 & 20 \\ 30 & 40 \end{bmatrix}$, with head dimension $d_k = 2$.
1. Compute raw dot-product scores $S = q K^T$.
2. Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.4142$.
3. Compute attention weights $A = \text{softmax}(S / \sqrt{d_k})$.
4. Compute final attended representation vector $o = A V$.

**Step-by-step Numerical Solution:**
1. Raw dot products:
   - Token 1: $q \cdot k_1 = [1.0, 0.0] \cdot [1.0, 0.0]^T = 1.0(1.0) + 0(0) = \mathbf{1.0}$
   - Token 2: $q \cdot k_2 = [1.0, 0.0] \cdot [0.0, 2.0]^T = 1.0(0.0) + 0(2.0) = \mathbf{0.0}$
   $$S = [1.0, 0.0]$$
2. Scale by $\sqrt{2} \approx 1.4142$:
   $$S / \sqrt{2} = [1.0 / 1.4142, 0.0 / 1.4142] \approx [\mathbf{0.7071}, \mathbf{0.0000}]$$
3. Softmax computation:
   - $e^{0.7071} \approx 2.0281$
   - $e^{0.0000} = 1.0000$
   - Sum $= 2.0281 + 1.0000 = 3.0281$
   Attention weights:
   $$A_1 = \frac{2.0281}{3.0281} \approx \mathbf{0.6698}, \quad A_2 = \frac{1.0000}{3.0281} \approx \mathbf{0.3302}$$
4. Final attention output $o = A_1 v_1 + A_2 v_2$:
   $$o = 0.6698 \begin{bmatrix} 10 \\ 20 \end{bmatrix} + 0.3302 \begin{bmatrix} 30 \\ 40 \end{bmatrix} = \begin{bmatrix} 6.698 + 9.906 \\ 13.396 + 13.208 \end{bmatrix} = \mathbf{\begin{bmatrix} 16.604 \\ 26.604 \end{bmatrix}}$$

---

### Problem Type 2: Layer Normalization Forward & Backward by Hand
**Problem:** Given input activation vector $x = [2.0, 4.0, 6.0]$ for a layer with learnable parameters $\gamma = 2.0, \beta = 0.0$ and $\epsilon = 0.0$:
1. Compute mean $\mu$ and variance $\sigma^2$.
2. Compute normalized activations $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2}}$.
3. Compute output activations $y_i = \gamma \hat{x}_i + \beta$.
4. Given upstream gradient $\frac{\partial \mathcal{L}}{\partial y} = [1.0, -1.0, 0.0]$, compute weight gradient $\frac{\partial \mathcal{L}}{\partial \gamma}$.

**Step-by-step Numerical Solution:**
1. Mean & Variance:
   $$\mu = \frac{2.0 + 4.0 + 6.0}{3} = \frac{12.0}{3} = \mathbf{4.0000}$$
   $$\sigma^2 = \frac{(2-4)^2 + (4-4)^2 + (6-4)^2}{3} = \frac{(-2)^2 + 0^2 + 2^2}{3} = \frac{4 + 0 + 4}{3} = \frac{8}{3} \approx \mathbf{2.6667}$$
   $$\sigma = \sqrt{8/3} = \frac{2\sqrt{2}}{\sqrt{3}} \approx \mathbf{1.6330}$$
2. Normalized vector $\hat{x}$:
   $$\hat{x}_1 = \frac{2.0 - 4.0}{1.6330} = \frac{-2.0}{1.6330} \approx \mathbf{-1.2247}$$
   $$\hat{x}_2 = \frac{4.0 - 4.0}{1.6330} = \mathbf{0.0000}$$
   $$\hat{x}_3 = \frac{6.0 - 4.0}{1.6330} = \frac{+2.0}{1.6330} \approx \mathbf{+1.2247}$$
3. Output vector $y = \gamma \hat{x} + \beta$ with $\gamma = 2.0, \beta = 0$:
   $$y = 2.0 \begin{bmatrix} -1.2247 \\ 0.0000 \\ 1.2247 \end{bmatrix} + 0 = \mathbf{\begin{bmatrix} -2.4495 \\ 0.0000 \\ +2.4495 \end{bmatrix}}$$
4. Compute scale parameter gradient $\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^3 \left(\frac{\partial \mathcal{L}}{\partial y_i}\right) \hat{x}_i$:
   $$\frac{\partial \mathcal{L}}{\partial \gamma} = (1.0)(-1.2247) + (-1.0)(0.0000) + (0.0)(1.2247) = \mathbf{-1.2247}$$

---

## 🧭 Course Progress Tracker & Chapter Directory

All 111 comprehensive textbook chapters can be navigated via:
- [00_index_and_tracker.md](./00_index_and_tracker.md) — Comprehensive chapter completion tracker.
- [01_linear_algebra/](./01_linear_algebra) — 9 chapters on vector spaces, SVD, and tensors.
- [02_multivariable_calculus/](./02_multivariable_calculus) — 7 chapters on gradients, Hessians, and autodiff.
- [03_probability_theory/](./03_probability_theory) — 7 chapters on distributions, Bayes, and information theory.
- [04_mathematical_statistics/](./04_mathematical_statistics) — 6 chapters on estimators, MLE, MAP, and sampling.
- [05_optimization/](./05_optimization) — 7 chapters on KKT, SGD, AdamW, and second-order methods.
- [06_deep_learning_foundations/](./06_deep_learning_foundations) — 8 chapters on MLPs, loss functions, and normalization.
- [07_convolutional_networks/](./07_convolutional_networks) — 4 chapters on CNN backprop, architectures, and ViTs.
- [08_recurrent_networks/](./08_recurrent_networks) — 4 chapters on sequence modeling, vanishing gradients, and LSTMs.
- [09_transformers_and_llms/](./09_transformers_and_llms) — 8 chapters on attention, RoPE, efficiency, and alignment.
- [10_generative_models/](./10_generative_models) — 4 chapters on VAEs, GANs, and Diffusion models.
- [11_reinforcement_learning/](./11_reinforcement_learning) — 28 chapters on MDPs, PPO, SAC, MCTS, and GRPO.
- [12_modern_llm_architectures/](./12_modern_llm_architectures) — 12 chapters on tokenizers, SwiGLU, MoE, LoRA, and DiTs.
- [13_reasoning_and_test_time_compute/](./13_reasoning_and_test_time_compute) — 9 chapters on CoT, PRMs, MCTS, and DeepSeek-R1.
