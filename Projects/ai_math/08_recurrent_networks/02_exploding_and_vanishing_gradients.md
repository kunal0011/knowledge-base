# Chapter 8.2: The Exploding and Vanishing Gradient Problem in RNNs

---

## 1. Intuition & 101 Motivation

In Chapter 8.1, we saw that Backpropagation Through Time (BPTT) unrolls a Recurrent Neural Network across $T$ temporal steps, forming a deep computational graph of depth $T$.
However, unlike a standard feedforward deep network where each layer has its own distinct weight matrix $\mathbf{W}_l$, an RNN applies the **exact same weight matrix $\mathbf{W}_{hh}$** at every single time step.

This structural repetition leads directly to the defining pathological challenge of recurrent sequence modeling:
- When calculating the influence of an early hidden state $\mathbf{h}_t$ on a distant future loss $\mathcal{L}_T$, the chain rule computes a product of $(T - t)$ Jacobian matrices.
- Just as repeatedly multiplying a scalar $w$ yields $w^{T-t}$:
  - If $|w| < 1$, $w^{100} \to 0$ (exponential decay $\to$ **Vanishing Gradient**).
  - If $|w| > 1$, $w^{100} \to \infty$ (exponential explosion $\to$ **Exploding Gradient**).

### The Fundamental Information Trade-off (Bengio et al., 1994)
Why couldn't early practitioners easily tune $\mathbf{W}_{hh}$ to avoid both extremes?
Yoshua Bengio et al. proved a mathematical dilemma:
- To store information robustly against input noise, an autonomous dynamical system must possess **attracting fixed points**, which requires the system map to be **contractive** ($\|\mathbf{J}\| < 1$).
- But contracting dynamics guarantee that backward error signals contract at the exact same exponential rate!
- Therefore, a Vanilla RNN cannot simultaneously be robust to input perturbations and capable of learning long-term temporal dependencies ($> 10-15$ steps).

---

## 2. Rigorous Mathematical Formulation

```
                 THE TEMPORAL JACOBIAN CHAIN PRODUCT
    dL/dh_T ──► [ J_T ] ──► [ J_T-1 ] ──► ... ──► [ J_t+1 ] ──► dL/dh_t
                 │            │                     │
               W_hh         W_hh                  W_hh
               
    If ||W_hh|| < 1 : Signal shrinks by factor of ||W_hh|| at every hop (Vanishing)
    If ||W_hh|| > 1 : Signal multiplies exponentially at every hop     (Exploding)
```

---

### 2.1 The Long-Term Jacobian Chain Product

Recall from Chapter 8.1 that the gradient of the loss $\mathcal{L}_T$ at time $T$ with respect to the hidden state $\mathbf{h}_t$ at an earlier time step $t < T$ is:
$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t}$$

By expanding the intermediate hidden state transitions:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^T \mathbf{J}_k$$

where each temporal Jacobian matrix $\mathbf{J}_k \in \mathbb{R}^{d_h \times d_h}$ is given by:
$$\mathbf{J}_k \equiv \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \operatorname{diag}\left(1 - \mathbf{h}_k^2\right) \mathbf{W}_{hh}$$
Letting $\mathbf{D}_k = \operatorname{diag}\left(1 - \mathbf{h}_k^2\right)$ be the diagonal matrix of $\tanh$ derivatives:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \mathbf{D}_T \mathbf{W}_{hh} \mathbf{D}_{T-1} \mathbf{W}_{hh} \dots \mathbf{D}_{t+1} \mathbf{W}_{hh} = \prod_{k=t+1}^T \left( \mathbf{D}_k \mathbf{W}_{hh} \right)$$

---

### 2.2 Upper and Lower Bounds: The Vanishing and Exploding Theorems

Let $\|\cdot\|$ denote the induced $L_2$ operator matrix norm (the maximum singular value $\sigma_{\max}$).

#### Theorem 1: Sufficient Condition for Vanishing Gradients
Using the submultiplicative property of matrix norms ($\|\mathbf{A}\mathbf{B}\| \le \|\mathbf{A}\| \|\mathbf{B}\|$):
$$\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\| \le \prod_{k=t+1}^T \|\mathbf{D}_k \mathbf{W}_{hh}\| \le \prod_{k=t+1}^T \|\mathbf{D}_k\| \|\mathbf{W}_{hh}\|$$

For the hyperbolic tangent activation function $\tanh(z)$:
$$|\tanh'(z)| = 1 - \tanh^2(z) \in (0, 1] \implies \|\mathbf{D}_k\| \le 1, \quad \forall k$$

Therefore:
$$\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\| \le \|\mathbf{W}_{hh}\|^{T - t} = \sigma_{\max}(\mathbf{W}_{hh})^{T - t}$$

**Conclusion:**
If the largest singular value of the recurrent transition matrix satisfies $\sigma_{\max}(\mathbf{W}_{hh}) < 1$:
$$\lim_{(T - t) \to \infty} \left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\| = 0$$
The gradient vanishes exponentially fast with temporal separation $(T - t)$. Early input tokens receive identically zero gradient, paralyzing long-term memory acquisition.

---

#### Theorem 2: Condition for Exploding Gradients
Now consider the case where the spectral radius $\rho(\mathbf{W}_{hh}) = \max_i |\lambda_i| > 1$.
If the pre-activations lie in the linear regime of $\tanh$ ($|a_k| \approx 0 \implies \tanh'(a_k) \approx 1 \implies \mathbf{D}_k \approx \mathbf{I}$), then:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \approx \mathbf{W}_{hh}^{T - t}$$
Let $\mathbf{v}$ be an eigenvector of $\mathbf{W}_{hh}$ corresponding to eigenvalue $\lambda$ with $|\lambda| > 1$. Then:
$$\mathbf{W}_{hh}^{T - t} \mathbf{v} = \lambda^{T - t} \mathbf{v}$$
Taking norms:
$$\left\| \mathbf{W}_{hh}^{T - t} \mathbf{v} \right\| = |\lambda|^{T - t} \|\mathbf{v}\| \xrightarrow{(T - t) \to \infty} \infty$$

**Conclusion:**
If $\rho(\mathbf{W}_{hh}) > 1$ and states do not heavily saturate, gradient norms grow exponentially:
$$\lim_{(T - t) \to \infty} \left\| \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_t} \right\| = \infty$$
During SGD optimization, an exploding gradient causes parameter updates $\Delta \mathbf{W} = -\eta \mathbf{g}$ to take gigantic steps, completely destroying learned representations and causing floating-point overflow (`NaN` or `Inf`).

---

### 2.3 Pascanu's Gradient Clipping by Norm (Pascanu et al., 2013)

To stabilize training in the presence of exploding gradients without artificially limiting sequence length, Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio introduced **Gradient Clipping by Norm**.

Let $\mathbf{g} = \nabla_{\theta} \mathcal{L}$ be the concatenated gradient vector of all trainable parameters.
Given a maximum allowable gradient norm threshold $\theta_{\text{clip}} > 0$:

$$\mathbf{g}_{\text{clipped}} = \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \le \theta_{\text{clip}} \\ \theta_{\text{clip}} \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \theta_{\text{clip}} \end{cases}$$

#### Why Norm-based Clipping is Superior to Element-wise Value Clipping:
1. **Direction Preservation:**
   $$\frac{\mathbf{g}_{\text{clipped}}}{\|\mathbf{g}_{\text{clipped}}\|} = \frac{\mathbf{g}}{\|\mathbf{g}\|}$$
   Norm clipping rescales the step size while keeping the descent direction **$100\%$ identical** to the true steepest descent direction.
2. **Value Clipping Ruin:**
   If one instead clips component-wise ($\text{clip}(g_i, -c, c)$), the angle between the clipped vector and the true gradient can approach $90^\circ$, causing the optimizer to move in an arbitrary, incorrect direction.

```
       Norm Clipping vs. Element-wise Value Clipping
                   g (True Gradient)
                  ▲
                 /
                /  <-- Same direction, scaled magnitude
               /
              * g_clipped (Norm Clipping)
             /
            /
           O ──────────────► g_value_clipped (Direction altered!)
```

---

### 2.4 Orthogonal & Unitary Weight Initialization

Can we prevent vanishing gradients at initialization?
If we initialize $\mathbf{W}_{hh}$ as an **orthogonal matrix**:
$$\mathbf{W}_{hh}^T \mathbf{W}_{hh} = \mathbf{I}$$
Then all singular values are identically equal to one:
$$\sigma_i(\mathbf{W}_{hh}) = 1, \quad \forall i \in \{1, \dots, d_h\}$$

An orthogonal transformation preserves Euclidean vector norms:
$$\|\mathbf{W}_{hh} \mathbf{v}\|_2 = \|\mathbf{v}\|_2$$
When backpropagating through the linear component:
$$\|\mathbf{W}_{hh}^{T - t} \delta\|_2 = \|\delta\|_2$$
Gradients neither explode nor vanish through the linear transition!
*(Note: While orthogonal initialization mitigates gradient decay at step 0, as training progresses and non-linearities saturate, gating mechanisms like LSTMs/GRUs become necessary to maintain constant gradient highways).*

---

### 2.5 Deep Derivation 8.2.1: Spectral Radius, Gelfand's Formula, and Jordan Decomposition of Recurrent Jacobians

In this derivation, we rigorously establish the asymptotic behavior of matrix powers in recurrent neural networks using spectral theory and the Jordan canonical form.

#### Step 1: Definition of the Spectral Radius
Let $\mathbf{W} \in \mathbb{R}^{d \times d}$ be a real square matrix. The spectrum $\sigma(\mathbf{W}) \subset \mathbb{C}$ is the set of all eigenvalues $\lambda \in \mathbb{C}$ satisfying $\det(\lambda \mathbf{I} - \mathbf{W}) = 0$.
The **spectral radius** $\rho(\mathbf{W})$ is defined as:
$$\rho(\mathbf{W}) \equiv \max_{\lambda \in \sigma(\mathbf{W})} |\lambda|$$

While the induced Euclidean matrix norm $\|\mathbf{W}\|_2 = \sigma_{\max}(\mathbf{W})$ (the largest singular value) satisfies $\rho(\mathbf{W}) \le \|\mathbf{W}\|_2$, equality holds if and only if $\mathbf{W}$ is a normal matrix ($\mathbf{W}^T \mathbf{W} = \mathbf{W} \mathbf{W}^T$). For non-normal matrices, $\|\mathbf{W}\|_2$ can be arbitrarily larger than $\rho(\mathbf{W})$.

#### Step 2: Gelfand's Spectral Radius Formula
To determine the asymptotic behavior of the matrix power $\mathbf{W}^k$ as $k \to \infty$, we invoke **Gelfand's Formula**:
$$\rho(\mathbf{W}) = \lim_{k \to \infty} \|\mathbf{W}^k\|^{1/k}$$
for any submultiplicative matrix norm $\|\cdot\|$.

**Proof Outline via Complex Analysis:**
The resolvent operator $R(z, \mathbf{W}) = (z \mathbf{I} - \mathbf{W})^{-1}$ is holomorphic on the open set $|z| > \rho(\mathbf{W})$.
Its Neumann series expansion:
$$(z \mathbf{I} - \mathbf{W})^{-1} = \frac{1}{z} \sum_{k=0}^\infty \left( \frac{\mathbf{W}}{z} \right)^k$$
converges if and only if $|z| > \rho(\mathbf{W})$.
By the Cauchy-Hadamard theorem for power series, the radius of convergence is $R = \limsup_{k \to \infty} \|\mathbf{W}^k\|^{1/k}$. Since the singularity occurs at $|z| = \rho(\mathbf{W})$, the limit exists and equals $\rho(\mathbf{W})$.

#### Step 3: Jordan Canonical Form Decomposition
Any matrix $\mathbf{W} \in \mathbb{R}^{d \times d}$ can be decomposed as $\mathbf{W} = \mathbf{P} \mathbf{J} \mathbf{P}^{-1}$, where $\mathbf{J} = \operatorname{diag}(\mathbf{J}_1, \dots, \mathbf{J}_m)$ is a block diagonal matrix of Jordan blocks:
$$\mathbf{J}_i = \begin{bmatrix}
\lambda_i & 1 & 0 & \dots & 0 \\
0 & \lambda_i & 1 & \dots & 0 \\
\vdots & \ddots & \ddots & \ddots & \vdots \\
0 & \dots & 0 & \lambda_i & 1 \\
0 & \dots & 0 & 0 & \lambda_i
\end{bmatrix} \in \mathbb{C}^{n_i \times n_i}$$
Decomposing each block as $\mathbf{J}_i = \lambda_i \mathbf{I} + \mathbf{N}_i$, where $\mathbf{N}_i$ is a strictly upper triangular nilpotent matrix ($\mathbf{N}_i^{n_i} = \mathbf{0}$).
By the binomial theorem (since $\lambda_i \mathbf{I}$ and $\mathbf{N}_i$ commute):
$$\mathbf{J}_i^k = \sum_{j=0}^{\min(k, n_i - 1)} \binom{k}{j} \lambda_i^{k - j} \mathbf{N}_i^j$$
For $k \ge n_i$:
$$\mathbf{J}_i^k = \begin{bmatrix}
\lambda_i^k & \binom{k}{1} \lambda_i^{k-1} & \binom{k}{2} \lambda_i^{k-2} & \dots & \binom{k}{n_i-1} \lambda_i^{k-n_i+1} \\
0 & \lambda_i^k & \binom{k}{1} \lambda_i^{k-1} & \dots & \binom{k}{n_i-2} \lambda_i^{k-n_i+2} \\
\vdots & \ddots & \ddots & \ddots & \vdots \\
0 & \dots & 0 & \lambda_i^k & \binom{k}{1} \lambda_i^{k-1} \\
0 & \dots & 0 & 0 & \lambda_i^k
\end{bmatrix}$$

#### Step 4: Asymptotic Regimes of BPTT
Taking the norm of the power $\mathbf{W}^k = \mathbf{P} \mathbf{J}^k \mathbf{P}^{-1}$:
$$\|\mathbf{W}^k\| \le \|\mathbf{P}\| \|\mathbf{P}^{-1}\| \max_i \|\mathbf{J}_i^k\| \le \kappa(\mathbf{P}) \cdot \max_i \left( \sum_{j=0}^{n_i-1} \binom{k}{j} |\lambda_i|^{k-j} \right)$$
where $\kappa(\mathbf{P}) = \|\mathbf{P}\| \|\mathbf{P}^{-1}\|$ is the condition number of the eigenvector basis.

1. **Sub-unitary Spectral Radius ($\rho(\mathbf{W}) < 1$):**
   Every eigenvalue satisfies $|\lambda_i| \le \rho < 1$.
   The polynomial term $\binom{k}{j} \le k^{n_i}$ is exponentially dominated by $|\lambda_i|^k$:
   $$\lim_{k \to \infty} k^{n_i} \rho^k = 0 \implies \lim_{k \to \infty} \|\mathbf{W}^k\| = 0$$
   The recurrent error signal $\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_t} \sim \mathbf{W}^{T-t}$ vanishes exponentially.
2. **Super-unitary Spectral Radius ($\rho(\mathbf{W}) > 1$):**
   There exists at least one eigenvalue $|\lambda_{\max}| > 1$.
   The dominant Jordan block satisfies $\|\mathbf{J}_{\max}^k\| \ge |\lambda_{\max}|^k \to \infty$.
   The error signal explodes exponentially.
3. **Unitary Spectral Radius ($\rho(\mathbf{W}) = 1$):**
   - If all Jordan blocks with $|\lambda_i| = 1$ are trivial ($n_i = 1$, i.e., semi-simple eigenvalues), then $\|\mathbf{W}^k\| \le \kappa(\mathbf{P})$, remaining strictly bounded for all $k$.
   - If a Jordan block with $|\lambda_i| = 1$ has size $n_i > 1$ (defective matrix), then $\|\mathbf{W}^k\| \sim \binom{k}{n_i-1} = \mathcal{O}(k^{n_i-1})$, causing **polynomial gradient growth** (secular resonance). $\blacksquare$

---

### 2.6 Deep Derivation 8.2.2: Convergence and Step-Size Guarantees of Norm-Based Gradient Clipping Under Lipschitz Smoothness

In this derivation, we prove that gradient clipping guarantees descent and prevents explosive updates on $L$-Lipschitz smooth loss surfaces.

#### Step 1: Definition of $L$-Lipschitz Gradient Smoothness
Let the objective function $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ be continuously differentiable and $L$-Lipschitz smooth:
$$\|\nabla \mathcal{L}(\mathbf{w}_1) - \nabla \mathcal{L}(\mathbf{w}_2)\| \le L \|\mathbf{w}_1 - \mathbf{w}_2\|, \quad \forall \mathbf{w}_1, \mathbf{w}_2 \in \mathbb{R}^d$$
By the Descent Lemma (integration along the line segment between $\mathbf{w}_1$ and $\mathbf{w}_2$):
$$\mathcal{L}(\mathbf{w}_{k+1}) \le \mathcal{L}(\mathbf{w}_k) + \langle \nabla \mathcal{L}(\mathbf{w}_k), \mathbf{w}_{k+1} - \mathbf{w}_k \rangle + \frac{L}{2} \|\mathbf{w}_{k+1} - \mathbf{w}_k\|^2$$

#### Step 2: Parameter Update with Norm-Based Clipping
Let $\mathbf{g}_k = \nabla \mathcal{L}(\mathbf{w}_k)$. The clipped gradient update with learning rate $\eta > 0$ is:
$$\mathbf{w}_{k+1} = \mathbf{w}_k - \eta \, \mathbf{g}_k^{\text{clip}}, \qquad \text{where } \mathbf{g}_k^{\text{clip}} = \min\left(1, \frac{\theta_{\text{clip}}}{\|\mathbf{g}_k\|}\right) \mathbf{g}_k$$

The step displacement is $\Delta \mathbf{w}_k = -\eta \mathbf{g}_k^{\text{clip}}$. Its magnitude is bounded unconditionally:
$$\|\Delta \mathbf{w}_k\| = \eta \|\mathbf{g}_k^{\text{clip}}\| \le \eta \theta_{\text{clip}}$$

#### Step 3: Guaranteed Descent Bound
Substitute $\Delta \mathbf{w}_k$ into the Descent Lemma:
$$\mathcal{L}(\mathbf{w}_{k+1}) \le \mathcal{L}(\mathbf{w}_k) - \eta \langle \mathbf{g}_k, \mathbf{g}_k^{\text{clip}} \rangle + \frac{L \eta^2}{2} \|\mathbf{g}_k^{\text{clip}}\|^2$$

Observe that $\mathbf{g}_k$ and $\mathbf{g}_k^{\text{clip}}$ are collinear:
$$\langle \mathbf{g}_k, \mathbf{g}_k^{\text{clip}} \rangle = \|\mathbf{g}_k\| \|\mathbf{g}_k^{\text{clip}}\|$$

Case A: $\|\mathbf{g}_k\| \le \theta_{\text{clip}}$ (Unclipped):
$$\mathcal{L}(\mathbf{w}_{k+1}) \le \mathcal{L}(\mathbf{w}_k) - \eta \left(1 - \frac{L \eta}{2}\right) \|\mathbf{g}_k\|^2$$
For any $\eta < \frac{2}{L}$, this yields standard monotone loss descent.

Case B: $\|\mathbf{g}_k\| > \theta_{\text{clip}}$ (Clipped):
Here $\|\mathbf{g}_k^{\text{clip}}\| = \theta_{\text{clip}}$, and $\langle \mathbf{g}_k, \mathbf{g}_k^{\text{clip}} \rangle = \|\mathbf{g}_k\| \theta_{\text{clip}}$.
$$\mathcal{L}(\mathbf{w}_{k+1}) \le \mathcal{L}(\mathbf{w}_k) - \eta \theta_{\text{clip}} \|\mathbf{g}_k\| + \frac{L \eta^2 \theta_{\text{clip}}^2}{2}$$
$$\mathcal{L}(\mathbf{w}_{k+1}) - \mathcal{L}(\mathbf{w}_k) \le -\eta \theta_{\text{clip}} \left( \|\mathbf{g}_k\| - \frac{L \eta \theta_{\text{clip}}}{2} \right)$$

Because $\|\mathbf{g}_k\| > \theta_{\text{clip}}$, if we choose the learning rate such that $\eta < \frac{2}{L}$:
$$\|\mathbf{g}_k\| - \frac{L \eta \theta_{\text{clip}}}{2} > \theta_{\text{clip}} \left(1 - \frac{L \eta}{2}\right) > 0$$
Thus:
$$\mathcal{L}(\mathbf{w}_{k+1}) - \mathcal{L}(\mathbf{w}_k) \le -\eta \theta_{\text{clip}}^2 \left(1 - \frac{L \eta}{2}\right) < 0$$

**Theorem:**
Gradient clipping strictly guarantees monotonic loss decrease even when encountering steep loss ravines where $\|\nabla \mathcal{L}\| \to \infty$. It caps the maximum step size at $\eta \theta_{\text{clip}}$, preventing the optimizer from being catapulted into remote, chaotic regions of parameter space. $\blacksquare$

---

### 2.7 Deep Derivation 8.2.3: Stiefel Manifolds, Skew-Symmetric Lie Algebras, and Unitary Recurrent Networks

To permanently prevent gradient explosion and vanishing during training, Unitary/Orthogonal RNNs constrain the weight matrix $\mathbf{W}_{hh}$ to reside on the orthogonal group $\mathcal{O}(d)$ for real weights or unitary group $\mathcal{U}(d)$ for complex weights.

#### Step 1: The Geometry of the Orthogonal Group (Stiefel Manifold)
The orthogonal group of dimension $d$ is the compact Lie group:
$$\mathcal{O}(d) = \left\{ \mathbf{W} \in \mathbb{R}^{d \times d} : \mathbf{W}^T \mathbf{W} = \mathbf{I} \right\}$$
Any matrix $\mathbf{W} \in \mathcal{O}(d)$ is an isometry of Euclidean space:
$$\|\mathbf{W} \mathbf{v}\|_2^2 = (\mathbf{W} \mathbf{v})^T (\mathbf{W} \mathbf{v}) = \mathbf{v}^T (\mathbf{W}^T \mathbf{W}) \mathbf{v} = \mathbf{v}^T \mathbf{I} \mathbf{v} = \|\mathbf{v}\|_2^2$$
All singular values satisfy $\sigma_i(\mathbf{W}) = 1$. Consequently, the operator norm is strictly unity: $\|\mathbf{W}\|_2 = 1$.

#### Step 2: The Lie Algebra $\mathfrak{so}(d)$ of Skew-Symmetric Matrices
The tangent space at the identity $\mathbf{I} \in \mathcal{O}(d)$ is the Lie algebra of skew-symmetric matrices:
$$\mathfrak{so}(d) = \left\{ \mathbf{A} \in \mathbb{R}^{d \times d} : \mathbf{A}^T = -\mathbf{A} \right\}$$
The dimension of $\mathfrak{so}(d)$ is $\frac{d(d-1)}{2}$ independent parameters (all diagonal entries are zero, $A_{i, i} = 0$).

#### Step 3: Parameterizing Orthogonal Matrices via the Matrix Exponential
The Lie group exponential map $\exp: \mathfrak{so}(d) \to \mathcal{SO}(d)$ maps any unconstrained skew-symmetric matrix $\mathbf{A}$ onto a proper orthogonal matrix:
$$\mathbf{W} = \exp(\mathbf{A}) = \sum_{k=0}^\infty \frac{\mathbf{A}^k}{k!}$$
**Proof of Orthogonality:**
$$\mathbf{W}^T = (\exp(\mathbf{A}))^T = \exp(\mathbf{A}^T) = \exp(-\mathbf{A})$$
Since $\mathbf{A}$ and $-\mathbf{A}$ commute:
$$\mathbf{W}^T \mathbf{W} = \exp(-\mathbf{A}) \exp(\mathbf{A}) = \exp(-\mathbf{A} + \mathbf{A}) = \exp(\mathbf{0}) = \mathbf{I}$$

#### Step 4: The Cayley Transform (Computationally Efficient Alternative)
Evaluating the matrix exponential requires computing Taylor series or Schur forms. The **Cayley Transform** provides a rational, matrix-inversion-based bijection between $\mathfrak{so}(d)$ and $\mathcal{SO}(d)$:
$$\mathbf{W} = \operatorname{Cayley}(\mathbf{A}) = (\mathbf{I} - \mathbf{A}) (\mathbf{I} + \mathbf{A})^{-1}$$

**Proof of Orthogonality:**
1. Note that $(\mathbf{I} - \mathbf{A})$ and $(\mathbf{I} + \mathbf{A})^{-1}$ commute:
   $$(\mathbf{I} - \mathbf{A}) (\mathbf{I} + \mathbf{A}) = \mathbf{I} - \mathbf{A}^2 = (\mathbf{I} + \mathbf{A}) (\mathbf{I} - \mathbf{A})$$
   Multiplying both sides by $(\mathbf{I} + \mathbf{A})^{-1}$ yields commutativity.
2. Transpose $\mathbf{W}$:
   $$\mathbf{W}^T = \left( (\mathbf{I} + \mathbf{A})^{-1} \right)^T (\mathbf{I} - \mathbf{A})^T = (\mathbf{I} + \mathbf{A}^T)^{-1} (\mathbf{I} - \mathbf{A}^T) = (\mathbf{I} - \mathbf{A})^{-1} (\mathbf{I} + \mathbf{A})$$
3. Compute $\mathbf{W}^T \mathbf{W}$:
   $$\mathbf{W}^T \mathbf{W} = (\mathbf{I} - \mathbf{A})^{-1} (\mathbf{I} + \mathbf{A}) (\mathbf{I} - \mathbf{A}) (\mathbf{I} + \mathbf{A})^{-1} = (\mathbf{I} - \mathbf{A})^{-1} (\mathbf{I} - \mathbf{A}) (\mathbf{I} + \mathbf{A}) (\mathbf{I} + \mathbf{A})^{-1} = \mathbf{I} \cdot \mathbf{I} = \mathbf{I}$$

By optimizing unconstrained skew-symmetric parameters $\mathbf{A} \in \mathbb{R}^{d \times d}$ (setting $A_{j, i} = -A_{i, j}$ and $A_{i, i} = 0$) and computing $\mathbf{W} = \operatorname{Cayley}(\mathbf{A})$, the recurrent weight matrix is **algebraically guaranteed to remain exactly orthogonal** throughout every step of gradient descent, eliminating exploding and vanishing gradients by construction. $\blacksquare$

---

## 3. Geometric & Algebraic Interpretation

### The Eigenspectrum on the Complex Unit Circle

Consider the eigendecomposition of the recurrent matrix $\mathbf{W}_{hh} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}$, where $\mathbf{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_{d_h})$ with $\lambda_i \in \mathbb{C}$:

$$\mathbf{W}_{hh}^k = \mathbf{Q} \begin{bmatrix} \lambda_1^k & & 0 \\ & \ddots & \\ 0 & & \lambda_{d_h}^k \end{bmatrix} \mathbf{Q}^{-1}$$

The complex plane partitions the dynamical behavior into three distinct regimes relative to the **unit circle** $\{z \in \mathbb{C} : |z| = 1\}$:
1. **Inside the Unit Circle ($|\lambda_i| < 1$):**
   $\lambda_i^k \to 0$ as $k \to \infty$. Defines the *stable subspace*. Gradients in this subspace vanish exponentially.
2. **Outside the Unit Circle ($|\lambda_i| > 1$):**
   $\lambda_i^k \to \infty$ as $k \to \infty$. Defines the *unstable subspace*. Gradients in this subspace explode exponentially.
3. **On the Unit Circle ($|\lambda_i| = 1$):**
   $|\lambda_i^k| = 1$ for all $k$. Defines the *center manifold*. Memory is preserved without amplification or decay (pure rotation in complex space).

```
                 COMPLEX EIGENSPECTRUM PLANE
                           Im(λ)
                             ▲
                             │   * (Unstable: |λ| > 1 -> Explodes)
                         ┌───┼───┐
                       ┌─┘   │   └─┐
                      ┌┘     │     └┐
                     ─┼──────┼──────┼─► Re(λ)
                      └┐  *  │     ┌┘
                       └─┐(Stable)─┘
                         └───┼───┘
                             │  Unit Circle (|λ| = 1)
```

---

## 4. Real-World Analogy

### The Public Address Feedback Loop vs. The Whispering Gallery
- **Exploding Gradient (The Acoustic Feedback Loop):**
  A singer accidentally points a live microphone at an amplifier speaker. The sound enters the mic, gets amplified by factor $G > 1$, exits the speaker, re-enters the mic, and multiplies again: $G \to G^2 \to G^3 \dots$ Within a second, the sound system produces a deafening, system-destroying acoustic screech (`NaN`). Gradient clipping is the electronic limiter that instantly clamps the audio signal before the speakers blow up.
- **Vanishing Gradient (The Telephone Game):**
  A message is whispered through a line of 100 people. Each person speaks at $90\%$ of the volume of the person before them ($0.9$). By person 50, the volume is $0.9^{50} \approx 0.005$; the original signal has dropped below the threshold of human hearing. No matter how important the first person's message was, the 100th person hears only silence.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace concrete numerical walkthroughs of both vanishing/exploding dynamics and gradient clipping arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Gradient Stability |
| :--- | :--- | :--- | :--- |
| $T$ | Temporal Horizon | Scalar ($4$) | Number of time steps across which gradients backpropagate |
| $w$ | Scalar Recurrent Weight | Scalar | $1$D recurrent transition parameter ($h_t = \tanh(w \cdot h_{t-1})$) |
| $J_k$ | Temporal Jacobian | Scalar | $J_k = (1 - h_k^2) \cdot w$ |
| $\frac{\partial h_4}{\partial h_0}$ | Accumulated Gradient | Scalar | $\prod_{k=1}^4 J_k = \text{sensitivity of } h_4 \text{ w.r.t. } h_0$ |
| $\mathbf{g}$ | Unclipped Gradient Vector | $(2,)$ | Raw parameter gradient vector entering optimizer |
| $\|\mathbf{g}\|_2$ | Gradient Norm | Scalar | Euclidean magnitude $\sqrt{g_1^2 + g_2^2}$ |
| $\theta_{\text{clip}}$ | Clipping Threshold | Scalar ($2.0$) | Maximum permitted Euclidean norm |
| $\mathbf{g}_{\text{clipped}}$ | Clipped Gradient Vector | $(2,)$ | Rescaled gradient vector ($\theta \frac{\mathbf{g}}{\|\mathbf{g}\|}$) |

---

### 5.2 Concrete Toy Numbers: Vanishing vs. Exploding Tracing ($T=4$)

Assume inputs $\mathbf{x} = \mathbf{0}$ and initial state $h_0 = 0$, so that all hidden states remain at the origin:
$$h_t = 0 \implies \tanh'(0) = 1 - (0)^2 = 1.0, \quad \forall t$$
Thus the Jacobian at each step simplifies to $J_k = 1.0 \times w = w$.

The gradient backpropagated across 4 time steps is:
$$\frac{\partial h_4}{\partial h_0} = \prod_{k=1}^4 J_k = w^4$$

#### Case A: Sub-unitary Weight $w = 0.5$ (Vanishing Regime)
- Step 4 to 3: $J_4 = 0.5$
- Step 4 to 2: $J_4 \cdot J_3 = (0.5)^2 = \mathbf{0.25}$
- Step 4 to 1: $J_4 \cdot J_3 \cdot J_2 = (0.5)^3 = \mathbf{0.125}$
- Step 4 to 0: $J_4 \cdot J_3 \cdot J_2 \cdot J_1 = (0.5)^4 = \mathbf{0.0625}$
Across just 4 steps, the gradient has decayed by **$93.75\%$**!
For $T = 20$: $(0.5)^{20} \approx 9.53 \times 10^{-7}$ (effective extinction).

#### Case B: Super-unitary Weight $w = 2.0$ (Exploding Regime)
- Step 4 to 3: $J_4 = 2.0$
- Step 4 to 2: $J_4 \cdot J_3 = (2.0)^2 = \mathbf{4.0}$
- Step 4 to 1: $J_4 \cdot J_3 \cdot J_2 = (2.0)^3 = \mathbf{8.0}$
- Step 4 to 0: $J_4 \cdot J_3 \cdot J_2 \cdot J_1 = (2.0)^4 = \mathbf{16.0}$
Across just 4 steps, the gradient has amplified by **$1,600\%$**!
For $T = 20$: $(2.0)^{20} \approx 1,048,576$ (massive gradient explosion).

---

### 5.3 Norm-based Gradient Clipping Arithmetic

Suppose backpropagation on a mini-batch yields an exploding parameter gradient vector:
$$\mathbf{g} = \begin{bmatrix} 6.0 \\ 8.0 \end{bmatrix}$$
Let the maximum allowed norm threshold be $\theta_{\text{clip}} = 2.0$.

1. **Compute Euclidean Norm $\|\mathbf{g}\|_2$:**
   $$\|\mathbf{g}\|_2 = \sqrt{(6.0)^2 + (8.0)^2} = \sqrt{36.0 + 64.0} = \sqrt{100.0} = \mathbf{10.0}$$

2. **Check Clipping Condition:**
   $$\|\mathbf{g}\|_2 = 10.0 > \theta_{\text{clip}} = 2.0 \implies \text{Condition triggered!}$$

3. **Compute Scaling Factor $\alpha$:**
   $$\alpha = \frac{\theta_{\text{clip}}}{\|\mathbf{g}\|_2} = \frac{2.0}{10.0} = \mathbf{0.2}$$

4. **Compute Clipped Gradient $\mathbf{g}_{\text{clipped}} = \alpha \mathbf{g}$:**
   $$g_{\text{clipped}, 1} = 0.2 \times 6.0 = \mathbf{1.2}$$
   $$g_{\text{clipped}, 2} = 0.2 \times 8.0 = \mathbf{1.6}$$
   $$\mathbf{g}_{\text{clipped}} = \begin{bmatrix} 1.2 \\ 1.6 \end{bmatrix}$$

5. **Verify Clipped Norm:**
   $$\|\mathbf{g}_{\text{clipped}}\|_2 = \sqrt{(1.2)^2 + (1.6)^2} = \sqrt{1.44 + 2.56} = \sqrt{4.00} = \mathbf{2.0} \equiv \theta_{\text{clip}}$$

The direction angle $\theta = \arctan(8/6) \approx 53.13^\circ$ is preserved exactly, while the magnitude is strictly restrained to $2.0$.

---

## 6. Solved Illustrations

### Illustration 1: Why ReLU in Recurrent Layers Causes Severe Catastrophic Explosion

**Problem:**
ReLU ($\sigma(z) = \max(0, z)$) revolutionized feedforward ConvNets by solving vanishing gradients. Why does replacing $\tanh$ with standard ReLU in a Vanilla RNN frequently cause catastrophic numerical explosion?

**Solution:**
For $\tanh(z)$, the derivative satisfies $|\tanh'(z)| \le 1$, and as activations grow large ($|z| \gg 1$), $\tanh'(z) \to 0$, providing a natural saturating "brake" against explosive growth.
For ReLU:
$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
Whenever neurons are active ($z > 0$), $\sigma'(z) \equiv 1.0$.
Thus:
$$\mathbf{J}_k = \mathbf{I} \cdot \mathbf{W}_{hh} = \mathbf{W}_{hh}$$
If $\sigma_{\max}(\mathbf{W}_{hh}) > 1$, there is **zero saturation damping**. Activations in the forward pass grow without bound ($h_t \sim \mathcal{O}(\lambda^t)$), and gradients in the backward pass explode without bound.
*(To use ReLU safely in an RNN, Quoc Le et al., 2015 introduced the IRNN, requiring $\mathbf{W}_{hh}$ to be initialized strictly to the Identity matrix $\mathbf{I}$ with tiny learning rates).*

---

### Illustration 2: Value Clipping vs. Norm Clipping Directional Distortion

**Problem:**
Let $\mathbf{g} = [100.0, 1.0]^T$.
Compare the resulting vector under:
1. Element-wise value clipping with bounds $[-5.0, 5.0]$.
2. Norm clipping with threshold $\theta = 5.0$.
Compute the cosine similarity of each clipped vector with the original gradient $\mathbf{g}$.

**Solution:**
1. **Original Direction:**
   $$\|\mathbf{g}\| = \sqrt{100^2 + 1^2} \approx 100.005$$
   $$\mathbf{u}_{\text{orig}} = \frac{\mathbf{g}}{\|\mathbf{g}\|} \approx [0.99995, 0.0099995]^T$$

2. **Norm Clipping ($\theta = 5.0$):**
   $$\mathbf{g}_{\text{norm}} = 5.0 \frac{\mathbf{g}}{\|\mathbf{g}\|} \approx [4.99975, 0.0499975]^T$$
   $$\text{Cosine Similarity} = \frac{\mathbf{g}^T \mathbf{g}_{\text{norm}}}{\|\mathbf{g}\| \|\mathbf{g}_{\text{norm}}\|} = \mathbf{1.0000} \quad (\text{Perfect preservation})$$

3. **Value Clipping (clamp between $[-5, 5]$):**
   $$\mathbf{g}_{\text{value}} = [\text{clip}(100, -5, 5), \text{clip}(1, -5, 5)]^T = [5.0, 1.0]^T$$
   $$\|\mathbf{g}_{\text{value}}\| = \sqrt{5^2 + 1^2} = \sqrt{26} \approx 5.099$$
   $$\text{Cosine Similarity} = \frac{100(5) + 1(1)}{100.005 \times 5.099} = \frac{501}{509.925} \approx \mathbf{0.9825}$$
   Angle of deflection: $\theta = \arccos(0.9825) \approx 10.74^\circ$.

Value clipping shifted the direction by over $10^\circ$, penalizing large components while leaving small components unscaled. In high dimensions ($d = 10,000$), value clipping can rotate the gradient by up to $80^\circ$ away from the true steepest descent direction!

---

### Illustration 3: Eigenspectrum and Singular Value Evolution Across 10 Time Steps

**Problem:**
Consider an RNN with a $2 \times 2$ diagonal recurrent weight matrix:
$$\mathbf{W}_{hh} = \begin{bmatrix} 1.2 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}$$
Operating in the linear regime ($\mathbf{D}_k = \mathbf{I}$).
1. Compute the matrix powers $\mathbf{W}_{hh}^t$ for $t \in \{1, 2, 5, 10\}$.
2. Trace the transformation of two orthogonal unit vectors: $\mathbf{v}_1 = [1, 0]^T$ and $\mathbf{v}_2 = [0, 1]^T$.
3. Compute the condition number $\kappa(\mathbf{W}_{hh}^t) = \frac{\sigma_{\max}(\mathbf{W}_{hh}^t)}{\sigma_{\min}(\mathbf{W}_{hh}^t)}$ at each step and explain the geometric distortion of the error surface.

**Solution:**

#### Step 1: Matrix Powers
Since $\mathbf{W}_{hh}$ is diagonal, $\mathbf{W}_{hh}^t = \begin{bmatrix} (1.2)^t & 0 \\ 0 & (0.5)^t \end{bmatrix}$:
- **$t = 1$:** $\mathbf{W}_{hh}^1 = \begin{bmatrix} 1.2 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}$
- **$t = 2$:** $\mathbf{W}_{hh}^2 = \begin{bmatrix} (1.2)^2 & 0 \\ 0 & (0.5)^2 \end{bmatrix} = \begin{bmatrix} 1.44 & 0.0 \\ 0.0 & 0.25 \end{bmatrix}$
- **$t = 5$:** $\mathbf{W}_{hh}^5 = \begin{bmatrix} (1.2)^5 & 0 \\ 0 & (0.5)^5 \end{bmatrix} = \begin{bmatrix} 2.488320 & 0.0 \\ 0.0 & 0.031250 \end{bmatrix}$
- **$t = 10$:** $\mathbf{W}_{hh}^{10} = \begin{bmatrix} (1.2)^{10} & 0 \\ 0 & (0.5)^{10} \end{bmatrix} \approx \begin{bmatrix} 6.191736 & 0.0 \\ 0.0 & 0.000977 \end{bmatrix}$

#### Step 2: Vector Magnitudes Across Time
- Mode 1 ($\mathbf{v}_1 = [1, 0]^T$, unstable manifold):
  $$\|\mathbf{W}_{hh}^t \mathbf{v}_1\|_2 = (1.2)^t \implies \{1.2, \, 1.44, \, 2.488, \, 6.192\}$$
  Mode 1 **explodes** by $6.19\times$ over 10 steps.
- Mode 2 ($\mathbf{v}_2 = [0, 1]^T$, stable manifold):
  $$\|\mathbf{W}_{hh}^t \mathbf{v}_2\|_2 = (0.5)^t \implies \{0.5, \, 0.25, \, 0.03125, \, 0.000977\}$$
  Mode 2 **vanishes** by a factor of $1,024\times$ over 10 steps.

#### Step 3: Condition Number Evolution
$$\kappa(\mathbf{W}_{hh}^t) = \frac{(1.2)^t}{(0.5)^t} = (2.4)^t$$
- $t = 1$: $\kappa = 2.4$
- $t = 2$: $\kappa = (2.4)^2 = 5.76$
- $t = 5$: $\kappa = (2.4)^5 \approx 79.63$
- $t = 10$: $\kappa = (2.4)^{10} \approx \mathbf{6,340.34}$

**Geometric Implication:**
Across 10 steps, the error ellipsoid stretches by a factor of over $6,340$. Any gradient backpropagated across 10 steps will align almost strictly with the $x$-axis ($\mathbf{v}_1$), completely obliterating any learning signal in the $y$-direction ($\mathbf{v}_2$).

---

### Illustration 4: Multi-Tensor Norm-Based Gradient Clipping Arithmetic

**Problem:**
In PyTorch's `torch.nn.utils.clip_grad_norm_`, gradients across all parameter tensors are concatenated into a single global vector.
Consider an RNN with two trainable tensors:
1. Recurrent weight matrix $\mathbf{W} \in \mathbb{R}^{2 \times 2}$ with unclipped gradient:
   $$\mathbf{G}_W = \begin{bmatrix} 3.0 & -4.0 \\ 0.0 & 5.0 \end{bmatrix}$$
2. Bias vector $\mathbf{b} \in \mathbb{R}^2$ with unclipped gradient:
   $$\mathbf{g}_b = \begin{bmatrix} 2.0 \\ -2.0 \end{bmatrix}$$
Let the maximum global norm threshold be $\theta_{\text{clip}} = 4.0$.
1. Compute the global Frobenius/Euclidean norm $\|\mathbf{g}_{\text{total}}\|_2$.
2. Determine whether clipping triggers, and compute the global scaling factor $\alpha$.
3. Compute the clipped tensors $\mathbf{G}_W^{\text{clip}}$ and $\mathbf{g}_b^{\text{clip}}$.
4. Verify that the combined norm of the clipped tensors equals exactly $\theta_{\text{clip}} = 4.0$.

**Solution:**

#### Step 1: Global Norm Calculation
Sum the squared elements across all tensors:
$$\|\mathbf{G}_W\|_F^2 = (3.0)^2 + (-4.0)^2 + (0.0)^2 + (5.0)^2 = 9.0 + 16.0 + 0.0 + 25.0 = 50.0$$
$$\|\mathbf{g}_b\|_2^2 = (2.0)^2 + (-2.0)^2 = 4.0 + 4.0 = 8.0$$
$$\|\mathbf{g}_{\text{total}}\|_2 = \sqrt{\|\mathbf{G}_W\|_F^2 + \|\mathbf{g}_b\|_2^2} = \sqrt{50.0 + 8.0} = \sqrt{58.0} \approx \mathbf{7.615773}$$

#### Step 2: Clipping Factor
Since $\|\mathbf{g}_{\text{total}}\|_2 = 7.615773 > \theta_{\text{clip}} = 4.0$, clipping is triggered:
$$\alpha = \frac{\theta_{\text{clip}}}{\|\mathbf{g}_{\text{total}}\|_2} = \frac{4.0}{\sqrt{58.0}} \approx \mathbf{0.525226}$$

#### Step 3: Rescale Tensors
- **Clipped Weight Gradient:**
  $$\mathbf{G}_W^{\text{clip}} = \alpha \mathbf{G}_W = 0.525226 \begin{bmatrix} 3.0 & -4.0 \\ 0.0 & 5.0 \end{bmatrix} = \begin{bmatrix} \mathbf{1.575677} & \mathbf{-2.100903} \\ \mathbf{0.000000} & \mathbf{2.626129} \end{bmatrix}$$
- **Clipped Bias Gradient:**
  $$\mathbf{g}_b^{\text{clip}} = \alpha \mathbf{g}_b = 0.525226 \begin{bmatrix} 2.0 \\ -2.0 \end{bmatrix} = \begin{bmatrix} \mathbf{1.050451} \\ \mathbf{-1.050451} \end{bmatrix}$$

#### Step 4: Verification of Clipped Global Norm
$$\|\mathbf{G}_W^{\text{clip}}\|_F^2 = (1.575677)^2 + (-2.100903)^2 + 0^2 + (2.626129)^2 \approx 2.482759 + 4.413793 + 6.896552 = 13.793104$$
$$\|\mathbf{g}_b^{\text{clip}}\|_2^2 = (1.050451)^2 + (-1.050451)^2 \approx 1.103448 + 1.103448 = 2.206896$$
$$\text{Total Squared Norm} = 13.793104 + 2.206896 = \mathbf{16.000000}$$
$$\|\mathbf{g}_{\text{total}}^{\text{clip}}\|_2 = \sqrt{16.000000} = \mathbf{4.000000} \equiv \theta_{\text{clip}}$$
The global norm is constrained to exactly $4.0$ while preserving the relative proportions between weights and biases.

---

### Illustration 5: Cayley Transform for Exact Orthogonal Weight Transition and Norm Preservation

**Problem:**
Let an unconstrained skew-symmetric matrix parameter $\mathbf{A} \in \mathfrak{so}(2)$ be:
$$\mathbf{A} = \begin{bmatrix} 0.0 & -0.5 \\ 0.5 & 0.0 \end{bmatrix}$$
1. Compute the Cayley transform $\mathbf{W} = (\mathbf{I} - \mathbf{A}) (\mathbf{I} + \mathbf{A})^{-1}$.
2. Verify algebraically that $\mathbf{W}^T \mathbf{W} = \mathbf{I}$.
3. For a test gradient vector $\delta = [3.0, 4.0]^T$, verify that $\|\mathbf{W} \delta\|_2 = \|\delta\|_2 = 5.0$, proving zero vanishing and zero exploding gradient flow.

**Solution:**

#### Step 1: Cayley Transform Computation
$$\mathbf{I} - \mathbf{A} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} - \begin{bmatrix} 0.0 & -0.5 \\ 0.5 & 0.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.5 \\ -0.5 & 1.0 \end{bmatrix}$$
$$\mathbf{I} + \mathbf{A} = \begin{bmatrix} 1.0 & -0.5 \\ 0.5 & 1.0 \end{bmatrix}$$

Determinant of $(\mathbf{I} + \mathbf{A})$:
$$\det(\mathbf{I} + \mathbf{A}) = (1.0)(1.0) - (-0.5)(0.5) = 1.0 + 0.25 = 1.25 = \frac{5}{4}$$

Inverse $(\mathbf{I} + \mathbf{A})^{-1}$:
$$(\mathbf{I} + \mathbf{A})^{-1} = \frac{1}{1.25} \begin{bmatrix} 1.0 & 0.5 \\ -0.5 & 1.0 \end{bmatrix} = 0.8 \begin{bmatrix} 1.0 & 0.5 \\ -0.5 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.8 & 0.4 \\ -0.4 & 0.8 \end{bmatrix}$$

Matrix Multiplication $\mathbf{W} = (\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})^{-1}$:
$$\mathbf{W} = \begin{bmatrix} 1.0 & 0.5 \\ -0.5 & 1.0 \end{bmatrix} \begin{bmatrix} 0.8 & 0.4 \\ -0.4 & 0.8 \end{bmatrix}$$
- $W_{1, 1} = (1.0)(0.8) + (0.5)(-0.4) = 0.8 - 0.2 = \mathbf{0.6}$
- $W_{1, 2} = (1.0)(0.4) + (0.5)(0.8) = 0.4 + 0.4 = \mathbf{0.8}$
- $W_{2, 1} = (-0.5)(0.8) + (1.0)(-0.4) = -0.4 - 0.4 = \mathbf{-0.8}$
- $W_{2, 2} = (-0.5)(0.4) + (1.0)(0.8) = -0.2 + 0.8 = \mathbf{0.6}$

$$\mathbf{W} = \begin{bmatrix} 0.6 & 0.8 \\ -0.8 & 0.6 \end{bmatrix}$$

#### Step 2: Verification of Orthogonality
$$\mathbf{W}^T \mathbf{W} = \begin{bmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{bmatrix} \begin{bmatrix} 0.6 & 0.8 \\ -0.8 & 0.6 \end{bmatrix} = \begin{bmatrix} 0.36 + 0.64 & 0.48 - 0.48 \\ 0.48 - 0.48 & 0.64 + 0.36 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} = \mathbf{I}$$

#### Step 3: Norm Preservation Check
For $\delta = [3.0, 4.0]^T$:
$$\|\delta\|_2 = \sqrt{3.0^2 + 4.0^2} = \sqrt{9.0 + 16.0} = \sqrt{25.0} = \mathbf{5.0}$$
Transforming $\delta$:
$$\mathbf{W} \delta = \begin{bmatrix} 0.6 & 0.8 \\ -0.8 & 0.6 \end{bmatrix} \begin{bmatrix} 3.0 \\ 4.0 \end{bmatrix} = \begin{bmatrix} 0.6(3.0) + 0.8(4.0) \\ -0.8(3.0) + 0.6(4.0) \end{bmatrix} = \begin{bmatrix} 1.8 + 3.2 \\ -2.4 + 2.4 \end{bmatrix} = \begin{bmatrix} 5.0 \\ 0.0 \end{bmatrix}$$
$$\|\mathbf{W} \delta\|_2 = \sqrt{5.0^2 + 0.0^2} = \mathbf{5.0}$$

The gradient vector norm is preserved to machine precision without scaling distortion.

---

## 7. Deep Learning Connection & Application

### 1. The Historical Catalyst for Gated Networks (LSTM & GRU)
The mathematical impossibility of training Vanilla RNNs across long horizons directly drove Sepp Hochreiter and Jürgen Schmidhuber (1997) to invent the **Long Short-Term Memory (LSTM)** network.
The core architectural insight of LSTM is to replace the multiplicative Jacobian chain:
$$\prod_{k=t+1}^T \mathbf{D}_k \mathbf{W}_{hh}$$
with an additive **Constant Error Carousel (CEC)**:
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
where the cell state $\mathbf{C}_t$ updates via linear addition. When the forget gate $\mathbf{f}_t \approx \mathbf{1}$, the Jacobian $\frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_t} \approx \mathbf{I}$, creating an unimpeded gradient superhighway spanning hundreds of time steps.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. Pure NumPy simulations comparing gradient norm decay ($w < 1$) and explosion ($w > 1$) across sequence lengths $T \in [2, 5, 10, 20, 50]$.
2. Exact numerical verification of the Part 5 Visual Grid hand arithmetic and gradient clipping equations.
3. Scratch implementation of Pascanu's Norm-based Gradient Clipping compared against PyTorch's `torch.nn.utils.clip_grad_norm_`.
4. Long-term gradient preservation experiment comparing Random Gaussian Initialization vs. Orthogonal Initialization.

See implementation in:
[`08_recurrent_networks/code/02_exploding_and_vanishing_gradients.py`](./code/02_exploding_and_vanishing_gradients.py)
