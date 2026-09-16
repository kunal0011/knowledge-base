# 6.1 The Perceptron, Linear Models & Perceptron Convergence Theorem

---

## Part 1: Intuition & 101 Motivation

At the core of every modern neural network—from a tiny multilayer perceptron to a trillion-parameter Large Language Model—lies an elementary computational unit: the **artificial neuron**.

The historical origins of deep learning trace back to 1943, when neurophysiologist Warren McCulloch and logician Walter Pitts proposed a mathematical abstraction of a biological brain cell: an electrical circuit that sums weighted electrical inputs from incoming synapses, compares the sum against a threshold, and fires an all-or-nothing binary action potential.

In 1958, Frank Rosenblatt at the Cornell Aeronautical Laboratory transformed this concept into a learning machine called the **Perceptron**. Rosenblatt built a custom physical machine—the Mark I Perceptron—fitted with an array of 400 cadmium-sulfide photocells and motorized potentiometers that mechanically turned dials to adjust weights.

```
Biological Neuron                    Artificial Perceptron (Rosenblatt 1958)
=================                    ======================================
Dendrites (Inputs)       ------>     Input Vector: x = [x_1, x_2, ..., x_d]^T
Synaptic Strengths       ------>     Synaptic Weights: w = [w_1, w_2, ..., w_d]^T
Soma (Cell Body Sum)     ------>     Linear Sum: z = w^T x + b = \sum w_i x_i + b
Axon Hillock (Threshold) ------>     Activation: y = sign(z) \in {-1, +1}
Axon (Output Pulse)      ------>     Binary Decision Output
```

The fundamental geometric goal of the perceptron is **binary linear classification**: given a dataset of points belonging to two distinct classes (labeled $+1$ and $-1$), find an affine hyperplane that slices through the feature space such that all positive points lie on one side and all negative points lie on the other.

In 1962, Albert Novikoff proved the celebrated **Perceptron Convergence Theorem**: if a dataset is linearly separable by any margin $\gamma > 0$, Rosenblatt's algorithm is mathematically guaranteed to find a separating hyperplane in a **finite number of steps**, regardless of how many training examples exist or what initial weights are used!

However, in 1969, Marvin Minsky and Seymour Papert published their landmark book *Perceptrons*, presenting a rigorous mathematical proof that a single perceptron cannot even compute the simple **XOR (exclusive-OR)** logical function. This finding dealt a catastrophic blow to the field, triggering the first "AI Winter." Decades later, researchers realized that stacking perceptrons into hierarchical layers with non-linear continuous activations completely dissolves the XOR barrier, giving birth to modern Deep Learning.

---

## Part 2: Rigorous Mathematical Formulation

### 1. The Perceptron Model

Let $x \in \mathbb{R}^d$ be an input feature vector, and let $y \in \{-1, +1\}$ be the binary class label.
The perceptron hypothesis $h_{w, b}: \mathbb{R}^d \to \{-1, +1\}$ is defined as:
$$h_{w, b}(x) = \text{sign}(w^T x + b) = \begin{cases} +1 & \text{if } w^T x + b \ge 0 \\ -1 & \text{if } w^T x + b < 0 \end{cases}$$
where $w \in \mathbb{R}^d$ is the weight vector and $b \in \mathbb{R}$ is the bias scalar.

#### Homogeneous Coordinate Representation
To simplify the algebra, we absorb the bias $b$ directly into the weight vector by appending a constant dummy feature $x_0 = 1$:
$$\tilde{x} = \begin{bmatrix} 1 \\ x_1 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^{d+1}, \quad \tilde{w} = \begin{bmatrix} b \\ w_1 \\ \vdots \\ w_d \end{bmatrix} \in \mathbb{R}^{d+1}$$
Then:
$$w^T x + b = \tilde{w}^T \tilde{x}$$
Henceforth, without loss of generality, we work in homogeneous coordinates and omit the tilde notation: $h_w(x) = \text{sign}(w^T x)$.

---

### 2. The Perceptron Learning Algorithm (PLA)

Given a training dataset $\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$:

1. **Initialization**: Set $w_0 = \vec{0} \in \mathbb{R}^d$.
2. **Online Iteration**: Iterate through the training examples. For sample $(x_i, y_i)$:
   - Compute prediction: $\hat{y}_i = \text{sign}(w_t^T x_i)$.
   - A classification mistake occurs if and only if:
     $$y_i (w_t^T x_i) \le 0$$
   - **Update Rule**: If a mistake occurs, update the weight vector:
     $$w_{t+1} = w_t + \eta y_i x_i$$
     where $\eta > 0$ is the learning rate (typically $\eta = 1$). If the prediction is correct ($y_i (w_t^T x_i) > 0$), do not modify $w$: $w_{t+1} = w_t$.
3. **Termination**: Repeat until all $N$ training examples are classified correctly with zero mistakes.

---

### 3. Novikoff's Perceptron Convergence Theorem (1962)

#### Theorem Statement
Let $\mathcal{D} = \{(x_1, y_1), \dots, (x_N, y_N)\}$ be a sequence of training examples. Suppose:
1. **Linear Separability**: There exists an optimal unit weight vector $w^* \in \mathbb{R}^d$ ($\|w^*\|_2 = 1$) and a strictly positive geometric margin $\gamma > 0$ such that:
   $$y_i (w^{*T} x_i) \ge \gamma \quad \forall i \in \{1, \dots, N\}$$
2. **Bounded Domain**: All input vectors lie within a hypersphere of radius $R > 0$:
   $$\|x_i\|_2 \le R \quad \forall i \in \{1, \dots, N\}$$

Then, starting from $w_0 = \vec{0}$ with learning rate $\eta = 1$, the Perceptron Learning Algorithm makes a total number of mistakes $k$ bounded by:
$$k \le \left( \frac{R}{\gamma} \right)^2$$

---

#### Step-by-Step Rigorous Proof

Let $w_k$ denote the weight vector after $k$ mistakes have occurred. Let $(x_{i_k}, y_{i_k})$ be the training sample on which the $k$-th mistake was made. By definition of a mistake:
$$y_{i_k} (w_{k-1}^T x_{i_k}) \le 0$$
and the update rule is:
$$w_k = w_{k-1} + y_{i_k} x_{i_k}$$

##### Step 1: Lower Bounding the Inner Product $w_k^T w^*$
Take the inner product of $w_k$ with the optimal unit vector $w^*$:
$$w_k^T w^* = (w_{k-1} + y_{i_k} x_{i_k})^T w^* = w_{k-1}^T w^* + y_{i_k} (w^{*T} x_{i_k})$$

By the linear separability assumption, $y_{i_k} (w^{*T} x_{i_k}) \ge \gamma$. Therefore:
$$w_k^T w^* \ge w_{k-1}^T w^* + \gamma$$

Applying this inequality recursively from $w_0 = \vec{0}$:
$$w_k^T w^* \ge w_0^T w^* + k \gamma = 0 + k \gamma = k \gamma$$
$$\mathbf{w_k^T w^* \ge k \gamma} \quad \text{(Inequality 1)}$$

##### Step 2: Upper Bounding the Squared Norm $\|w_k\|_2^2$
Compute the squared Euclidean norm of $w_k$:
$$\|w_k\|_2^2 = \|w_{k-1} + y_{i_k} x_{i_k}\|_2^2 = \|w_{k-1}\|_2^2 + 2 y_{i_k} (w_{k-1}^T x_{i_k}) + y_{i_k}^2 \|x_{i_k}\|_2^2$$

Since $y_{i_k} \in \{-1, +1\}$, we have $y_{i_k}^2 = 1$.
Because a mistake occurred, the cross-term satisfies $y_{i_k} (w_{k-1}^T x_{i_k}) \le 0$.
By the bounded domain assumption, $\|x_{i_k}\|_2^2 \le R^2$.

Substituting these bounds:
$$\|w_k\|_2^2 \le \|w_{k-1}\|_2^2 + 0 + R^2 = \|w_{k-1}\|_2^2 + R^2$$

Applying this recurrence from $w_0 = \vec{0}$:
$$\|w_k\|_2^2 \le \|w_0\|_2^2 + k R^2 = 0 + k R^2 = k R^2$$
$$\mathbf{\|w_k\|_2 \le \sqrt{k} R} \quad \text{(Inequality 2)}$$

##### Step 3: Combining via the Cauchy-Schwarz Inequality
By the Cauchy-Schwarz inequality, for any two vectors:
$$w_k^T w^* \le \|w_k\|_2 \|w^*\|_2$$

Substitute Inequality 1 for the left-hand side, Inequality 2 for $\|w_k\|_2$, and the fact that $\|w^*\|_2 = 1$:
$$k \gamma \le w_k^T w^* \le (\sqrt{k} R) \cdot 1 = \sqrt{k} R$$

Dividing both sides by $\sqrt{k} > 0$:
$$\sqrt{k} \gamma \le R$$

Squaring both sides:
$$k \gamma^2 \le R^2 \implies \mathbf{k \le \left( \frac{R}{\gamma} \right)^2}$$

$\blacksquare$ **Q.E.D.**

#### Profound Implications of the Theorem
1. **Guaranteed Finite Halting**: The algorithm never loops infinitely on linearly separable data. It stops in at most $\lfloor R^2 / \gamma^2 \rfloor$ steps.
2. **Dimension Independence**: The mistake bound depends *only* on the maximum data radius $R$ and the geometric margin $\gamma$, **independent of the dimension $d$ of the feature space!** Even in a 1,000,000-dimensional space, if the margin $\gamma$ is large, the perceptron converges quickly.
3. **Invariance to Learning Rate $\eta$**: Multiplying $\eta$ scales $w_k$ by $\eta$, but does not alter the decision boundary $\{x : w^T x = 0\}$.

---

### 4. Deep Mathematical Derivations

#### Deep Derivation 6.1.1: The Perceptron Criterion Loss Function, Clarke Subdifferential, and Stochastic Subgradient Descent Equivalence

We prove that Rosenblatt's Perceptron Learning Algorithm (PLA) is mathematically equivalent to Stochastic Subgradient Descent (SGD) minimizing a convex surrogate loss function known as the **Perceptron Criterion**.

**1. Formulation of the Perceptron Loss**:
In binary classification with labels $y_i \in \{-1, +1\}$, the ideal empirical risk is the 0-1 classification error:
$$\mathcal{L}_{0-1}(w) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i (w^T x_i) \le 0)$$
Because the indicator function $\mathbb{I}(\cdot)$ is discontinuous with zero gradient almost everywhere, it cannot be optimized by gradient methods.
The **Perceptron Criterion** (Bishop, 2006) replaces this with a continuous, piece-wise linear convex surrogate loss:
$$\ell_{\text{perc}}(w; x_i, y_i) = \max\left(0, -y_i (w^T x_i)\right)$$
Notice the behavior:
- If sample $i$ is correctly classified ($y_i (w^T x_i) > 0$), the penalty is $\ell_{\text{perc}} = 0$.
- If sample $i$ is misclassified ($y_i (w^T x_i) \le 0$), the penalty is $-y_i (w^T x_i) \ge 0$, which increases linearly with the magnitude of the error.
The total empirical risk is:
$$\mathcal{L}_{\text{perc}}(w) = \frac{1}{N} \sum_{i=1}^N \max\left(0, -y_i (w^T x_i)\right)$$

**2. Convexity of the Loss Function**:
Let $h(z) = \max(0, -z)$. For any $z_1, z_2 \in \mathbb{R}$ and $\lambda \in [0, 1]$:
$$h(\lambda z_1 + (1-\lambda)z_2) = \max(0, -\lambda z_1 - (1-\lambda)z_2) \le \lambda \max(0, -z_1) + (1-\lambda) \max(0, -z_2)$$
Hence $h(z)$ is convex. Since $z(w) = y_i w^T x_i$ is an affine function of $w$, the composition $f_i(w) = h(z(w))$ is convex. Since the sum of convex functions is convex, $\mathcal{L}_{\text{perc}}(w)$ is a convex function on $\mathbb{R}^d$.

**3. Clarke Subdifferential Calculus**:
Because $\ell_{\text{perc}}$ has a "kink" at $y_i w^T x_i = 0$, it is non-differentiable at the decision boundary. We compute its **Clarke subdifferential** $\partial \ell_{\text{perc}}(w)$.
For a scalar function $h(z) = \max(0, -z)$:
$$\partial h(z) = \begin{cases} \{0\} & \text{if } z > 0 \\ [-1, 0] & \text{if } z = 0 \\ \{-1\} & \text{if } z < 0 \end{cases}$$

By the chain rule for subdifferentials of convex functions composed with affine maps $\nabla_w z = y_i x_i$:
$$\partial_w \ell_{\text{perc}}(w; x_i, y_i) = \left\{ c \cdot (-y_i x_i) : c \in \partial h(y_i w^T x_i) \right\}$$
Evaluating this piece-wise:
$$\partial_w \ell_{\text{perc}}(w; x_i, y_i) = \begin{cases} \{\vec{0}\} & \text{if } y_i w^T x_i > 0 \quad (\text{Correctly classified}) \\ \text{conv}\left(\{\vec{0}, -y_i x_i\}\right) & \text{if } y_i w^T x_i = 0 \quad (\text{Boundary case}) \\ \{-y_i x_i\} & \text{if } y_i w^T x_i < 0 \quad (\text{Misclassified}) \end{cases}$$
where $\text{conv}(A)$ denotes the convex hull.

**4. Equivalence to the Perceptron Learning Algorithm**:
Consider online Stochastic Subgradient Descent (SGD) on $\mathcal{L}_{\text{perc}}(w)$ with step size $\eta > 0$.
At step $t$, we observe a single sample $(x_i, y_i)$ and select a subgradient $g_t \in \partial_w \ell_{\text{perc}}(w_t; x_i, y_i)$:
$$w_{t+1} = w_t - \eta g_t$$
- **Case 1 (Correct Classification: $y_i w_t^T x_i > 0$)**:
  The subdifferential is the singleton $\{\vec{0}\}$. Thus $g_t = \vec{0}$, yielding:
  $$w_{t+1} = w_t - \eta(\vec{0}) = w_t \quad (\text{No update})$$
- **Case 2 (Misclassification or Boundary: $y_i w_t^T x_i \le 0$)**:
  Choosing the active subgradient $g_t = -y_i x_i$:
  $$w_{t+1} = w_t - \eta (-y_i x_i) = w_t + \eta y_i x_i$$

Setting $\eta = 1$, this is **identically Rosenblatt's Perceptron Learning Algorithm**.
*Theoretical Conclusion*: Rosenblatt's 1958 heuristic bio-inspired rule is fundamentally a Stochastic Subgradient Descent method minimizing the convex Perceptron Criterion!

---

#### Deep Derivation 6.1.2: Dimension-Free Generalization and Rademacher Complexity Bounds for Margin Perceptrons

A central mystery of early machine learning was why linear perceptrons with large margins do not overfit in high-dimensional spaces ($d \gg N$). We prove a dimension-free generalization error bound using **Empirical Rademacher Complexity**.

**1. Formal Setup**:
- Let $\mathcal{X} = \{x \in \mathbb{R}^d : \|x\|_2 \le R\}$ be a bounded input domain.
- Consider the class of linear score functions with bounded $\ell_2$ norm:
  $$\mathcal{F} = \{x \mapsto w^T x : \|w\|_2 \le 1\}$$
- Given a sample $S = \{x_1, \dots, x_N\} \subset \mathcal{X}$, the **Empirical Rademacher Complexity** $\hat{\mathcal{R}}_S(\mathcal{F})$ measures the capacity of $\mathcal{F}$ to fit random noise:
  $$\hat{\mathcal{R}}_S(\mathcal{F}) = \mathbb{E}_{\sigma}\left[ \sup_{\|w\|_2 \le 1} \frac{1}{N} \sum_{i=1}^N \sigma_i (w^T x_i) \right]$$
  where $\sigma_1, \dots, \sigma_N$ are independent Rademacher random variables taking values in $\{-1, +1\}$ with equal probability $1/2$.

**2. Bounding the Rademacher Complexity**:
Rewrite the inner sum as a vector dot product:
$$\sum_{i=1}^N \sigma_i (w^T x_i) = w^T \left( \sum_{i=1}^N \sigma_i x_i \right)$$
By the Cauchy-Schwarz inequality, the supremum over $\|w\|_2 \le 1$ is achieved when $w$ is aligned with the vector $\sum_{i=1}^N \sigma_i x_i$:
$$\sup_{\|w\|_2 \le 1} w^T \left( \sum_{i=1}^N \sigma_i x_i \right) = \left\| \sum_{i=1}^N \sigma_i x_i \right\|_2$$
Therefore:
$$\hat{\mathcal{R}}_S(\mathcal{F}) = \frac{1}{N} \mathbb{E}_{\sigma}\left[ \left\| \sum_{i=1}^N \sigma_i x_i \right\|_2 \right]$$

Applying Jensen's Inequality to the concave function $g(u) = \sqrt{u}$:
$$\mathbb{E}_{\sigma}\left[ \left\| \sum_{i=1}^N \sigma_i x_i \right\|_2 \right] \le \sqrt{\mathbb{E}_{\sigma}\left[ \left\| \sum_{i=1}^N \sigma_i x_i \right\|_2^2 \right]}$$

Expanding the squared Euclidean norm:
$$\left\| \sum_{i=1}^N \sigma_i x_i \right\|_2^2 = \sum_{i=1}^N \sum_{j=1}^N \sigma_i \sigma_j (x_i^T x_j) = \sum_{i=1}^N \sigma_i^2 \|x_i\|_2^2 + \sum_{i \ne j} \sigma_i \sigma_j (x_i^T x_j)$$
Since $\sigma_i^2 = 1$ and $\mathbb{E}[\sigma_i \sigma_j] = \mathbb{E}[\sigma_i]\mathbb{E}[\sigma_j] = 0$ for $i \ne j$:
$$\mathbb{E}_{\sigma}\left[ \left\| \sum_{i=1}^N \sigma_i x_i \right\|_2^2 \right] = \sum_{i=1}^N \|x_i\|_2^2 \le N R^2$$

Taking the square root and dividing by $N$:
$$\hat{\mathcal{R}}_S(\mathcal{F}) \le \frac{1}{N} \sqrt{N R^2} = \frac{R}{\sqrt{N}}$$

**3. Generalization Bound with Geometric Margin $\gamma$**:
Define the margin loss function $\phi_\gamma: \mathbb{R} \to [0, 1]$:
$$\phi_\gamma(u) = \begin{cases} 1 & \text{if } u \le 0 \\ 1 - u/\gamma & \text{if } 0 < u < \gamma \\ 0 & \text{if } u \ge \gamma \end{cases}$$
Notice that $\mathbb{I}(u \le 0) \le \phi_\gamma(u)$, and $\phi_\gamma$ is $\frac{1}{\gamma}$-Lipschitz continuous.
By the **Talagrand Contraction Lemma**, the Rademacher complexity of the composed function class satisfies:
$$\hat{\mathcal{R}}_S(\phi_\gamma \circ \mathcal{F}) \le \frac{1}{\gamma} \hat{\mathcal{R}}_S(\mathcal{F}) \le \frac{R}{\gamma \sqrt{N}}$$

Applying the standard Rademacher generalization theorem (Bartlett & Mendelson, 2002): with probability at least $1 - \delta$ over the choice of training sample $S$, the true 0-1 generalization error $R(w) = \mathbb{P}(y \ne \text{sign}(w^T x))$ satisfies:
$$R(w) \le \frac{1}{N} \sum_{i=1}^N \phi_\gamma(y_i w^T x_i) + \frac{2 R}{\gamma \sqrt{N}} + \sqrt{\frac{\log(1/\delta)}{2N}}$$

If the training data is separated with margin $\gamma$, the empirical margin loss is zero: $\frac{1}{N} \sum_{i=1}^N \phi_\gamma(y_i w^T x_i) = 0$.
Hence:
$$R(w) \le \mathcal{O}\left( \frac{R}{\gamma \sqrt{N}} \right)$$

**Significance**:
The generalization error depends strictly on the scale ratio $\frac{R}{\gamma}$ and the sample size $N$. It has **zero explicit dependence on the dimension $d$**! A linear model operating in infinite dimensions (e.g., reproducing kernel Hilbert spaces) generalizes reliably as long as the geometric margin $\gamma$ remains bounded away from zero.

---

#### Deep Derivation 6.1.3: Dual Representation, the Kernel Trick, and the Non-Linear Kernel Perceptron

We derive the dual representation of the perceptron weight vector and demonstrate how the Kernel Trick resolves non-linear classification problems (such as XOR) without expanding the explicit coordinate representation.

**1. The Dual Representation**:
The perceptron learning algorithm initializes $w_0 = \vec{0}$ and executes updates of the form:
$$w_{t+1} = w_t + y_i x_i \quad \text{whenever sample } i \text{ is misclassified.}$$
By unrolling this recurrence from $w_0$:
$$w = \sum_{i=1}^N \alpha_i y_i x_i$$
where the integer coefficient $\alpha_i \in \mathbb{N}_0 = \{0, 1, 2, \dots\}$ represents the **total number of times sample $i$ was misclassified** during training.
- Samples that were never misclassified have $\alpha_i = 0$.
- Points near the decision boundary or hard-to-classify samples have large $\alpha_i$ (the direct analog of *Support Vectors*).

**2. Dual Prediction Function**:
Substitute the dual weight representation into the primal hypothesis $h_w(x) = \text{sign}(w^T x)$:
$$h(x) = \text{sign}\left( \left( \sum_{i=1}^N \alpha_i y_i x_i \right)^T x \right) = \text{sign}\left( \sum_{i=1}^N \alpha_i y_i (x_i^T x) \right)$$
Notice that the feature vector $x$ appears **only through inner products** with the training examples $x_i$!

**3. The Kernel Trick (Aizerman et al., 1964)**:
Let $\Phi: \mathbb{R}^d \to \mathcal{H}$ be a non-linear mapping from the input space to a higher- (or infinite-) dimensional Hilbert feature space $\mathcal{H}$.
By **Mercer's Theorem**, any symmetric, positive semi-definite kernel function $K(x, x') = \langle \Phi(x), \Phi(x') \rangle_\mathcal{H}$ computes the inner product in $\mathcal{H}$ without explicitly evaluating or storing $\Phi(x)$.

Replacing $x_i^T x$ with $K(x_i, x)$, the **Kernel Perceptron hypothesis** becomes:
$$h(x) = \text{sign}\left( \sum_{i=1}^N \alpha_i y_i K(x_i, x) \right)$$

**4. The Kernel Perceptron Learning Algorithm**:
1. **Precomputation**: Compute the $N \times N$ symmetric Gram matrix $K \in \mathbb{R}^{N \times N}$ where $K_{i, j} = K(x_i, x_j)$.
2. **Initialization**: Set $\alpha = [0, 0, \dots, 0]^T \in \mathbb{N}_0^N$.
3. **Training Loop**: For each sample $(x_i, y_i)$:
   - Compute prediction score:
     $$s_i = \sum_{j=1}^N \alpha_j y_j K(x_j, x_i) = \sum_{j=1}^N \alpha_j y_j K_{j, i}$$
   - Check margin: If $y_i s_i \le 0$ (mistake):
     $$\alpha_i \leftarrow \alpha_i + 1$$
4. **Convergence Guarantee**: By Novikoff's theorem applied in Hilbert space $\mathcal{H}$, if the data is separable in $\mathcal{H}$ with margin $\gamma_\Phi > 0$, the Kernel Perceptron terminates in at most:
   $$k \le \frac{R_\Phi^2}{\gamma_\Phi^2} = \frac{\max_i K(x_i, x_i)}{\gamma_\Phi^2} \text{ steps.}$$

---

## Part 3: Geometric & Algebraic Interpretation

### 1. The Geometry of the Decision Boundary
The equation $w^T x + b = 0$ defines an affine hyperplane $\mathcal{H}$ of dimension $d-1$ embedded in $\mathbb{R}^d$:
- The weight vector $w$ is the **normal vector** perpendicular to the hyperplane.
- The signed perpendicular distance from any point $x$ to the hyperplane is:
  $$\text{dist}(x, \mathcal{H}) = \frac{w^T x + b}{\|w\|_2}$$
- The class prediction is determined purely by which half-space $x$ falls into:
  $$\mathcal{H}^+ = \{x : w^T x + b > 0\}, \quad \mathcal{H}^- = \{x : w^T x + b < 0\}$$

```
                   w (Normal Vector perpendicular to Hyperplane)
                   ^
                   |         Positive Half-Space H^+: sign(w^T x + b) = +1
                   |            (+)           (+)
                   |                  (+)
  =================+======================================== Hyperplane: w^T x + b = 0
                   |
                   |            (-)           (-)
                   |         Negative Half-Space H^-: sign(w^T x + b) = -1
```

### 2. Geometry of the Update Rule: Angular Correction
Why does the update $w_{t+1} = w_t + y_i x_i$ correct a mistake?
Suppose $y_i = +1$, but the perceptron predicted $-1$, meaning $w_t^T x_i \le 0$ (the angle $\theta$ between $w_t$ and $x_i$ is $\ge 90^\circ$).
Updating $w_{t+1} = w_t + x_i$ computes the new inner product:
$$w_{t+1}^T x_i = (w_t + x_i)^T x_i = w_t^T x_i + \|x_i\|_2^2$$
Since $\|x_i\|_2^2 > 0$, the inner product is strictly increased by $\|x_i\|_2^2$. Geometrically, adding $x_i$ **rotates the normal vector $w$ toward $x_i$**, pulling $x_i$ closer to the positive half-space!

---

## Part 4: Real-World Analogy

### 1. The Club Bouncer with a Mental Scorecard
Imagine a nightclub bouncer deciding who gets admitted into VIP based on two attributes: $x_1$ (Age) and $x_2$ (Dress Code score from 1 to 10).
- The bouncer keeps mental weights: $w_1 \cdot \text{Age} + w_2 \cdot \text{Dress} - \text{Threshold}$.
- If a person is admitted ($+1$) but starts a fight inside (true label $-1$), the bouncer made a false-positive mistake! The bouncer immediately updates his mental criteria by **subtracting** that person's attributes: he lowers the weight for that dress style and raises his admission threshold.
- If a well-behaved patron was wrongfully turned away (true label $+1$), the bouncer made a false-negative mistake. He updates his criteria by **adding** that person's attributes.
- Because there exists a real distinction between well-behaved and unruly guests (linear separability margin $\gamma$), after a finite number of mistakes, the bouncer's policy stabilizes permanently.

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us execute Rosenblatt's Perceptron Learning Algorithm by hand, step-by-step, until full convergence on a concrete 2D dataset.

### Dataset Definition (Augmented with Bias $x_0 = 1$)
We have 4 training samples in $\mathbb{R}^2$. Appending $x_0 = 1$ gives homogeneous coordinates $\tilde{x}_i \in \mathbb{R}^3$:

1. Sample 1: $x_1 = [1, 2]^T, y_1 = +1 \implies \tilde{x}_1 = [1, 1, 2]^T$
2. Sample 2: $x_2 = [2, 1]^T, y_2 = +1 \implies \tilde{x}_2 = [1, 2, 1]^T$
3. Sample 3: $x_3 = [0, 0]^T, y_3 = -1 \implies \tilde{x}_3 = [1, 0, 0]^T$
4. Sample 4: $x_4 = [-1, 1]^T, y_4 = -1 \implies \tilde{x}_4 = [1, -1, 1]^T$

Initial weight: $w_0 = [0.0, 0.0, 0.0]^T$, learning rate $\eta = 1.0$.

---

### Step-by-Step Execution Trace

#### Epoch 1:
- **Test Sample 1**: $\tilde{x}_1 = [1, 1, 2]^T$, true $y_1 = +1$.
  - Score: $w_0^T \tilde{x}_1 = 0(1) + 0(1) + 0(2) = 0.0$.
  - Margin test: $y_1 (w_0^T \tilde{x}_1) = 1(0.0) = 0 \le 0 \implies$ **MISTAKE!**
  - Update:
    $$w_1 = w_0 + y_1 \tilde{x}_1 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} + (+1) \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 1.0 \\ 2.0 \end{bmatrix}$$

- **Test Sample 2**: $\tilde{x}_2 = [1, 2, 1]^T$, true $y_2 = +1$.
  - Score: $w_1^T \tilde{x}_2 = 1(1) + 1(2) + 2(1) = 1 + 2 + 2 = 5.0$.
  - Margin test: $y_2 (w_1^T \tilde{x}_2) = 1(5.0) = 5.0 > 0 \implies$ **CORRECT! (No update)**
  - Weight remains: $w_2 = w_1 = [1.0, 1.0, 2.0]^T$.

- **Test Sample 3**: $\tilde{x}_3 = [1, 0, 0]^T$, true $y_3 = -1$.
  - Score: $w_2^T \tilde{x}_3 = 1(1) + 1(0) + 2(0) = 1.0$.
  - Margin test: $y_3 (w_2^T \tilde{x}_3) = (-1)(1.0) = -1.0 \le 0 \implies$ **MISTAKE!**
  - Update:
    $$w_3 = w_2 + y_3 \tilde{x}_3 = \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix} + (-1) \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 1.0 \\ 2.0 \end{bmatrix}$$

- **Test Sample 4**: $\tilde{x}_4 = [1, -1, 1]^T$, true $y_4 = -1$.
  - Score: $w_3^T \tilde{x}_4 = 0(1) + 1(-1) + 2(1) = 0 - 1 + 2 = 1.0$.
  - Margin test: $y_4 (w_3^T \tilde{x}_4) = (-1)(1.0) = -1.0 \le 0 \implies$ **MISTAKE!**
  - Update:
    $$w_4 = w_3 + y_4 \tilde{x}_4 = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} + (-1) \begin{bmatrix} 1 \\ -1 \\ 1 \end{bmatrix} = \begin{bmatrix} -1.0 \\ 2.0 \\ 1.0 \end{bmatrix}$$

---

#### Epoch 2:
- **Test Sample 1**: $\tilde{x}_1 = [1, 1, 2]^T, y_1 = +1$.
  - Score: $w_4^T \tilde{x}_1 = -1(1) + 2(1) + 1(2) = -1 + 2 + 2 = 3.0$.
  - Margin test: $1(3.0) = 3.0 > 0 \implies$ **CORRECT!**
- **Test Sample 2**: $\tilde{x}_2 = [1, 2, 1]^T, y_2 = +1$.
  - Score: $w_4^T \tilde{x}_2 = -1(1) + 2(2) + 1(1) = -1 + 4 + 1 = 4.0$.
  - Margin test: $1(4.0) = 4.0 > 0 \implies$ **CORRECT!**
- **Test Sample 3**: $\tilde{x}_3 = [1, 0, 0]^T, y_3 = -1$.
  - Score: $w_4^T \tilde{x}_3 = -1(1) + 2(0) + 1(0) = -1.0$.
  - Margin test: $(-1)(-1.0) = +1.0 > 0 \implies$ **CORRECT!**
- **Test Sample 4**: $\tilde{x}_4 = [1, -1, 1]^T, y_4 = -1$.
  - Score: $w_4^T \tilde{x}_4 = -1(1) + 2(-1) + 1(1) = -1 - 2 + 1 = -2.0$.
  - Margin test: $(-1)(-2.0) = +2.0 > 0 \implies$ **CORRECT!**

**ZERO MISTAKES IN EPOCH 2! ALGORITHM TERMINATES.**
The final learned separating hyperplane is:
$$w^T \tilde{x} = -1.0 + 2.0 x_1 + 1.0 x_2 = 0 \iff \mathbf{x_2 = -2 x_1 + 1}$$

---

### Visual Grid Walkthrough Table

```
+----------------------------------------------------------------------------------------------------+
|                                    PERCEPTRON HAND CALCULATION TRACE                               |
+-------+--------+------------------+-------+--------------------+----------------+------------------+
| Epoch | Sample | Feature Vector x | Label | Score: w^T x       | Margin Check   | Updated Weight w |
|       |        | [x_0, x_1, x_2]  | y     |                    | y * (w^T x)    | [b, w_1, w_2]    |
+-------+--------+------------------+-------+--------------------+----------------+------------------+
| Start | -      | -                | -     | -                  | -              | [ 0.0, 0.0, 0.0] |
| Ep 1  | 1      | [1,  1,  2]      | +1    | 0(1)+0(1)+0(2)=  0 | 1(0)  =  0 <=0 | [ 1.0, 1.0, 2.0] |
| Ep 1  | 2      | [1,  2,  1]      | +1    | 1(1)+1(2)+2(1)=  5 | 1(5)  =  5 > 0 | [ 1.0, 1.0, 2.0] |
| Ep 1  | 3      | [1,  0,  0]      | -1    | 1(1)+1(0)+2(0)=  1 | -1(1) = -1 <=0 | [ 0.0, 1.0, 2.0] |
| Ep 1  | 4      | [1, -1,  1]      | -1    | 0(1)+1(-1)+2(1)= 1 | -1(1) = -1 <=0 | [-1.0, 2.0, 1.0] |
+-------+--------+------------------+-------+--------------------+----------------+------------------+
| Ep 2  | 1      | [1,  1,  2]      | +1    | -1+2(1)+1(2)  =  3 | 1(3)  =  3 > 0 | [-1.0, 2.0, 1.0] |
| Ep 2  | 2      | [1,  2,  1]      | +1    | -1+2(2)+1(1)  =  4 | 1(4)  =  4 > 0 | [-1.0, 2.0, 1.0] |
| Ep 2  | 3      | [1,  0,  0]      | -1    | -1+2(0)+1(0)  = -1 | -1(-1)=  1 > 0 | [-1.0, 2.0, 1.0] |
| Ep 2  | 4      | [1, -1,  1]      | -1    | -1+2(-1)+1(1) = -2 | -1(-2)=  2 > 0 | [-1.0, 2.0, 1.0] |
+-------+--------+------------------+-------+--------------------+----------------+------------------+
| SUCCESS: 0 Mistakes in Epoch 2. Converged to Hyperplane: 2*x_1 + x_2 - 1 = 0                       |
+----------------------------------------------------------------------------------------------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning Analog |
| :--- | :--- | :--- | :--- |
| $x_i \in \mathbb{R}^d$ | $\mathbb{R}^d$ vector | Coordinates in feature space | Input features or token embedding activations |
| $\tilde{x}_i \in \mathbb{R}^{d+1}$ | $\mathbb{R}^{d+1}$ vector | Homogeneous coordinate vector with $x_0 = 1$ | Input vector concatenated with constant bias input |
| $y_i \in \{-1, +1\}$ | Discrete scalar | Ground truth binary classification label | Target binary class in binary classification |
| $w \in \mathbb{R}^d$ | $\mathbb{R}^d$ vector | Normal vector orthogonal to separating hyperplane | Weight tensor in `nn.Linear` (`weight`) |
| $b \in \mathbb{R}$ | $\mathbb{R}$ scalar | Offset / intercept from origin | Bias scalar in `nn.Linear` (`bias`) |
| $w^T x + b$ | $\mathbb{R}$ scalar | Signed algebraic distance to hyperplane | Pre-activation logit $z$ |
| $\gamma > 0$ | $\mathbb{R}_{>0}$ scalar | Minimum geometric margin to separating hyperplane | Margin hyperparameter in SVM and loss boundaries |
| $R > 0$ | $\mathbb{R}_{>0}$ scalar | Maximum radius (norm) of training data points | Maximum input feature bound $\|x\|_2 \le R$ |
| $k$ | Integer $\ge 0$ | Total number of classification mistakes | Number of backpropagation parameter update steps |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Novikoff's Mistake Bound Calculation
**Statement**: Consider the dataset from Part 5.
1. Find the maximum norm $R = \max_i \|\tilde{x}_i\|_2$.
2. Given that the unit normal vector $w^* = \frac{1}{\sqrt{6}} \begin{bmatrix} -1 \\ 2 \\ 1 \end{bmatrix}$ separates the dataset:
   - Compute the margin $\gamma_i = y_i (w^{*T} \tilde{x}_i)$ for each of the 4 samples.
   - Find the minimal dataset margin $\gamma = \min_i \gamma_i$.
3. Compute Novikoff's theoretical maximum mistake bound $k_{\max} = (R/\gamma)^2$. Compare this against the actual number of mistakes made in our hand trace ($k_{\text{actual}} = 3$).

**Solution**:
1. **Compute Maximum Radius $R$**:
   - $\|\tilde{x}_1\|_2 = \sqrt{1^2 + 1^2 + 2^2} = \sqrt{1 + 1 + 4} = \sqrt{6} \approx 2.4495$
   - $\|\tilde{x}_2\|_2 = \sqrt{1^2 + 2^2 + 1^2} = \sqrt{6} \approx 2.4495$
   - $\|\tilde{x}_3\|_2 = \sqrt{1^2 + 0^2 + 0^2} = 1.0$
   - $\|\tilde{x}_4\|_2 = \sqrt{1^2 + (-1)^2 + 1^2} = \sqrt{3} \approx 1.7321$
   $$R = \max_i \|\tilde{x}_i\|_2 = \sqrt{6} \approx \mathbf{2.4495}$$

2. **Compute Margins $\gamma_i$ with $w^* = \frac{1}{\sqrt{6}} [-1, 2, 1]^T$**:
   Notice $\|w^*\|_2 = \sqrt{\frac{1}{6}((-1)^2 + 2^2 + 1^2)} = \sqrt{\frac{6}{6}} = 1.0$ (unit vector).
   - Sample 1: $\gamma_1 = 1 \cdot \left( \frac{-1(1) + 2(1) + 1(2)}{\sqrt{6}} \right) = \frac{3}{\sqrt{6}}$
   - Sample 2: $\gamma_2 = 1 \cdot \left( \frac{-1(1) + 2(2) + 1(1)}{\sqrt{6}} \right) = \frac{4}{\sqrt{6}}$
   - Sample 3: $\gamma_3 = -1 \cdot \left( \frac{-1(1) + 2(0) + 1(0)}{\sqrt{6}} \right) = -1 \left( \frac{-1}{\sqrt{6}} \right) = \frac{1}{\sqrt{6}}$
   - Sample 4: $\gamma_4 = -1 \cdot \left( \frac{-1(1) + 2(-1) + 1(1)}{\sqrt{6}} \right) = -1 \left( \frac{-2}{\sqrt{6}} \right) = \frac{2}{\sqrt{6}}$

   The minimal margin is:
   $$\gamma = \min_i \gamma_i = \gamma_3 = \frac{1}{\sqrt{6}} \approx \mathbf{0.4082}$$

3. **Theoretical Mistake Bound**:
   $$k_{\max} = \left( \frac{R}{\gamma} \right)^2 = \left( \frac{\sqrt{6}}{1/\sqrt{6}} \right)^2 = (\sqrt{6} \cdot \sqrt{6})^2 = 6^2 = \mathbf{36\text{ mistakes}}$$
   *Comparison*: Novikoff's theorem guarantees the algorithm will make at most 36 mistakes. In reality, our algorithm converged after only **3 mistakes** ($k_{\text{actual}} = 3 \le 36$), perfectly verifying the theorem!

---

### Problem 2: The XOR Impossibility Proof (Minsky & Papert 1969)
**Statement**: The Exclusive-OR (XOR) logic gate takes inputs $x_1, x_2 \in \{0, 1\}$ and returns $y \in \{0, 1\}$ according to:
- $(0, 0) \to 0$
- $(0, 1) \to 1$
- $(1, 0) \to 1$
- $(1, 1) \to 0$

Provide an uncompromised algebraic proof showing that **no single linear perceptron can solve the XOR problem.**

**Proof**:
Assume for the sake of contradiction that there exists a linear perceptron with real weights $w_1, w_2 \in \mathbb{R}$ and bias $b \in \mathbb{R}$ such that:
$$\hat{y}(x_1, x_2) = \begin{cases} 1 & \text{if } w_1 x_1 + w_2 x_2 + b > 0 \\ 0 & \text{if } w_1 x_1 + w_2 x_2 + b \le 0 \end{cases}$$

Evaluating this condition on all 4 input pairs yields a system of four simultaneous linear inequalities:
1. For $(0, 0) \to 0$:
   $$w_1(0) + w_2(0) + b \le 0 \implies \mathbf{b \le 0} \quad \text{(Inequality A)}$$
2. For $(0, 1) \to 1$:
   $$w_1(0) + w_2(1) + b > 0 \implies \mathbf{w_2 + b > 0} \quad \text{(Inequality B)}$$
3. For $(1, 0) \to 1$:
   $$w_1(1) + w_2(0) + b > 0 \implies \mathbf{w_1 + b > 0} \quad \text{(Inequality C)}$$
4. For $(1, 1) \to 0$:
   $$w_1(1) + w_2(1) + b \le 0 \implies \mathbf{w_1 + w_2 + b \le 0} \quad \text{(Inequality D)}$$

Now, add Inequalities B and C together:
$$(w_2 + b) + (w_1 + b) > 0 + 0 \implies \mathbf{w_1 + w_2 + 2b > 0}$$

Rewrite this as:
$$(w_1 + w_2 + b) + b > 0$$

From Inequality D, we know that $w_1 + w_2 + b \le 0$.
From Inequality A, we know that $b \le 0$.

Therefore, the sum of two non-positive numbers must be non-positive:
$$(w_1 + w_2 + b) + b \le 0 + 0 \implies (w_1 + w_2 + b) + b \le 0$$

This yields a direct contradiction:
$$\underbrace{(w_1 + w_2 + b) + b > 0}_{\text{From B and C}} \quad \text{and} \quad \underbrace{(w_1 + w_2 + b) + b \le 0}_{\text{From A and D}}$$

No real numbers $w_1, w_2, b$ can simultaneously satisfy both statements.
$\blacksquare$ Therefore, a single linear perceptron **cannot compute XOR**.

---

### Problem 3: Solving XOR via a 2-Layer Multilayer Perceptron
**Statement**: How does stacking perceptrons resolve the XOR paradox? Construct an explicit 2-layer network with step activation $H(z) = \mathbb{I}(z \ge 0)$ that computes XOR.

**Construction**:
Notice that XOR can be rewritten using basic logic gates:
$$\text{XOR}(x_1, x_2) = (x_1 \text{ OR } x_2) \text{ AND } \text{NOT}(x_1 \text{ AND } x_2)$$

We construct a hidden layer with 2 neurons:
- **Neuron $h_1$ (OR gate)**: $h_1 = H(x_1 + x_2 - 0.5)$
- **Neuron $h_2$ (NAND gate)**: $h_2 = H(-x_1 - x_2 + 1.5)$

Output neuron $y$ (AND gate on $h_1$ and $h_2$):
$$y = H(h_1 + h_2 - 1.5)$$

Let us verify on all 4 input combinations:
1. $(0, 0)$: $h_1 = H(-0.5) = 0$, $h_2 = H(1.5) = 1$. Output: $y = H(0 + 1 - 1.5) = H(-0.5) = \mathbf{0}$.
2. $(0, 1)$: $h_1 = H(0.5) = 1$, $h_2 = H(0.5) = 1$. Output: $y = H(1 + 1 - 1.5) = H(0.5) = \mathbf{1}$.
3. $(1, 0)$: $h_1 = H(0.5) = 1$, $h_2 = H(0.5) = 1$. Output: $y = H(1 + 1 - 1.5) = H(0.5) = \mathbf{1}$.
4. $(1, 1)$: $h_1 = H(1.5) = 1$, $h_2 = H(-0.5) = 0$. Output: $y = H(1 + 0 - 1.5) = H(-0.5) = \mathbf{0}$.

*Conclusion*: A two-layer network solves XOR effortlessly by warping the feature space into a linearly separable representation!

---

### Problem 4: Solving XOR with a Non-Linear Kernel Perceptron by Hand
**Statement**: In Problem 2, we proved that a primal linear perceptron cannot solve the XOR problem. We now solve XOR using a **Kernel Perceptron** with the inhomogeneous degree-2 polynomial kernel:
$$K(u, v) = (u^T v + 1)^2$$
The dataset with labels $y \in \{-1, +1\}$ is:
- Sample 1: $x_1 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, y_1 = -1$
- Sample 2: $x_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, y_2 = +1$
- Sample 3: $x_3 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, y_3 = +1$
- Sample 4: $x_4 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, y_4 = -1$
1. Compute the $4 \times 4$ Gram matrix $K_{i, j} = K(x_i, x_j)$.
2. Trace the dual coefficients $\alpha = [\alpha_1, \alpha_2, \alpha_3, \alpha_4]^T$ initialized at $\vec{0}$ across training epochs until convergence.
3. Verify that the final dual hypothesis correctly classifies all 4 samples with strictly positive margins $y_i s_i > 0$.
4. Give the explicit feature map $\Phi(x)$ and explain how the kernel trick bypasses the XOR limitation.

**Solution**:

#### 1. Exact Gram Matrix Evaluation:
Compute all pairwise kernel values $K(x_i, x_j) = (x_i^T x_j + 1)^2$:
- $x_1 = [0, 0]^T$:
  $$x_1^T x_1 = 0 \implies K_{11} = (0 + 1)^2 = \mathbf{1}$$
  $$x_1^T x_2 = 0 \implies K_{12} = \mathbf{1}, \quad x_1^T x_3 = 0 \implies K_{13} = \mathbf{1}, \quad x_1^T x_4 = 0 \implies K_{14} = \mathbf{1}$$
- $x_2 = [0, 1]^T$:
  $$x_2^T x_2 = 1 \implies K_{22} = (1 + 1)^2 = \mathbf{4}$$
  $$x_2^T x_3 = 0 \implies K_{23} = \mathbf{1}, \quad x_2^T x_4 = 1 \implies K_{24} = (1 + 1)^2 = \mathbf{4}$$
- $x_3 = [1, 0]^T$:
  $$x_3^T x_3 = 1 \implies K_{33} = (1 + 1)^2 = \mathbf{4}, \quad x_3^T x_4 = 1 \implies K_{34} = (1 + 1)^2 = \mathbf{4}$$
- $x_4 = [1, 1]^T$:
  $$x_4^T x_4 = 1 + 1 = 2 \implies K_{44} = (2 + 1)^2 = \mathbf{9}$$

The full symmetric Gram matrix is:
$$K = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & 4 & 1 & 4 \\ 1 & 1 & 4 & 4 \\ 1 & 4 & 4 & 9 \end{bmatrix}$$

---

#### 2. Dual Perceptron Execution Trace:
Score formula for sample $i$:
$$s_i = \sum_{j=1}^4 \alpha_j y_j K_{j, i}$$
Mistake condition: $y_i s_i \le 0 \implies \alpha_i \leftarrow \alpha_i + 1$.

##### Epoch 1 ($\alpha_{\text{init}} = [0, 0, 0, 0]$):
- **Sample 1** ($y_1 = -1$): $s_1 = 0 \implies y_1 s_1 = 0 \le 0$ (**Mistake!**). $\alpha \leftarrow [1, 0, 0, 0]^T$.
- **Sample 2** ($y_2 = +1$): $s_2 = 1(-1)(1) = -1 \implies y_2 s_2 = -1 \le 0$ (**Mistake!**). $\alpha \leftarrow [1, 1, 0, 0]^T$.
- **Sample 3** ($y_3 = +1$): $s_3 = 1(-1)(1) + 1(+1)(1) = 0 \implies y_3 s_3 = 0 \le 0$ (**Mistake!**). $\alpha \leftarrow [1, 1, 1, 0]^T$.
- **Sample 4** ($y_4 = -1$): $s_4 = 1(-1)(1) + 1(+1)(4) + 1(+1)(4) = -1 + 4 + 4 = 7 \implies y_4 s_4 = (-1)(7) = -7 \le 0$ (**Mistake!**). $\alpha \leftarrow [1, 1, 1, 1]^T$.
End of Epoch 1: $\alpha = [1, 1, 1, 1]^T$ (4 mistakes).

##### Epochs 2 to 7 (Evolution of Dual Weights):
- **Epoch 2**: $\alpha$ increments uniformly: $\alpha = [2, 2, 2, 2]^T$ (4 mistakes).
- **Epoch 3**: $\alpha = [3, 3, 3, 3]^T$ (4 mistakes).
- **Epoch 4**: $\alpha = [4, 4, 4, 4]^T$ (4 mistakes).
- **Epoch 5**: Symmetry breaks! On Sample 4, $s_4 = 5(-1)(1) + 5(1)(4) + 5(1)(4) + 4(-1)(9) = -5 + 20 + 20 - 36 = -1 \implies y_4 s_4 = +1 > 0$ (Correct!).
  $\alpha \leftarrow [5, 5, 5, 4]^T$ (3 mistakes).
- **Epoch 6**: Only Sample 1 triggers a mistake: $\alpha \leftarrow [6, 5, 5, 4]^T$ (1 mistake).
- **Epoch 7**: Only Sample 1 triggers a mistake: $\alpha \leftarrow [7, 5, 5, 4]^T$ (1 mistake).

##### Epoch 8 (Full Verification of Convergence):
Current dual state: $\alpha^* = [7, 5, 5, 4]^T$.
- **Sample 1** ($x_1 = [0, 0]^T, y_1 = -1$):
  $$s_1 = 7(-1)(1) + 5(+1)(1) + 5(+1)(1) + 4(-1)(1) = -7 + 5 + 5 - 4 = -1.0$$
  $$\text{Margin Check}: y_1 s_1 = (-1)(-1.0) = \mathbf{+1.000 > 0} \quad \text{(CORRECT!)}$$
- **Sample 2** ($x_2 = [0, 1]^T, y_2 = +1$):
  $$s_2 = 7(-1)(1) + 5(+1)(4) + 5(+1)(1) + 4(-1)(4) = -7 + 20 + 5 - 16 = +2.0$$
  $$\text{Margin Check}: y_2 s_2 = (+1)(+2.0) = \mathbf{+2.000 > 0} \quad \text{(CORRECT!)}$$
- **Sample 3** ($x_3 = [1, 0]^T, y_3 = +1$):
  $$s_3 = 7(-1)(1) + 5(+1)(1) + 5(+1)(4) + 4(-1)(4) = -7 + 5 + 20 - 16 = +2.0$$
  $$\text{Margin Check}: y_3 s_3 = (+1)(+2.0) = \mathbf{+2.000 > 0} \quad \text{(CORRECT!)}$$
- **Sample 4** ($x_4 = [1, 1]^T, y_4 = -1$):
  $$s_4 = 7(-1)(1) + 5(+1)(4) + 5(+1)(4) + 4(-1)(9) = -7 + 20 + 20 - 36 = -3.0$$
  $$\text{Margin Check}: y_4 s_4 = (-1)(-3.0) = \mathbf{+3.000 > 0} \quad \text{(CORRECT!)}$$

**0 MISTAKES IN EPOCH 8! ALGORITHM CONVERGES COMPLETELY.**

---

#### 3. How the Kernel Trick Bypasses the XOR Barrier:
Expanding the polynomial kernel:
$$K(u, v) = (u_1 v_1 + u_2 v_2 + 1)^2 = u_1^2 v_1^2 + u_2^2 v_2^2 + 2 u_1 u_2 v_1 v_2 + 2 u_1 v_1 + 2 u_2 v_2 + 1$$
This corresponds to the inner product $\Phi(u)^T \Phi(v)$ in a 6-dimensional Hilbert feature space:
$$\Phi(x) = \begin{bmatrix} x_1^2 \\ x_2^2 \\ \sqrt{2} x_1 x_2 \\ \sqrt{2} x_1 \\ \sqrt{2} x_2 \\ 1 \end{bmatrix} \in \mathbb{R}^6$$

Evaluating $\Phi(x)$ on the 4 XOR samples:
- $\Phi(x_1) = [0, 0, 0, 0, 0, 1]^T$ ($y_1 = -1$)
- $\Phi(x_2) = [0, 1, 0, 0, \sqrt{2}, 1]^T$ ($y_2 = +1$)
- $\Phi(x_3) = [1, 0, 0, \sqrt{2}, 0, 1]^T$ ($y_3 = +1$)
- $\Phi(x_4) = [1, 1, \sqrt{2}, \sqrt{2}, \sqrt{2}, 1]^T$ ($y_4 = -1$)

Notice the third coordinate $\phi_3(x) = \sqrt{2} x_1 x_2$: it evaluates to $\sqrt{2}$ **only** for $(1, 1)$, and $0$ for all others!
In this 6D space, the data is linearly separable by a single affine hyperplane, proving that the Kernel Perceptron completely shatters the Minsky-Papert XOR barrier without requiring deep multilayer architectures.

---

### Problem 5: Multi-Class Perceptron (Kesler / Crammer-Singer) Hand Trace
**Statement**: Consider a 3-class classification problem ($C \in \{1, 2, 3\}$) in $\mathbb{R}^2$. Using augmented homogeneous coordinates ($x_0 = 1$), we have 3 training examples:
- Sample 1: $\tilde{x}_1 = [1, 1, 0]^T$, true class $y_1 = 1$
- Sample 2: $\tilde{x}_2 = [1, 0, 1]^T$, true class $y_2 = 2$
- Sample 3: $\tilde{x}_3 = [1, -1, -1]^T$, true class $y_3 = 3$

We maintain three weight vectors $w_1, w_2, w_3 \in \mathbb{R}^3$, all initialized to $\vec{0}$.
Prediction rule:
$$\hat{y} = \arg\max_{c \in \{1, 2, 3\}} w_c^T \tilde{x}$$
(Ties are broken by choosing the smallest class index).
Update rule on mistake ($\hat{y} \ne y_i$):
$$w_{y_i} \leftarrow w_{y_i} + \tilde{x}_i, \quad w_{\hat{y}} \leftarrow w_{\hat{y}} - \tilde{x}_i$$
All other weight vectors remain unchanged.
Execute the multi-class perceptron step-by-step until full convergence.

**Solution**:

#### Initial State:
$$w_1 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad w_2 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \quad w_3 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

---

#### Epoch 1:
- **Sample 1** ($\tilde{x}_1 = [1, 1, 0]^T$, true $y_1 = 1$):
  - Class scores: $w_1^T \tilde{x}_1 = 0$, $w_2^T \tilde{x}_1 = 0$, $w_3^T \tilde{x}_1 = 0$.
  - Tie-breaking picks $\hat{y} = 1$.
  - $\hat{y} = y_1 \implies$ **CORRECT! (No update)**.

- **Sample 2** ($\tilde{x}_2 = [1, 0, 1]^T$, true $y_2 = 2$):
  - Class scores: $w_1^T \tilde{x}_2 = 0$, $w_2^T \tilde{x}_2 = 0$, $w_3^T \tilde{x}_2 = 0$.
  - Tie-breaking picks $\hat{y} = 1 \ne 2 \implies$ **MISTAKE!**
  - Update: Promote true class $w_2$, penalize false class $w_1$:
    $$w_2 \leftarrow w_2 + \tilde{x}_2 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$
    $$w_1 \leftarrow w_1 - \tilde{x}_2 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} - \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 0 \\ -1 \end{bmatrix}$$
    $$w_3 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

- **Sample 3** ($\tilde{x}_3 = [1, -1, -1]^T$, true $y_3 = 3$):
  - Class scores:
    $$w_1^T \tilde{x}_3 = -1(1) + 0(-1) - 1(-1) = -1 + 0 + 1 = 0$$
    $$w_2^T \tilde{x}_3 = 1(1) + 0(-1) + 1(-1) = 1 + 0 - 1 = 0$$
    $$w_3^T \tilde{x}_3 = 0(1) + 0(-1) + 0(-1) = 0$$
  - All scores are $0$. Tie-breaker picks $\hat{y} = 1 \ne 3 \implies$ **MISTAKE!**
  - Update: Promote $w_3$, penalize $w_1$:
    $$w_3 \leftarrow w_3 + \tilde{x}_3 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ -1 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \\ -1 \end{bmatrix}$$
    $$w_1 \leftarrow w_1 - \tilde{x}_3 = \begin{bmatrix} -1 \\ 0 \\ -1 \end{bmatrix} - \begin{bmatrix} 1 \\ -1 \\ -1 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix}$$
    $$w_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$

---

#### Epoch 2:
- **Sample 1** ($\tilde{x}_1 = [1, 1, 0]^T$, true $y_1 = 1$):
  - Class scores:
    $$w_1^T \tilde{x}_1 = -2(1) + 1(1) + 0(0) = -1$$
    $$w_2^T \tilde{x}_1 = 1(1) + 0(1) + 1(0) = +1$$
    $$w_3^T \tilde{x}_1 = 1(1) - 1(1) - 1(0) = 0$$
  - Predicted: $\hat{y} = \arg\max(-1, +1, 0) = 2 \ne 1 \implies$ **MISTAKE!**
  - Update: Promote $w_1$, penalize $w_2$:
    $$w_1 \leftarrow w_1 + \tilde{x}_1 = \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \\ 0 \end{bmatrix}$$
    $$w_2 \leftarrow w_2 - \tilde{x}_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \\ 1 \end{bmatrix}$$
    $$w_3 = \begin{bmatrix} 1 \\ -1 \\ -1 \end{bmatrix}$$

- **Sample 2** ($\tilde{x}_2 = [1, 0, 1]^T$, true $y_2 = 2$):
  - Class scores:
    $$w_1^T \tilde{x}_2 = -1(1) + 2(0) + 0(1) = -1$$
    $$w_2^T \tilde{x}_2 = 0(1) - 1(0) + 1(1) = +1$$
    $$w_3^T \tilde{x}_2 = 1(1) - 1(0) - 1(1) = 0$$
  - Predicted: $\hat{y} = \arg\max(-1, +1, 0) = 2 == y_2 \implies$ **CORRECT!**

- **Sample 3** ($\tilde{x}_3 = [1, -1, -1]^T$, true $y_3 = 3$):
  - Class scores:
    $$w_1^T \tilde{x}_3 = -1(1) + 2(-1) + 0(-1) = -3$$
    $$w_2^T \tilde{x}_3 = 0(1) - 1(-1) + 1(-1) = 0$$
    $$w_3^T \tilde{x}_3 = 1(1) - 1(-1) - 1(-1) = 1 + 1 + 1 = +3$$
  - Predicted: $\hat{y} = \arg\max(-3, 0, +3) = 3 == y_3 \implies$ **CORRECT!**

---

#### Epoch 3 (Verification):
- **Sample 1**: Scores: $w_1^T \tilde{x}_1 = +1$, $w_2^T \tilde{x}_1 = -1$, $w_3^T \tilde{x}_1 = 0 \implies \hat{y} = 1$ (**CORRECT!**)
- **Sample 2**: Scores: $w_1^T \tilde{x}_2 = -1$, $w_2^T \tilde{x}_2 = +1$, $w_3^T \tilde{x}_2 = 0 \implies \hat{y} = 2$ (**CORRECT!**)
- **Sample 3**: Scores: $w_1^T \tilde{x}_3 = -3$, $w_2^T \tilde{x}_3 = 0$, $w_3^T \tilde{x}_3 = +3 \implies \hat{y} = 3$ (**CORRECT!**)

**0 MISTAKES IN EPOCH 3! ALGORITHM CONVERGES.**

Final Learned Multi-Class Linear Model:
$$w_1 = \begin{bmatrix} -1 \\ 2 \\ 0 \end{bmatrix}, \quad w_2 = \begin{bmatrix} 0 \\ -1 \\ 1 \end{bmatrix}, \quad w_3 = \begin{bmatrix} 1 \\ -1 \\ -1 \end{bmatrix}$$
Every class region forms a convex polyhedral cone in $\mathbb{R}^2$, providing the exact mathematical template for modern linear classification heads in PyTorch (`nn.Linear(d_in, 3)`).

---

## Part 7: Deep Learning Connection & Application

### 1. From Rosenblatt's Perceptron to Modern Linear Layers
In modern deep learning architectures, every dense layer is a vectorized bank of perceptrons:
$$\text{Perceptron: } z = w^T x + b \implies \text{PyTorch Linear Layer: } z = W x + b$$
where $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$.
Instead of Rosenblatt's non-differentiable step function $\text{sign}(z)$, modern deep learning uses smooth, differentiable continuous activations (ReLU, GELU, Swish, Sigmoid) so that backpropagation can compute $\nabla_\theta \mathcal{L}$ via the multivariable chain rule.

### 2. Linear Separability of Latent Embeddings
A foundational principle of modern deep representation learning is:
> **Deep neural networks transform non-linearly separable raw data into a latent feature space where the classes become linearly separable by a single final linear layer.**

In CNNs (ResNet) and Transformers (BERT, GPT), the penultimate layer outputs an embedding vector $h(x) \in \mathbb{R}^d$. The final classification head is simply:
$$\hat{y} = \text{softmax}(W h(x) + b)$$
which is nothing other than a multi-class perceptron operating on the learned representation space!

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `06_deep_learning_foundations/code/01_perceptron_and_linear_models.py`. It implements:
1. Exact visual grid verification of Rosenblatt's algorithm matching the hand calculations in Part 5.
2. Novikoff's theorem verification: empirical mistake count $k \le (R/\gamma)^2$.
3. Exact algebraic proof of XOR non-separability.
4. An explicit 2-layer perceptron solving XOR in NumPy and PyTorch.

Save the code and run it directly in Python 3.
