# 6.4 Full Backpropagation Derivation & Scratch Autograd Engine

---

## Part 1: Intuition & 101 Motivation

At the beating heart of all modern deep learning is an algorithm: **Backpropagation** (Rumelhart, Hinton, & Williams 1986).

To train a neural network, we define an objective (loss function $\mathcal{L}$) that measures the discrepancy between model predictions and ground-truth targets. We then adjust each of the millions or billions of parameters $\theta$ using gradient descent:
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

This presents the fundamental **credit assignment problem**: when an error occurs at the final output of a 100-layer network, exactly how much did a weight in layer 12 contribute to that error?

### Why Numerical Differentiation Is Impossible
One naive approach to computing $\nabla_\theta \mathcal{L}$ is finite differences:
$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta + \epsilon e_i) - \mathcal{L}(\theta)}{\epsilon}$$
To compute the gradient for a model with $P$ parameters, finite differences requires **$P + 1$ complete forward passes**.
For a 70-billion parameter LLaMA model, computing a single gradient step would require 70 billion forward passes. Running at 100 forward passes per second on an H100 GPU cluster, a single optimization step would take **22 years**!

### The Magic of Reverse-Mode Automatic Differentiation
Backpropagation is an exact, analytic implementation of **Reverse-Mode Automatic Differentiation**. By exploiting the multivariable **Chain Rule** of calculus, backpropagation computes the exact analytical gradients for **all $P$ parameters simultaneously in a single backward sweep**, requiring only $\sim 2\times$ the computational cost of a single forward pass!

```
FORWARD PASS (Compute and Cache Activations):
Input x  ------> [Layer 1] ------> [Layer 2] ------> [Output Layer] ------> Loss L
            a^(0)          a^(1)          a^(2)                 y_hat

BACKWARD PASS (Propagate Error Signals via Chain Rule):
Input    <------ [delta^(1)] <---- [delta^(2)] <---- [delta^(L)]     <------ dL/dy_hat
                   |                 |                 |
                   v                 v                 v
               dL/dW^(1)         dL/dW^(2)         dL/dW^(L)
```

---

## Part 2: Rigorous Mathematical Formulation

### 1. Forward Pass Recurrence

Consider an $L$-layer feedforward neural network with input $a^{(0)} = x \in \mathbb{R}^{d_0}$.
For each layer $l \in \{1, 2, \dots, L\}$:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma^{(l)}(z^{(l)})$$

Where:
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ is the weight matrix.
- $b^{(l)} \in \mathbb{R}^{d_l}$ is the bias vector.
- $z^{(l)} \in \mathbb{R}^{d_l}$ is the pre-activation vector.
- $a^{(l)} \in \mathbb{R}^{d_l}$ is the post-activation feature vector.
- $\sigma^{(l)}: \mathbb{R} \to \mathbb{R}$ is the activation function applied element-wise.

The final network output is $\hat{y} = a^{(L)}$, and the scalar loss is $\mathcal{L}(\hat{y}, y)$.

---

### 2. Definition of Error Sensitivity (The Adjoint / $\delta$)

The central quantity in backpropagation is the **error sensitivity vector** $\delta^{(l)} \in \mathbb{R}^{d_l}$, defined as the partial derivative of the scalar loss with respect to the pre-activation $z^{(l)}$:
$$\delta^{(l)} \equiv \nabla_{z^{(l)}} \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial z_1^{(l)}} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial z_{d_l}^{(l)}} \end{bmatrix}$$

---

### 3. The Four Fundamental Equations of Backpropagation

#### Theorem: The Backpropagation Equations
For any differentiable feedforward neural network, the gradients of the scalar loss $\mathcal{L}$ with respect to all weights and biases are given by:

1. **Output Layer Error ($\delta^{(L)}$)**:
   $$\mathbf{\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \odot \sigma'^{(L)}(z^{(L)})}$$
   *(In component form: $\delta_i^{(L)} = \frac{\partial \mathcal{L}}{\partial a_i^{(L)}} \sigma'^{(L)}(z_i^{(L)})$)*

2. **Hidden Layer Error Recurrence ($\delta^{(l)}$)**:
   $$\mathbf{\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'^{(l)}(z^{(l)})}$$
   *(Propagates error backward from layer $l+1$ to layer $l$)*

3. **Weight Gradient ($\frac{\partial \mathcal{L}}{\partial W^{(l)}}$)**:
   $$\mathbf{\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T \in \mathbb{R}^{d_l \times d_{l-1}}}$$
   *(In component form: $\frac{\partial \mathcal{L}}{\partial W_{jk}^{(l)}} = \delta_j^{(l)} a_k^{(l-1)}$)*

4. **Bias Gradient ($\frac{\partial \mathcal{L}}{\partial b^{(l)}}$)**:
   $$\mathbf{\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)} \in \mathbb{R}^{d_l}}$$

---

#### Complete Step-by-Step Proof of the 4 Equations

##### Proof of Equation 1 (Output Error $\delta^{(L)}$)
By definition, $\delta_j^{(L)} = \frac{\partial \mathcal{L}}{\partial z_j^{(L)}}$.
The loss $\mathcal{L}$ depends on $z_j^{(L)}$ exclusively through the activation $a_j^{(L)} = \sigma(z_j^{(L)})$.
Applying the single-variable chain rule:
$$\frac{\partial \mathcal{L}}{\partial z_j^{(L)}} = \frac{\partial \mathcal{L}}{\partial a_j^{(L)}} \cdot \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} = \frac{\partial \mathcal{L}}{\partial a_j^{(L)}} \cdot \sigma'(z_j^{(L)})$$
In vector notation with Hadamard product $\odot$:
$$\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \odot \sigma'(z^{(L)})$$
$\blacksquare$

##### Proof of Equation 2 (Hidden Error Recurrence $\delta^{(l)}$)
We seek $\delta_j^{(l)} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}}$.
Notice that $z_j^{(l)}$ affects all pre-activations $z_k^{(l+1)}$ in the next layer through the forward relationship:
$$z_k^{(l+1)} = \sum_{p=1}^{d_l} W_{kp}^{(l+1)} a_p^{(l)} + b_k^{(l+1)} = \sum_{p=1}^{d_l} W_{kp}^{(l+1)} \sigma(z_p^{(l)}) + b_k^{(l+1)}$$

Applying the multivariable chain rule summing across all neurons $k \in \{1, \dots, d_{l+1}\}$ in layer $l+1$:
$$\frac{\partial \mathcal{L}}{\partial z_j^{(l)}} = \sum_{k=1}^{d_{l+1}} \frac{\partial \mathcal{L}}{\partial z_k^{(l+1)}} \cdot \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}}$$

By definition of next-layer error: $\frac{\partial \mathcal{L}}{\partial z_k^{(l+1)}} = \delta_k^{(l+1)}$.
Differentiating $z_k^{(l+1)}$ with respect to $z_j^{(l)}$:
$$\frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} = \frac{\partial}{\partial z_j^{(l)}} \left( \sum_p W_{kp}^{(l+1)} \sigma(z_p^{(l)}) + b_k^{(l+1)} \right) = W_{kj}^{(l+1)} \sigma'(z_j^{(l)})$$

Substituting this back:
$$\delta_j^{(l)} = \sum_{k=1}^{d_{l+1}} \delta_k^{(l+1)} W_{kj}^{(l+1)} \sigma'(z_j^{(l)}) = \left( \sum_{k=1}^{d_{l+1}} W_{kj}^{(l+1)} \delta_k^{(l+1)} \right) \sigma'(z_j^{(l)})$$

Notice that $\sum_{k=1}^{d_{l+1}} W_{kj}^{(l+1)} \delta_k^{(l+1)} = \sum_k (W^{(l+1)T})_{jk} \delta_k^{(l+1)} = \left[ (W^{(l+1)})^T \delta^{(l+1)} \right]_j$.
In vector notation:
$$\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'(z^{(l)})$$
$\blacksquare$

##### Proof of Equation 3 (Weight Gradient $\frac{\partial \mathcal{L}}{\partial W^{(l)}}$)
The pre-activation equation is $z_j^{(l)} = \sum_k W_{jk}^{(l)} a_k^{(l-1)} + b_j^{(l)}$.
The loss $\mathcal{L}$ depends on $W_{jk}^{(l)}$ exclusively through $z_j^{(l)}$.
Applying chain rule:
$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{(l)}} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial W_{jk}^{(l)}} = \delta_j^{(l)} \cdot a_k^{(l-1)}$$
This is an outer product of vector $\delta^{(l)} \in \mathbb{R}^{d_l}$ and vector $a^{(l-1)} \in \mathbb{R}^{d_{l-1}}$:
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$
$\blacksquare$

##### Proof of Equation 4 (Bias Gradient $\frac{\partial \mathcal{L}}{\partial b^{(l)}}$)
Similarly:
$$\frac{\partial \mathcal{L}}{\partial b_j^{(l)}} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial b_j^{(l)}} = \delta_j^{(l)} \cdot 1 = \delta_j^{(l)}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$
$\blacksquare$ **Q.E.D.**

---

### 4. Batched Tensor Formulation (Mini-Batch Training)

In production deep learning (PyTorch, JAX), computations are executed over a mini-batch of size $B$:
- Input batch: $A^{(0)} = X \in \mathbb{R}^{B \times d_0}$
- Layer $l$ forward:
  $$Z^{(l)} = A^{(l-1)} (W^{(l)})^T + \mathbf{1}_B (b^{(l)})^T \in \mathbb{R}^{B \times d_l}$$
  $$A^{(l)} = \sigma(Z^{(l)}) \in \mathbb{R}^{B \times d_l}$$
- Layer $l$ backward error matrix $\Delta^{(l)} \in \mathbb{R}^{B \times d_l}$:
  $$\Delta^{(l)} = \left( \Delta^{(l+1)} W^{(l+1)} \right) \odot \sigma'(Z^{(l)})$$
- Parameter gradients averaged over batch:
  $$\mathbf{\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{B} (\Delta^{(l)})^T A^{(l-1)} \in \mathbb{R}^{d_l \times d_{l-1}}}$$
  $$\mathbf{\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{B} \sum_{i=1}^B \Delta_{i, :}^{(l)} \in \mathbb{R}^{d_l}}$$

---

### 5. Computational Graphs & Topological Sorting

A computational graph is a **Directed Acyclic Graph (DAG)** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
- **Nodes $\mathcal{V}$**: Represent mathematical values/tensors (constants, parameters, intermediate activations).
- **Edges $\mathcal{E}$**: Represent elementary functional operations (`+`, `*`, `matmul`, `relu`).

```
Computational Graph DAG for L = 0.5 * (W_2 * relu(W_1 * x + b_1) + b_2 - y)^2:

  x ------> [matmul] ------> [+] ------> [relu] ------> [matmul] ------> [+] ------> [-] ------> [square] ------> [* 0.5] ------> Loss
              ^               ^                            ^              ^           ^
              |               |                            |              |           |
             W_1             b_1                          W_2            b_2          y
```

#### The Autograd Execution Protocol
1. **Forward Pass**: Nodes are evaluated in **topological order** from input sources to the loss sink. Every node caches its computed tensor and pointers to its parent nodes.
2. **Reverse Pass**: Starting from the scalar loss node ($\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1.0$), nodes are traversed in **reverse topological order**. Each node executes its local backward method, accumulating incoming gradients into its parents:
   $$\text{parent.grad} \mathrel{+}= \text{local\_derivative} \times \text{child.grad}$$

---

## Part 3: Geometric & Graph Interpretation

### 1. Reverse-Mode vs. Forward-Mode Automatic Differentiation
Consider a function $f: \mathbb{R}^n \to \mathbb{R}^m$:
- **Forward-Mode Autograd (Dual Numbers / Tangent Mode)**:
  Carries directional derivatives $\frac{\partial v}{\partial x_i}$ forward alongside function evaluations.
  To compute the complete Jacobian $J \in \mathbb{R}^{m \times n}$, forward-mode requires **$n$ sweeps** (one per input dimension).
- **Reverse-Mode Autograd (Adjoint Mode / Backprop)**:
  Carries sensitivities $\frac{\partial \mathcal{L}}{\partial v}$ backward from outputs to inputs.
  To compute the Jacobian, reverse-mode requires **$m$ sweeps** (one per output dimension).

In deep learning:
$$f: \mathbb{R}^{10^{11}} \to \mathbb{R}^1 \quad (n = 10^{11} \text{ parameters}, \; m = 1 \text{ scalar loss})$$
- Forward-mode would require $10^{11}$ forward sweeps.
- **Reverse-mode requires EXACTLY 1 backward sweep!**
This asymmetry is why modern deep learning is built exclusively on reverse-mode autograd.

---

## Part 4: Real-World Analogy

### 1. Corporate Blame Allocation
Imagine a corporation that produced a defective product, losing \$1,000,000 in customer lawsuits ($\mathcal{L}$):
- **Output Layer (The CEO)**: The board holds the CEO accountable: $\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial \hat{y}} = -\$1\text{M}$.
- **Hidden Layer (The Engineering VP)**: The CEO calls the VP into his office. The VP's culpability is scaled by how directly his department influenced the CEO's decision ($W^{(L)T} \delta^{(L)}$).
- **Early Layer (The Software Engineer)**: The VP reprimands the engineering team ($W^{(1)T} \delta^{(2)}$). The individual programmer's responsibility ($\frac{\partial \mathcal{L}}{\partial W^{(1)}}$) is proportional to the blame assigned to his team ($\delta^{(1)}$) multiplied by how actively he worked on the codebase ($a^{(0)T}$).
- In a single pass down the corporate hierarchy, every employee's exact contribution to the \$1,000,000 loss is determined!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us calculate by hand the exact cell-by-cell forward pass activations and backward pass gradients for a concrete 2-layer neural network with scalar inputs and weights.

### Architecture & Numerical Parameters
- **Input vector**: $x = \begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix}$, true target scalar $y = 1.0$.
- **Layer 1 (Hidden, 2 neurons, ReLU activation)**:
  $$W^{(1)} = \begin{bmatrix} 0.5 & -0.5 \\ 0.2 & 0.8 \end{bmatrix}, \quad b^{(1)} = \begin{bmatrix} 0.1 \\ -0.2 \end{bmatrix}$$
- **Layer 2 (Output, 1 neuron, Linear activation)**:
  $$W^{(2)} = \begin{bmatrix} 0.4 & 0.6 \end{bmatrix}, \quad b^{(2)} = [0.1]$$
- **Loss function**: Mean Squared Error $\mathcal{L} = \frac{1}{2} (\hat{y} - y)^2$.

---

### Step-by-Step Forward Pass Walkthrough

#### 1. Layer 1 Pre-activation $z^{(1)} = W^{(1)} x + b^{(1)}$:
$$z_1^{(1)} = (0.5)(1.0) + (-0.5)(2.0) + 0.1 = 0.5 - 1.0 + 0.1 = \mathbf{-0.4000}$$
$$z_2^{(1)} = (0.2)(1.0) + (0.8)(2.0) - 0.2 = 0.2 + 1.6 - 0.2 = \mathbf{+1.6000}$$
$$z^{(1)} = \begin{bmatrix} -0.4000 \\ +1.6000 \end{bmatrix}$$

#### 2. Layer 1 Activation $a^{(1)} = \text{ReLU}(z^{(1)})$:
$$a_1^{(1)} = \text{ReLU}(-0.4000) = \mathbf{0.0000} \quad (\text{Neuron is INACTIVE / DEAD!})$$
$$a_2^{(1)} = \text{ReLU}(+1.6000) = \mathbf{+1.6000} \quad (\text{Neuron is ACTIVE})$$
$$a^{(1)} = \begin{bmatrix} 0.0000 \\ 1.6000 \end{bmatrix}$$

#### 3. Layer 2 Pre-activation & Output $z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}$:
$$z^{(2)} = (0.4)(0.0000) + (0.6)(1.6000) + 0.1 = 0.0 + 0.9600 + 0.1 = \mathbf{1.0600}$$
$$\hat{y} = a^{(2)} = z^{(2)} = \mathbf{1.0600}$$

#### 4. Scalar Loss $\mathcal{L}$:
$$\mathcal{L} = \frac{1}{2} (1.0600 - 1.0000)^2 = \frac{1}{2} (0.0600)^2 = \frac{1}{2} (0.0036) = \mathbf{0.001800}$$

---

### Step-by-Step Backward Pass Walkthrough

#### 1. Output Error $\delta^{(2)} = \frac{\partial \mathcal{L}}{\partial z^{(2)}}$:
$$\delta^{(2)} = \frac{\partial}{\partial \hat{y}}\left[ \frac{1}{2} (\hat{y} - y)^2 \right] \cdot \sigma'^{(2)}(z^{(2)}) = (\hat{y} - y) \cdot 1 = 1.0600 - 1.0000 = \mathbf{+0.0600}$$

#### 2. Layer 2 Gradients $\frac{\partial \mathcal{L}}{\partial W^{(2)}}$ and $\frac{\partial \mathcal{L}}{\partial b^{(2)}}$:
$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \delta^{(2)} (a^{(1)})^T = (+0.0600) \begin{bmatrix} 0.0000 & 1.6000 \end{bmatrix} = \begin{bmatrix} \mathbf{0.0000} & \mathbf{0.0960} \end{bmatrix}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \delta^{(2)} = \mathbf{+0.0600}$$

#### 3. Backpropagate Error to Layer 1 $\delta^{(1)}$:
First, pull back through weights:
$$(W^{(2)})^T \delta^{(2)} = \begin{bmatrix} 0.4 \\ 0.6 \end{bmatrix} (+0.0600) = \begin{bmatrix} 0.0240 \\ 0.0360 \end{bmatrix}$$

Next, gate by ReLU derivative $\sigma'(z^{(1)}) = \begin{bmatrix} \mathbb{I}(z_1 > 0) \\ \mathbb{I}(z_2 > 0) \end{bmatrix} = \begin{bmatrix} \mathbb{I}(-0.4 > 0) \\ \mathbb{I}(1.6 > 0) \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$:
$$\delta^{(1)} = \begin{bmatrix} 0.0240 \\ 0.0360 \end{bmatrix} \odot \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{0.0000} \\ \mathbf{0.0360} \end{bmatrix}$$
*(Notice: Because Neuron 1 was inactive during forward pass, its backpropagated error is clamped to EXACTLY ZERO!)*

#### 4. Layer 1 Gradients $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$ and $\frac{\partial \mathcal{L}}{\partial b^{(1)}}$:
$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \delta^{(1)} x^T = \begin{bmatrix} 0.0000 \\ 0.0360 \end{bmatrix} \begin{bmatrix} 1.0000 & 2.0000 \end{bmatrix} = \begin{bmatrix} (0)(1) & (0)(2) \\ (0.036)(1) & (0.036)(2) \end{bmatrix} = \begin{bmatrix} \mathbf{0.0000} & \mathbf{0.0000} \\ \mathbf{0.0360} & \mathbf{0.0720} \end{bmatrix}$$
$$\frac{\partial \mathcal{L}}{\partial b^{(1)}} = \delta^{(1)} = \begin{bmatrix} \mathbf{0.0000} \\ \mathbf{0.0360} \end{bmatrix}$$

---

### Comparative Visual Grid: Forward & Backward Quantities

```
+----------------------------------------------------------------------------------------------------+
|                                    FORWARD PASS VALUES (Input x = [1.0, 2.0])                      |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Layer             | Pre-activation z   | Activation f(z)    | Output Tensor      | Current Loss L  |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Layer 1 (Hidden)  | [-0.4000, +1.6000] | ReLU               | a^(1) = [0.0, 1.6] | -               |
| Layer 2 (Output)  | [1.0600]           | Linear             | y_hat = 1.0600     | 0.001800        |
+-------------------+--------------------+--------------------+--------------------+-----------------+

+----------------------------------------------------------------------------------------------------+
|                                    BACKWARD PASS GRADIENTS                                         |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Layer             | Error Adjoint delta| Weight Gradient    | Bias Gradient      | Notes           |
|                   | (dL/dz)            | (dL/dW)            | (dL/db)            |                 |
+-------------------+--------------------+--------------------+--------------------+-----------------+
| Layer 2 (Output)  | [+0.0600]          | [0.0000, 0.0960]   | [+0.0600]          | Exact Match     |
| Layer 1 (Hidden)  | [0.0000, +0.0360]  | [[0.0000, 0.0000], | [0.0000, +0.0360]  | Row 1 zeroed    |
|                   |                    |  [0.0360, 0.0720]] |                    | by dead ReLU!   |
+-------------------+--------------------+--------------------+--------------------+-----------------+
```

---

### "What Refers to What" Legend Protocol

| Mathematical Symbol | Dimensions / Type | Pure Mathematics Meaning | Deep Learning Implementation |
| :--- | :--- | :--- | :--- |
| $z^{(l)}$ | $\mathbb{R}^{d_l}$ vector | Layer $l$ pre-activation | Intermediate tensor before activation |
| $a^{(l)}$ | $\mathbb{R}^{d_l}$ vector | Layer $l$ post-activation output | Activated tensor passed forward (`a = act(z)`) |
| $\delta^{(l)}$ | $\mathbb{R}^{d_l}$ vector | Adjoint / Error sensitivity $\partial \mathcal{L} / \partial z^{(l)}$ | Backpropagated tensor gradient |
| $W^{(l)}$ | $\mathbb{R}^{d_l \times d_{l-1}}$ | Forward linear operator | Trainable weight tensor (`linear.weight`) |
| $\frac{\partial \mathcal{L}}{\partial W^{(l)}}$ | $\mathbb{R}^{d_l \times d_{l-1}}$ | Matrix derivative of scalar loss | Gradient buffer (`linear.weight.grad`) |
| $\sigma'(z^{(l)})$ | $\mathbb{R}^{d_l}$ vector | Local Jacobian diagonal | Multiplicative gating tensor |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | Directed Acyclic Graph | Graph topology of computational trace | Autograd tape engine (`grad_fn` chain) |

---

## Part 6: Step-by-Step Solved Illustrations

### Problem 1: Numerical Gradient Check (Finite Differences Verification)
**Statement**: Verify by hand that the analytical gradient $\frac{\partial \mathcal{L}}{\partial W_{22}^{(1)}} = 0.0720$ matches the two-sided finite difference numerical approximation with $\epsilon = 10^{-4}$.

**Calculation**:
We perturb $W_{22}^{(1)}$ by $\pm \epsilon = \pm 0.0001$. Originally, $W_{22}^{(1)} = 0.8$.
- **Perturb positive**: $W_{22}^+ = 0.8001$.
  $$z_2^{(1)+} = 0.2(1.0) + (0.8001)(2.0) - 0.2 = 0.2 + 1.6002 - 0.2 = 1.6002$$
  $$a_2^{(1)+} = 1.6002$$
  $$z^{(2)+} = 0.4(0.0) + 0.6(1.6002) + 0.1 = 0.96012 + 0.1 = 1.06012$$
  $$\mathcal{L}^+ = \frac{1}{2} (1.06012 - 1.0)^2 = \frac{1}{2} (0.06012)^2 = \frac{1}{2} (0.0036144144) \approx \mathbf{0.0018072072}$$

- **Perturb negative**: $W_{22}^- = 0.7999$.
  $$z_2^{(1)-} = 0.2(1.0) + (0.7999)(2.0) - 0.2 = 0.2 + 1.5998 - 0.2 = 1.5998$$
  $$a_2^{(1)-} = 1.5998$$
  $$z^{(2)-} = 0.4(0.0) + 0.6(1.5998) + 0.1 = 0.95988 + 0.1 = 1.05988$$
  $$\mathcal{L}^- = \frac{1}{2} (1.05988 - 1.0)^2 = \frac{1}{2} (0.05988)^2 = \frac{1}{2} (0.0035856144) \approx \mathbf{0.0017928072}$$

- **Two-Sided Finite Difference Ratio**:
  $$\frac{\mathcal{L}^+ - \mathcal{L}^-}{2 \epsilon} = \frac{0.0018072072 - 0.0017928072}{2(0.0001)} = \frac{0.0000144000}{0.0002000} = \mathbf{0.072000}$$

**Exact Match!** The numerical finite difference yields $0.072000$, matching our analytical backpropagation derivation to 6 decimal places!

---

### Problem 2: Softmax + Cross-Entropy Gradient Cancellation
**Statement**: In classification networks, the output layer combines Softmax with Cross-Entropy Loss:
$$S_i = \frac{e^{z_i}}{\sum_k e^{z_k}}, \quad \mathcal{L} = -\sum_{i=1}^K y_i \log S_i$$
where $y$ is a one-hot ground-truth label vector ($\sum y_i = 1$).
Prove that the error sensitivity with respect to the pre-activation logits $z$ simplifies miraculously to:
$$\mathbf{\delta_i = \frac{\partial \mathcal{L}}{\partial z_i} = S_i - y_i}$$
*(Zero $K \times K$ Jacobian matrix multiplication required!)*

**Proof**:
By the multivariable chain rule, summing over all output classes $k$:
$$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{k=1}^K \frac{\partial \mathcal{L}}{\partial S_k} \frac{\partial S_k}{\partial z_i}$$

1. Differentiating Cross-Entropy with respect to probabilities:
   $$\frac{\partial \mathcal{L}}{\partial S_k} = -\frac{y_k}{S_k}$$
2. From Chapter 6.2, the Softmax Jacobian is:
   $$\frac{\partial S_k}{\partial z_i} = S_k (\delta_{ki} - S_i)$$
3. Substituting both into the chain rule:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{k=1}^K \left( -\frac{y_k}{S_k} \right) \cdot \left[ S_k (\delta_{ki} - S_i) \right] = -\sum_{k=1}^K y_k (\delta_{ki} - S_i)$$
   Splitting the summation:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_{k=1}^K y_k \delta_{ki} + \sum_{k=1}^K y_k S_i$$
4. Since $\delta_{ki} = 1$ only when $k = i$ (and 0 otherwise):
   $$\sum_{k=1}^K y_k \delta_{ki} = y_i$$
5. Since $\sum_{k=1}^K y_k = 1$ (one-hot probability distribution):
   $$\sum_{k=1}^K y_k S_i = S_i \sum_{k=1}^K y_k = S_i \cdot 1 = S_i$$
6. Combining terms:
   $$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i + S_i = \mathbf{S_i - y_i}$$
$\blacksquare$ In vector form: $\mathbf{\delta = S - y}$.
This elegant algebraic cancellation explains why modern deep learning frameworks never instantiate the full Softmax Jacobian: the gradient is simply the **predicted probability minus target one-hot indicator**!

---

## Part 7: Deep Learning Connection & Application

### 1. How PyTorch Autograd Operates
Every `torch.Tensor` with `requires_grad=True` acts as a node in a dynamic tape-based DAG:
- When an operation is executed (e.g., `c = a * b`), PyTorch allocates an internal C++ `Node` object (`MulBackward0`) and sets `c.grad_fn = MulBackward0`.
- The node stores the saved tensors needed for backprop (e.g., `a` and `b`).
- When `loss.backward()` is called, PyTorch creates a topologically sorted dependency list and walks backward from sink to sources, invoking each node's `apply()` method to accumulate gradients into `tensor.grad`.

### 2. Activation Checkpointing (Gradient Checkpointing)
In large models (e.g., 70B Transformers), caching all intermediate activations $a^{(l)}$ across 80 layers exhausts GPU VRAM.
**Activation Checkpointing** (Chen et al. 2016) discards intermediate activations during the forward pass, retaining only the inputs to each Transformer block. During the backward pass, it recomputes the intermediate activations on-the-fly via a miniature forward pass. This trades $30\%$ extra compute for a **$5\times$ to $10\times$ reduction in peak memory**!

---

## Part 8: Code Implementation & Verification

Below is the verified, standalone test suite `06_deep_learning_foundations/code/04_backpropagation_and_scratch_autograd.py`. It implements:
1. Exact visual grid verification of all forward activations and backward gradients against Part 5 hand calculations.
2. A lightweight, complete **Scratch Autograd Engine** (`Tensor` class with reverse-mode DAG, topological sorting, `add`, `mul`, `matmul`, `relu`, and `sum`).
3. Numerical gradient checking (`gradcheck`) verifying analytical backprop against finite differences.
4. Strict parity check between scratch autograd gradients and PyTorch C++ autograd gradients to machine precision ($< 10^{-7}$).

Save the code and run it directly in Python 3.
