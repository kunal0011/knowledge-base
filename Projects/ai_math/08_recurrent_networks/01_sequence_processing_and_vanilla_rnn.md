# Chapter 8.1: Sequence Processing, Autoregressive Models & Vanilla RNN (BPTT)

---

## 1. Intuition & 101 Motivation

Feedforward networks and Convolutional Neural Networks operate under the foundational assumption that input samples are **independent and identically distributed (I.I.D.)**:
$$\mathcal{D} = \{(\mathbf{x}^{(1)}, \mathbf{y}^{(1)}), (\mathbf{x}^{(2)}, \mathbf{y}^{(2)}), \dots, (\mathbf{x}^{(N)}, \mathbf{y}^{(N)})\}$$
Processing sample $\mathbf{x}^{(i)}$ has zero mathematical dependence on sample $\mathbf{x}^{(i-1)}$.

However, natural signals in the physical universe—human speech, written language, DNA nucleotides, musical compositions, algorithmic trades, and robot sensory streams—are inherently **temporal sequences**:
$$\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$$
where the meaning, syntax, and truth value of token $\mathbf{x}_t$ depend inextricably on the preceding context $(\mathbf{x}_1, \dots, \mathbf{x}_{t-1})$.

### The Challenges of Sequential Modeling
1. **Variable Input Length:** Sentences vary from 3 to 1,000 words; feedforward networks require fixed-dimensional vector inputs.
2. **Long-Range Dependencies:** A grammatical subject introduced at $t=1$ dictates the pluralization of a verb at $t=50$.
3. **Temporal Invariance:** The phrase *"the cat sat"* has the same semantic syntax whether it appears at the start, middle, or end of a paragraph.

### The Autoregressive Factorization
By the probability chain rule, any joint distribution over sequences factorizes without loss of generality into an autoregressive product of conditional probabilities:
$$p(\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T) = \prod_{t=1}^T p(\mathbf{x}_t \mid \mathbf{x}_1, \dots, \mathbf{x}_{t-1}) = \prod_{t=1}^T p(\mathbf{x}_t \mid \mathbf{x}_{<t})$$

To compute this without maintaining an exponentially growing history of past inputs, the Recurrent Neural Network (RNN) introduces a **hidden state vector** $\mathbf{h}_t \in \mathbb{R}^{d_h}$:
$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$$
The hidden state $\mathbf{h}_t$ acts as a lossy, fixed-dimensional dynamical memory that recursively updates with each incoming token.

---

## 2. Rigorous Mathematical Formulation

```
      UNFOLDED RECURRENT NEURAL NETWORK (COMPUTATIONAL GRAPH)
       L_1                  L_2                  L_T
        ▲                    ▲                    ▲
        │                    │                    │
       y_hat_1              y_hat_2              y_hat_T
        ▲                    ▲                    ▲
        │ W_hy               │ W_hy               │ W_hy
   ┌────┴─────┐  W_hh   ┌────┴─────┐  W_hh   ┌────┴─────┐
h_0│   h_1    │────────►│   h_2    │── ... ─►│   h_T    │
──►│          │         │          │         │          │
   └────▲─────┘         └────▲─────┘         └────▲─────┘
        │ W_xh               │ W_xh               │ W_xh
        │                    │                    │
       x_1                  x_2                  x_T
```

---

### 2.1 The Forward Pass (Elman RNN Formulation)

Let $\mathbf{x} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$ be a sequence of input vectors $\mathbf{x}_t \in \mathbb{R}^{d_x}$.
Let $\mathbf{h}_t \in \mathbb{R}^{d_h}$ be the hidden state vector at time step $t$, initialized with $\mathbf{h}_0 = \mathbf{0}$.

At each time step $t \in \{1, \dots, T\}$:
1. **Pre-activation Linear Combination:**
   $$\mathbf{a}_t = \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h$$
   where:
   - $\mathbf{W}_{hh} \in \mathbb{R}^{d_h \times d_h}$ is the recurrent hidden-to-hidden transition weight matrix,
   - $\mathbf{W}_{xh} \in \mathbb{R}^{d_h \times d_x}$ is the input-to-hidden projection weight matrix,
   - $\mathbf{b}_h \in \mathbb{R}^{d_h}$ is the hidden bias vector.

2. **Non-linear Activation:**
   $$\mathbf{h}_t = \tanh(\mathbf{a}_t)$$

3. **Output Emission (for sequence-to-sequence or language modeling):**
   $$\mathbf{z}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$
   $$\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{z}_t) \in \mathbb{R}^{d_y}$$
   where $\mathbf{W}_{hy} \in \mathbb{R}^{d_y \times d_h}$ and $\mathbf{b}_y \in \mathbb{R}^{d_y}$.

4. **Sequence Objective Loss Function:**
   $$\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t(\hat{\mathbf{y}}_t, \mathbf{y}_t)$$
   For multi-class classification / language modeling, $\mathcal{L}_t$ is the cross-entropy loss:
   $$\mathcal{L}_t = - \sum_{k=1}^{d_y} y_{t, k} \log \hat{y}_{t, k}$$

---

### 2.2 Backpropagation Through Time (BPTT) Derivation

Because parameters $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{b}_h, \mathbf{W}_{hy}, \mathbf{b}_y$ are **shared identically across all time steps**, the total gradient w.r.t. any parameter $\theta$ is the sum of its partial gradients across time:
$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \theta}$$

---

#### 1. Output Layer Gradients ($\mathbf{W}_{hy}, \mathbf{b}_y$)
For cross-entropy loss with softmax activation, the error at time step $t$ is:
$$\delta_t^y \equiv \frac{\partial \mathcal{L}_t}{\partial \mathbf{z}_t} = \hat{\mathbf{y}}_t - \mathbf{y}_t \in \mathbb{R}^{d_y}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hy}} = \sum_{t=1}^T \delta_t^y \, \mathbf{h}_t^T \in \mathbb{R}^{d_y \times d_h}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_y} = \sum_{t=1}^T \delta_t^y \in \mathbb{R}^{d_y}$$

---

#### 2. Hidden State Recurrent Gradients ($\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$)
At any time step $t$, the hidden state $\mathbf{h}_t$ influences the loss in two distinct pathways:
1. **Immediately:** through the output prediction $\hat{\mathbf{y}}_t$ and loss $\mathcal{L}_t$.
2. **Into the Future:** through the next hidden state $\mathbf{h}_{t+1}$, which influences all future losses $\sum_{k=t+1}^T \mathcal{L}_k$.

Applying the multivariable chain rule backwards in time from $t = T, T-1, \dots, 1$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \underbrace{\mathbf{W}_{hy}^T \delta_t^y}_{\text{Direct loss contribution at } t} + \underbrace{\left( \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}}}_{\text{Recurrent gradient from future } t+1}$$

To compute the Jacobian $\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$:
Recall: $\mathbf{h}_{t+1} = \tanh(\mathbf{a}_{t+1})$, so $\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{a}_{t+1}} = \operatorname{diag}\left(1 - \mathbf{h}_{t+1}^2\right)$.
Since $\mathbf{a}_{t+1} = \mathbf{W}_{hh} \mathbf{h}_t + \mathbf{W}_{xh} \mathbf{x}_{t+1} + \mathbf{b}_h$, we have $\frac{\partial \mathbf{a}_{t+1}}{\partial \mathbf{h}_t} = \mathbf{W}_{hh}$.

By the chain rule:
$$\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \operatorname{diag}\left(1 - \mathbf{h}_{t+1}^2\right) \mathbf{W}_{hh}$$

Therefore, the backward recurrence relation for the hidden gradient is:
$$\delta_t^h \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \mathbf{W}_{hy}^T \delta_t^y + \mathbf{W}_{hh}^T \left( \delta_{t+1}^h \odot (1 - \mathbf{h}_{t+1}^2) \right)$$
with the terminal boundary condition at $t = T$:
$$\delta_T^h = \mathbf{W}_{hy}^T \delta_T^y$$

---

#### 3. Pre-activation Gradient ($\frac{\partial \mathcal{L}}{\partial \mathbf{a}_t}$)
Define:
$$\delta_t^a \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_t} = \delta_t^h \odot (1 - \mathbf{h}_t^2) \in \mathbb{R}^{d_h}$$

---

#### 4. Recurrent and Input Weight Gradients ($\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{b}_h$)
Accumulating across all time steps:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \delta_t^a \, \mathbf{h}_{t-1}^T \in \mathbb{R}^{d_h \times d_h}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \sum_{t=1}^T \delta_t^a \, \mathbf{x}_t^T \in \mathbb{R}^{d_h \times d_x}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_h} = \sum_{t=1}^T \delta_t^a \in \mathbb{R}^{d_h}$$

---

### 2.3 Truncated Backpropagation Through Time (TBPTT)

For sequence lengths $T \ge 1,000$, standard BPTT suffers from:
1. **Memory Exhaustion:** Storing all $T$ hidden state vectors $\{\mathbf{h}_1, \dots, \mathbf{h}_T\}$ in memory consumes $O(T \cdot B \cdot d_h)$ GPU RAM.
2. **Severe Gradient Instability:** Backpropagating across 1,000 time steps entails 1,000 successive matrix multiplications with $\mathbf{W}_{hh}^T$, causing gradients to vanish or explode exponentially.

**TBPTT Protocol:**
- Unroll the forward pass continuously across long text, maintaining the hidden state $\mathbf{h}_t$.
- Truncate the backward pass to a fixed horizon of $k_1$ steps.
- Update weights after every $k_2$ steps.
- While TBPTT solves the memory bottleneck, it destroys the network's mathematical capacity to learn statistical dependencies spanning longer than $k_1$ time steps.

---

### 2.4 Deep Derivation 8.1.1: Rigorous Formulation of Backpropagation Through Time (BPTT) via Total Derivatives and Adjoint State Methods

In this derivation, we prove the exact BPTT gradient equations using the calculus of total variations and Lagrangian duality (the discrete adjoint state method of Pontryagin's Minimum Principle).

#### Step 1: Definition of the Constrained Optimization Problem
Let the sequence of inputs be $\mathbf{x} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$, and let the recurrent hidden dynamics be governed by:
$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t; \mathbf{W}) = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h), \quad \mathbf{h}_0 = \mathbf{0}$$
At each step, a prediction is emitted: $\hat{\mathbf{y}}_t = g(\mathbf{h}_t; \mathbf{W}_{hy})$, yielding scalar step loss $\ell_t(\hat{\mathbf{y}}_t, \mathbf{y}_t)$.
The total objective function is:
$$\mathcal{L}(\mathbf{W}) = \sum_{t=1}^T \ell_t(g(\mathbf{h}_t; \mathbf{W}_{hy}), \mathbf{y}_t)$$
subject to the state transition constraints:
$$\mathbf{c}_t(\mathbf{h}_t, \mathbf{h}_{t-1}, \mathbf{W}) \equiv f(\mathbf{h}_{t-1}, \mathbf{x}_t; \mathbf{W}) - \mathbf{h}_t = \mathbf{0}, \quad \forall t \in \{1, \dots, T\}$$

#### Step 2: The Lagrangian Formulation and Adjoint Multipliers
Form the Lagrangian functional by introducing vector Lagrange multipliers (adjoint vectors) $\lambda_t \in \mathbb{R}^{d_h}$ for each constraint:
$$\mathcal{J}(\{\mathbf{h}_t\}_{t=1}^T, \mathbf{W}, \{\lambda_t\}_{t=1}^T) = \sum_{t=1}^T \ell_t(\mathbf{h}_t) + \sum_{t=1}^T \lambda_t^T \left( f(\mathbf{h}_{t-1}, \mathbf{x}_t; \mathbf{W}) - \mathbf{h}_t \right)$$

By the principle of stationary action, the total derivative $\frac{d\mathcal{L}}{d\mathbf{W}}$ is obtained by setting the variation of $\mathcal{J}$ with respect to all internal states $\mathbf{h}_t$ to zero:
$$d_{\mathbf{h}_t} \mathcal{J} = \mathbf{0}, \quad \forall t \in \{1, \dots, T\}$$

#### Step 3: Derivation of the Backward Adjoint Equations
Differentiating $\mathcal{J}$ with respect to state $\mathbf{h}_t$ (for $1 \le t < T$):
$$\frac{\partial \mathcal{J}}{\partial \mathbf{h}_t} = \frac{\partial \ell_t}{\partial \mathbf{h}_t} - \lambda_t^T + \lambda_{t+1}^T \frac{\partial f(\mathbf{h}_t, \mathbf{x}_{t+1}; \mathbf{W})}{\partial \mathbf{h}_t} = \mathbf{0}^T$$

Transposing and isolating $\lambda_t$:
$$\lambda_t = \left( \frac{\partial \ell_t}{\partial \mathbf{h}_t} \right)^T + \left( \frac{\partial f(\mathbf{h}_t, \mathbf{x}_{t+1})}{\partial \mathbf{h}_t} \right)^T \lambda_{t+1}$$

For the terminal step $t = T$, since $\mathbf{h}_T$ does not enter any future constraint:
$$\frac{\partial \mathcal{J}}{\partial \mathbf{h}_T} = \frac{\partial \ell_T}{\partial \mathbf{h}_T} - \lambda_T^T = \mathbf{0}^T \implies \lambda_T = \left( \frac{\partial \ell_T}{\partial \mathbf{h}_T} \right)^T$$

Evaluating the state transition Jacobian:
$$\frac{\partial f(\mathbf{h}_t, \mathbf{x}_{t+1})}{\partial \mathbf{h}_t} = \operatorname{diag}\left(1 - \mathbf{h}_{t+1}^2\right) \mathbf{W}_{hh} \equiv \mathbf{D}_{t+1} \mathbf{W}_{hh}$$
where $\mathbf{D}_{t+1} = \operatorname{diag}\left(1 - \mathbf{h}_{t+1}^2\right)$.

Substituting this Jacobian into the adjoint recursion:
$$\lambda_t = \left( \frac{\partial \ell_t}{\partial \mathbf{h}_t} \right)^T + \mathbf{W}_{hh}^T \mathbf{D}_{t+1} \lambda_{t+1}$$
Identifying $\lambda_t \equiv \delta_t^h = \left( \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \right)^T$ recovers the exact backward recurrence of BPTT.

#### Step 4: Total Parameter Derivative Accumulation
At the saddle point where $\frac{\partial \mathcal{J}}{\partial \mathbf{h}_t} = \mathbf{0}$ and constraints $\mathbf{c}_t = \mathbf{0}$ are satisfied, the total derivative of the loss with respect to parameters equals the partial derivative of $\mathcal{J}$:
$$\frac{d\mathcal{L}}{d\mathbf{W}_{hh}} = \frac{\partial \mathcal{J}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \lambda_t^T \frac{\partial f(\mathbf{h}_{t-1}, \mathbf{x}_t; \mathbf{W})}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \left( \mathbf{D}_t \lambda_t \right) \mathbf{h}_{t-1}^T = \sum_{t=1}^T \delta_t^a \, \mathbf{h}_{t-1}^T$$
This proves that BPTT is mathematically exact and identical to the discrete adjoint state method of optimal control. $\blacksquare$

---

### 2.5 Deep Derivation 8.1.2: Real-Time Recurrent Learning (RTRL) vs. BPTT: Algorithmic Duality and Complexity Analysis

In this derivation, we formulate **Real-Time Recurrent Learning (RTRL)**, prove its mathematical equivalence to BPTT, and analyze the fundamental space-time duality between forward-mode and reverse-mode automatic differentiation in recurrent systems.

#### Step 1: Forward Propagation of Parameter Sensitivities
Instead of propagating adjoint errors backwards in time after sequence completion, RTRL propagates the sensitivity of the current hidden state with respect to parameters **forward in time**:
$$\mathbf{S}_t \equiv \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}} \in \mathbb{R}^{d_h \times (d_h \times d_h)}$$

Differentiating the recurrent equation $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$ with respect to $\mathbf{W}_{hh}$ using the multivariable chain rule:
$$\frac{\partial \mathbf{h}_t}{\partial W_{i, j}} = \left(1 - \mathbf{h}_t^2\right) \odot \left( \mathbf{W}_{hh} \frac{\partial \mathbf{h}_{t-1}}{\partial W_{i, j}} + \frac{\partial \mathbf{W}_{hh}}{\partial W_{i, j}} \mathbf{h}_{t-1} \right)$$
where $\frac{\partial \mathbf{W}_{hh}}{\partial W_{i, j}} = \mathbf{E}_{i, j}$ is the single-entry indicator matrix with $1$ at $(i, j)$ and $0$ elsewhere.
Thus:
$$\frac{\partial \mathbf{W}_{hh}}{\partial W_{i, j}} \mathbf{h}_{t-1} = h_{t-1, j} \, \mathbf{e}_i$$
where $\mathbf{e}_i$ is the $i$-th standard basis vector in $\mathbb{R}^{d_h}$.

Therefore, each column of the sensitivity tensor evolves via the linear forward recurrence:
$$\frac{\partial \mathbf{h}_t}{\partial W_{i, j}} = \operatorname{diag}\left(1 - \mathbf{h}_t^2\right) \left[ \mathbf{W}_{hh} \frac{\partial \mathbf{h}_{t-1}}{\partial W_{i, j}} + h_{t-1, j} \, \mathbf{e}_i \right]$$
with initial condition $\frac{\partial \mathbf{h}_0}{\partial W_{i, j}} = \mathbf{0}$ for all $i, j$.

#### Step 2: Instantaneous Online Gradient Evaluation
At any time step $t$, the gradient of the immediate step loss $\ell_t$ is computed without waiting for future steps:
$$\frac{\partial \ell_t}{\partial W_{i, j}} = \left( \frac{\partial \ell_t}{\partial \mathbf{h}_t} \right) \frac{\partial \mathbf{h}_t}{\partial W_{i, j}} = \left( \delta_t^y \mathbf{W}_{hy} \right) \frac{\partial \mathbf{h}_t}{\partial W_{i, j}}$$
The total gradient across the sequence is:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \frac{\partial \ell_t}{\partial \mathbf{W}_{hh}}$$

#### Step 3: Algorithmic Duality and Complexity Comparison

| Dimension / Metric | Backpropagation Through Time (BPTT) | Real-Time Recurrent Learning (RTRL) |
| :--- | :--- | :--- |
| **AD Mode** | **Reverse-Mode AD** (Adjoint accumulation) | **Forward-Mode AD** (Sensitivity propagation) |
| **Execution Direction** | Forward pass $1 \to T$, then backward pass $T \to 1$ | Strictly forward pass $1 \to T$ concurrently with inputs |
| **Temporal Streaming** | Non-streaming (requires full sequence history) | **Fully Online & Streaming** (can run infinitely) |
| **Peak Memory Complexity** | $\mathcal{O}(T \cdot d_h)$ (stores all hidden trajectories) | $\mathcal{O}(d_h^3 + d_h^2 d_x)$ (**Independent of sequence length $T$!**) |
| **Time Complexity per Step** | $\mathcal{O}(d_h^2)$ (vector-matrix products) | $\mathcal{O}(d_h^4)$ (tensor contractions) |
| **Total Sequence Compute** | $\mathcal{O}(T \cdot d_h^2)$ | $\mathcal{O}(T \cdot d_h^4)$ |

**Conclusion:**
RTRL solves the lifelong streaming memory problem: memory consumption is strictly $\mathcal{O}(1)$ with respect to sequence length $T$. However, its quartic computation $\mathcal{O}(d_h^4)$ per step renders it intractable for standard hidden dimensions (e.g., $d_h = 1024 \implies 1024^4 \approx 10^{12}$ ops/step). BPTT remains the algorithm of choice for offline model training due to its $\mathcal{O}(d_h^2)$ efficiency. $\blacksquare$

---

### 2.6 Deep Derivation 8.1.3: Asymptotic Memory Complexity, Truncated BPTT Bias Bounds, and Optimal Checkpointing

In this derivation, we derive the exact truncation bias introduced by Truncated BPTT and prove the optimal sublinear memory bound achieved by gradient checkpointing.

#### Step 1: Truncation Bias in Truncated BPTT (TBPTT)
In full BPTT, the gradient of the loss at time step $t$ with respect to parameters $\mathbf{W}$ incorporates temporal dependencies tracing back to step $1$:
$$\nabla_{\mathbf{W}} \ell_t = \sum_{\tau=1}^t \left( \frac{\partial \ell_t}{\partial \mathbf{h}_t} \left( \prod_{k=\tau+1}^t \mathbf{J}_k \right) \frac{\partial^+ \mathbf{h}_\tau}{\partial \mathbf{W}} \right)$$
where $\mathbf{J}_k = \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \operatorname{diag}\left(1 - \mathbf{h}_k^2\right) \mathbf{W}_{hh}$, and $\frac{\partial^+ \mathbf{h}_\tau}{\partial \mathbf{W}}$ denotes direct partial derivative.

In Truncated BPTT with horizon $K \ll t$, the summation is cut off after $K$ steps:
$$\nabla_{\mathbf{W}}^{\text{TBPTT}} \ell_t = \sum_{\tau=\max(1, t-K+1)}^t \left( \frac{\partial \ell_t}{\partial \mathbf{h}_t} \left( \prod_{k=\tau+1}^t \mathbf{J}_k \right) \frac{\partial^+ \mathbf{h}_\tau}{\partial \mathbf{W}} \right)$$

The truncation error vector is the discarded historical tail:
$$\mathbf{E}_K(t) = \nabla_{\mathbf{W}} \ell_t - \nabla_{\mathbf{W}}^{\text{TBPTT}} \ell_t = \sum_{\tau=1}^{t-K} \left( \frac{\partial \ell_t}{\partial \mathbf{h}_t} \left( \prod_{k=\tau+1}^t \mathbf{J}_k \right) \frac{\partial^+ \mathbf{h}_\tau}{\partial \mathbf{W}} \right)$$

Assume the dynamical system is strictly dissipative with spectral contraction bound:
$$\sup_k \|\mathbf{J}_k\|_2 \le \rho < 1$$
Let $\|\frac{\partial \ell_t}{\partial \mathbf{h}_t}\|_2 \le M_1$ and $\|\frac{\partial^+ \mathbf{h}_\tau}{\partial \mathbf{W}}\|_2 \le M_2$. Taking Euclidean norms:
$$\|\mathbf{E}_K(t)\|_2 \le M_1 M_2 \sum_{\tau=1}^{t-K} \prod_{k=\tau+1}^t \|\mathbf{J}_k\|_2 \le M_1 M_2 \sum_{\tau=1}^{t-K} \rho^{t - \tau}$$
Let $j = t - \tau$. As $\tau$ ranges from $1$ to $t-K$, $j$ ranges from $K$ to $t-1$:
$$\|\mathbf{E}_K(t)\|_2 \le M_1 M_2 \sum_{j=K}^{t-1} \rho^j \le M_1 M_2 \sum_{j=K}^\infty \rho^j = M_1 M_2 \frac{\rho^K}{1 - \rho}$$

**Theorem (TBPTT Truncation Bound):**
If the recurrent transition operator is strictly contractive ($\rho < 1$), the truncation bias decays exponentially with horizon $K$:
$$\|\nabla_{\mathbf{W}} \ell_t - \nabla_{\mathbf{W}}^{\text{TBPTT}} \ell_t\|_2 \le \mathcal{O}(\rho^K)$$
However, if $\rho \ge 1$ (critical or chaotic regime, necessary for storing long-term context), the error does not decay, and TBPTT introduces an $O(1)$ asymptotic bias that permanently blinds the network to temporal patterns longer than $K$ steps.

#### Step 2: Optimal Memory Checkpointing (Griewank's Theorem)
To compute the exact full BPTT gradient on an uncontracted sequence of length $T$ without storing all $T$ states:
1. Partition the sequence of length $T$ into $M$ segments of length $S = T / M$.
2. In the forward pass, store only the $M$ boundary states $\{\mathbf{h}_0, \mathbf{h}_S, \mathbf{h}_{2S}, \dots, \mathbf{h}_T\}$.
3. During the backward pass, recompute the forward activations within segment $m$ on the fly from its boundary state $\mathbf{h}_{(m-1)S}$, then immediately backpropagate across that segment and discard internal states.

The total memory required is:
$$\text{Memory}(M) = \underbrace{M \cdot d_h}_{\text{Boundary Checkpoints}} + \underbrace{\frac{T}{M} \cdot d_h}_{\text{Segment Recomputation Buffer}}$$
To minimize total memory w.r.t. $M$:
$$\frac{d}{dM} \text{Memory}(M) = d_h - \frac{T}{M^2} d_h = 0 \implies M^* = \sqrt{T}$$

Substituting $M^* = \sqrt{T}$:
$$\text{Memory}^* = 2 \sqrt{T} \cdot d_h = \mathcal{O}(\sqrt{T} \cdot d_h)$$
The computational overhead is exactly one additional forward pass ($2\times$ forward FLOPs).
Thus, checkpointing reduces peak memory from $\mathcal{O}(T)$ to $\mathcal{O}(\sqrt{T})$ without introducing any approximation bias! $\blacksquare$

---

## 3. Geometric & Algebraic Interpretation

### The Discrete-Time Non-Linear Dynamical System
An RNN is fundamentally an autonomous non-linear dynamical map:
$$\mathbf{h}_t = \Phi(\mathbf{h}_{t-1}; \mathbf{x}_t)$$
Operating in hidden state phase space $\mathbb{R}^{d_h}$:
- **Fixed Points:** Coordinates $\mathbf{h}^*$ satisfying $\mathbf{h}^* = \tanh(\mathbf{W}_{hh} \mathbf{h}^* + \mathbf{b}_h)$ when input $\mathbf{x} = \mathbf{0}$.
- **Attractors:** Valleys in phase space where trajectories converge. In memory models, distinct attractors represent distinct memorized concepts (Hopfield network analogy).
- **Jacobian Spectrum:** The stability of the dynamical trajectory is governed by the eigenvalues of the Jacobian matrix:
  $$\mathbf{J}_t = \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \operatorname{diag}(1 - \mathbf{h}_t^2) \mathbf{W}_{hh}$$
  If the spectral radius $\rho(\mathbf{J}) > 1$, trajectories diverge (chaotic regime / exploding gradients). If $\rho(\mathbf{J}) < 1$, trajectories contract toward the origin (dissipative regime / vanishing gradients).

---

## 4. Real-World Analogy

### The Courtroom Stenographer vs. The Snapshot Photographer
- **Feedforward / CNN (The Photographer):** Takes a single instant photograph of a static room. Every detail must be visible in that single frame. The camera has no memory of what happened 5 minutes ago.
- **RNN (The Courtroom Stenographer):** The stenographer listens to continuous spoken testimony over 6 hours.
  - They cannot record every raw audio waveform in short-term working memory.
  - Instead, after every spoken word $\mathbf{x}_t$, they update their running mental model $\mathbf{h}_t$ ("The witness just contradicted their earlier statement").
  - When the judge asks at 5:00 PM "Why did the witness pause at 10:00 AM?", the error signal $\delta$ must trace backwards through thousands of sequential mental states. If their memory degraded at 11:00 AM, the historical connection is permanently severed.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a full forward pass and complete BPTT backward pass on a concrete toy sequence of length $T = 2$.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in Vanilla RNN |
| :--- | :--- | :--- | :--- |
| $T$ | Sequence Length | Scalar ($2$) | Number of temporal time steps |
| $\mathbf{x}_1, \mathbf{x}_2$ | Input Vectors | $(2,)$ | Inputs at time steps $t=1$ and $t=2$ |
| $\mathbf{h}_0$ | Initial Hidden State | $(2,)$ | Starting state (initialized to $[0, 0]^T$) |
| $\mathbf{W}_{xh}$ | Input-to-Hidden Weights | $(2, 2)$ | Projects input $\mathbf{x}_t$ into hidden space |
| $\mathbf{W}_{hh}$ | Hidden-to-Hidden Weights | $(2, 2)$ | Recurrent state transition matrix |
| $\mathbf{a}_1, \mathbf{a}_2$ | Pre-activations | $(2,)$ | Linear combinations before $\tanh$ |
| $\mathbf{h}_1, \mathbf{h}_2$ | Hidden State Activations | $(2,)$ | Hidden memories $\mathbf{h}_t = \tanh(\mathbf{a}_t)$ |
| $\mathbf{W}_{hy}$ | Hidden-to-Output Weights | $(2, 2)$ | Maps hidden state to output logits |
| $\mathbf{z}_1, \mathbf{z}_2$ | Output Logits | $(2,)$ | Unnormalized scores for prediction |
| $\delta_t^z \equiv \hat{\mathbf{y}}_t - \mathbf{y}_t$ | Output Loss Gradient | $(2,)$ | Error at output time step $t$ |
| $\delta_t^h \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t}$ | Hidden Gradient | $(2,)$ | Sensitivity of total loss w.r.t. hidden state at $t$ |
| $\delta_t^a \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{a}_t}$ | Pre-activation Gradient | $(2,)$ | $\delta_t^h \odot (1 - \mathbf{h}_t^2)$ |

---

### 5.2 Concrete Toy Numbers

#### Dimensions:
$d_x = 2, \quad d_h = 2, \quad d_y = 2, \quad T = 2$

#### Inputs:
$$\mathbf{x}_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad \mathbf{x}_2 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}, \quad \mathbf{h}_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

#### Model Weights (biases $= \mathbf{0}$ for crystal clarity):
$$\mathbf{W}_{xh} = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}, \quad \mathbf{W}_{hh} = \begin{bmatrix} 0.4 & 0.1 \\ 0.0 & 0.4 \end{bmatrix}, \quad \mathbf{W}_{hy} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$$

#### Targets:
At each time step, suppose downstream targets and softmax produce output errors:
$$\delta_1^z \equiv \frac{\partial \mathcal{L}_1}{\partial \mathbf{z}_1} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}, \quad \delta_2^z \equiv \frac{\partial \mathcal{L}_2}{\partial \mathbf{z}_2} = \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix}$$

---

### 5.3 Forward Pass Arithmetic

#### Time Step $t = 1$:
1. **Pre-activation $\mathbf{a}_1 = \mathbf{W}_{hh} \mathbf{h}_0 + \mathbf{W}_{xh} \mathbf{x}_1$:**
   $$\mathbf{W}_{hh} \mathbf{h}_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
   $$\mathbf{W}_{xh} \mathbf{x}_1 = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix}$$
   $$\mathbf{a}_1 = \begin{bmatrix} 0.5 \\ 0.0 \end{bmatrix}$$

2. **Hidden State $\mathbf{h}_1 = \tanh(\mathbf{a}_1)$:**
   Using $\tanh(0.5) \approx 0.462117$, $\tanh(0.0) = 0.0$:
   $$\mathbf{h}_1 = \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix}$$

3. **Output Logits $\mathbf{z}_1 = \mathbf{W}_{hy} \mathbf{h}_1$:**
   $$\mathbf{z}_1 = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix}$$

---

#### Time Step $t = 2$:
1. **Pre-activation $\mathbf{a}_2 = \mathbf{W}_{hh} \mathbf{h}_1 + \mathbf{W}_{xh} \mathbf{x}_2$:**
   $$\mathbf{W}_{hh} \mathbf{h}_1 = \begin{bmatrix} 0.4 & 0.1 \\ 0.0 & 0.4 \end{bmatrix} \begin{bmatrix} 0.462117 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.4 \times 0.462117 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.184847 \\ 0.0 \end{bmatrix}$$
   $$\mathbf{W}_{xh} \mathbf{x}_2 = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix} \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.5 \end{bmatrix}$$
   $$\mathbf{a}_2 = \begin{bmatrix} 0.184847 + 0.0 \\ 0.0 + 0.5 \end{bmatrix} = \begin{bmatrix} 0.184847 \\ 0.5 \end{bmatrix}$$

2. **Hidden State $\mathbf{h}_2 = \tanh(\mathbf{a}_2)$:**
   Using $\tanh(0.184847) \approx 0.182773$, $\tanh(0.5) \approx 0.462117$:
   $$\mathbf{h}_2 = \begin{bmatrix} 0.182773 \\ 0.462117 \end{bmatrix}$$

3. **Output Logits $\mathbf{z}_2 = \mathbf{W}_{hy} \mathbf{h}_2$:**
   $$\mathbf{z}_2 = \begin{bmatrix} 0.182773 \\ 0.462117 \end{bmatrix}$$

---

### 5.4 Backward Pass (BPTT) Arithmetic

#### 1. Backward at Time Step $t = 2$:
- **Hidden Gradient at $t = 2$:**
  Since $t=2$ is the terminal step ($T=2$), there is no future gradient:
  $$\delta_2^h = \mathbf{W}_{hy}^T \delta_2^z = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix}$$

- **Pre-activation Gradient at $t = 2$ ($\delta_2^a = \delta_2^h \odot (1 - \mathbf{h}_2^2)$):**
  $$1 - h_{2, 1}^2 = 1 - (0.182773)^2 = 1 - 0.033406 = 0.966594$$
  $$1 - h_{2, 2}^2 = 1 - (0.462117)^2 = 1 - 0.213552 = 0.786448$$
  $$\delta_2^a = \begin{bmatrix} 0.2 \times 0.966594 \\ 0.1 \times 0.786448 \end{bmatrix} = \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix}$$

- **Parameter Gradient Contributions at $t = 2$:**
  $$\frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{hh}} = \delta_2^a \, \mathbf{h}_1^T = \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix} \begin{bmatrix} 0.462117 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.089336 & 0.0 \\ 0.036343 & 0.0 \end{bmatrix}$$
  $$\frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{xh}} = \delta_2^a \, \mathbf{x}_2^T = \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix} \begin{bmatrix} 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 0.0 & 0.193319 \\ 0.0 & 0.078645 \end{bmatrix}$$

---

#### 2. Backward at Time Step $t = 1$ (The Temporal Recurrence):
- **Hidden Gradient at $t = 1$ ($\delta_1^h$):**
  $$\delta_1^h = \underbrace{\mathbf{W}_{hy}^T \delta_1^z}_{\text{Direct from } \mathcal{L}_1} + \underbrace{\mathbf{W}_{hh}^T \delta_2^a}_{\text{Recurrent from } t=2}$$
  $$\text{Direct Term} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$
  $$\text{Recurrent Term} = \begin{bmatrix} 0.4 & 0.0 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 0.193319 \\ 0.078645 \end{bmatrix} = \begin{bmatrix} 0.4(0.193319) \\ 0.1(0.193319) + 0.4(0.078645) \end{bmatrix} = \begin{bmatrix} 0.077328 \\ 0.019332 + 0.031458 \end{bmatrix} = \begin{bmatrix} 0.077328 \\ 0.050790 \end{bmatrix}$$
  $$\delta_1^h = \begin{bmatrix} 0.5 + 0.077328 \\ -0.5 + 0.050790 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.577328 \\ -0.449210 \end{bmatrix}}$$

- **Pre-activation Gradient at $t = 1$ ($\delta_1^a = \delta_1^h \odot (1 - \mathbf{h}_1^2)$):**
  $$1 - h_{1, 1}^2 = 1 - (0.462117)^2 = 0.786448$$
  $$1 - h_{1, 2}^2 = 1 - 0.0 = 1.0$$
  $$\delta_1^a = \begin{bmatrix} 0.577328 \times 0.786448 \\ -0.449210 \times 1.0 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.454038 \\ -0.449210 \end{bmatrix}}$$

- **Parameter Gradient Contributions at $t = 1$:**
  $$\frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{hh}} = \delta_1^a \, \mathbf{h}_0^T = \delta_1^a \begin{bmatrix} 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$$
  $$\frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{xh}} = \delta_1^a \, \mathbf{x}_1^T = \begin{bmatrix} 0.454038 \\ -0.449210 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 \end{bmatrix} = \begin{bmatrix} 0.454038 & 0.0 \\ -0.449210 & 0.0 \end{bmatrix}$$

---

#### 3. Total Parameter Gradients (Accumulation Over Time):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{hh}} + \frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{hh}} = \begin{bmatrix} 0.089336 & 0.0 \\ 0.036343 & 0.0 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \frac{\partial \mathcal{L}_1}{\partial \mathbf{W}_{xh}} + \frac{\partial \mathcal{L}_2}{\partial \mathbf{W}_{xh}} = \begin{bmatrix} 0.454038 & 0.193319 \\ -0.449210 & 0.078645 \end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hy}} = \delta_1^z \mathbf{h}_1^T + \delta_2^z \mathbf{h}_2^T = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} [0.462117, 0] + \begin{bmatrix} 0.2 \\ 0.1 \end{bmatrix} [0.182773, 0.462117]$$
$$= \begin{bmatrix} 0.231059 & 0 \\ -0.231059 & 0 \end{bmatrix} + \begin{bmatrix} 0.036555 & 0.092423 \\ 0.018277 & 0.046212 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.267614 & 0.092423 \\ -0.212782 & 0.046212 \end{bmatrix}}$$

Every calculation above is exact and mathematically transparent.

---

## 6. Solved Illustrations

### Illustration 1: Teacher Forcing vs. Free-Running Autoregression

**Problem:**
Explain the mathematical distinction between **Teacher Forcing** during training and **Free-Running Generation** during inference for an autoregressive sequence model. What is **Exposure Bias**?

**Solution:**
1. **Teacher Forcing (Training Phase):**
   At time step $t$, the ground-truth previous token $\mathbf{y}_{t-1}^*$ is fed as input $\mathbf{x}_t$, regardless of what the network predicted at step $t-1$:
   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{y}_{t-1}^*)$$
   - *Advantage:* Fast, stable parallel training; errors do not cascade early in training.
2. **Free-Running (Inference / Sampling Phase):**
   Ground-truth future tokens do not exist. The model must feed its own sampled output $\hat{\mathbf{y}}_{t-1}$ as input to step $t$:
   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \hat{\mathbf{y}}_{t-1})$$
3. **Exposure Bias:**
   Because the model was trained exclusively on pristine ground-truth history ($p(\mathbf{x}_t \mid \mathbf{y}_{<t}^*)$), it never learned to recover from mistakes. During free-running inference, an early minor error puts the hidden state into an out-of-distribution regime, causing errors to compound uncontrollably down the sequence.

---

### Illustration 2: Why RNNs Cannot Parallelize Training (The $O(T)$ Sequential Bottleneck)

**Problem:**
Contrast the computational graph depth and training parallelizability of an RNN vs. a Transformer on a sequence of length $T$.

**Solution:**
- **RNN:**
  The forward recurrence $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t)$ forms an inherently sequential chain. Step $t$ cannot begin until step $t-1$ finishes.
  - Compute graph depth across time: $\mathcal{O}(T)$ sequential operations.
  - Modern GPUs have thousands of parallel cores that remain underutilized because time cannot be unrolled in parallel.
- **Transformer:**
  All $T$ tokens are projected and attended simultaneously via matrix multiplications:
  $$\mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{T \times T}$$
  - Compute graph depth across time: $\mathcal{O}(1)$ parallel operations.
  - Maximizes GPU hardware occupancy, which is the primary reason Transformers replaced RNNs for large foundation models.

---

### Illustration 3: Complete 3-Step BPTT Scalar Trace with Analytical Verification

**Problem:**
Let a scalar RNN ($d_h = 1, d_x = 1, d_y = 1$) process a sequence of $T = 3$ time steps with weights:
$$w_{hh} = 0.5, \quad w_{xh} = 1.0, \quad b_h = 0.0, \quad w_{hy} = 1.0, \quad b_y = 0.0$$
The input sequence is $x_1 = 1.0, x_2 = 0.5, x_3 = -0.5$ with initial state $h_0 = 0.0$.
Downstream supervision produces scalar output error signals:
$$\delta_1^z = 0.1, \quad \delta_2^z = -0.2, \quad \delta_3^z = 0.3$$
1. Compute the forward hidden trajectory $h_1, h_2, h_3$.
2. Compute the backward adjoint signals $\delta_3^h, \delta_2^h, \delta_1^h$ and pre-activation gradients $\delta_3^a, \delta_2^a, \delta_1^a$.
3. Compute the exact parameter gradients $\frac{\partial \mathcal{L}}{\partial w_{hh}}$ and $\frac{\partial \mathcal{L}}{\partial w_{xh}}$.

**Solution:**

#### Step 1: Forward Pass Arithmetic
- **Step $t = 1$:**
  $$a_1 = w_{hh} h_0 + w_{xh} x_1 = 0.5(0.0) + 1.0(1.0) = 1.0$$
  $$h_1 = \tanh(1.0) \approx \mathbf{0.761594}$$
- **Step $t = 2$:**
  $$a_2 = w_{hh} h_1 + w_{xh} x_2 = 0.5(0.761594) + 1.0(0.5) = 0.380797 + 0.5 = 0.880797$$
  $$h_2 = \tanh(0.880797) \approx \mathbf{0.706822}$$
- **Step $t = 3$:**
  $$a_3 = w_{hh} h_2 + w_{xh} x_3 = 0.5(0.706822) + 1.0(-0.5) = 0.353411 - 0.5 = -0.146589$$
  $$h_3 = \tanh(-0.146589) \approx \mathbf{-0.145553}$$

#### Step 2: Backward Adjoint Recurrence (BPTT)
- **Time Step $t = 3$ (Terminal):**
  $$\delta_3^h = w_{hy} \delta_3^z = 1.0(0.3) = \mathbf{0.300000}$$
  $$1 - h_3^2 = 1 - (-0.145553)^2 = 1 - 0.021186 = 0.978814$$
  $$\delta_3^a = \delta_3^h (1 - h_3^2) = 0.300000 \times 0.978814 = \mathbf{0.293644}$$

- **Time Step $t = 2$:**
  $$\delta_2^h = w_{hy} \delta_2^z + w_{hh} \delta_3^a = 1.0(-0.2) + 0.5(0.293644) = -0.2 + 0.146822 = \mathbf{-0.053178}$$
  $$1 - h_2^2 = 1 - (0.706822)^2 = 1 - 0.499597 = 0.500403$$
  $$\delta_2^a = \delta_2^h (1 - h_2^2) = -0.053178 \times 0.500403 = \mathbf{-0.026610}$$

- **Time Step $t = 1$:**
  $$\delta_1^h = w_{hy} \delta_1^z + w_{hh} \delta_2^a = 1.0(0.1) + 0.5(-0.026610) = 0.1 - 0.013305 = \mathbf{0.086695}$$
  $$1 - h_1^2 = 1 - (0.761594)^2 = 1 - 0.580026 = 0.419974$$
  $$\delta_1^a = \delta_1^h (1 - h_1^2) = 0.086695 \times 0.419974 = \mathbf{0.036409}$$

#### Step 3: Parameter Gradient Accumulation
- **Recurrent Weight $w_{hh}$:**
  $$\frac{\partial \mathcal{L}}{\partial w_{hh}} = \delta_1^a h_0 + \delta_2^a h_1 + \delta_3^a h_2$$
  $$= (0.036409)(0.0) + (-0.026610)(0.761594) + (0.293644)(0.706822)$$
  $$= 0.0 - 0.020266 + 0.207554 = \mathbf{0.187288}$$

- **Input Weight $w_{xh}$:**
  $$\frac{\partial \mathcal{L}}{\partial w_{xh}} = \delta_1^a x_1 + \delta_2^a x_2 + \delta_3^a x_3$$
  $$= (0.036409)(1.0) + (-0.026610)(0.5) + (0.293644)(-0.5)$$
  $$= 0.036409 - 0.013305 - 0.146822 = \mathbf{-0.123718}$$

---

### Illustration 4: Real-Time Recurrent Learning (RTRL) Step-by-Step Forward Sensitivity Trace

**Problem:**
Using the exact model parameters and input sequence from Illustration 3:
$$w_{hh} = 0.5, \quad w_{xh} = 1.0, \quad x = [1.0, 0.5, -0.5], \quad \delta^z = [0.1, -0.2, 0.3]$$
1. Propagate the forward parameter sensitivities $S_t^{hh} \equiv \frac{\partial h_t}{\partial w_{hh}}$ and $S_t^{xh} \equiv \frac{\partial h_t}{\partial w_{xh}}$ from $t = 1$ to $t = 3$.
2. Compute the total loss gradients via the instantaneous sensitivity inner product $\frac{\partial \mathcal{L}}{\partial w} = \sum_{t=1}^3 \delta_t^z w_{hy} S_t$.
3. Prove numerical parity with the BPTT results from Illustration 3.

**Solution:**

#### Step 1: Forward Sensitivity Recurrence
Initial conditions at $t = 0$: $S_0^{hh} = 0.0, S_0^{xh} = 0.0$.

- **Time Step $t = 1$ ($h_1 = 0.761594, 1 - h_1^2 = 0.419974$):**
  $$\frac{\partial a_1}{\partial w_{hh}} = w_{hh} S_0^{hh} + h_0 = 0.5(0.0) + 0.0 = 0.0 \implies S_1^{hh} = (1 - h_1^2)(0.0) = \mathbf{0.0}$$
  $$\frac{\partial a_1}{\partial w_{xh}} = w_{hh} S_0^{xh} + x_1 = 0.5(0.0) + 1.0 = 1.0 \implies S_1^{xh} = (1 - h_1^2)(1.0) = \mathbf{0.419974}$$

- **Time Step $t = 2$ ($h_2 = 0.706822, 1 - h_2^2 = 0.500403$):**
  $$\frac{\partial a_2}{\partial w_{hh}} = w_{hh} S_1^{hh} + h_1 = 0.5(0.0) + 0.761594 = 0.761594 \implies S_2^{hh} = (0.500403)(0.761594) = \mathbf{0.381104}$$
  $$\frac{\partial a_2}{\partial w_{xh}} = w_{hh} S_1^{xh} + x_2 = 0.5(0.419974) + 0.5 = 0.209987 + 0.5 = 0.709987 \implies S_2^{xh} = (0.500403)(0.709987) = \mathbf{0.355279}$$

- **Time Step $t = 3$ ($h_3 = -0.145553, 1 - h_3^2 = 0.978814$):**
  $$\frac{\partial a_3}{\partial w_{hh}} = w_{hh} S_2^{hh} + h_2 = 0.5(0.381104) + 0.706822 = 0.190552 + 0.706822 = 0.897374$$
  $$S_3^{hh} = (0.978814)(0.897374) = \mathbf{0.878362}$$
  $$\frac{\partial a_3}{\partial w_{xh}} = w_{hh} S_2^{xh} + x_3 = 0.5(0.355279) + (-0.5) = 0.177640 - 0.5 = -0.322360$$
  $$S_3^{xh} = (0.978814)(-0.322360) = \mathbf{-0.315531}$$

#### Step 2: Online Gradient Accumulation
$$\frac{\partial \mathcal{L}}{\partial w_{hh}} = \sum_{t=1}^3 \delta_t^z w_{hy} S_t^{hh} = (0.1)(1.0)(0.0) + (-0.2)(1.0)(0.381104) + (0.3)(1.0)(0.878362)$$
$$= 0.0 - 0.076221 + 0.263509 = \mathbf{0.187288}$$

$$\frac{\partial \mathcal{L}}{\partial w_{xh}} = \sum_{t=1}^3 \delta_t^z w_{hy} S_t^{xh} = (0.1)(1.0)(0.419974) + (-0.2)(1.0)(0.355279) + (0.3)(1.0)(-0.315531)$$
$$= 0.041997 - 0.071056 - 0.094659 = \mathbf{-0.123718}$$

#### Step 3: Exact Parity Check
- BPTT $\frac{\partial \mathcal{L}}{\partial w_{hh}} = 0.187288$ vs. RTRL $\frac{\partial \mathcal{L}}{\partial w_{hh}} = 0.187288$ ($\Delta = 0.0$)
- BPTT $\frac{\partial \mathcal{L}}{\partial w_{xh}} = -0.123718$ vs. RTRL $\frac{\partial \mathcal{L}}{\partial w_{xh}} = -0.123718$ ($\Delta = 0.0$)
Both methods yield identical mathematical gradients to machine precision.

---

### Illustration 5: Truncated BPTT Horizon Bias Analysis on a Decaying Sequence

**Problem:**
Evaluate the gradient approximation error of Truncated BPTT on the 3-step sequence from Illustration 3 for truncation horizons $K = 1$ and $K = 2$.
Compare the observed truncation error against the theoretical upper bound $\frac{M \rho^K}{1 - \rho}$.

**Solution:**

#### Step 1: Gradients under Truncated BPTT
- **Horizon $K = 1$ (Zero Temporal Recurrence across steps):**
  At each step $t$, the backward pass terminates after $1$ step ($\delta_t^h = w_{hy} \delta_t^z$ without receiving terms from $\delta_{t+1}^h$):
  - $t = 3$: $\delta_3^a = 0.293644 \implies \nabla_{w_{hh}}^{(3)} = \delta_3^a h_2 = (0.293644)(0.706822) = 0.207554$
  - $t = 2$: $\delta_2^h = 1.0(-0.2) = -0.200000 \implies \delta_2^a = (-0.2)(0.500403) = -0.100081 \implies \nabla_{w_{hh}}^{(2)} = (-0.100081)(0.761594) = -0.076221$
  - $t = 1$: $\delta_1^h = 1.0(0.1) \implies \nabla_{w_{hh}}^{(1)} = \delta_1^a h_0 = 0.0$
  $$\nabla_{w_{hh}}^{\text{TBPTT}(K=1)} = 0.0 - 0.076221 + 0.207554 = \mathbf{0.131333}$$
  - Exact Full BPTT: $0.187288$
  - **Truncation Error ($K=1$):** $|0.187288 - 0.131333| = \mathbf{0.055955}$ ($29.9\%$ relative error).

- **Horizon $K = 2$ (Allows 1 Step of Temporal History):**
  Step $3$ backpropagates to step $2$, but terminates before step $1$:
  - Recurrent contribution from step $3$ into step $2$ is included: $\delta_2^h = -0.2 + 0.146822 = -0.053178 \implies \nabla_{w_{hh}}^{(2)} = -0.020266$.
  - But step $1$ still receives zero recurrent feedback from step $2$: $\nabla_{w_{hh}}^{(1)} = 0.0$.
  $$\nabla_{w_{hh}}^{\text{TBPTT}(K=2)} = 0.0 - 0.020266 + 0.207554 = \mathbf{0.187288}$$
  - **Truncation Error ($K=2$):** $|0.187288 - 0.187288| = \mathbf{0.000000}$ (since $h_0 = 0$, step 1's contribution to $w_{hh}$ is identically zero).

#### Step 2: Verification Against Theoretical Contraction Bound
The maximum Jacobian norm across steps is:
$$\rho = \max_t |(1 - h_t^2) w_{hh}| = \max(0.419974, 0.500403, 0.978814) \times 0.5 = 0.978814 \times 0.5 \approx \mathbf{0.4894} < 1$$
Because $\rho \approx 0.49 < 1$, the dynamical system is strictly contractive. The truncation error decays by a factor of $\approx \rho = 0.49$ with each increment of $K$, confirming the exponential convergence guaranteed by Deep Derivation 8.1.3.

---

## 7. Deep Learning Connection & Application

### 1. Modern Roles of RNNs / State Space Models
While Transformers dominate language modeling, the recurrent formulation $\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$ has experienced a massive resurgence through **Linear Recurrent Networks and State Space Models (SSMs)**:
- **Mamba & S4 (Gu & Dao, 2023):** Modern continuous-time linear dynamical systems that eliminate non-linear $\tanh$ recurrences, allowing training to be formulated as a parallel 1D convolution ($\mathcal{O}(\log T)$ parallel scan) while retaining $\mathcal{O}(1)$ inference memory per step.
- **Ultra-low Latency Audio & Biosensors:** Microcontrollers and edge wearables running on milliwatt power budgets rely on lightweight RNNs because $O(1)$ streaming inference requires no multi-gigabyte KV-cache.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `VanillaRNNScratch`: Pure NumPy implementation of forward pass and backward BPTT.
2. Exact numerical verification of the Part 5 Visual Grid hand arithmetic.
3. High-precision finite-difference numerical gradient checking for $\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy}$.
4. Full parity comparison against PyTorch's `nn.RNN` module with gradient matching to $< 10^{-12}$.

See implementation in:
[`08_recurrent_networks/code/01_sequence_processing_and_vanilla_rnn.py`](./code/01_sequence_processing_and_vanilla_rnn.py)
