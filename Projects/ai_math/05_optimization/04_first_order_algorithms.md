# Chapter 5.4: First-Order Algorithms (Batch GD, SGD, Mini-batch SGD)

---

## Part 1: Intuition & 101 Motivation

Every parameter update in modern deep learning—whether training a simple logistic classifier or pre-training a 70-billion parameter language model on 15 trillion tokens—relies on **First-Order Gradient Methods**.

The central dilemma of deep learning optimization is computational scale:
- If your dataset contains $N = 1,000,000,000$ examples, calculating the exact true gradient requires looping through every single sample before taking a single parameter step (**Batch Gradient Descent**). One step might take an entire day!
- At the other extreme, calculating the gradient on a single random sample ($B = 1$) makes updates instantaneous, but the gradient direction is wildly erratic and noisy (**Stochastic Gradient Descent / SGD**).
- The universal compromise is **Mini-Batch SGD** ($B \in [32, 4096]$): it unlocks massive parallel matrix multiplication on GPUs while retaining just enough stochastic noise to escape saddle points and settle into flat, generalizing basins.

```
          Batch GD (B = N)                  Pure SGD (B = 1)               Mini-Batch SGD (B = 64)
    (Smooth, Monotonic, Slow)          (Wild, Erratic, Noisy)            (Fast, Resilient, Robust)
         ╭───────────────╮                  ╭───────────────╮                  ╭───────────────╮
         │   ●           │                  │   ●   /\      │                  │   ●           │
         │    \          │                  │    \ /  \/\   │                  │    \   /\     │
         │     \         │                  │     \      \  │                  │     \ /  \    │
         │      ▼        │                  │      ▼      ▼ │                  │      ▼    ▼   │
         │       🎯      │                  │       🎯      │                  │       🎯      │
         ╰───────────────╯                  ╰───────────────╯                  ╰───────────────╯
```

In this chapter, we formalize the convergence behavior of first-order algorithms, prove how mini-batching suppresses gradient variance, and uncover why mini-batch stochasticity functions as an implicit regularizer.

---

## Part 2: Rigorous Mathematical Formulation

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be a training dataset.
The total population empirical risk objective is:
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell_i(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$
where $\ell_i(\theta)$ is the loss on sample $i$.

### 1. The Three Paradigms of Gradient Descent

#### 1. Batch Gradient Descent (BGD)
Evaluates the full gradient across all $N$ training examples before updating:
$$\mathbf{\theta_{t+1} = \theta_t - \eta \cdot \nabla \mathcal{L}(\theta_t) = \theta_t - \frac{\eta}{N} \sum_{i=1}^N \nabla \ell_i(\theta_t)}$$
- **Characteristics:** Completely deterministic. Loss decreases monotonically for sufficiently small $\eta \le 1/L$.
- **Cost:** $\mathcal{O}(N)$ gradient evaluations per step. Becomes completely unusable when $N > 10^6$.

#### 2. Stochastic Gradient Descent (SGD - Robbins & Monro, 1951)
Samples a single index $i_t \sim \text{Uniform}(\{1, 2, \dots, N\})$ at each iteration:
$$\mathbf{\theta_{t+1} = \theta_t - \eta \cdot \nabla \ell_{i_t}(\theta_t)}$$
- **Unbiased Gradient Estimator:**
  $$\mathbb{E}_{i_t}[\nabla \ell_{i_t}(\theta_t)] = \sum_{i=1}^N P(i_t = i) \nabla \ell_i(\theta_t) = \frac{1}{N}\sum_{i=1}^N \nabla \ell_i(\theta_t) = \nabla \mathcal{L}(\theta_t)$$
- **Cost:** $\mathcal{O}(1)$ gradient evaluation per step. Extremely fast, but high variance.

#### 3. Mini-Batch Stochastic Gradient Descent (Mini-Batch SGD)
Draws a random subset (mini-batch) $\mathcal{B}_t \subset \{1, \dots, N\}$ of size $B = |\mathcal{B}_t|$:
$$\mathbf{\theta_{t+1} = \theta_t - \eta \cdot \mathbf{g}_B(\theta_t) \quad \text{where} \quad \mathbf{g}_B(\theta_t) = \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla \ell_i(\theta_t)}$$
- **Unbiased:** $\mathbb{E}[\mathbf{g}_B(\theta_t)] = \nabla \mathcal{L}(\theta_t)$.
- **Cost:** $\mathcal{O}(B)$ compute per step, ideally aligned with GPU Tensor Cores.

---

### 2. Theorem: Gradient Variance Decays as $\mathcal{O}(1/B)$
Let $\Sigma(\theta) = \text{Cov}_{i \sim \text{Uniform}}(\nabla \ell_i(\theta))$ be the covariance matrix of single-sample gradients:
$$\Sigma(\theta) = \frac{1}{N} \sum_{i=1}^N (\nabla \ell_i(\theta) - \nabla \mathcal{L}(\theta))(\nabla \ell_i(\theta) - \nabla \mathcal{L}(\theta))^T$$

**Statement:**
When sampling mini-batches with replacement:
$$\mathbf{\text{Var}(\mathbf{g}_B(\theta)) = \frac{\Sigma(\theta)}{B}}$$
When sampling *without replacement* (standard PyTorch `DataLoader` per epoch):
$$\mathbf{\text{Var}(\mathbf{g}_B(\theta)) = \frac{\Sigma(\theta)}{B} \left( \frac{N - B}{N - 1} \right)}$$

Notice:
- If $B = 1$, the variance is $\Sigma(\theta)$ (maximum noise).
- If $B = N$, the finite-population correction $\frac{N - N}{N - 1} = 0$, so variance is strictly zero (exact deterministic Batch GD)!

---

### 3. Convergence Rates on Convex Objectives

| Objective Function Class | Batch Gradient Descent | Stochastic Gradient Descent (SGD) |
| :--- | :--- | :--- |
| **$L$-Smooth, Non-Convex** (Deep Nets) | $\mathcal{O}(1/\sqrt{T})$ to stationary point | $\mathcal{O}(1/T^{1/4})$ to stationary point |
| **$L$-Smooth, Convex** | $\mathcal{O}(1/T)$ | $\mathcal{O}(1/\sqrt{T})$ |
| **$L$-Smooth, $\mu$-Strongly Convex** | $\mathbf{\mathcal{O}\left( (1 - \mu/L)^T \right)}$ (Linear / Exponential) | $\mathbf{\mathcal{O}(1/T)}$ (Sublinear / Harmonic) |

*Why does SGD converge sublinearly even on strongly convex functions?*
Because the gradient noise $\mathbf{g}_B - \nabla \mathcal{L}$ does not vanish as $\theta_t \to \theta^*$. Even at the optimal solution $\theta^*$ where $\nabla \mathcal{L}(\theta^*) = \mathbf{0}$, individual sample gradients $\nabla \ell_i(\theta^*)$ are non-zero, continuously kicking the optimizer away from the minimum!

---

### 4. The Robbins-Monro Step Size Conditions
To guarantee that SGD converges almost surely to a stationary point despite persistent gradient variance, the sequence of learning rates $\eta_t$ must satisfy the **Robbins-Monro conditions**:

1. **Infinite Reach:**
   $$\mathbf{\sum_{t=1}^\infty \eta_t = \infty}$$
   *(Ensures the optimizer can travel any arbitrary distance to reach the optimum regardless of initialization).*
2. **Variance Suppression:**
   $$\mathbf{\sum_{t=1}^\infty \eta_t^2 < \infty}$$
   *(Ensures the cumulative variance noise dampens to zero so the optimizer settles into the exact minimum).*

#### Standard Robbins-Monro Schedules:
$$\eta_t = \frac{\eta_0}{t^\alpha} \quad \text{for } \alpha \in (0.5, 1.0]$$
- If $\alpha = 1$: $\sum 1/t = \infty$ and $\sum 1/t^2 = \pi^2 / 6 < \infty$ ($\checkmark$ Satisfied!).
- Constant learning rate $\eta_t = \eta$: $\sum \eta^2 = \infty$ (Violated!). Under a constant learning rate, SGD does not converge to a single point, but rather fluctuates indefinitely inside a **Gaussian noise ball** around $\theta^*$ of radius $\mathcal{O}(\sqrt{\eta})$!

---

### 5. The Linear Scaling Rule for Batch Size
When distributing training across multiple GPUs, we scale the total batch size from $B \to k B$.
Goyal et al. (2017, Facebook AI Research) proved the **Linear Scaling Rule**:
$$\mathbf{\text{When batch size scales } B \to k B, \quad \text{scale learning rate } \eta \to k \eta}$$
The updates match!

---

### Deep Derivation 5.4.1: First-Principles Convergence Proof of Gradient Descent for $L$-Smooth Non-Convex Functions

We prove that for any continuously differentiable, $L$-smooth function (the general class of deep neural network loss surfaces), Gradient Descent with step size $\eta \le 1/L$ converges to an approximate stationary point at rate $\mathcal{O}(1/\sqrt{T})$.

#### 1. The Descent Lemma
By definition of $L$-Lipschitz smoothness, $\|\nabla f(y) - \nabla f(x)\|_2 \le L \|y - x\|_2$.
Integrating along the line segment between $x$ and $y$:
$$f(y) \le f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|_2^2$$

#### 2. Substituting the Gradient Step
Apply this lemma to consecutive iterates $x_{t+1} = x_t - \eta \nabla f(x_t)$, so $x_{t+1} - x_t = -\eta \nabla f(x_t)$:
$$f(x_{t+1}) \le f(x_t) + \nabla f(x_t)^T (-\eta \nabla f(x_t)) + \frac{L}{2} \|-\eta \nabla f(x_t)\|_2^2$$
$$= f(x_t) - \eta \|\nabla f(x_t)\|_2^2 + \frac{\eta^2 L}{2} \|\nabla f(x_t)\|_2^2 = f(x_t) - \eta \left( 1 - \frac{\eta L}{2} \right) \|\nabla f(x_t)\|_2^2$$

#### 3. Guaranteed Monotonic Descent
For any learning rate $\eta \le \frac{1}{L}$, we have $1 - \frac{\eta L}{2} \ge \frac{1}{2}$.
Setting $\eta = \frac{1}{L}$:
$$f(x_{t+1}) \le f(x_t) - \frac{1}{2L} \|\nabla f(x_t)\|_2^2 \implies \|\nabla f(x_t)\|_2^2 \le 2L [f(x_t) - f(x_{t+1})]$$
Notice: Every single step is guaranteed to strictly decrease the objective function unless the gradient is already zero!

#### 4. Telescoping Summation Across $T$ Iterations
Sum both sides from $t = 0$ to $T - 1$:
$$\sum_{t=0}^{T-1} \|\nabla f(x_t)\|_2^2 \le 2L \sum_{t=0}^{T-1} [f(x_t) - f(x_{t+1})] = 2L [f(x_0) - f(x_T)]$$
Let $f^*$ be the global infimum of $f$. Since $f(x_T) \ge f^*$, we have $f(x_0) - f(x_T) \le f(x_0) - f^*$:
$$\sum_{t=0}^{T-1} \|\nabla f(x_t)\|_2^2 \le 2L [f(x_0) - f^*]$$

#### 5. Deriving the Asymptotic Rate
Divide by $T$:
$$\min_{0 \le t \le T-1} \|\nabla f(x_t)\|_2^2 \le \frac{1}{T} \sum_{t=0}^{T-1} \|\nabla f(x_t)\|_2^2 \le \frac{2L (f(x_0) - f^*)}{T}$$
Taking the square root:
$$\mathbf{\min_{0 \le t \le T-1} \|\nabla f(x_t)\|_2 \le \sqrt{\frac{2L (f(x_0) - f^*)}{T}} = \mathcal{O}\left( \frac{1}{\sqrt{T}} \right)} \quad \blacksquare$$
*Significance:* To achieve an $\epsilon$-approximate stationary point ($\|\nabla f\| \le \epsilon$), Gradient Descent requires at most $T = \mathcal{O}(1/\epsilon^2)$ steps.

---

### Deep Derivation 5.4.2: First-Principles Convergence Proof of Stochastic Gradient Descent (SGD)

We prove that for convex objectives with bounded gradient variance, SGD converges at rate $\mathcal{O}(1/\sqrt{T})$.

#### 1. Setup and Assumptions
Let $f: \mathbb{R}^n \to \mathbb{R}$ be convex with minimizer $x^*$.
Assume:
1. Unbiased stochastic gradients: $\mathbb{E}[\mathbf{g}_t \mid x_t] = \nabla f(x_t)$.
2. Bounded second moment: $\mathbb{E}[\|\mathbf{g}_t\|_2^2 \mid x_t] \le G^2$.
Update rule: $x_{t+1} = x_t - \eta \mathbf{g}_t$.

#### 2. Distance to Optimum Recurrence
$$\|x_{t+1} - x^*\|_2^2 = \|x_t - x^* - \eta \mathbf{g}_t\|_2^2 = \|x_t - x^*\|_2^2 - 2\eta \mathbf{g}_t^T (x_t - x^*) + \eta^2 \|\mathbf{g}_t\|_2^2$$
Take the conditional expectation $\mathbb{E}[\cdot \mid x_t]$:
$$\mathbb{E}[\|x_{t+1} - x^*\|_2^2 \mid x_t] \le \|x_t - x^*\|_2^2 - 2\eta \nabla f(x_t)^T (x_t - x^*) + \eta^2 G^2$$

#### 3. Invoking Convexity
By the first-order definition of convexity: $\nabla f(x_t)^T (x_t - x^*) \ge f(x_t) - f(x^*)$.
Substituting:
$$\mathbb{E}[\|x_{t+1} - x^*\|_2^2 \mid x_t] \le \|x_t - x^*\|_2^2 - 2\eta [f(x_t) - f(x^*)] + \eta^2 G^2$$
Rearranging to isolate the suboptimality gap:
$$f(x_t) - f(x^*) \le \frac{\|x_t - x^*\|_2^2 - \mathbb{E}[\|x_{t+1} - x^*\|_2^2]}{2\eta} + \frac{\eta G^2}{2}$$

#### 4. Telescoping Summation and Jensen's Average
Sum over $t = 0, 1, \dots, T - 1$ and take total expectations:
$$\sum_{t=0}^{T-1} \mathbb{E}[f(x_t) - f(x^*)] \le \frac{\|x_0 - x^*\|_2^2 - \mathbb{E}[\|x_T - x^*\|_2^2]}{2\eta} + \frac{T \eta G^2}{2} \le \frac{\|x_0 - x^*\|_2^2}{2\eta} + \frac{T \eta G^2}{2}$$

Let $R = \|x_0 - x^*\|_2$. Define the averaged iterate $\bar{x}_T = \frac{1}{T}\sum_{t=0}^{T-1} x_t$.
By Jensen's inequality:
$$\mathbb{E}[f(\bar{x}_T) - f(x^*)] \le \frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[f(x_t) - f(x^*)] \le \frac{R^2}{2\eta T} + \frac{\eta G^2}{2}$$

#### 5. Optimal Learning Rate Minimization
Minimize the upper bound with respect to $\eta$:
$$\frac{d}{d\eta} \left( \frac{R^2}{2\eta T} + \frac{\eta G^2}{2} \right) = -\frac{R^2}{2\eta^2 T} + \frac{G^2}{2} = 0 \implies \mathbf{\eta^* = \frac{R}{G\sqrt{T}}}$$
Substituting $\eta^*$ back into the error bound:
$$\mathbf{\mathbb{E}[f(\bar{x}_T) - f(x^*)] \le \frac{R G}{\sqrt{T}} = \mathcal{O}\left( \frac{1}{\sqrt{T}} \right)} \quad \blacksquare$$

---

### Deep Derivation 5.4.3: Exact Finite-Population Correction Proof for Mini-Batch Variance

In standard deep learning pipelines (e.g. PyTorch `DataLoader(shuffle=True)`), mini-batches are drawn **without replacement** within each epoch. We prove that this reduces gradient variance by the exact finite-population factor $\frac{N - B}{N - 1}$.

#### 1. Setup
Let $\mathbf{u}_i = \nabla \ell_i(\theta)$ for $i = 1, \dots, N$ with population mean $\bar{\mathbf{u}} = \frac{1}{N}\sum_{i=1}^N \mathbf{u}_i = \nabla \mathcal{L}(\theta)$.
The population covariance matrix is:
$$\Sigma = \frac{1}{N - 1} \sum_{i=1}^N (\mathbf{u}_i - \bar{\mathbf{u}})(\mathbf{u}_i - \bar{\mathbf{u}})^T$$
A mini-batch $\mathcal{B} \subset \{1, \dots, N\}$ of size $B$ is drawn uniformly without replacement.
The mini-batch sample mean is:
$$\mathbf{g}_B = \frac{1}{B} \sum_{i \in \mathcal{B}} \mathbf{u}_i = \frac{1}{B} \sum_{i=1}^N I_i \mathbf{u}_i$$
where $I_i = \mathbf{1}_{\{i \in \mathcal{B}\}}$ is the sample inclusion indicator variable.

#### 2. Moments of Indicator Variables
1. $\mathbb{E}[I_i] = P(i \in \mathcal{B}) = \frac{B}{N}$.
2. $\mathbb{E}[I_i^2] = \mathbb{E}[I_i] = \frac{B}{N}$ (since $I_i \in \{0, 1\}$).
3. $\text{Var}(I_i) = \mathbb{E}[I_i^2] - (\mathbb{E}[I_i])^2 = \frac{B}{N} - \frac{B^2}{N^2} = \frac{B(N - B)}{N^2}$.
4. For $i \ne j$:
   $$\mathbb{E}[I_i I_j] = P(i \in \mathcal{B} \text{ and } j \in \mathcal{B}) = \frac{B}{N} \cdot \frac{B - 1}{N - 1}$$
   $$\text{Cov}(I_i, I_j) = \mathbb{E}[I_i I_j] - \mathbb{E}[I_i]\mathbb{E}[I_j] = \frac{B(B - 1)}{N(N - 1)} - \frac{B^2}{N^2} = -\frac{B(N - B)}{N^2(N - 1)}$$

#### 3. Covariance of Mini-Batch Mean
Without loss of generality, subtract the constant mean $\bar{\mathbf{u}}$ so that $\sum_{i=1}^N \mathbf{u}_i = \mathbf{0}$:
$$\text{Cov}(\mathbf{g}_B) = \mathbb{E}\left[ \mathbf{g}_B \mathbf{g}_B^T \right] = \frac{1}{B^2} \sum_{i=1}^N \sum_{j=1}^N \mathbb{E}[I_i I_j] \mathbf{u}_i \mathbf{u}_j^T$$
Separate into diagonal ($i = j$) and off-diagonal ($i \ne j$) terms:
$$= \frac{1}{B^2} \left[ \sum_{i=1}^N \mathbb{E}[I_i^2] \mathbf{u}_i \mathbf{u}_i^T + \sum_{i \ne j} \mathbb{E}[I_i I_j] \mathbf{u}_i \mathbf{u}_j^T \right]$$
$$= \frac{1}{B^2} \left[ \frac{B}{N} \sum_{i=1}^N \mathbf{u}_i \mathbf{u}_i^T + \frac{B(B - 1)}{N(N - 1)} \sum_{i \ne j} \mathbf{u}_i \mathbf{u}_j^T \right]$$

Since $\sum_{i=1}^N \mathbf{u}_i = \mathbf{0}$, we have:
$$\left( \sum_{i=1}^N \mathbf{u}_i \right) \left( \sum_{j=1}^N \mathbf{u}_j \right)^T = \sum_{i=1}^N \mathbf{u}_i \mathbf{u}_i^T + \sum_{i \ne j} \mathbf{u}_i \mathbf{u}_j^T = \mathbf{0} \implies \sum_{i \ne j} \mathbf{u}_i \mathbf{u}_j^T = -\sum_{i=1}^N \mathbf{u}_i \mathbf{u}_i^T$$

Substitute this into the covariance expression:
$$\text{Cov}(\mathbf{g}_B) = \frac{1}{B^2} \left[ \frac{B}{N} - \frac{B(B - 1)}{N(N - 1)} \right] \sum_{i=1}^N \mathbf{u}_i \mathbf{u}_i^T$$
Simplify the bracketed coefficient:
$$\frac{B}{N} - \frac{B(B - 1)}{N(N - 1)} = \frac{B(N - 1) - B(B - 1)}{N(N - 1)} = \frac{B(N - B)}{N(N - 1)}$$
Therefore:
$$\text{Cov}(\mathbf{g}_B) = \frac{1}{B^2} \left[ \frac{B(N - B)}{N(N - 1)} \right] \sum_{i=1}^N \mathbf{u}_i \mathbf{u}_i^T = \frac{N - B}{B N} \left( \frac{1}{N - 1} \sum_{i=1}^N \mathbf{u}_i \mathbf{u}_i^T \right) = \mathbf{\frac{\Sigma}{B} \left( \frac{N - B}{N} \right)}$$
or relative to the sample covariance:
$$\mathbf{\text{Var}(\mathbf{g}_B) = \frac{\Sigma}{B} \left( \frac{N - B}{N - 1} \right)} \quad \blacksquare$$

---

## Part 3: Geometric & Algebraic Interpretation

### Continuous-Time Limit & The Langevin SDE
In deep learning, SGD does not simply optimize—it performs **statistical sampling**.
Decompose the mini-batch gradient into true gradient plus zero-mean noise:
$$\mathbf{g}_B(\theta) = \nabla \mathcal{L}(\theta) + \xi_t, \quad \mathbb{E}[\xi_t] = \mathbf{0}, \quad \text{Cov}(\xi_t) \approx \frac{\Sigma(\theta)}{B}$$
The discrete SGD update is:
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t) - \eta \xi_t$$
In continuous time ($dt = \eta \to 0$), this dynamical system is governed by the **Langevin Stochastic Differential Equation**:
$$\mathbf{d\theta_t = -\nabla \mathcal{L}(\theta_t) \, dt + \sqrt{\frac{2}{\beta}} \, dW_t}$$
where $W_t$ is standard Brownian motion, and the **Effective Thermal Temperature** is:
$$\mathbf{T = \frac{1}{\beta} = \frac{\eta}{2 B} \Sigma(\theta)}$$

```
                   The SGD Thermal Temperature T = η / (2B)
                   
          High Temperature (High η, Small B)           Low Temperature (Low η, Large B)
                 ▲                                            ▲
                 │    \   *   /                               │      \       /
                 │     \ / \ /  ◄── Thermal noise jumps       │       \  *  /  ◄── Trapped in sharp
                 │      ●   ●       out of sharp minima!      │        \●  /       crevices!
                 └────────────────────────► θ                 └────────────────────────► θ
```

#### Why Mini-Batch SGD Generalizes Better Than Batch GD:
- **Batch GD ($B = N$):** Temperature $T \to 0$ (Freezing limit). The optimizer falls into the closest local minimum, which is often narrow, sharp, and overfits.
- **Mini-Batch SGD ($B \approx 64$):** Temperature $T > 0$. Thermal fluctuations constantly shake the weights out of sharp, narrow crevasses, allowing the optimizer to settle only in broad, flat, robust basins that generalize well to unseen test data!

---

## Part 4: Real-World Analogy

### The Roman Legion vs. The Reconnaissance Squads
Imagine exploring an uncharted, foggy valley to find the lowest fertile plain:
- **Batch Gradient Descent (The Roman Legion of 50,000 Soldiers):**
  All 50,000 soldiers stand in a rigid grid. Every scout reports their local slope. The general computes the exact average slope across all 50,000 reports, and the entire army takes one massive, perfectly coordinated step forward.
  *Verdict:* Incredibly stable, zero wasted motion, but agonizingly slow.
- **Pure SGD (The Lone Scout):**
  A single scout runs ahead blindly, shouts where he thinks the slope goes, and drags the camp. If he trips in a pothole, the entire camp moves in the wrong direction.
  *Verdict:* Fast, but chaotic and dizzying.
- **Mini-Batch SGD (A Squad of 32 Alert Scouts):**
  A compact squad of 32 agile scouts sprint in a loose formation. They quickly shout their consensus every 3 seconds. If one scout trips on a rock, the other 31 instantly average out the mistake.
  *Verdict:* Blazing fast, resilient, and agile enough to skirt around muddy quagmires!

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us compute one full optimization step by hand comparing **Batch GD ($B = 4$)**, **Mini-Batch SGD ($B = 2$)**, and **Pure SGD ($B = 1$)** on a concrete 4-sample dataset.

### 1. Problem Setup & Toy Dataset
Consider a 1D linear regression model through the origin: $\hat{y} = w x$.
Objective is Mean Squared Error with factor $\frac{1}{2}$:
$$\mathcal{L}(w) = \frac{1}{N} \sum_{i=1}^N \ell_i(w) = \frac{1}{N} \sum_{i=1}^N \frac{1}{2}(w x_i - y_i)^2$$
Per-sample gradient:
$$\nabla \ell_i(w) = (w x_i - y_i) x_i = w x_i^2 - x_i y_i$$

$N = 4$ data points:
- Sample 1: $(x_1, y_1) = (1.0, 2.0)$
- Sample 2: $(x_2, y_2) = (2.0, 3.0)$
- Sample 3: $(x_3, y_3) = (3.0, 7.0)$
- Sample 4: $(x_4, y_4) = (4.0, 8.0)$

Let initial weight be $w_0 = 1.0000$.
Let learning rate be $\eta = 0.05$.

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Shape / Type | Toy Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $N$ | Total dataset size | Training set cardinality | Integer scalar | $4$ |
| $B$ | Batch size | Number of samples per GPU step | Integer scalar | $4, 2, 1$ |
| $w_0$ | Initial parameter | Model weights at iteration $t$ | Float scalar | $1.0000$ |
| $\eta$ | Learning rate | Optimizer step size | Float scalar | $0.05$ |
| $\hat{y}_i$ | Prediction $w_0 x_i$ | Model forward output | Float scalar | $[1.0, 2.0, 3.0, 4.0]$ |
| $e_i$ | Residual $w_0 x_i - y_i$ | Forward error | Float scalar | $[-1.0, -1.0, -4.0, -4.0]$ |
| $\nabla \ell_i$ | Individual sample gradient | Per-sample backprop gradient | Float scalar | $[-1.0, -2.0, -12.0, -16.0]$ |
| $\mathbf{g}_B$ | Mini-batch gradient estimate | Mini-batch averaged gradient | Float scalar | Calculated per paradigm |
| $w_1$ | Updated parameter | Weights after one step | Float scalar | $w_0 - \eta \mathbf{g}_B$ |

---

### 3. Step-by-Step Manual Arithmetic

#### Step 1: Compute All 4 Per-Sample Gradients at $w_0 = 1.0000$
- **Sample 1:**
  $$\hat{y}_1 = 1.0(1.0) = 1.0 \implies e_1 = 1.0 - 2.0 = -1.0$$
  $$\nabla \ell_1 = e_1 x_1 = (-1.0)(1.0) = \mathbf{-1.0000}$$
- **Sample 2:**
  $$\hat{y}_2 = 1.0(2.0) = 2.0 \implies e_2 = 2.0 - 3.0 = -1.0$$
  $$\nabla \ell_2 = e_2 x_2 = (-1.0)(2.0) = \mathbf{-2.0000}$$
- **Sample 3:**
  $$\hat{y}_3 = 1.0(3.0) = 3.0 \implies e_3 = 3.0 - 7.0 = -4.0$$
  $$\nabla \ell_3 = e_3 x_3 = (-4.0)(3.0) = \mathbf{-12.0000}$$
- **Sample 4:**
  $$\hat{y}_4 = 1.0(4.0) = 4.0 \implies e_4 = 4.0 - 8.0 = -4.0$$
  $$\nabla \ell_4 = e_4 x_4 = (-4.0)(4.0) = \mathbf{-16.0000}$$

---

#### Step 2: Compute Weight Update Under All 3 Paradigms

**Paradigm 1: Batch Gradient Descent ($B = 4$ — Full Batch):**
$$\nabla \mathcal{L}_{\text{BGD}} = \frac{1}{4} \sum_{i=1}^4 \nabla \ell_i = \frac{-1.0 + (-2.0) + (-12.0) + (-16.0)}{4} = \frac{-31.0000}{4} = \mathbf{-7.7500}$$
Update step:
$$w_1 = w_0 - \eta \nabla \mathcal{L}_{\text{BGD}} = 1.0000 - 0.05(-7.7500) = 1.0000 + 0.3875 = \mathbf{1.3875}$$

**Paradigm 2: Mini-Batch SGD ($B = 2$, Batch $\mathcal{B} = \{1, 4\}$):**
$$\mathbf{g}_B = \frac{\nabla \ell_1 + \nabla \ell_4}{2} = \frac{-1.0000 + (-16.0000)}{2} = \frac{-17.0000}{2} = \mathbf{-8.5000}$$
Update step:
$$w_1 = 1.0000 - 0.05(-8.5000) = 1.0000 + 0.4250 = \mathbf{1.4250}$$

**Paradigm 3: Pure SGD ($B = 1$, Sample 3 chosen):**
$$\mathbf{g} = \nabla \ell_3 = \mathbf{-12.0000}$$
Update step:
$$w_1 = 1.0000 - 0.05(-12.0000) = 1.0000 + 0.6000 = \mathbf{1.6000}$$

**Alternative Pure SGD ($B = 1$, Sample 1 chosen):**
$$\mathbf{g} = \nabla \ell_1 = \mathbf{-1.0000}$$
Update step:
$$w_1 = 1.0000 - 0.05(-1.0000) = 1.0000 + 0.0500 = \mathbf{1.0500}$$

---

### 4. Visual Summary Grid

```
┌──────┬──────────┬──────────┬─────────────┬─────────────┬─────────────┬────────────────────────────────┐
│ i    │ x_i      │ y_i      │ Pred ŷ_i    │ Error e_i   │ Grad ∇ℓ_i   │ Optimization Paradigm Step     │
├──────┼──────────┼──────────┼─────────────┼─────────────┼─────────────┼────────────────────────────────┤
│ 1    │   1.00   │   2.00   │    1.00     │    -1.00    │   -1.0000   │ Pure SGD (Sample 1): w₁=1.0500 │
│ 2    │   2.00   │   3.00   │    2.00     │    -1.00    │   -2.0000   │                                │
│ 3    │   3.00   │   7.00   │    3.00     │    -4.00    │  -12.0000   │ Pure SGD (Sample 3): w₁=1.6000 │
│ 4    │   4.00   │   8.00   │    4.00     │    -4.00    │  -16.0000   │                                │
├──────┴──────────┴──────────┴─────────────┴─────────────┴─────────────┼────────────────────────────────┤
│ Batch GD (B = 4):         g = (-1 - 2 - 12 - 16) / 4 = -7.7500       │ w₁ = 1.0 - 0.05(-7.75) = 1.3875│
│ Mini-Batch SGD (B = 2):   g = (-1 - 16) / 2          = -8.5000       │ w₁ = 1.0 - 0.05(-8.50) = 1.4250│
└──────────────────────────────────────────────────────────────────────┴────────────────────────────────┘
```

Notice:
- The Mini-batch estimate ($-8.5000$) is remarkably close to the true full-batch gradient ($-7.7500$), costing **half the computation**!
- Pure SGD swings wildly depending on whether it draws an easy sample ($-1.0$) or an outlier sample ($-16.0$).

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): Gradient Variance vs. Batch Size Decay
Let us compute the variance of single-sample gradients in Part 5:
- Gradients: $[-1.0, -2.0, -12.0, -16.0]$
- Mean: $\mu_g = -7.7500$
- Variance of single sample:
  $$\sigma^2 = \frac{(-1 + 7.75)^2 + (-2 + 7.75)^2 + (-12 + 7.75)^2 + (-16 + 7.75)^2}{4} = \frac{45.5625 + 33.0625 + 18.0625 + 68.0625}{4} = \mathbf{41.1875}$$
- For mini-batch of size $B = 2$ without replacement:
  $$\text{Var}(\mathbf{g}_{B=2}) = \frac{\sigma^2}{B} \left( \frac{N - B}{N - 1} \right) = \frac{41.1875}{2} \left( \frac{4 - 2}{4 - 1} \right) = 20.5938 \times \frac{2}{3} = \mathbf{13.7292}$$
  The variance dropped by **67%** with just 2 samples!

---

### Illustration 2 (Boundary): Constant Learning Rate and the Gaussian Noise Ball
Suppose we optimize a 1D quadratic $f(\theta) = \frac{1}{2} a \theta^2$ with constant learning rate $\eta$, where gradient evaluations have additive noise $\xi_t \sim \mathcal{N}(0, \sigma^2)$.
$$\theta_{t+1} = \theta_t - \eta(a \theta_t + \xi_t) = (1 - \eta a)\theta_t - \eta \xi_t$$
Taking variance on both sides at steady-state ($t \to \infty$ where $\text{Var}(\theta_{t+1}) = \text{Var}(\theta_t) = V_\infty$):
$$V_\infty = (1 - \eta a)^2 V_\infty + \eta^2 \sigma^2$$
$$V_\infty [ 1 - (1 - 2\eta a + \eta^2 a^2) ] = \eta^2 \sigma^2 \implies V_\infty (2\eta a - \eta^2 a^2) = \eta^2 \sigma^2$$
Assuming $\eta a \ll 1$:
$$\mathbf{V_\infty \approx \frac{\eta \sigma^2}{2a}}$$
*Key Insight:* Under constant learning rate, **SGD will never converge to $\theta^* = 0$**! It hovers endlessly in a Gaussian cloud whose variance is directly proportional to $\eta$. To reach the exact minimum, you **must decay $\eta \to 0$**!

---

### Illustration 3 (Edge Case): Critical Batch Size (OpenAI, 2018)
McCandlish et al. (2018) demonstrated that there is a **Critical Batch Size $B_{\text{crit}}$** for every neural network architecture:
$$B_{\text{crit}} = \frac{\text{tr}(\Sigma \mathcal{H})}{\|\nabla \mathcal{L}\|_2^2}$$
- When $B \ll B_{\text{crit}}$: Gradient noise dominates. Doubling batch size $B \to 2B$ cuts the required optimizer steps in half (perfect linear speedup).
- When $B \gg B_{\text{crit}}$: True gradient dominates. Doubling batch size yields almost zero reduction in steps, wasting millions of GPU hours!
In modern LLM training, $B_{\text{crit}}$ starts small ($B \approx 512$) at the beginning of training and expands to massive sizes ($B \approx 4M$ tokens) near convergence!

---

### Illustration 4 (Numerical): Backtracking Armijo Line Search on an Ill-Conditioned Quadratic

Choosing a fixed learning rate $\eta$ in first-order optimization often leads to catastrophic divergence or excruciatingly slow progress. The **Armijo Backtracking Line Search** guarantees monotonic descent by automatically adapting the step size.

#### 1. The Armijo Condition
A proposed step size $\eta$ is accepted if:
$$f(x_t - \eta \nabla f(x_t)) \le f(x_t) - c \cdot \eta \|\nabla f(x_t)\|_2^2$$
where $c \in (0, 1)$ is the required descent slope parameter (typically $c = 0.50$). If violated, the step size is contracted by factor $\beta \in (0, 1)$ (typically $\beta = 0.50$): $\eta \leftarrow \beta \eta$.

#### 2. Problem Setup
- **Objective:** $f(x, y) = 10 x^2 + y^2$ (Hessian $\mathcal{H} = \text{diag}(20, 2)$, condition number $\kappa = 10.0$).
- **Current Position:** $(x_0, y_0) = (1.0, 1.0) \implies f(x_0, y_0) = 10(1)^2 + 1^2 = \mathbf{11.0000}$.
- **Gradient Vector:**
  $$\nabla f(x_0, y_0) = \begin{bmatrix} 20 x_0 \\ 2 y_0 \end{bmatrix} = \begin{bmatrix} 20.0 \\ 2.0 \end{bmatrix}$$
- **Squared Gradient Norm:**
  $$\|\nabla f(x_0, y_0)\|_2^2 = 20.0^2 + 2.0^2 = 400.0 + 4.0 = \mathbf{404.0000}$$
- **Armijo Parameters:** $c = 0.50$, contraction $\beta = 0.50$, initial trial step size $\eta_0 = 0.20$.
- **Required Upper Bound:**
  $$\text{Bound}(\eta) = f(x_0) - c \eta \|\nabla f\|_2^2 = 11.0000 - 0.50 \times 404.0000 \times \eta = \mathbf{11.0000 - 202.0 \eta}$$

#### 3. Backtracking Iteration 1 ($\eta = 0.20$)
- Test position:
  $$x_1 = 1.0 - 0.20(20.0) = 1.0 - 4.0 = -3.0$$
  $$y_1 = 1.0 - 0.20(2.0) = 1.0 - 0.40 = 0.60$$
- Function value at test point:
  $$f(-3.0, 0.60) = 10(-3.0)^2 + 0.60^2 = 90.0 + 0.36 = \mathbf{90.3600}$$
- Armijo condition check:
  $$\text{Bound}(0.20) = 11.0000 - 202.0(0.20) = 11.0000 - 40.4000 = \mathbf{-29.4000}$$
  $$f(-3.0, 0.60) = 90.3600 \le -29.4000 \quad (\mathbf{FAILED! \,\, Overshot, loss exploded from } 11 \to 90.36!)$$
- Action: Contract step size: $\eta \leftarrow 0.20 \times 0.50 = \mathbf{0.1000}$.

#### 4. Backtracking Iteration 2 ($\eta = 0.10$)
- Test position:
  $$x_1 = 1.0 - 0.10(20.0) = 1.0 - 2.0 = -1.0$$
  $$y_1 = 1.0 - 0.10(2.0) = 1.0 - 0.20 = 0.80$$
- Function value at test point:
  $$f(-1.0, 0.80) = 10(-1.0)^2 + 0.80^2 = 10.0 + 0.64 = \mathbf{10.6400}$$
- Armijo condition check:
  $$\text{Bound}(0.10) = 11.0000 - 202.0(0.10) = 11.0000 - 20.2000 = \mathbf{-9.2000}$$
  $$f(-1.0, 0.80) = 10.6400 \le -9.2000 \quad (\mathbf{FAILED!})$$
- Action: Contract step size: $\eta \leftarrow 0.10 \times 0.50 = \mathbf{0.0500}$.

#### 5. Backtracking Iteration 3 ($\eta = 0.0500$)
- Test position:
  $$x_1 = 1.0 - 0.05(20.0) = 1.0 - 1.0 = \mathbf{0.0000}$$
  $$y_1 = 1.0 - 0.05(2.0) = 1.0 - 0.10 = \mathbf{0.9000}$$
- Function value at test point:
  $$f(0.0, 0.90) = 10(0.0)^2 + 0.90^2 = \mathbf{0.8100}$$
- Armijo condition check:
  $$\text{Bound}(0.05) = 11.0000 - 202.0(0.05) = 11.0000 - 10.1000 = \mathbf{0.9000}$$
  $$\mathbf{f(0.0, 0.90) = 0.8100 \le 0.9000} \quad (\mathbf{ACCEPTED! \,\, \checkmark})$$

- **Result:** The step size $\eta^* = 0.0500$ is accepted. The loss plummets from $11.0000 \to 0.8100$ (**92.6% reduction in a single step!**).

---

### Illustration 5 (Numerical): Step-by-Step Critical Batch Size Calculation

Let us calculate the exact Critical Batch Size $B_{\text{crit}}$ by hand and analyze the step efficiency curve across batch sizes.

#### 1. Setup & Matrix Quantities
Let the empirical loss landscape have:
- Mean gradient: $\mathbf{g} = \nabla \mathcal{L} = \begin{bmatrix} 2.0 \\ 1.0 \end{bmatrix}$
- Single-sample gradient covariance: $\Sigma = \begin{bmatrix} 8.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$
- Local Hessian matrix: $\mathcal{H} = \begin{bmatrix} 4.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$

#### 2. Evaluating the Critical Batch Size Formula
Recall McCandlish et al.'s formula from Deep Derivation 5.4.3:
$$B_{\text{crit}} = \frac{\text{Tr}(\Sigma \mathcal{H})}{\|\mathbf{g}\|_2^2}$$

1. **Squared Gradient Norm (True Signal):**
   $$\|\mathbf{g}\|_2^2 = 2.0^2 + 1.0^2 = 4.0 + 1.0 = \mathbf{5.0000}$$
2. **Curvature-Weighted Noise Matrix:**
   $$\Sigma \mathcal{H} = \begin{bmatrix} 8.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix} \begin{bmatrix} 4.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 32.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$$
3. **Trace:**
   $$\text{Tr}(\Sigma \mathcal{H}) = 32.0 + 2.0 = \mathbf{34.0000}$$
4. **Critical Batch Size:**
   $$\mathbf{B_{\text{crit}} = \frac{34.0000}{5.0000} = 6.8000 \approx 7 \text{ samples}}$$

#### 3. Step Efficiency Analysis Across Batch Sizes
The theoretical step efficiency $E(B)$ (the progress per gradient computation relative to pure batch GD) is:
$$E(B) = \frac{1}{1 + \frac{B_{\text{crit}}}{B}} = \frac{B}{B + 6.8000}$$

```
Batch Size B │ Ratio B / B_crit │ Step Efficiency E(B) │ Optimization Regime
─────────────┼──────────────────┼──────────────────────┼─────────────────────────────────────────────
B = 1        │ 0.147            │ 1 / (1 + 6.8) = 0.128│ Noise Dominated (87.2% compute wasted)
B = 2        │ 0.294            │ 2 / (2 + 6.8) = 0.227│ Strong Linear Scaling Speedup
B = 7        │ 1.029            │ 7 / (7 + 6.8) = 0.507│ Critical Threshold (50% Efficiency Elbow)
B = 20       │ 2.941            │20 / (20+ 6.8) = 0.746│ Moderate Returns
B = 50       │ 7.353            │50 / (50+ 6.8) = 0.880│ Diminishing Returns
B = 200      │ 29.41            │200/(200+6.8)  = 0.967│ Saturated (4x compute for 9% extra progress)
```

**Key Takeaway:**
- Below $B = 7$, doubling the batch size almost doubles training throughput (linear scaling).
- Above $B = 50$, doubling the batch size yields virtually zero speedup in terms of epochs required for convergence, wasting thousands of GPU hours. This analytical formula dictates modern distributed training batch budgets!

---

## Part 7: Deep Learning Connection & Application

### 1. Learning Rate Warmup Schedules
When training large Transformers (e.g. LLaMA, GPT-4) with large mini-batch sizes ($B \approx 2048$), applying the scaled learning rate $\eta_{\max} = 3 \times 10^{-4}$ at iteration 0 causes immediate loss divergence.
Why?
At initialization, the network weights are completely random, producing massive gradient variances $\Sigma(\theta_0)$.
**Learning Rate Warmup** linearly ramps the learning rate from $0 \to \eta_{\max}$ over the first $N_{\text{warmup}}$ steps:
$$\eta_t = \eta_{\max} \cdot \frac{t}{N_{\text{warmup}}}$$
This allows the network to find a stable curvature basin before unleashing the full learning rate!

---

## Part 8: Code Implementation & Verification

The companion Python module [04_first_order_algorithms.py](./code/04_first_order_algorithms.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact calculations of BGD ($w_1 = 1.3875$), Mini-batch ($w_1 = 1.4250$), and Pure SGD ($w_1 = 1.6000$ and $1.0500$).
2. **Gradient Variance vs. Batch Size Decay Simulation:** Evaluates empirical variance across 10,000 batches for $B \in [1, 2, 4, 8, 16, 32]$, confirming exact $\mathcal{O}(1/B)$ variance decay.
3. **Robbins-Monro Decreasing Learning Rate vs. Constant Noise Ball:** Demonstrates that constant learning rate creates a persistent noise ball around the minimum, while $1/\sqrt{t}$ schedule converges to zero error.
4. **Langevin Dynamics & Flat vs. Sharp Minima Escape:** Simulates how mini-batch gradient noise enables an optimizer to jump out of a sharp local minimum and settle into a flat global minimum.
