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
*Justification:*
Over $k$ consecutive mini-batch steps of size $B$, the net weight change is:
$$\Delta \theta_{\text{small}} = -\eta \sum_{j=1}^k \mathbf{g}_B^{(j)}(\theta) \approx -k \eta \nabla \mathcal{L}(\theta)$$
A single giant mini-batch step of size $k B$ with scaled learning rate $\hat{\eta} = k \eta$ yields:
$$\Delta \theta_{\text{large}} = -\hat{\eta} \mathbf{g}_{kB}(\theta) = -(k \eta) \nabla \mathcal{L}(\theta) \approx \Delta \theta_{\text{small}}$$
The updates match!

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
