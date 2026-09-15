# Chapter 3.7: Information Theory (Entropy, Cross-Entropy, KL Divergence, Mutual Information)

---

## Part 1: Intuition & 101 Motivation

Why does a modern deep learning practitioner need **Information Theory**?

Claude Shannon founded information theory in 1948 not to study neural networks, but to answer two fundamental engineering questions:
1. What is the ultimate data compression limit of a signal? (Entropy $H(X)$)
2. What is the ultimate transmission rate over a noisy channel? (Mutual Information $I(X; Y)$)

In modern Artificial Intelligence and Deep Learning:
- **Loss Functions:** The standard classification loss function across Computer Vision, NLP, and Large Language Models (LLMs)—**Cross-Entropy Loss**—is derived straight from Shannon's source coding theorems.
- **Variational Autoencoders (VAEs):** The training objective is an explicit optimization balancing reconstruction fidelity against a **Kullback-Leibler (KL) Divergence** penalty that prevents latent space explosion.
- **Generative Adversarial Networks (GANs):** Goodfellow et al. (2014) proved that the original minimax GAN objective minimizes the **Jensen-Shannon Divergence (JSD)** between the data distribution and the generator's distribution.
- **Self-Supervised & Contrastive Learning:** State-of-the-art representations (e.g., CLIP, SimCLR, CPC) maximize a lower bound on the **Mutual Information** between different augmented views of data via the **InfoNCE loss**.

Information theory gives us the exact mathematical language to quantify **uncertainty**, **information flow**, and the **distance between probability distributions**.

---

## Part 2: Rigorous Mathematical Formulation

### 1. Self-Information (Surprisal)
How much "information" does learning that an event $x$ occurred provide?
If someone tells you the sun rose this morning ($p \approx 1$), they have told you virtually nothing. If they tell you an earthquake just struck your city ($p \ll 1$), they have conveyed massive information.

**Axiomatic Requirements for Information Measure $I(x)$:**
1. Monotonicity: Less probable events convey strictly more information ($p_1 < p_2 \implies I(x_1) > I(x_2)$).
2. Non-negativity: $I(x) \ge 0$ for all $x$, with $I(x) = 0$ if $p(x) = 1$.
3. Additivity: For two independent events $x, y$ where $P(x, y) = P(x) P(y)$, the total information gained must sum:
   $$I(x, y) = I(x) + I(y)$$

The unique continuous function satisfying these axioms is the negative logarithm:
$$I(x) = -\log_b P(x) = \log_b \frac{1}{P(x)}$$

- If $b = 2$, information is measured in **bits** (or **shannons**).
- If $b = e$, information is measured in **nats** ($1 \text{ nat} = \frac{1}{\ln 2} \approx 1.4427 \text{ bits}$).
- If $b = 10$, information is measured in **hartleys** (or **bans**).

*Convention in Deep Learning:* The natural base $e$ is standard due to calculus with PyTorch `torch.log` and softmax exponentials.

---

### 2. Shannon Entropy $H(X)$
The **entropy** of a discrete random variable $X \in \mathcal{X}$ is the **expected surprisal**:
$$H(X) = \mathbb{E}_{X \sim P}[I(X)] = -\sum_{x \in \mathcal{X}} P(x) \log P(x)$$

#### Boundary Convention
By L'Hôpital's rule:
$$\lim_{p \to 0^+} p \log p = 0$$
Hence, if $P(x) = 0$, we define $0 \log 0 = 0$.

#### Fundamental Properties of Discrete Entropy:
1. **Non-negativity:** $H(X) \ge 0$.
2. **Determinism:** $H(X) = 0$ if and only if $X$ is deterministic ($P(x_0) = 1$ for some $x_0$).
3. **Maximum Entropy:** For a finite alphabet $|\mathcal{X}| = K$:
   $$0 \le H(X) \le \log K$$
   with equality if and only if $X$ is uniformly distributed: $P(x) = \frac{1}{K}$ for all $x$.

---

### 3. Joint and Conditional Entropy
For a pair of discrete random variables $(X, Y) \sim P(x, y)$:

#### Joint Entropy:
$$H(X, Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log P(x, y)$$

#### Conditional Entropy:
The remaining uncertainty in $Y$ given that $X$ is observed:
$$H(Y \mid X) = \sum_{x \in \mathcal{X}} P(x) H(Y \mid X = x) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log P(y \mid x)$$

#### Chain Rule of Entropy:
$$H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$$

*Proof:*
$$\begin{aligned}
H(X, Y) &= -\sum_{x, y} P(x, y) \log [P(x) P(y \mid x)] \\
&= -\sum_{x, y} P(x, y) \log P(x) - \sum_{x, y} P(x, y) \log P(y \mid x) \\
&= -\sum_x P(x) \log P(x) + H(Y \mid X) = H(X) + H(Y \mid X) \quad \blacksquare
\end{aligned}$$

---

### 4. Continuous (Differential) Entropy $h(X)$
For a continuous random variable $X \sim f(x)$:
$$h(X) = -\int_{-\infty}^\infty f(x) \ln f(x) \, dx$$

> [!WARNING]
> Unlike discrete entropy, **differential entropy can be negative**!
> Example: For $X \sim \text{Uniform}(0, a)$, $h(X) = \ln(a)$. If $a = 0.1$, $h(X) = \ln(0.1) \approx -2.3026 < 0$. Differential entropy reflects volume scaling rather than pure bit count.

#### Maximum Differential Entropy Theorem:
Among all continuous distributions on $\mathbb{R}^d$ with a fixed mean $\mu$ and fixed covariance matrix $\Sigma$, the **Multivariate Normal distribution $\mathcal{N}(\mu, \Sigma)$ maximizes differential entropy**:
$$h(\mathcal{N}(\mu, \Sigma)) = \frac{1}{2} \ln \left( (2\pi e)^d \det(\Sigma) \right)$$
*Deep Learning Meaning:* The Gaussian distribution is the most conservative (unbiased/least-assumption) prior when only second-order statistics are known.

---

### 5. Kullback-Leibler (KL) Divergence (Relative Entropy)
Given two probability distributions $P$ and $Q$ defined over the same sample space $\mathcal{X}$, the **KL Divergence** from $Q$ to $P$ measures the inefficiency or information lost when approximating the true distribution $P$ by $Q$:

$$D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{X \sim P}\left[ \log \frac{P(X)}{Q(X)} \right]$$

For continuous densities:
$$D_{\text{KL}}(P \parallel Q) = \int_{\mathbb{R}^d} p(x) \log \frac{p(x)}{q(x)} \, dx$$

#### Strict Support Condition:
The KL divergence is well-defined only if $P$ is **absolutely continuous** with respect to $Q$ ($P \ll Q$):
$$Q(x) = 0 \implies P(x) = 0$$
If there exists any $x$ where $P(x) > 0$ but $Q(x) = 0$, then $D_{\text{KL}}(P \parallel Q) = +\infty$!

---

### 6. Theorem: Gibbs' Inequality ($D_{\text{KL}} \ge 0$)
**Statement:** For any valid probability distributions $P$ and $Q$:
$$D_{\text{KL}}(P \parallel Q) \ge 0$$
with equality if and only if $P(x) = Q(x)$ almost everywhere.

**Rigorous Proof (via Jensen's Inequality):**
Consider the negative KL divergence:
$$-D_{\text{KL}}(P \parallel Q) = -\sum_{x} P(x) \ln \frac{P(x)}{Q(x)} = \sum_x P(x) \ln \frac{Q(x)}{P(x)}$$
Let $g(t) = \ln(t)$. The logarithm function is strictly **concave** on $(0, \infty)$ (since $g''(t) = -1/t^2 < 0$).
By Jensen's Inequality for strictly concave functions ($\mathbb{E}[g(Z)] \le g(\mathbb{E}[Z])$):
$$\sum_x P(x) \ln \frac{Q(x)}{P(x)} \le \ln \left( \sum_x P(x) \frac{Q(x)}{P(x)} \right)$$
Simplify the inner sum:
$$\sum_x P(x) \frac{Q(x)}{P(x)} = \sum_x Q(x) = 1$$
Therefore:
$$-D_{\text{KL}}(P \parallel Q) \le \ln(1) = 0 \implies D_{\text{KL}}(P \parallel Q) \ge 0 \quad \blacksquare$$
Since $\ln(t)$ is strictly concave, equality holds if and only if the random variable $\frac{Q(X)}{P(X)}$ is constant almost everywhere. Since both distributions sum to 1, this constant must be 1, so $P(x) = Q(x)$ for all $x$. $\blacksquare$

---

### 7. Asymmetry of KL Divergence
KL divergence is **not a metric** because:
1. It is **asymmetric**: $D_{\text{KL}}(P \parallel Q) \ne D_{\text{KL}}(Q \parallel P)$ in general.
2. It does **not satisfy the triangle inequality**.

#### Forward KL vs. Reverse KL in Deep Learning:
Suppose $P$ is a complex true multimodal distribution, and $Q_\theta$ is a simpler model distribution (e.g., Gaussian):

| Direction | Optimization | Behavior | Application |
| :--- | :--- | :--- | :--- |
| **Forward KL:** $D_{\text{KL}}(P \parallel Q_\theta)$ | $\min_\theta \mathbb{E}_{x \sim P}[\log \frac{P(x)}{Q_\theta(x)}]$ | **Zero-avoiding / Mode-covering:** $Q_\theta$ must cover all regions where $P(x) > 0$. High penalty if $Q_\theta(x) \approx 0$ while $P(x) > 0$. | Maximum Likelihood Estimation (MLE), Supervised Classification. |
| **Reverse KL:** $D_{\text{KL}}(Q_\theta \parallel P)$ | $\min_\theta \mathbb{E}_{x \sim Q_\theta}[\log \frac{Q_\theta(x)}{P(x)}]$ | **Zero-forcing / Mode-seeking:** $Q_\theta$ locks onto a single high-density mode of $P$ and ignores other modes to avoid evaluating $P(x)$ in low-density regions. | Variational Inference, VAEs, Policy Gradient RL, Distillation. |

---

### 8. Cross-Entropy $H(P, Q)$
The **cross-entropy** between distribution $P$ and distribution $Q$ is:
$$H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x) = \mathbb{E}_{X \sim P}[-\log Q(X)]$$

#### Fundamental Identity:
$$H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)$$

*Proof:*
$$\begin{aligned}
H(P, Q) &= -\sum_x P(x) \log Q(x) \\
&= -\sum_x P(x) \log P(x) + \sum_x P(x) \log \frac{P(x)}{Q(x)} \\
&= H(P) + D_{\text{KL}}(P \parallel Q) \quad \blacksquare
\end{aligned}$$

> [!IMPORTANT]
> **Why do we minimize Cross-Entropy in Deep Learning?**
> In supervised classification, $P$ is the true empirical label distribution.
> For fixed training data, $P$ is constant, so its entropy $H(P)$ is a **constant** independent of the model parameters $\theta$!
> $$\arg\min_\theta H(P, Q_\theta) = \arg\min_\theta \left[ H(P) + D_{\text{KL}}(P \parallel Q_\theta) \right] = \arg\min_\theta D_{\text{KL}}(P \parallel Q_\theta)$$
> **Minimizing Cross-Entropy is mathematically identical to minimizing the KL Divergence to the true data distribution!**

---

### 9. Jensen-Shannon Divergence (JSD)
Because KL divergence is asymmetric and unbounded ($+\infty$), Endres & Schindelin (2003) introduced the **Jensen-Shannon Divergence**:
$$JSD(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$$
where $M = \frac{1}{2}(P + Q)$ is the mixture distribution.

#### Properties of JSD:
1. **Symmetric:** $JSD(P \parallel Q) = JSD(Q \parallel P)$.
2. **Bounded:** When using base 2 logarithm:
   $$0 \le JSD(P \parallel Q) \le 1 \text{ bit}$$
   When using natural log: $0 \le JSD(P \parallel Q) \le \ln 2 \approx 0.69315$.
3. **True Metric:** $\sqrt{JSD(P \parallel Q)}$ satisfies all metric axioms including the triangle inequality (the *Jensen-Shannon metric*).
4. **GAN Theorem (Goodfellow et al., 2014):**
   The global minimum of the minimax GAN objective $V(D, G)$ occurs when $P_{\text{model}} = P_{\text{data}}$, yielding:
   $$V(D^*, G) = -\ln 4 + 2 \cdot JSD(P_{\text{data}} \parallel P_{\text{model}})$$

---

### 10. Mutual Information $I(X; Y)$
How much information does observing random variable $Y$ share with random variable $X$?
$$I(X; Y) = D_{\text{KL}}(P_{X, Y} \parallel P_X P_Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log \frac{P(x, y)}{P(x) P(y)}$$

#### Equivalent Decompositions:
$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)$$

#### Key Properties:
1. **Symmetry:** $I(X; Y) = I(Y; X)$.
2. **Non-negativity:** $I(X; Y) \ge 0$, with equality if and only if $X \perp Y$ (independent).
3. **Self-Information Relationship:** $I(X; X) = H(X)$ (the information a variable shares with itself is its entropy).
4. **Data Processing Inequality (DPI):**
   If $X \to Y \to Z$ forms a Markov chain (i.e., $Z$ depends on $X$ only through $Y$):
   $$I(X; Z) \le I(X; Y)$$
   *Deep Learning Meaning:* No post-processing transformation (no matter how deep or complex a neural network is) can create new information about $X$ that was not already present in the representation $Y$!

---

## Part 3: Geometric & Algebraic Interpretation

### 1. The Information Venn Diagram
Information measures admit a beautiful set-theoretic duality:

```
        ┌─────────────────────────┐
        │         H(X, Y)         │
   ┌────┴────────────┬────────────┴────┐
   │      H(X)       │      H(Y)       │
   │   ┌─────────────┼─────────────┐   │
   │   │  H(X | Y)   │   I(X; Y)   │   │
   │   │             │             │   │
   │   │ (Uncertainty│ (Shared     │   │
   │   │  in X only) │  Information│   │
   │   │             │   Overlap)  │   │
   │   └─────────────┼─────────────┘   │
   │                 │   H(Y | X)      │
   │                 │                 │
   └─────────────────┴─────────────────┘
```

- Total union area $= H(X, Y)$ (Joint Entropy)
- Left circle $= H(X)$, Right circle $= H(Y)$
- Overlap intersection $= I(X; Y)$ (Mutual Information)
- Left crescent $= H(X \mid Y) = H(X) - I(X; Y)$
- Right crescent $= H(Y \mid X) = H(Y) - I(X; Y)$

---

### 2. The Statistical Manifold & Fisher-Rao Geometry
In Information Geometry (Amari, 1985), the space of probability distributions forms a Riemannian manifold $\mathcal{M}$.
For two infinitesimally close distributions $P_\theta$ and $P_{\theta + d\theta}$:
$$D_{\text{KL}}(P_\theta \parallel P_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F(\theta) d\theta$$
where $F(\theta)$ is the **Fisher Information Matrix**:
$$F(\theta) = \mathbb{E}_{x \sim P_\theta}\left[ \nabla_\theta \log P_\theta(x) \nabla_\theta \log P_\theta(x)^T \right]$$
The KL divergence acts as the local **squared Riemannian distance metric** on the statistical manifold! This is the direct theoretical foundation of **Natural Gradient Descent** (Amari) and **Trust Region Policy Optimization (TRPO)** in reinforcement learning.

---

## Part 4: Real-World Analogy

### 1. The 20 Questions Game (Entropy)
Imagine playing the 20 Questions game to guess an unknown object.
- Each optimal question divides the remaining probability space exactly in half ($p = 0.5$).
- A single yes/no answer conveys exactly $-\log_2(0.5) = \mathbf{1 \text{ bit}}$ of information.
- If the object space has $N = 1{,}048{,}576 = 2^{20}$ equally likely candidates, its entropy is $H(X) = \log_2(2^{20}) = \mathbf{20 \text{ bits}}$.
- Thus, you require **at least 20 optimal questions** on average to identify the secret object!

### 2. The Faulty Morse Code Transmitter (KL Divergence)
Suppose an ancient telegraph operator designs a Morse code transmission system optimized for the English language, where the letter "E" ($p = 0.13$) receives the shortest code (`.`) and "Q" ($p = 0.001$) receives a long code (`--.-`).
- **Optimal code length** for symbol $x$ is $-\log_2 P(x)$ bits. Average transmission length is $H(P)$.
- Now, suppose the telegraph is suddenly used to transmit encrypted text or Turkish text, which follows a radically different distribution $Q$.
- If you use code lengths optimized for $Q$ when the true text follows $P$, your average transmission length balloons to $H(P, Q)$ bits.
- The extra, wasted wire-time per character is precisely the **KL Divergence**:
  $$\Delta \text{Bits} = H(P, Q) - H(P) = D_{\text{KL}}(P \parallel Q)$$

---

## Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)

Let us compute **Entropy**, **Cross-Entropy**, **KL Divergence**, and **Jensen-Shannon Divergence** by hand with concrete numbers for a 3-class classification problem.

### 1. Problem Setup & Toy Probability Vectors
Consider a 3-class computer vision model classifying an image as `[Cat, Dog, Bird]`:
- **True Distribution $P$:** Ground truth probability distribution (e.g., human annotator soft consensus):
  $$P = [p_1, p_2, p_3] = [0.60, 0.30, 0.10]$$
- **Model A Predictions $Q_A$:** High-quality calibrated model:
  $$Q_A = [q_{A1}, q_{A2}, q_{A3}] = [0.50, 0.35, 0.15]$$
- **Model B Predictions $Q_B$:** Poor, confused model:
  $$Q_B = [q_{B1}, q_{B2}, q_{B3}] = [0.10, 0.20, 0.70]$$

All calculations use **natural logarithms (nats)**.
*Reference Logarithms Table:*
- $\ln(0.60) \approx -0.510826$
- $\ln(0.30) \approx -1.203973$
- $\ln(0.10) \approx -2.302585$
- $\ln(0.50) \approx -0.693147$
- $\ln(0.35) \approx -1.049822$
- $\ln(0.15) \approx -1.897120$
- $\ln(0.20) \approx -1.609438$
- $\ln(0.70) \approx -0.356675$

---

### 2. "What Refers to What" Legend Protocol

| Symbol | Pure Math Meaning | Deep Learning Meaning | Dimension / Shape | Walkthrough Value |
| :--- | :--- | :--- | :--- | :--- |
| $K$ | Alphabet / Support cardinality | Number of classes | Scalar integer | $3$ |
| $P$ | Target probability distribution | Ground truth target labels | $\mathbb{R}^K$ row vector | $[0.60, 0.30, 0.10]$ |
| $Q_A$ | Approximating distribution A | Model A softmax output logits $\hat{y}_A$ | $\mathbb{R}^K$ row vector | $[0.50, 0.35, 0.15]$ |
| $Q_B$ | Approximating distribution B | Model B softmax output logits $\hat{y}_B$ | $\mathbb{R}^K$ row vector | $[0.10, 0.20, 0.70]$ |
| $H(P)$ | Shannon entropy of $P$ | Intrinsic label uncertainty | Scalar $\ge 0$ (nats) | $0.8979$ |
| $H(P, Q)$ | Cross-entropy between $P$ and $Q$ | Supervised classification loss | Scalar $\ge H(P)$ | $H(P, Q_A) \approx 0.9583$ |
| $D_{\text{KL}}(P \parallel Q)$ | Relative entropy | Information lost by using $Q$ | Scalar $\ge 0$ | $D_{\text{KL}}(P \parallel Q_A) \approx 0.0604$ |
| $JSD(P \parallel Q)$ | Jensen-Shannon divergence | GAN discriminator equilibrium loss | Scalar $\in [0, \ln 2]$ | $JSD(P \parallel Q_A) \approx 0.0146$ |

---

### 3. Step-by-Step Manual Arithmetic

#### Step 1: True Distribution Entropy $H(P)$
$$H(P) = -\sum_{i=1}^3 p_i \ln p_i$$
- Class 1: $-0.60 \times \ln(0.60) = -0.60 \times (-0.510826) = \mathbf{+0.306495}$
- Class 2: $-0.30 \times \ln(0.30) = -0.30 \times (-1.203973) = \mathbf{+0.361192}$
- Class 3: $-0.10 \times \ln(0.10) = -0.10 \times (-2.302585) = \mathbf{+0.230259}$
$$\mathbf{H(P) = 0.306495 + 0.361192 + 0.230259 = 0.897946 \text{ nats}}$$

---

#### Step 2: Cross-Entropy for Model A: $H(P, Q_A)$
$$H(P, Q_A) = -\sum_{i=1}^3 p_i \ln q_{Ai}$$
- Class 1: $-0.60 \times \ln(0.50) = -0.60 \times (-0.693147) = \mathbf{+0.415888}$
- Class 2: $-0.30 \times \ln(0.35) = -0.30 \times (-1.049822) = \mathbf{+0.314947}$
- Class 3: $-0.10 \times \ln(0.15) = -0.10 \times (-1.897120) = \mathbf{+0.189712}$
$$\mathbf{H(P, Q_A) = 0.415888 + 0.314947 + 0.189712 = 0.920547 \text{ nats}}$$

---

#### Step 3: KL Divergence for Model A: $D_{\text{KL}}(P \parallel Q_A)$
Using the identity $D_{\text{KL}}(P \parallel Q_A) = H(P, Q_A) - H(P)$:
$$D_{\text{KL}}(P \parallel Q_A) = 0.920547 - 0.897946 = \mathbf{0.022601 \text{ nats}}$$

Let us verify by direct summation $\sum_i p_i \ln \frac{p_i}{q_{Ai}}$:
- Class 1: $0.60 \times \ln(0.60 / 0.50) = 0.60 \times \ln(1.20) = 0.60 \times 0.182322 = \mathbf{+0.109393}$
- Class 2: $0.30 \times \ln(0.30 / 0.35) = 0.30 \times \ln(0.857143) = 0.30 \times (-0.154151) = \mathbf{-0.046245}$
- Class 3: $0.10 \times \ln(0.10 / 0.15) = 0.10 \times \ln(0.666667) = 0.10 \times (-0.405465) = \mathbf{-0.040547}$
$$\mathbf{D_{\text{KL}}(P \parallel Q_A) = 0.109393 - 0.046245 - 0.040547 = 0.022601 \text{ nats} \ge 0 \quad \checkmark}$$

---

#### Step 4: Cross-Entropy & KL Divergence for Model B (Bad Model)
$$H(P, Q_B) = -\sum_{i=1}^3 p_i \ln q_{Bi}$$
- Class 1: $-0.60 \times \ln(0.10) = -0.60 \times (-2.302585) = \mathbf{+1.381551}$
- Class 2: $-0.30 \times \ln(0.20) = -0.30 \times (-1.609438) = \mathbf{+0.482831}$
- Class 3: $-0.10 \times \ln(0.70) = -0.10 \times (-0.356675) = \mathbf{+0.035667}$
$$\mathbf{H(P, Q_B) = 1.381551 + 0.482831 + 0.035667 = 1.900049 \text{ nats}}$$

Compute $D_{\text{KL}}(P \parallel Q_B)$:
$$D_{\text{KL}}(P \parallel Q_B) = H(P, Q_B) - H(P) = 1.900049 - 0.897946 = \mathbf{1.002103 \text{ nats}}$$
*Observation:* Model B's KL divergence is **44 times larger** than Model A's, driving a much steeper gradient in cross-entropy training!

---

#### Step 5: Jensen-Shannon Divergence Calculation $JSD(P \parallel Q_A)$
Compute the midpoint distribution $M = \frac{1}{2}(P + Q_A)$:
$$M = [0.55, 0.325, 0.125]$$

1. **$D_{\text{KL}}(P \parallel M)$:**
   $$\begin{aligned}
   0.60 \ln(0.60 / 0.55) &= 0.60 \ln(1.090909) = 0.60 \times 0.087011 = \mathbf{0.052207} \\
   0.30 \ln(0.30 / 0.325) &= 0.30 \ln(0.923077) = 0.30 \times (-0.080043) = \mathbf{-0.024013} \\
   0.10 \ln(0.10 / 0.125) &= 0.10 \ln(0.800000) = 0.10 \times (-0.223144) = \mathbf{-0.022314} \\
   \sum &= 0.052207 - 0.024013 - 0.022314 = \mathbf{0.005880}
   \end{aligned}$$

2. **$D_{\text{KL}}(Q_A \parallel M)$:**
   $$\begin{aligned}
   0.50 \ln(0.50 / 0.55) &= 0.50 \ln(0.909091) = 0.50 \times (-0.095310) = \mathbf{-0.047655} \\
   0.35 \ln(0.35 / 0.325) &= 0.35 \ln(1.076923) = 0.35 \times 0.074108 = \mathbf{0.025938} \\
   0.15 \ln(0.15 / 0.125) &= 0.15 \ln(1.200000) = 0.15 \times 0.182322 = \mathbf{0.027348} \\
   \sum &= -0.047655 + 0.025938 + 0.027348 = \mathbf{0.005631}
   \end{aligned}$$

3. **Total $JSD(P \parallel Q_A)$:**
   $$JSD(P \parallel Q_A) = \frac{1}{2}(0.005880) + \frac{1}{2}(0.005631) = \mathbf{0.005756 \text{ nats}}$$

---

### 4. Visual Summary Grid

```
┌─────────┬──────────────────────┬──────────────────────┬──────────────────────┬─────────────┐
│ Metric  │ Class 1 (Cat)        │ Class 2 (Dog)        │ Class 3 (Bird)       │ TOTAL (nats)│
├─────────┼──────────────────────┼──────────────────────┼──────────────────────┼─────────────┤
│ P       │  0.60                │  0.30                │  0.10                │  1.0000     │
│ Q_A     │  0.50                │  0.35                │  0.15                │  1.0000     │
│ Q_B     │  0.10                │  0.20                │  0.70                │  1.0000     │
├─────────┼──────────────────────┼──────────────────────┼──────────────────────┼─────────────┤
│ -p ln p │  +0.3065             │  +0.3612             │  +0.2303             │ H(P) =0.8979│
├─────────┼──────────────────────┼──────────────────────┼──────────────────────┼─────────────┤
│-p ln Q_A│  +0.4159             │  +0.3149             │  +0.1897             │ H(P,QA)=0.92│
│-p ln Q_B│  +1.3816             │  +0.4828             │  +0.0357             │ H(P,QB)=1.90│
├─────────┼──────────────────────┼──────────────────────┼──────────────────────┼─────────────┤
│KL(P||QA)│  +0.1094             │  -0.0462             │  -0.0405             │ KL = 0.0226 │
│KL(P||QB)│  +1.0751             │  +0.1216             │  -0.1946             │ KL = 1.0021 │
├─────────┼──────────────────────┼──────────────────────┼──────────────────────┼─────────────┤
│JSD(P||A)│       —              │       —              │       —              │ JSD = 0.0058│
└─────────┴──────────────────────┴──────────────────────┴──────────────────────┴─────────────┘
```

---

## Part 6: Solved Illustrations (Standard, Boundary, Edge Cases)

### Illustration 1 (Standard): One-Hot Supervised Classification
In typical supervised classification, the ground truth label is one-hot: $P = [1, 0, 0]$ (the sample is definitely a Cat).
Let model prediction be $Q = [0.70, 0.20, 0.10]$.

1. **Entropy of Truth:**
   $$H(P) = -1 \ln(1) - 0 \ln(0) - 0 \ln(0) = \mathbf{0.0000 \text{ nats}}$$
2. **Cross-Entropy:**
   $$H(P, Q) = -1 \ln(0.70) - 0 \ln(0.20) - 0 \ln(0.10) = -\ln(0.70) \approx \mathbf{0.3567 \text{ nats}}$$
3. **KL Divergence:**
   $$D_{\text{KL}}(P \parallel Q) = H(P, Q) - H(P) = 0.3567 - 0 = \mathbf{0.3567 \text{ nats}}$$
*Conclusion:* For one-hot targets, Cross-Entropy is **identically equal** to the KL Divergence:
$$\mathcal{L}_{\text{CE}} = -\ln q_{\text{correct}} = D_{\text{KL}}(P_{\text{one-hot}} \parallel Q)$$

---

### Illustration 2 (Boundary): Zero Probability Support Mismatch
Suppose $P = [0.80, 0.20]$ and a neural network outputs $Q = [1.00, 0.00]$ (e.g., due to extreme uncalibrated confidence).
Compute $D_{\text{KL}}(P \parallel Q)$:
$$D_{\text{KL}}(P \parallel Q) = 0.80 \ln\left(\frac{0.80}{1.00}\right) + 0.20 \ln\left(\frac{0.20}{0.00}\right) = 0.80(-0.223) + 0.20(+\infty) = \mathbf{+\infty}$$
*Deep Learning Implication:* If your model outputs a probability of exactly zero for an event that actually occurs in reality, your cross-entropy loss **explodes to infinity**. This is why numerically stable implementations (like `torch.nn.CrossEntropyLoss`) combine log-sum-exp and why label smoothing is used to prevent $Q$ from approaching $\{0, 1\}$.

---

### Illustration 3 (Edge Case): Mode-Covering vs. Mode-Seeking in Multimodal Fits
Let the true distribution $P(x) = \frac{1}{2}\mathcal{N}(-3, 1) + \frac{1}{2}\mathcal{N}(+3, 1)$ be a bimodal Gaussian mixture.
Suppose we fit a unimodal Gaussian $Q_\theta = \mathcal{N}(\mu, \sigma^2)$:

1. **Minimizing Forward KL: $\min_\theta D_{\text{KL}}(P \parallel Q_\theta)$**
   $$\mathbb{E}_{x \sim P}[\ln Q_\theta(x)] \implies \mu = \mathbb{E}_P[X] = 0, \quad \sigma^2 = \text{Var}_P(X) = 1 + 3^2 = 10$$
   $Q$ stretches out its variance to cover **both modes**, placing substantial probability mass in the dead valley between $-3$ and $+3$ (**zero-avoiding / mode-covering**).
2. **Minimizing Reverse KL: $\min_\theta D_{\text{KL}}(Q_\theta \parallel P)$**
   To minimize $\mathbb{E}_{x \sim Q}[\ln \frac{Q(x)}{P(x)}]$, $Q$ will position itself tightly around **one of the modes** (either $\mu = -3$ or $\mu = +3$) with $\sigma^2 \approx 1$, completely ignoring the other mode (**zero-forcing / mode-seeking**).

---

## Part 7: Deep Learning Connection & Application

### 1. Variational Autoencoders (VAEs) & The ELBO
The marginal log-likelihood of data $x$ in a latent variable model is:
$$\log p_\theta(x) = \mathbb{E}_{z \sim q_\phi(z \mid x)}\left[ \log \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] + D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))$$
Rearranging gives the **Evidence Lower Bound (ELBO)**:
$$\mathcal{L}_{\text{ELBO}}(\theta, \phi; x) = \underbrace{\mathbb{E}_{z \sim q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{Reconstruction Fidelity}} - \underbrace{D_{\text{KL}}(q_\phi(z \mid x) \parallel p(z))}_{\text{Latent Prior Regularizer}}$$
When the latent prior is standard normal $p(z) = \mathcal{N}(0, I)$ and the encoder outputs diagonal Gaussian parameters $q_\phi(z \mid x) = \mathcal{N}(\mu(x), \text{diag}(\sigma^2(x)))$, the KL divergence has an **exact closed-form formula**:
$$D_{\text{KL}}(q_\phi(z \mid x) \parallel \mathcal{N}(0, I)) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

---

### 2. Contrastive Representation Learning (InfoNCE & Mutual Information)
Modern foundational models (CLIP, SimCLR) train encoders $f(x)$ using the **InfoNCE loss**:
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[ \log \frac{\exp(f(x)^T f(x^+) / \tau)}{\exp(f(x)^T f(x^+) / \tau) + \sum_{j=1}^{K-1} \exp(f(x)^T f(x_j^-) / \tau)} \right]$$
**Theorem (Oord et al., 2018):**
Minimizing $\mathcal{L}_{\text{InfoNCE}}$ is mathematically equivalent to maximizing a variational lower bound on the **Mutual Information** between positive views $X$ and $X^+$:
$$I(X; X^+) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}$$
As the batch size $K$ grows, the lower bound tightens, proving why scaling batch sizes in contrastive learning directly boosts representation quality!

---

## Part 8: Code Implementation & Verification

The companion Python module [07_information_theory.py](./code/07_information_theory.py) provides comprehensive numerical verification:
1. **Part 5 Visual Grid Verification:** Validates exact calculations of $H(P)$, $H(P, Q_A)$, $H(P, Q_B)$, $D_{\text{KL}}(P \parallel Q_A)$, and $JSD(P \parallel Q_A)$ to 6 decimal places.
2. **Gibbs' Inequality Verification:** Monte Carlo tests over 10,000 random probability simplex pairs verifying $D_{\text{KL}}(P \parallel Q) \ge 0$ with strict equality only when $P = Q$.
3. **Mutual Information & Chain Rule:** Verifies the identity $I(X; Y) = H(X) + H(Y) - H(X, Y)$ on a bivariate discrete contingency table.
4. **Closed-Form Gaussian KL Divergence:** Verifies analytical VAE latent KL against numerical Monte Carlo integration.
5. **PyTorch Automatic Differentiation of Cross-Entropy:** Confirms manual gradient $\nabla_z H(P, \text{softmax}(z)) = \hat{y} - y$ matches PyTorch autograd engine exactly.
