# Chapter 27: Reinforcement Learning from Human Feedback (RLHF) with PPO

---

## 1. Intuition & 101 Motivation

When a large language model (like GPT-3 or LLaMA) is pre-trained on hundreds of billions of Internet tokens, its objective is strictly **self-supervised next-token prediction**:
$$\max_\theta \sum_{t} \ln P(w_t \mid w_{<t})$$

Because the Internet contains toxicity, falsehoods, hate speech, and unhelpful rants, a raw pre-trained model will happily generate harmful, deceptive, or offensive text if prompted.

While **Supervised Fine-Tuning (SFT)** on curated question-answer pairs helps, human demonstration data is expensive, scarce, and brittle:
- For complex tasks (writing code, summarizing research papers, explaining quantum physics), humans find it **much easier to compare and rank two responses** than to author the perfect response from scratch!

Enter **Reinforcement Learning from Human Feedback (RLHF)** (Christiano et al., NeurIPS 2017; Ouyang et al., InstructGPT 2022):
1. **Step 1 (Preference Data Collection):** The LLM generates two candidate answers $(y_w, y_l)$ for a prompt $x$. A human annotator selects the better answer ($y_w \succ y_l$).
2. **Step 2 (Reward Modeling):** Train a **Reward Model $r_\psi(x, y)$** that scores how helpful, honest, and harmless a response is using the Bradley-Terry preference framework.
3. **Step 3 (RL Fine-Tuning with PPO):** Treat the language model as an **RL Actor**:
   - The **Prompt $x$** is the initial state $s_0$.
   - Each **generated Token $y_t$** is an action $a_t$.
   - The **Environment** is the frozen Reward Model, which delivers a scalar reward when the sentence ends.
   - A **Kullback-Leibler (KL) divergence penalty** anchors the RL policy to the original SFT model, preventing linguistic collapse and reward hacking!

RLHF is the foundational breakthrough that transformed raw predictive models into reliable, conversational AI assistants (ChatGPT, Claude, Gemini).

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Bradley-Terry Preference Model

Suppose human evaluators are presented with prompt $x$ and two candidate responses $y_1, y_2$. The probability that human judges prefer $y_w$ over $y_l$ ($y_w \succ y_l$) is modeled using the **Bradley-Terry logistic model**:

$$P(y_w \succ y_l \mid x) = \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) = \frac{1}{1 + \exp\big( -(r_\psi(x, y_w) - r_\psi(x, y_l)) \big)}$$

where $r_\psi(x, y) \in \mathbb{R}$ is a scalar reward model parameterized by a Transformer network.

#### The Reward Model Loss:
Given a dataset of pairwise preferences $\mathcal{D}_{\text{pref}} = \{(x, y_w, y_l)\}$, the reward model is trained by minimizing the binary cross-entropy loss:

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \ln \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$

---

### 2.2 The RLHF Policy Optimization Objective

Once $r_\psi$ is trained, it is frozen. We wish to optimize the parameters $\theta$ of the language model policy $\pi_\theta(y \mid x)$ to maximize the expected reward while penalizing divergence from the frozen SFT reference model $\pi_{\text{ref}}$:

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\psi(x, y) - \beta D_{\text{KL}}\left( \pi_\theta(\cdot \mid x) \;\Vert\; \pi_{\text{ref}}(\cdot \mid x) \right) \right]$$

where $\beta > 0$ controls the strength of the KL divergence penalty.

---

### 2.3 Token-Level Reward Decomposition

In standard RL, rewards can occur at each step. In language modeling, the reward model $r_\psi(x, y)$ evaluates the *entire completed sequence* $y = (y_1, \dots, y_T)$.
To apply actor-critic temporal-difference learning, the sequence-level reward and KL divergence are decomposed into **per-token rewards $R_t$**:

$$R_t = \begin{cases}
-\beta \left( \ln \pi_\theta(y_t \mid x, y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right), & \text{for } t = 1, \dots, T-1 \\
r_\psi(x, y) - \beta \left( \ln \pi_\theta(y_T \mid x, y_{<T}) - \ln \pi_{\text{ref}}(y_T \mid x, y_{<T}) \right), & \text{for } t = T
\end{cases}$$

Notice:
- For all intermediate tokens $t < T$, the reward is purely the negative token-level KL divergence.
- At the final terminal token $t = T$, the agent receives the full reward model score $r_\psi(x, y)$ plus the final token's KL penalty.

The sum of token rewards reproduces the exact sequence objective:
$$\sum_{t=1}^T R_t = r_\psi(x, y) - \beta \sum_{t=1}^T \left( \ln \pi_\theta(y_t \mid x, y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right)$$

---

### 2.4 Token-Level GAE & PPO-Clipped Update

Four separate models are coordinated during RLHF training:
1. **Actor Policy $\pi_\theta$:** The language model being trained.
2. **Reference Policy $\pi_{\text{ref}}$:** Frozen copy of the SFT model (provides the KL baseline).
3. **Reward Model $r_\psi$:** Frozen preference scoring model.
4. **Critic / Value Network $V_\phi$:** Language model with scalar head predicting expected future token returns from prefix $(x, y_{\le t})$.

#### Token-Level TD Residual & GAE:
With discount $\gamma = 1.0$:
$$\delta_t = R_t + V_\phi(x, y_{\le t+1}) - V_\phi(x, y_{\le t})$$
$$\hat{A}_t = \sum_{l=0}^{T-t-1} \lambda^l \delta_{t+l}$$

#### PPO Actor Loss:
$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$$
$$\mathcal{L}_{\text{actor}}(\theta) = -\frac{1}{T} \sum_{t=1}^T \min \left( r_t(\theta) \hat{A}_t, \; \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right)$$

#### Critic Loss:
$$\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{2 T} \sum_{t=1}^T \left( V_\phi(x, y_{\le t}) - (V_{\text{old}}(x, y_{\le t}) + \hat{A}_t) \right)^2$$

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.27.1: Bradley-Terry-Luce Preference Model and Cross-Entropy Loss

##### Part 1: Problem Statement & Mathematical Goal
Let $\mathcal{X}$ denote the space of input prompts and $\mathcal{Y}$ the space of textual completions. Consider an arbitrary prompt $x \in \mathcal{X}$ and a pair of candidate responses $(y_1, y_2) \in \mathcal{Y} \times \mathcal{Y}$. In reinforcement learning from human feedback (RLHF), human evaluators or verifier models compare candidate completions, declaring one response strictly superior. Let $y_w \succ y_l \mid x$ denote the event that response $y_w$ (the chosen completion) is preferred over $y_l$ (the rejected completion) given context $x$.

Our mathematical goals are:
1. Formulate the latent utility framework (Thurstone Case V / Gumbel Random Utility Model) and prove from first principles that the pairwise preference probability is governed by the logistic sigmoid of the scalar latent reward difference:
   $$P(y_w \succ y_l \mid x) = \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) = \frac{1}{1 + \exp\big( -(r_\theta(x, y_w) - r_\theta(x, y_l)) \big)}$$
   where $r_\theta: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ is a parameterized reward model with parameter vector $\theta \in \mathbb{R}^d$.
2. Formulate the negative log-likelihood (binary cross-entropy) training objective over a dataset of pairwise preferences $\mathcal{D}_{\text{pref}} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$:
   $$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \ln \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$
3. Compute the exact analytical gradient $\nabla_\theta \mathcal{L}_{\text{RM}}(\theta)$ with respect to network parameters $\theta$, proving that the parameter update decomposes into symmetric, opposing forces that reinforce the winning representation and penalize the losing representation, self-modulated by the prediction residual $(1 - P)$.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Random Utility Maximization (RUM):** The unobserved true utility $U(x, y)$ assigned to response $y$ given prompt $x$ decomposes into a deterministic parameterized component $r_\theta(x, y)$ and a stochastic noise variable $\epsilon_y$:
   $$U(x, y) = r_\theta(x, y) + \epsilon_y$$
   where $\epsilon_w$ and $\epsilon_l$ are independent and identically distributed (i.i.d.) standard Gumbel (Type I Extreme Value) random variables with zero location and unit scale:
   $$\epsilon \sim \operatorname{Gumbel}(0, 1), \quad F(\epsilon) = \exp(-e^{-\epsilon}), \quad f(\epsilon) = e^{-\epsilon} \exp(-e^{-\epsilon})$$
2. **Strict Binary Preference (No Ties):** Judges must select either $y_w \succ y_l$ or $y_l \succ y_w$ almost surely; that is, $P(y_w \succ y_l \mid x) + P(y_l \succ y_w \mid x) = 1$. Indifference or ties have measure zero.
3. **Smooth Parameterization:** The reward function $r_\theta(x, y)$ is continuously differentiable ($\mathcal{C}^1$) with respect to parameter vector $\theta \in \mathbb{R}^d$ almost everywhere on its domain.
4. **Dataset Regularity & Boundedness:** The preference dataset $\mathcal{D}_{\text{pref}}$ consists of independent and identically distributed draws from an underlying population distribution $P(x, y_w, y_l)$. The reward predictions $|r_\theta(x, y)| \le M < \infty$ remain uniformly bounded on compact parameter subsets, preventing gradient explosion.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- **Energy-Based Logistic Mapping:** In latent reward space, the scalar values $r_\theta(x, y_w)$ and $r_\theta(x, y_l)$ represent the "goodness energies" of the candidate texts. The decision boundary depends strictly on the scalar difference $\Delta r \equiv r_\theta(x, y_w) - r_\theta(x, y_l)$. The logistic sigmoid function $\sigma: \mathbb{R} \to (0, 1)$ smoothly squashes the infinite real line into calibrated probabilities. When $\Delta r = 0$, $P = 0.5$ (maximum decision entropy); when $\Delta r \to +\infty$, $P \to 1.0$; when $\Delta r \to -\infty$, $P \to 0.0$.
- **Dynamic Force Regulation via Residual Error:** In physical terms, the gradient magnitude acts as an elastic restoring force governed by the prediction error $(1 - P)$. If the reward model already correctly ranks $y_w$ much higher than $y_l$ ($P \to 1.0$), the error residual $(1 - P) \to 0$, producing virtually zero gradient and preventing overfitting on clear examples. Conversely, if the reward model incorrectly ranks the loser higher ($P < 0.5$), the residual $(1 - P)$ approaches $1.0$, exerting maximal gradient torque.
- **Opposing Dipole in Parameter Space:** The parameter gradient $\nabla_\theta \mathcal{L}$ applies an equal and opposite electrostatic force along the representation manifold: pulling parameters along $+\nabla_\theta r_\theta(x, y_w)$ to elevate the winner, while pushing parameters along $-\nabla_\theta r_\theta(x, y_l)$ to suppress the loser.

##### Part 4: End-to-End Step-by-Step Algebraic Proof

###### Step 1: Derivation of the Logistic Probability from Gumbel Utility Noise
Under the Random Utility Maximization framework, judge preference $y_w \succ y_l$ occurs if and only if the latent utility of the winner exceeds that of the loser:
$$P(y_w \succ y_l \mid x) = P(U(x, y_w) > U(x, y_l)) = P(r_\theta(x, y_w) + \epsilon_w > r_\theta(x, y_l) + \epsilon_l)$$
Let $\Delta r \equiv r_\theta(x, y_w) - r_\theta(x, y_l)$. Rearranging the inequality:
$$P(y_w \succ y_l \mid x) = P(\epsilon_l - \epsilon_w < \Delta r) = P(\epsilon_l < \epsilon_w + \Delta r)$$
Because $\epsilon_w$ and $\epsilon_l$ are independent continuous random variables with probability density $f(\epsilon_w)$ and cumulative distribution $F(\epsilon_l)$, we condition on $\epsilon_w$ and integrate across the real line:
$$P(\epsilon_l < \epsilon_w + \Delta r) = \int_{-\infty}^{\infty} f(\epsilon_w) F(\epsilon_w + \Delta r) \, d\epsilon_w$$
Substitute the standard Gumbel density $f(\epsilon_w) = e^{-\epsilon_w} \exp(-e^{-\epsilon_w})$ and CDF $F(z) = \exp(-e^{-z})$:
$$\begin{aligned}
P(y_w \succ y_l \mid x) &= \int_{-\infty}^{\infty} e^{-\epsilon_w} \exp\left(-e^{-\epsilon_w}\right) \exp\left(-e^{-(\epsilon_w + \Delta r)}\right) \, d\epsilon_w \\
&= \int_{-\infty}^{\infty} e^{-\epsilon_w} \exp\left(-e^{-\epsilon_w}\right) \exp\left(-e^{-\epsilon_w} e^{-\Delta r}\right) \, d\epsilon_w \\
&= \int_{-\infty}^{\infty} e^{-\epsilon_w} \exp\left(-e^{-\epsilon_w} \left(1 + e^{-\Delta r}\right)\right) \, d\epsilon_w
\end{aligned}$$
Perform the substitution:
$$u \equiv e^{-\epsilon_w} \implies du = -e^{-\epsilon_w} \, d\epsilon_w \implies e^{-\epsilon_w} \, d\epsilon_w = -du$$
Determine the integration limits for $u$:
- As $\epsilon_w \to -\infty$, $u = e^{-(-\infty)} \to \infty$.
- As $\epsilon_w \to +\infty$, $u = e^{-\infty} \to 0$.

Substituting into the integral:
$$\begin{aligned}
P(y_w \succ y_l \mid x) &= \int_{\infty}^{0} \exp\left(-u \left(1 + e^{-\Delta r}\right)\right) (-du) \\
&= \int_{0}^{\infty} \exp\left(-u \left(1 + e^{-\Delta r}\right)\right) \, du
\end{aligned}$$
Evaluating this standard exponential integral:
$$\begin{aligned}
P(y_w \succ y_l \mid x) &= \left[ \frac{-\exp\left(-u \left(1 + e^{-\Delta r}\right)\right)}{1 + e^{-\Delta r}} \right]_{u=0}^{u=\infty} \\
&= \left( \lim_{u \to \infty} \frac{-\exp\left(-u \left(1 + e^{-\Delta r}\right)\right)}{1 + e^{-\Delta r}} \right) - \left( \frac{-\exp(0)}{1 + e^{-\Delta r}} \right) \\
&= 0 - \left( -\frac{1}{1 + e^{-\Delta r}} \right) \\
&= \frac{1}{1 + e^{-\Delta r}} \\
&= \sigma(\Delta r) = \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right)
\end{aligned}$$
This proves from first principles that the Bradley-Terry preference model corresponds exactly to utility maximization under Gumbel perturbation.

###### Step 2: Binary Cross-Entropy Loss Formulation
Given a preferred pair where $y_w \succ y_l$, the empirical target label is $t = 1$. The likelihood of observing this preference under model parameters $\theta$ is $P(y_w \succ y_l \mid x)$. Taking the negative logarithm yields the sample loss:
$$\ell(\theta; x, y_w, y_l) = -\ln P(y_w \succ y_l \mid x) = -\ln \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right)$$
Taking the expectation over the preference dataset $\mathcal{D}_{\text{pref}}$:
$$\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \ln \sigma\left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

###### Step 3: Exact Derivative of the Logistic Sigmoid
Let $z \in \mathbb{R}$ and define $\sigma(z) \equiv \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$.
Differentiating with respect to $z$ via the power rule and chain rule:
$$\begin{aligned}
\frac{d\sigma(z)}{dz} &= -(1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z}) \\
&= -(1 + e^{-z})^{-2} \cdot (-e^{-z}) \\
&= \frac{e^{-z}}{(1 + e^{-z})^2} \\
&= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} \\
&= \sigma(z) \cdot \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} \\
&= \sigma(z) \left( 1 - \frac{1}{1 + e^{-z}} \right) \\
&= \sigma(z) (1 - \sigma(z))
\end{aligned}$$

###### Step 4: Derivative of Negative Log-Likelihood with Respect to Reward Margin $\Delta r$
Let $\Delta r \equiv r_\theta(x, y_w) - r_\theta(x, y_l)$. The loss for a single preference pair is $\ell = -\ln \sigma(\Delta r)$.
Applying the chain rule:
$$\begin{aligned}
\frac{\partial \ell}{\partial \Delta r} &= -\frac{1}{\sigma(\Delta r)} \cdot \frac{d\sigma(\Delta r)}{d\Delta r} \\
&= -\frac{1}{\sigma(\Delta r)} \cdot \big[ \sigma(\Delta r) (1 - \sigma(\Delta r)) \big] \\
&= -(1 - \sigma(\Delta r)) \\
&= \sigma(\Delta r) - 1
\end{aligned}$$

###### Step 5: Parameter Gradient $\nabla_\theta \mathcal{L}_{\text{RM}}(\theta)$ via Multivariate Chain Rule
Applying the multivariate chain rule to differentiate $\ell$ with respect to vector $\theta \in \mathbb{R}^d$:
$$\nabla_\theta \ell = \frac{\partial \ell}{\partial \Delta r} \cdot \nabla_\theta (\Delta r)$$
Compute the gradient of the margin $\Delta r$:
$$\nabla_\theta (\Delta r) = \nabla_\theta \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) = \nabla_\theta r_\theta(x, y_w) - \nabla_\theta r_\theta(x, y_l)$$
Substituting $\frac{\partial \ell}{\partial \Delta r} = -(1 - \sigma(\Delta r))$:
$$\nabla_\theta \ell(\theta) = -(1 - \sigma(\Delta r)) \left[ \nabla_\theta r_\theta(x, y_w) - \nabla_\theta r_\theta(x, y_l) \right]$$
Taking expectations over $\mathcal{D}_{\text{pref}}$, the complete gradient is:
$$\mathbf{\nabla_\theta \mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \left(1 - \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right) \Big( \nabla_\theta r_\theta(x, y_w) - \nabla_\theta r_\theta(x, y_l) \Big) \right]}$$

In a gradient descent step with learning rate $\eta > 0$:
$$\begin{aligned}
\theta_{t+1} &= \theta_t - \eta \nabla_\theta \mathcal{L}_{\text{RM}}(\theta_t) \\
&= \theta_t + \eta \underbrace{\left(1 - \sigma(\Delta r)\right)}_{\text{Error Residual } \in (0, 1)} \cdot \underbrace{\nabla_\theta r_\theta(x, y_w)}_{\text{Boost Winner}} - \eta \underbrace{\left(1 - \sigma(\Delta r)\right)}_{\text{Error Residual } \in (0, 1)} \cdot \underbrace{\nabla_\theta r_\theta(x, y_l)}_{\text{Penalize Loser}} \quad \blacksquare
\end{aligned}$$

---

#### Derivation 11.27.2: Closed-Form Optimal Policy Under KL-Constrained RLHF

##### Part 1: Problem Statement & Mathematical Goal
Let $x \in \mathcal{X}$ be a prompt drawn from prompt distribution $\mathcal{D}$, and let $y \in \mathcal{Y}$ be a textual response. Let $\pi_{\text{ref}}(\cdot \mid x)$ denote the frozen reference policy (e.g., the supervised fine-tuned SFT base model). Let $r(x, y) \in \mathbb{R}$ be a known, frozen reward function (either an explicit reward model or oracle preference function).

The KL-constrained RLHF policy optimization problem seeks a policy $\pi$ that maximizes expected reward while penalizing divergence from $\pi_{\text{ref}}$:
$$\max_{\pi} \mathcal{J}(\pi) \equiv \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi(\cdot \mid x)} \big[ r(x, y) \big] - \beta D_{\text{KL}}\left( \pi(\cdot \mid x) \;\Vert\; \pi_{\text{ref}}(\cdot \mid x) \right) \right]$$
where $\beta > 0$ is the regularization temperature parameter, and $D_{\text{KL}}$ is the Kullback-Leibler divergence:
$$D_{\text{KL}}\left( \pi(\cdot \mid x) \;\Vert\; \pi_{\text{ref}}(\cdot \mid x) \right) \equiv \sum_{y \in \mathcal{Y}} \pi(y \mid x) \ln \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right)$$

Our mathematical goals are:
1. Derive the closed-form global non-parametric optimal policy $\pi^*(y \mid x)$ over the probability simplex $\Delta(\mathcal{Y})$ without any parametric restrictions:
   $$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
   where the prompt partition function is:
   $$Z(x) \equiv \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right) = \mathbb{E}_{y \sim \pi_{\text{ref}}(\cdot \mid x)} \left[ \exp\left( \frac{1}{\beta} r(x, y) \right) \right]$$
2. Prove that the maximum achievable objective value is:
   $$\mathcal{J}(\pi^*) = \beta \mathbb{E}_{x \sim \mathcal{D}} \left[ \ln Z(x) \right]$$
3. Derive the exact algebraic inversion that expresses the reward function $r(x, y)$ in terms of the optimal policy ratio, establishing the mathematical foundation of Direct Preference Optimization (DPO):
   $$r(x, y) = \beta \ln \left( \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) + \beta \ln Z(x)$$

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Prompt-Wise Separability:** The distribution $\mathcal{D}(x)$ has full support over prompt space $\mathcal{X}$. Because there are no cross-prompt constraints on $\pi(\cdot \mid x)$, maximizing the expectation $\mathbb{E}_{x \sim \mathcal{D}}[\mathcal{J}_x(\pi)]$ is equivalent to independently maximizing the inner objective $\mathcal{J}_x(\pi)$ for each prompt $x \in \mathcal{X}$ pointwise.
2. **Probability Simplex Constraints:** For every prompt $x \in \mathcal{X}$, the policy must be a valid probability distribution:
   $$\pi(y \mid x) \ge 0 \quad \forall y \in \mathcal{Y}, \quad \text{and} \quad \sum_{y \in \mathcal{Y}} \pi(y \mid x) = 1$$
3. **Absolute Continuity / Support Compatibility:** The policy $\pi$ must be absolutely continuous with respect to $\pi_{\text{ref}}$:
   $$\operatorname{supp}(\pi(\cdot \mid x)) \subseteq \operatorname{supp}(\pi_{\text{ref}}(\cdot \mid x))$$
   If $\pi_{\text{ref}}(y \mid x) = 0$ for some $y$, then $\pi(y \mid x) = 0$ necessarily, because otherwise $D_{\text{KL}}(\pi \parallel \pi_{\text{ref}}) = +\infty$, rendering the objective $-\infty$.
4. **Integrability & Finite Log-Partition Function:** The partition function $Z(x)$ satisfies $0 < Z(x) < \infty$ for all $x \in \mathcal{X}$. This is guaranteed if the reward function is bounded above: $\sup_{y \in \mathcal{Y}} r(x, y) \le M < \infty$.
5. **Strict Positive Regularization:** $\beta \in (0, \infty)$.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- **Thermodynamic Free Energy Equivalence:** In statistical physics, the equilibrium state of an ensemble in a heat bath at temperature $T$ balances energy minimization $\langle E \rangle$ with entropy maximization $S$:
  $$F = \langle E \rangle - T S$$
  Defining reward as negative energy $r(x, y) = -E(y)$ and setting temperature $T = \beta$, the RLHF objective is identically the negative Helmholtz Free Energy $-F(\pi)$. The unique thermodynamic equilibrium that minimizes free energy (and maximizes $-F$) is the celebrated **Gibbs-Boltzmann distribution**, where each state's probability is weighted by $\exp(-E/T) = \exp(r/\beta)$.
- **Information Projection ($I$-Projection):** Geometrically, the objective can be rewritten identically as:
  $$\mathcal{J}_x(\pi) = \beta \ln Z(x) - \beta D_{\text{KL}}\left( \pi(\cdot \mid x) \;\Vert\; \pi^*(\cdot \mid x) \right) $$
  Because KL divergence is non-negative and equals zero if and only if its arguments coincide, maximizing $\mathcal{J}_x(\pi)$ is equivalent to projecting $\pi$ onto the tilted distribution $\pi^*$.
- **Temperature Asymptotes:**
  - When $\beta \to \infty$ (infinite conservative penalty): $\frac{r(x, y)}{\beta} \to 0 \implies \exp(r/\beta) \to 1 \implies \pi^* \to \pi_{\text{ref}}$. The policy ignores the reward and sticks to the base model.
  - When $\beta \to 0^+$ (zero penalty, unconstrained RL): $\pi^*$ concentrates all probability mass exclusively on the greedy argmax $\operatorname{argmax}_{y \in \mathcal{Y}} r(x, y)$, collapsing exploratory diversity into deterministic reward-hacking.

##### Part 4: End-to-End Step-by-Step Algebraic Proof

###### Step 1: Pointwise Decomposition of the Objective
Fix an arbitrary prompt $x \in \mathcal{X}$. The single-prompt objective $\mathcal{J}_x(\pi)$ is:
$$\mathcal{J}_x(\pi) = \sum_{y \in \mathcal{Y}} \pi(y \mid x) r(x, y) - \beta \sum_{y \in \mathcal{Y}} \pi(y \mid x) \ln \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right)$$
Factor out the common scalar $-\beta$ from both summations:
$$\mathcal{J}_x(\pi) = -\beta \sum_{y \in \mathcal{Y}} \pi(y \mid x) \left[ \ln \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) - \frac{1}{\beta} r(x, y) \right]$$

###### Step 2: Incorporating the Reward into the Logarithm
Express the scalar term $\frac{1}{\beta} r(x, y)$ as the logarithm of its exponential:
$$\frac{1}{\beta} r(x, y) = \ln \left( \exp\left( \frac{1}{\beta} r(x, y) \right) \right)$$
Using the logarithmic quotient identity $\ln A - \ln B = \ln(A / B)$:
$$\begin{aligned}
\ln \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) - \frac{1}{\beta} r(x, y) &= \ln \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) - \ln \left( \exp\left( \frac{1}{\beta} r(x, y) \right) \right) \\
&= \ln \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)} \right)
\end{aligned}$$

###### Step 3: Defining the Unnormalized Measure and Partition Function
Define the unnormalized measure $\tilde{\pi}^*(y \mid x)$:
$$\tilde{\pi}^*(y \mid x) \equiv \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
and its total mass (the partition function $Z(x)$):
$$Z(x) \equiv \sum_{y \in \mathcal{Y}} \tilde{\pi}^*(y \mid x) = \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
The normalized candidate probability distribution $\pi^*(y \mid x)$ is:
$$\pi^*(y \mid x) \equiv \frac{\tilde{\pi}^*(y \mid x)}{Z(x)} = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
Notice that $\tilde{\pi}^*(y \mid x) = Z(x) \pi^*(y \mid x)$.

###### Step 4: Rewriting the Logarithmic Argument
Substitute $\tilde{\pi}^*(y \mid x) = Z(x) \pi^*(y \mid x)$ into the log term:
$$\begin{aligned}
\ln \left( \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)} \right) &= \ln \left( \frac{\pi(y \mid x)}{\tilde{\pi}^*(y \mid x)} \right) \\
&= \ln \left( \frac{\pi(y \mid x)}{Z(x) \pi^*(y \mid x)} \right) \\
&= \ln \left( \frac{\pi(y \mid x)}{\pi^*(y \mid x)} \right) - \ln Z(x)
\end{aligned}$$

###### Step 5: Algebraic Completion of the KL Divergence
Substitute this expression back into the single-prompt objective $\mathcal{J}_x(\pi)$:
$$\begin{aligned}
\mathcal{J}_x(\pi) &= -\beta \sum_{y \in \mathcal{Y}} \pi(y \mid x) \left[ \ln \left( \frac{\pi(y \mid x)}{\pi^*(y \mid x)} \right) - \ln Z(x) \right] \\
&= -\beta \sum_{y \in \mathcal{Y}} \pi(y \mid x) \ln \left( \frac{\pi(y \mid x)}{\pi^*(y \mid x)} \right) + \beta \ln Z(x) \sum_{y \in \mathcal{Y}} \pi(y \mid x)
\end{aligned}$$
Because $\pi(\cdot \mid x)$ is a valid probability distribution on the simplex, $\sum_{y \in \mathcal{Y}} \pi(y \mid x) = 1$. The first summation is precisely the definition of $D_{\text{KL}}(\pi(\cdot \mid x) \parallel \pi^*(\cdot \mid x))$:
$$\mathbf{\mathcal{J}_x(\pi) = -\beta D_{\text{KL}}\left( \pi(\cdot \mid x) \;\Vert\; \pi^*(\cdot \mid x) \right) + \beta \ln Z(x)}$$

###### Step 6: Maximization via Non-Negativity of KL Divergence
By Gibbs' Inequality (the Information Inequality):
$$D_{\text{KL}}(P \parallel Q) \ge 0$$
for any probability distributions $P, Q$, with strict equality $D_{\text{KL}}(P \parallel Q) = 0$ if and only if $P(y) = Q(y)$ for all $y \in \mathcal{Y}$.
Since $\beta > 0$:
$$-\beta D_{\text{KL}}\left( \pi(\cdot \mid x) \;\Vert\; \pi^*(\cdot \mid x) \right) \le 0$$
with maximum value of $0$ attained uniquely when:
$$\pi(y \mid x) \equiv \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$
Therefore, the maximum value of the single-prompt objective is:
$$\max_{\pi} \mathcal{J}_x(\pi) = \mathcal{J}_x(\pi^*) = \beta \ln Z(x)$$
Taking expectations over $x \sim \mathcal{D}$:
$$\mathbf{\max_{\pi} \mathcal{J}(\pi) = \beta \mathbb{E}_{x \sim \mathcal{D}} \left[ \ln Z(x) \right]}$$

###### Step 7: Dual Verification via Constrained Calculus of Variations (Lagrange Multipliers)
To verify this result independently, consider the constrained optimization problem on the probability simplex:
$$\max_{\pi} \sum_{y} \pi(y) r(x, y) - \beta \sum_y \pi(y) \ln \frac{\pi(y)}{\pi_{\text{ref}}(y)} \quad \text{s.t.} \quad \sum_y \pi(y) = 1$$
Construct the Lagrangian $\Lambda(\pi, \lambda)$ with multiplier $\lambda \in \mathbb{R}$:
$$\Lambda(\pi, \lambda) = \sum_y \pi(y) r(x, y) - \beta \sum_y \pi(y) \left[ \ln \pi(y) - \ln \pi_{\text{ref}}(y) \right] - \lambda \left( \sum_y \pi(y) - 1 \right)$$
Compute the partial derivative with respect to $\pi(y)$ for each $y \in \mathcal{Y}$:
$$\begin{aligned}
\frac{\partial \Lambda}{\partial \pi(y)} &= r(x, y) - \beta \left[ \ln \pi(y) + \pi(y) \cdot \frac{1}{\pi(y)} - \ln \pi_{\text{ref}}(y) \right] - \lambda \\
&= r(x, y) - \beta \ln \left( \frac{\pi(y)}{\pi_{\text{ref}}(y)} \right) - \beta - \lambda
\end{aligned}$$
Set $\frac{\partial \Lambda}{\partial \pi(y)} = 0$ for all $y$:
$$\beta \ln \left( \frac{\pi(y)}{\pi_{\text{ref}}(y)} \right) = r(x, y) - (\lambda + \beta) \implies \ln \left( \frac{\pi(y)}{\pi_{\text{ref}}(y)} \right) = \frac{1}{\beta} r(x, y) - \frac{\lambda + \beta}{\beta}$$
Exponentiate both sides:
$$\pi(y) = \pi_{\text{ref}}(y) \exp\left( \frac{1}{\beta} r(x, y) \right) \exp\left( -\frac{\lambda + \beta}{\beta} \right)$$
Enforcing the simplex constraint $\sum_y \pi(y) = 1$:
$$1 = \exp\left( -\frac{\lambda + \beta}{\beta} \right) \sum_{y} \pi_{\text{ref}}(y) \exp\left( \frac{1}{\beta} r(x, y) \right) = \exp\left( -\frac{\lambda + \beta}{\beta} \right) Z(x)$$
Hence:
$$\exp\left( -\frac{\lambda + \beta}{\beta} \right) = \frac{1}{Z(x)}$$
Substituting back into $\pi(y)$ yields identically:
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

###### Step 8: Direct Preference Optimization (DPO) Inversion
Taking the natural logarithm of the optimal policy:
$$\ln \pi^*(y \mid x) = \ln \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \ln Z(x)$$
Rearranging terms to isolate reward $r(x, y)$:
$$\frac{1}{\beta} r(x, y) = \ln \pi^*(y \mid x) - \ln \pi_{\text{ref}}(y \mid x) + \ln Z(x) = \ln \left( \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) + \ln Z(x)$$
Multiplying by $\beta$:
$$\mathbf{r(x, y) = \beta \ln \left( \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) + \beta \ln Z(x)}$$
When evaluated on a pairwise comparison $(y_w, y_l)$, the difference in rewards is:
$$\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left[ \beta \ln \left( \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} \right) + \beta \ln Z(x) \right] - \left[ \beta \ln \left( \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) + \beta \ln Z(x) \right] \\
&= \beta \ln \left( \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} \right) - \beta \ln \left( \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
\end{aligned}$$
Notice that the intractable partition function $\beta \ln Z(x)$ cancels completely! Substituting this into the Bradley-Terry cross-entropy loss bypasses training a separate reward model and actor-critic network entirely, yielding the DPO loss function:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \ln \sigma\left( \beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] \quad \blacksquare$$

---

#### Derivation 11.27.3: PPO-RLHF Augmented Reward Function and Per-Token KL Decomposition

##### Part 1: Problem Statement & Mathematical Goal
In RLHF with Proximal Policy Optimization (PPO), a generative language model $\pi_\theta$ generates a sequence of tokens $y = (y_1, y_2, \dots, y_T) \in \mathcal{V}^T$ of length $T$ autoregressively in response to prompt $x \in \mathcal{X}$. The sequence-level objective is:
$$\max_\theta \mathcal{J}_{\text{RLHF}}(\theta) \equiv \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta D_{\text{KL}}\left( \pi_\theta(\cdot \mid x) \;\Vert\; \pi_{\text{ref}}(\cdot \mid x) \right) \right]$$
where $r_\phi(x, y)$ is the terminal scalar score from a learned reward model, $\pi_{\text{ref}}$ is the frozen SFT baseline policy, and $\beta > 0$ is the KL penalty coefficient.

Under autoregressive factorizations:
$$\pi_\theta(y \mid x) = \prod_{t=1}^T \pi_\theta(y_t \mid x, y_{<t}), \quad \pi_{\text{ref}}(y \mid x) = \prod_{t=1}^T \pi_{\text{ref}}(y_t \mid x, y_{<t})$$
where $y_{<t} = (y_1, \dots, y_{t-1})$ is the token prefix with $y_{<1} = \emptyset$.

Our mathematical goals are:
1. Prove from first principles that the joint sequence-level KL divergence decomposes into the exact sum of expected token-level conditional KL divergences across the generation steps:
   $$\mathbb{E}_{y \sim \pi_\theta} \left[ D_{\text{KL}}\left( \pi_\theta(y \mid x) \;\Vert\; \pi_{\text{ref}}(y \mid x) \right) \right] = \sum_{t=1}^T \mathbb{E}_{y_{<t} \sim \pi_\theta} \left[ D_{\text{KL}}\left( \pi_\theta(\cdot \mid x, y_{<t}) \;\Vert\; \pi_{\text{ref}}(\cdot \mid x, y_{<t}) \right) \right]$$
2. Prove that defining the per-token reward signal:
   $$R_t = \begin{cases} -\beta \left( \ln \pi_\theta(y_t \mid x, y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right), & \text{for } t = 1, \dots, T-1 \\ r_\phi(x, y) - \beta \left( \ln \pi_\theta(y_T \mid x, y_{<T}) - \ln \pi_{\text{ref}}(y_T \mid x, y_{<T}) \right), & \text{for } t = T \end{cases}$$
   guarantees that the cumulative trajectory return $G \equiv \sum_{t=1}^T R_t$ is an exact, unbiased sample estimator of the sequence-level RLHF objective:
   $$\mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^T R_t \right] = \mathcal{J}_{\text{RLHF}}(\theta)$$
3. Formulate the token-level Markov Decision Process (MDP) state transitions, Bellman value equations for Critic $V_\psi(s_t)$, and Generalized Advantage Estimation (GAE) at the token resolution.

##### Part 2: Explicit Assumptions & Regularity Conditions
1. **Autoregressive Factorization & Causal Masking:** Both policy networks $\pi_\theta$ and $\pi_{\text{ref}}$ satisfy the strict causal Markov chain rule on vocabulary $\mathcal{V}$ ($|\mathcal{V}| < \infty$). Probabilities are conditioned strictly on preceding tokens $(x, y_{<t})$.
2. **Finite-Horizon Stopping Time:** Sequences terminate almost surely at a finite horizon $T \le T_{\max} < \infty$ upon emitting a specialized End-of-Sequence token `EOS` (or reaching maximum generation length).
3. **Common Support / Non-Vanishing Reference:** For every prefix $(x, y_{<t})$ and every token $v \in \mathcal{V}$:
   $$\pi_\theta(v \mid x, y_{<t}) > 0 \implies \pi_{\text{ref}}(v \mid x, y_{<t}) > 0$$
   ensuring the token log-ratio $\ln \frac{\pi_\theta}{\pi_{\text{ref}}}$ is finite and well-defined everywhere.
4. **Undiscounted Language MDP:** The discount factor $\gamma = 1.0$, because language generation is an episodic task where grammatical structure and semantic coherence depend uniformly on all tokens in the sequence.

##### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- **Solving Sparse Reward Temporal Credit Assignment:** If a model generates 256 tokens and receives a scalar reward $r_\phi = 3.2$ only after the final period, credit assignment is severely underdetermined: which of the 256 tokens contributed to the success, and which were filler or hallucinations? Decomposing the KL penalty per token turns the sparse sequence evaluation into a densely rewarded MDP.
- **Immediate Information Tax:** The per-token reward $R_t$ imposes an instantaneous "information tax" of $\beta \Delta \ln \pi_t$ nats at each generation step. If the actor selects a high-probability natural token consistent with $\pi_{\text{ref}}$, the tax is zero. If the actor takes an exploratory or out-of-distribution gamble ($\pi_\theta \gg \pi_{\text{ref}}$), it incurs an immediate penalty.
- **Value Network as Prefix Return Predictor:** The Critic network $V_\psi(x, y_{<t})$ estimates the sum of remaining KL penalties plus the terminal reward from the current prefix state $s_t = (x, y_{<t})$. By subtracting this baseline from the token return, GAE isolates the true advantage $\hat{A}_t$ of choosing token $y_t$ over alternative vocabulary words.

##### Part 4: End-to-End Step-by-Step Algebraic Proof

###### Step 1: Chain Rule Expansion of the Sequence-Level KL Divergence
Let $x$ be fixed and suppress conditioning on $x$ for clarity: denote $\pi_\theta(y \mid x)$ by $P(y)$ and $\pi_{\text{ref}}(y \mid x)$ by $Q(y)$.
The sequence-level KL divergence between joint distributions $P(y)$ and $Q(y)$ on $\mathcal{Y}^T$ is:
$$D_{\text{KL}}(P \parallel Q) = \sum_{y \in \mathcal{Y}^T} P(y) \ln \left( \frac{P(y)}{Q(y)} \right)$$
Substitute the autoregressive product decompositions $P(y) = \prod_{t=1}^T P(y_t \mid y_{<t})$ and $Q(y) = \prod_{t=1}^T Q(y_t \mid y_{<t})$:
$$\frac{P(y)}{Q(y)} = \prod_{t=1}^T \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})}$$
Taking the natural logarithm:
$$\ln \left( \frac{P(y)}{Q(y)} \right) = \ln \left( \prod_{t=1}^T \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right) = \sum_{t=1}^T \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right)$$
Substitute this sum of log-ratios back into the KL expectation:
$$\begin{aligned}
D_{\text{KL}}(P \parallel Q) &= \sum_{y \in \mathcal{Y}^T} P(y) \left[ \sum_{t=1}^T \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right) \right] \\
&= \sum_{t=1}^T \left[ \sum_{y \in \mathcal{Y}^T} P(y) \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right) \right]
\end{aligned}$$
where the exchange of finite summations is justified by Fubini's theorem.

###### Step 2: Marginalization of Uninvolved Tokens
For each fixed timestep $t \in \{1, \dots, T\}$, decompose the full sequence vector into past, present, and future tokens:
$$y = (y_{<t}, y_t, y_{>t}) \quad \text{where } y_{<t} \in \mathcal{Y}^{t-1}, \; y_t \in \mathcal{Y}, \; y_{>t} = (y_{t+1}, \dots, y_T) \in \mathcal{Y}^{T-t}$$
The joint probability distribution factors as:
$$P(y) = P(y_{<t}) \cdot P(y_t \mid y_{<t}) \cdot P(y_{>t} \mid y_{\le t})$$
The summation over all $y \in \mathcal{Y}^T$ decomposes into nested sums over past, present, and future tokens:
$$\sum_{y \in \mathcal{Y}^T} P(y) \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right) = \sum_{y_{<t}} \sum_{y_t} \sum_{y_{>t}} P(y_{<t}) P(y_t \mid y_{<t}) P(y_{>t} \mid y_{\le t}) \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right)$$
Notice that the term $\ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right)$ does not depend on future tokens $y_{>t}$. Factor it out of the innermost summation:
$$\sum_{y_{<t}} P(y_{<t}) \sum_{y_t} P(y_t \mid y_{<t}) \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right) \left[ \sum_{y_{>t}} P(y_{>t} \mid y_{\le t}) \right]$$
By the axiom of total probability, conditional probabilities sum to unity:
$$\sum_{y_{>t}} P(y_{>t} \mid y_{\le t}) = 1$$
Therefore, the innermost sum evaluates to 1:
$$\sum_{y_{<t}} P(y_{<t}) \left[ \sum_{y_t} P(y_t \mid y_{<t}) \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right) \right]$$
The inner sum over $y_t$ is exactly the conditional KL divergence between step distributions given prefix $y_{<t}$:
$$\sum_{y_t} P(y_t \mid y_{<t}) \ln \left( \frac{P(y_t \mid y_{<t})}{Q(y_t \mid y_{<t})} \right) = D_{\text{KL}}\left( P(\cdot \mid y_{<t}) \;\Vert\; Q(\cdot \mid y_{<t}) \right)$$
The outer sum over $y_{<t}$ is the expectation under the prefix distribution $y_{<t} \sim P$:
$$\sum_{y_{<t}} P(y_{<t}) D_{\text{KL}}\left( P(\cdot \mid y_{<t}) \;\Vert\; Q(\cdot \mid y_{<t}) \right) = \mathbb{E}_{y_{<t} \sim P} \left[ D_{\text{KL}}\left( P(\cdot \mid y_{<t}) \;\Vert\; Q(\cdot \mid y_{<t}) \right) \right]$$
Summing across all timesteps $t = 1, \dots, T$:
$$\mathbf{D_{\text{KL}}(P \parallel Q) = \sum_{t=1}^T \mathbb{E}_{y_{<t} \sim P} \left[ D_{\text{KL}}\left( P(\cdot \mid y_{<t}) \;\Vert\; Q(\cdot \mid y_{<t}) \right) \right]}$$
This proves the exact chain rule of relative entropy for autoregressive language models.

###### Step 3: Exact Equivalence of Cumulative Trajectory Return
Define the per-token reward $R_t$ for a sampled completion $y = (y_1, \dots, y_T)$:
$$R_t \equiv -\beta \left( \ln \pi_\theta(y_t \mid x, y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right) + \delta_{t, T} r_\phi(x, y)$$
where $\delta_{t, T}$ is the Kronecker delta ($\delta_{t, T} = 1$ if $t = T$, and $0$ otherwise).
Summing $R_t$ over all generation steps $t = 1, \dots, T$:
$$\begin{aligned}
G \equiv \sum_{t=1}^T R_t &= \sum_{t=1}^T \left[ -\beta \left( \ln \pi_\theta(y_t \mid x, y_{<t}) - \ln \pi_{\text{ref}}(y_t \mid x, y_{<t}) \right) + \delta_{t, T} r_\phi(x, y) \right] \\
&= \left( \sum_{t=1}^T \delta_{t, T} r_\phi(x, y) \right) - \beta \sum_{t=1}^T \ln \left( \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} \right) \\
&= r_\phi(x, y) - \beta \ln \left( \prod_{t=1}^T \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} \right) \\
&= r_\phi(x, y) - \beta \ln \left( \frac{\prod_{t=1}^T \pi_\theta(y_t \mid x, y_{<t})}{\prod_{t=1}^T \pi_{\text{ref}}(y_t \mid x, y_{<t})} \right) \\
&= r_\phi(x, y) - \beta \ln \left( \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right)
\end{aligned}$$
Taking the expectation of this trajectory return with respect to generated sequences $y \sim \pi_\theta(\cdot \mid x)$:
$$\begin{aligned}
\mathbb{E}_{y \sim \pi_\theta}[G] &= \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^T R_t \right] \\
&= \mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)] - \beta \mathbb{E}_{y \sim \pi_\theta} \left[ \ln \left( \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) \right] \\
&= \mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)] - \beta \sum_{y \in \mathcal{Y}^T} \pi_\theta(y \mid x) \ln \left( \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right) \\
&= \mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)] - \beta D_{\text{KL}}\left( \pi_\theta(\cdot \mid x) \;\Vert\; \pi_{\text{ref}}(\cdot \mid x) \right) \\
&= \mathcal{J}_{\text{RLHF}}(\theta)
\end{aligned}$$
This proves that the expectation of the sum of token rewards is identically equal to the sequence-level RLHF objective, establishing that the token decomposition is strictly unbiased.

###### Step 4: Formulation of the Token MDP, Bellman Equations, and GAE
Formulate the token generation process as a Markov Decision Process $\mathcal{M}_{\text{token}} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
- **State space $\mathcal{S}$:** Prefix context $s_t = (x, y_{<t})$. The initial state is $s_1 = (x)$.
- **Action space $\mathcal{A}$:** Next token $a_t = y_t \in \mathcal{V}$.
- **Deterministic Transition Kernel $\mathcal{P}$:** Concatenation $s_{t+1} = (x, y_{\le t}) = [s_t, a_t]$.
- **Reward Function $\mathcal{R}(s_t, a_t)$:** Given by token reward $R_t$.
- **Discount Factor:** $\gamma = 1.0$.

Under policy $\pi_\theta$, the state-value function $V^{\pi_\theta}(s_t)$ satisfies the Bellman expectation equation:
$$V^{\pi_\theta}(s_t) = \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ R_t + \gamma V^{\pi_\theta}(s_{t+1}) \right]$$
For parameter Critic network $V_\psi(s_t)$, the one-step temporal-difference (TD) error at token step $t$ is:
$$\delta_t = R_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$
with boundary condition at terminal state $s_{T+1}$ of $V_\psi(s_{T+1}) \equiv 0$, giving $\delta_T = R_T - V_\psi(s_T)$.

The Generalized Advantage Estimator (GAE) with exponential decay $\lambda \in [0, 1]$ is:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$
which satisfies the backward recurrence:
$$\hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}, \quad \text{with } \hat{A}_T = \delta_T$$
This completes the end-to-end mathematical derivation of the PPO-RLHF token reward decomposition and credit assignment framework. $\blacksquare$

---

## 3. Geometric & Physical Interpretation

### 3.1 The Restoring Spring of the KL Penalty
Think of the policy parameter space as a physical landscape:
```
Reward Surface r_psi(x, y)
 ^
 |             /-- Over-optimized Adversarial Peak (Gibberish / Sycophancy)
 |            /
 |           /     * <-- Optimal Trade-off Point (High Reward + Natural Language)
 |          /     /
 |   [ SFT Reference pi_ref ] <==== KL Spring (Hooke's Law: -beta * grad KL) ===
 |________________________________________________________> Policy Space
```
Without the KL penalty ($\beta = 0$), the policy optimizes ruthlessly against the reward model, finding adversarial blind spots (e.g., repeating the word *"helpful"* 50 times, or sycophantically agreeing with false user statements). The KL penalty acts as a **Hooke's Law elastic spring**, pulling the policy back toward fluent, human-like distribution support.

---

## 4. Real-World Analogy: Speechwriter & Public Opinion Poll

Imagine a politician preparing a major address:
- **The Speechwriter ($\pi_\theta$):** Writes sentences word by word.
- **The Opinion Poll ($r_\psi$):** Measures voter sentiment ($+10$ for inspiring visions, $-10$ for gaffes).
- **The Politician's Core Identity ($\pi_{\text{ref}}$):** The politician's authentic speaking style and principles.
- **The KL Penalty ($\beta$):** If the speechwriter writes pandering, robotic buzzwords just to spike focus-group meters, the politician vetoes it: *"That doesn't sound like me at all!"*

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical calculation of **Token-Level Rewards, TD Errors, and GAE Advantages** by hand for a 2-token generation sequence.

---

### 5.1 System & Sequence Parameters
- **Prompt:** $x$
- **Generated Response:** 2 tokens $y = (y_1, y_2)$
- **KL Coefficient:** $\beta = 0.1000$
- **Discount:** $\gamma = 1.0000$
- **GAE Parameter:** $\lambda = 0.9500$
- **Token Probabilities:**
  - Token 1:
    $$\pi_\theta(y_1 \mid x) = 0.4000, \quad \pi_{\text{ref}}(y_1 \mid x) = 0.5000$$
  - Token 2 (Terminal):
    $$\pi_\theta(y_2 \mid x, y_1) = 0.8000, \quad \pi_{\text{ref}}(y_2 \mid x, y_1) = 0.4000$$
- **Reward Model Output at EOS:**
  $$r_\psi(x, y) = 3.0000$$
- **Critic Value Predictions:**
  - $V(s_1) = 2.5000$ (after seeing prompt $x$)
  - $V(s_2) = 2.8000$ (after seeing $x, y_1$)
  - $V(s_3) = 0.0000$ (terminal state after $y_2$)

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $\pi_\theta(y_t), \pi_{\text{ref}}(y_t)$ | `pi_curr[t], pi_ref[t]` | Current and reference policy token probabilities |
| $\Delta \ln \pi_t$ | `log_ratio[t]` | Per-token KL divergence component: $\ln \pi_\theta(y_t) - \ln \pi_{\text{ref}}(y_t)$ |
| $R_t$ | `reward_t[t]` | Decomposed token-level reward incorporating reward model & KL |
| $V(s_t)$ | `critic_v[t]` | Baseline value estimate of expected remaining reward |
| $\delta_t$ | `td_delta[t]` | Token TD error: $R_t + \gamma V(s_{t+1}) - V(s_t)$ |
| $\hat{A}_t$ | `gae_adv[t]` | Generalized Advantage Estimate for token $y_t$ |

---

### 5.3 Step-by-Step Hand Calculations

#### Step 1: Compute Token Log-Probabilities & KL Penalties
- **Token 1 ($t = 1$):**
  $$\ln \pi_\theta(y_1) = \ln(0.4000) \approx \mathbf{-0.916291}$$
  $$\ln \pi_{\text{ref}}(y_1) = \ln(0.5000) \approx \mathbf{-0.693147}$$
  $$\Delta \ln \pi_1 = -0.916291 - (-0.693147) = \mathbf{-0.223144}$$
  *(Since $\pi_\theta < \pi_{\text{ref}}$, the policy is less confident than reference, so KL penalty is negative, meaning a positive bonus!)*
  $$R_1 = -\beta \cdot \Delta \ln \pi_1 = -0.1000 \times (-0.223144) = \mathbf{+0.022314}$$

- **Token 2 ($t = 2$, Terminal):**
  $$\ln \pi_\theta(y_2) = \ln(0.8000) \approx \mathbf{-0.223144}$$
  $$\ln \pi_{\text{ref}}(y_2) = \ln(0.4000) \approx \mathbf{-0.916291}$$
  $$\Delta \ln \pi_2 = -0.223144 - (-0.916291) = \mathbf{+0.693147}$$
  Terminal reward:
  $$R_2 = r_\psi(x, y) - \beta \cdot \Delta \ln \pi_2 = 3.0000 - 0.1000 \times 0.693147 = 3.0000 - 0.069315 = \mathbf{2.930685}$$

#### Step 2: Compute Token TD Residuals ($\gamma = 1.0$)
- **Residual for Token 1:**
  $$\delta_1 = R_1 + V(s_2) - V(s_1) = 0.022314 + 2.800000 - 2.500000 = 0.022314 + 0.300000 = \mathbf{0.322314}$$

- **Residual for Token 2:**
  $$\delta_2 = R_2 + V(s_3) - V(s_2) = 2.930685 + 0.000000 - 2.800000 = \mathbf{0.130685}$$

#### Step 3: Compute GAE Advantages ($\lambda = 0.95$)
We compute backward from terminal token $t = 2$:
$$\hat{A}_2 = \delta_2 = \mathbf{0.130685}$$
$$\hat{A}_1 = \delta_1 + (\gamma \lambda) \hat{A}_2 = 0.322314 + (1.0 \times 0.95) \times 0.130685$$
$$= 0.322314 + 0.124151 = \mathbf{0.446465}$$

---

### 5.4 Summary Visual Grid: RLHF Token PPO Ledger

| Token | $\pi_\theta$ | $\pi_{\text{ref}}$ | $\Delta \ln \pi$ | RM $r_\psi$ | Token Reward $R_t$ | Critic $V(s)$ | TD Error $\delta_t$ | GAE $\hat{A}_t$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$y_1$** | $0.40$ | $0.50$ | $-0.2231$ | — | **$+0.0223$** | $2.5000$ | **$+0.3223$** | $\mathbf{+0.4465}$ |
| **$y_2$ (EOS)** | $0.80$ | $0.40$ | $+0.6931$ | $3.0000$ | **$+2.9307$** | $2.8000$ | **$+0.1307$** | $\mathbf{+0.1307}$ |
| **$s_3$ (End)** | — | — | — | — | — | $0.0000$ | — | — |

---

## 6. Solved Illustrations

### Illustration 1: Bradley-Terry Pairwise Loss Gradient
**Problem:**
Suppose prompt $x$ produces two completions. The current reward model outputs:
$$r_\psi(x, y_w) = 1.2000, \quad r_\psi(x, y_l) = 0.2000$$
1. Calculate the predicted preference probability $P(y_w \succ y_l)$.
2. Calculate the scalar gradient of the loss with respect to $r_\psi(x, y_w)$ and $r_\psi(x, y_l)$.

**Solution:**
1. **Preference Probability:**
   $$\Delta r = 1.2000 - 0.2000 = 1.0000$$
   $$P(y_w \succ y_l) = \sigma(\Delta r) = \frac{1}{1 + e^{-1.0}} \approx \frac{1}{1 + 0.367879} = \frac{1}{1.367879} \approx \mathbf{0.731059}$$

2. **Loss Gradients:**
   $$\mathcal{L} = -\ln \sigma(\Delta r)$$
   $$\frac{\partial \mathcal{L}}{\partial \Delta r} = -(1 - \sigma(\Delta r)) = -(1.0 - 0.731059) = \mathbf{-0.268941}$$
   Therefore:
   $$\nabla_{r(y_w)} \mathcal{L} = \mathbf{-0.268941} \quad (\text{gradient descent pushes } r(y_w) \text{ UP!})$$
   $$\nabla_{r(y_l)} \mathcal{L} = \mathbf{+0.268941} \quad (\text{gradient descent pushes } r(y_l) \text{ DOWN!}) \quad \blacksquare$$

---

### Illustration 2: Bradley-Terry Reward Model Loss and Weight Gradient Update
**Problem:**
Let a parameterized reward model have a linear readout layer on top of transformer representation $\mathbf{h}(x, y) \in \mathbb{R}^2$:
$$r_{\mathbf{w}, b}(x, y) = \mathbf{w}^T \mathbf{h}(x, y) + b$$
where current parameters are $\mathbf{w}_0 = [1.5000, -0.5000]^T$ and $b = 0.2500$.
A prompt $x$ generates two completions with feature representations:
- Chosen completion $y_w$: $\mathbf{h}(x, y_w) = [1.2000, 0.5000]^T$
- Rejected completion $y_l$: $\mathbf{h}(x, y_l) = [0.4000, 0.9000]^T$

1. Verify the forward reward predictions $r(x, y_w)$ and $r(x, y_l)$ and calculate the margin $\Delta r$.
2. Compute the predicted preference probability $P(y_w \succ y_l) = \sigma(\Delta r)$ and the cross-entropy loss $\mathcal{L}_{\text{RM}}$.
3. Compute the scalar error residual $\frac{\partial \mathcal{L}}{\partial \Delta r}$, the parameter gradient $\nabla_{\mathbf{w}} \mathcal{L}$, and the bias gradient $\frac{\partial \mathcal{L}}{\partial b}$.
4. Perform one gradient descent step with learning rate $\eta = 0.1000$ to obtain updated weights $\mathbf{w}_1$, and verify that the updated loss strictly decreases.

**Solution:**
1. **Forward Reward Predictions:**
   $$\begin{aligned}
   r(x, y_w) &= \mathbf{w}_0^T \mathbf{h}(x, y_w) + b = (1.5000)(1.2000) + (-0.5000)(0.5000) + 0.2500 \\
   &= 1.8000 - 0.2500 + 0.2500 = \mathbf{1.8000}
   \end{aligned}$$
   $$\begin{aligned}
   r(x, y_l) &= \mathbf{w}_0^T \mathbf{h}(x, y_l) + b = (1.5000)(0.4000) + (-0.5000)(0.9000) + 0.2500 \\
   &= 0.6000 - 0.4500 + 0.2500 = \mathbf{0.4000}
   \end{aligned}$$
   Reward Margin:
   $$\Delta r = r(x, y_w) - r(x, y_l) = 1.8000 - 0.4000 = \mathbf{1.4000}$$

2. **Preference Probability & Loss:**
   $$e^{-\Delta r} = e^{-1.4000} \approx 0.246597$$
   $$P(y_w \succ y_l) = \sigma(1.4000) = \frac{1}{1 + 0.246597} = \frac{1}{1.246597} \approx \mathbf{0.802184}$$
   $$\mathcal{L}_{\text{RM}} = -\ln \sigma(\Delta r) = -\ln(0.802184) \approx \mathbf{0.220417}$$

3. **Gradients:**
   Scalar error residual:
   $$\frac{\partial \mathcal{L}}{\partial \Delta r} = -(1 - \sigma(\Delta r)) = -(1.000000 - 0.802184) = \mathbf{-0.197816}$$
   Difference in feature representations:
   $$\Delta \mathbf{h} \equiv \mathbf{h}(x, y_w) - \mathbf{h}(x, y_l) = \begin{bmatrix} 1.2000 - 0.4000 \\ 0.5000 - 0.9000 \end{bmatrix} = \begin{bmatrix} +0.8000 \\ -0.4000 \end{bmatrix}$$
   Weight gradient:
   $$\nabla_{\mathbf{w}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \Delta r} \cdot \Delta \mathbf{h} = -0.197816 \times \begin{bmatrix} +0.8000 \\ -0.4000 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.158253} \\ \mathbf{+0.079126} \end{bmatrix}$$
   Bias gradient:
   $$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial \Delta r} \left( \frac{\partial r(y_w)}{\partial b} - \frac{\partial r(y_l)}{\partial b} \right) = -0.197816 \times (1 - 1) = \mathbf{0.000000}$$

4. **Parameter Update & Re-Evaluation:**
   With learning rate $\eta = 0.1000$:
   $$\mathbf{w}_1 = \mathbf{w}_0 - \eta \nabla_{\mathbf{w}} \mathcal{L} = \begin{bmatrix} 1.5000 \\ -0.5000 \end{bmatrix} - 0.1000 \begin{bmatrix} -0.158253 \\ 0.079126 \end{bmatrix} = \begin{bmatrix} \mathbf{1.515825} \\ \mathbf{-0.507913} \end{bmatrix}$$
   New predicted rewards:
   $$r_1(y_w) = 1.515825(1.2000) - 0.507913(0.5000) + 0.2500 = 1.818990 - 0.253956 + 0.2500 = \mathbf{1.815034}$$
   $$r_1(y_l) = 1.515825(0.4000) - 0.507913(0.9000) + 0.2500 = 0.606330 - 0.457122 + 0.2500 = \mathbf{0.399209}$$
   $$\Delta r_1 = 1.815034 - 0.399209 = \mathbf{1.415825}$$
   $$P_1 = \sigma(1.415825) \approx \mathbf{0.804683}$$
   $$\mathcal{L}_1 = -\ln(0.804683) \approx \mathbf{0.217307} < 0.220417 \quad (\text{Strict loss reduction verified!}) \quad \blacksquare$$

---

### Illustration 3: Per-Token KL Penalty and Advantage Computation over a 4-Token Sequence
**Problem:**
An actor model $\pi_\theta$ generates a completion of $T = 4$ tokens $y = (y_1, y_2, y_3, y_4)$ terminating at $y_4 = \text{EOS}$.
Given hyperparameters and predictions:
- KL penalty coefficient: $\beta = 0.0500$
- Discount factor: $\gamma = 1.0000$
- GAE decay parameter: $\lambda = 0.9500$
- Terminal reward model score: $r_\phi(x, y) = 2.5000$
- Token probabilities:
  - Step 1: $\pi_\theta(y_1 \mid x) = 0.6000, \quad \pi_{\text{ref}}(y_1 \mid x) = 0.4000$
  - Step 2: $\pi_\theta(y_2 \mid x, y_1) = 0.2500, \quad \pi_{\text{ref}}(y_2 \mid x, y_1) = 0.5000$
  - Step 3: $\pi_\theta(y_3 \mid x, y_{\le 2}) = 0.7000, \quad \pi_{\text{ref}}(y_3 \mid x, y_{\le 2}) = 0.3500$
  - Step 4: $\pi_\theta(y_4 \mid x, y_{\le 3}) = 0.8000, \quad \pi_{\text{ref}}(y_4 \mid x, y_{\le 3}) = 0.4000$
- Critic value estimates:
  - $V(s_1) = 2.2000, \; V(s_2) = 2.3000, \; V(s_3) = 2.4000, \; V(s_4) = 2.4500, \; V(s_5) \equiv 0.0000$

1. Compute the per-token log-ratios $\Delta \ln \pi_t$ and the augmented token rewards $R_t$ for all steps $t \in \{1, 2, 3, 4\}$.
2. Verify that the cumulative return $\sum_{t=1}^4 R_t$ identically matches the sequence objective $r_\phi - \beta \sum_t \Delta \ln \pi_t$.
3. Compute the temporal-difference residuals $\delta_t$ for each step.
4. Compute the GAE advantages $\hat{A}_t$ backward from terminal step $t = 4$.

**Solution:**
1. **Token Log-Ratios & Rewards:**
   - **Token 1 ($t = 1$):**
     $$\Delta \ln \pi_1 = \ln\left(\frac{0.6000}{0.4000}\right) = \ln(1.5000) \approx \mathbf{+0.405465}$$
     $$R_1 = -\beta \Delta \ln \pi_1 = -0.0500 \times 0.405465 = \mathbf{-0.020273}$$
   - **Token 2 ($t = 2$):**
     $$\Delta \ln \pi_2 = \ln\left(\frac{0.2500}{0.5000}\right) = \ln(0.5000) \approx \mathbf{-0.693147}$$
     $$R_2 = -\beta \Delta \ln \pi_2 = -0.0500 \times (-0.693147) = \mathbf{+0.034657}$$
     *(Note: When policy is more conservative than reference, $\Delta \ln \pi < 0$, giving an immediate positive exploration bonus!)*
   - **Token 3 ($t = 3$):**
     $$\Delta \ln \pi_3 = \ln\left(\frac{0.7000}{0.3500}\right) = \ln(2.0000) \approx \mathbf{+0.693147}$$
     $$R_3 = -\beta \Delta \ln \pi_3 = -0.0500 \times 0.693147 = \mathbf{-0.034657}$$
   - **Token 4 ($t = 4$, Terminal EOS):**
     $$\Delta \ln \pi_4 = \ln\left(\frac{0.8000}{0.4000}\right) = \ln(2.0000) \approx \mathbf{+0.693147}$$
     Terminal step incorporates both the full reward model score and the final token KL penalty:
     $$R_4 = r_\phi(x, y) - \beta \Delta \ln \pi_4 = 2.5000 - 0.0500 \times 0.693147 = 2.5000 - 0.034657 = \mathbf{+2.465343}$$

2. **Return Consistency Verification:**
   $$\sum_{t=1}^4 R_t = -0.020273 + 0.034657 - 0.034657 + 2.465343 = \mathbf{2.445069}$$
   Sequence objective calculation:
   $$\sum_{t=1}^4 \Delta \ln \pi_t = 0.405465 - 0.693147 + 0.693147 + 0.693147 = 1.098612$$
   $$r_\phi(x, y) - \beta \sum_{t=1}^4 \Delta \ln \pi_t = 2.5000 - 0.0500 \times 1.098612 = 2.5000 - 0.054931 = \mathbf{2.445069} \quad \checkmark$$

3. **Temporal-Difference Errors ($\gamma = 1.0$):**
   $$\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$$
   - $\delta_1 = -0.020273 + 2.300000 - 2.200000 = -0.020273 + 0.100000 = \mathbf{+0.079727}$
   - $\delta_2 = +0.034657 + 2.400000 - 2.300000 = +0.034657 + 0.100000 = \mathbf{+0.134657}$
   - $\delta_3 = -0.034657 + 2.450000 - 2.400000 = -0.034657 + 0.050000 = \mathbf{+0.015343}$
   - $\delta_4 = +2.465343 + 0.000000 - 2.450000 = \mathbf{+0.015343}$

4. **GAE Advantages ($\lambda = 0.95$):**
   $$\hat{A}_4 = \delta_4 = \mathbf{+0.015343}$$
   $$\hat{A}_3 = \delta_3 + \lambda \hat{A}_4 = 0.015343 + 0.9500 \times 0.015343 = 0.015343 + 0.014576 = \mathbf{+0.029918}$$
   $$\hat{A}_2 = \delta_2 + \lambda \hat{A}_3 = 0.134657 + 0.9500 \times 0.029918 = 0.134657 + 0.028422 = \mathbf{+0.163080}$$
   $$\hat{A}_1 = \delta_1 + \lambda \hat{A}_2 = 0.079727 + 0.9500 \times 0.163080 = 0.079727 + 0.154926 = \mathbf{+0.234652}$$

**Ledger Summary Table:**
| Step $t$ | Token $y_t$ | $\pi_\theta$ | $\pi_{\text{ref}}$ | $\Delta \ln \pi_t$ | Reward $R_t$ | Critic $V(s_t)$ | TD Residual $\delta_t$ | Advantage $\hat{A}_t$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1$** | $y_1$ | $0.6000$ | $0.4000$ | $+0.4055$ | **$-0.0203$** | $2.2000$ | **$+0.0797$** | $\mathbf{+0.2347}$ |
| **$2$** | $y_2$ | $0.2500$ | $0.5000$ | $-0.6931$ | **$+0.0347$** | $2.3000$ | **$+0.1347$** | $\mathbf{+0.1631}$ |
| **$3$** | $y_3$ | $0.7000$ | $0.3500$ | $+0.6931$ | **$-0.0347$** | $2.4000$ | **$+0.0153$** | $\mathbf{+0.0299}$ |
| **$4$** | $y_4$ (EOS) | $0.8000$ | $0.4000$ | $+0.6931$ | **$+2.4653$** | $2.4500$ | **$+0.0153$** | $\mathbf{+0.0153}$ |
| **$5$** | End State | — | — | — | — | $0.0000$ | — | — | $\blacksquare$

---

### Illustration 4: Closed-Form Optimal Policy Distribution across 3 Candidate Responses vs. Greedy Policy
**Problem:**
For a prompt $x$, a language model can generate 3 candidate responses $\mathcal{Y} = \{y_1, y_2, y_3\}$.
The reference policy probabilities $\pi_{\text{ref}}$ and evaluated scalar rewards $r(x, y)$ are:
- Candidate $y_1$: $\pi_{\text{ref}}(y_1 \mid x) = 0.6000, \quad r(x, y_1) = 1.0000$
- Candidate $y_2$: $\pi_{\text{ref}}(y_2 \mid x) = 0.3000, \quad r(x, y_2) = 2.5000$
- Candidate $y_3$: $\pi_{\text{ref}}(y_3 \mid x) = 0.1000, \quad r(x, y_3) = 4.0000$
Notice that $y_3$ yields the highest reward ($4.0000$), but is an infrequent tail generation under the reference model ($\pi_{\text{ref}} = 0.1000$).

1. Compute the exact closed-form optimal policy distribution $\pi^*(y \mid x)$ and the partition function $Z(x)$ under regularization $\beta = 1.0000$.
2. Compute the expected reward $\mathbb{E}_{\pi^*}[r]$, the KL divergence $D_{\text{KL}}(\pi^* \parallel \pi_{\text{ref}})$, and the net objective $\mathcal{J}(\pi^*)$. Verify that $\mathcal{J}(\pi^*) = \beta \ln Z(x)$.
3. Recompute $\pi^*(y \mid x)$ under stronger optimization $\beta = 0.5000$.
4. Compare with the unconstrained greedy policy $\pi_{\text{greedy}} = [0, 0, 1]^T$ and explain why greedy generation is strictly suboptimal under the KL-regularized objective.

**Solution:**
1. **Optimal Policy under $\beta = 1.0000$:**
   Compute unnormalized probabilities $\tilde{\pi}^*(y_i) = \pi_{\text{ref}}(y_i) \exp(r(x, y_i) / \beta)$:
   - $\tilde{\pi}^*(y_1) = 0.6000 \times e^{1.0000 / 1.0} = 0.6000 \times 2.718282 = \mathbf{1.630969}$
   - $\tilde{\pi}^*(y_2) = 0.3000 \times e^{2.5000 / 1.0} = 0.3000 \times 12.182494 = \mathbf{3.654748}$
   - $\tilde{\pi}^*(y_3) = 0.1000 \times e^{4.0000 / 1.0} = 0.1000 \times 54.598150 = \mathbf{5.459815}$
   Partition function:
   $$Z(x) = 1.630969 + 3.654748 + 5.459815 = \mathbf{10.745532}$$
   Normalized optimal policy $\pi^*(y_i) = \tilde{\pi}^*(y_i) / Z(x)$:
   - $\pi^*(y_1) = \frac{1.630969}{10.745532} \approx \mathbf{0.151781}$
   - $\pi^*(y_2) = \frac{3.654748}{10.745532} \approx \mathbf{0.340118}$
   - $\pi^*(y_3) = \frac{5.459815}{10.745532} \approx \mathbf{0.508101}$
   *(Verification of sum: $0.151781 + 0.340118 + 0.508101 = 1.000000$)*

2. **Expected Reward, KL Divergence, and Objective Verification:**
   Expected reward:
   $$\begin{aligned}
   \mathbb{E}_{\pi^*}[r] &= 0.151781(1.0000) + 0.340118(2.5000) + 0.508101(4.0000) \\
   &= 0.151781 + 0.850295 + 2.032404 = \mathbf{3.034480}
   \end{aligned}$$
   KL divergence:
   $$\begin{aligned}
   D_{\text{KL}}(\pi^* \parallel \pi_{\text{ref}}) &= 0.151781 \ln\left(\frac{0.151781}{0.6000}\right) + 0.340118 \ln\left(\frac{0.340118}{0.3000}\right) + 0.508101 \ln\left(\frac{0.508101}{0.1000}\right) \\
   &= 0.151781 \ln(0.252968) + 0.340118 \ln(1.133727) + 0.508101 \ln(5.081010) \\
   &= 0.151781(-1.374492) + 0.340118(+0.125510) + 0.508101(+1.625510) \\
   &= -0.208622 + 0.042688 + 0.825924 = \mathbf{0.659990}
   \end{aligned}$$
   Net objective:
   $$\mathcal{J}(\pi^*) = \mathbb{E}_{\pi^*}[r] - \beta D_{\text{KL}} = 3.034480 - 1.0000 \times 0.659990 = \mathbf{2.374490}$$
   Closed-form partition function prediction:
   $$\beta \ln Z(x) = 1.0000 \times \ln(10.745532) = \mathbf{2.374490} \quad (\text{Exact analytical agreement!}) \quad \checkmark$$

3. **Optimal Policy under $\beta = 0.5000$:**
   - $\tilde{\pi}^*(y_1) = 0.6000 \times e^{1.0 / 0.5} = 0.6000 \times e^2 = 0.6000 \times 7.389056 = \mathbf{4.433434}$
   - $\tilde{\pi}^*(y_2) = 0.3000 \times e^{2.5 / 0.5} = 0.3000 \times e^5 = 0.3000 \times 148.413159 = \mathbf{44.523948}$
   - $\tilde{\pi}^*(y_3) = 0.1000 \times e^{4.0 / 0.5} = 0.1000 \times e^8 = 0.1000 \times 2980.957987 = \mathbf{298.095799}$
   $$Z(x) = 4.433434 + 44.523948 + 298.095799 = \mathbf{347.053181}$$
   Normalized probabilities:
   $$\pi^*(y_1) = \mathbf{0.012775}, \quad \pi^*(y_2) = \mathbf{0.128291}, \quad \pi^*(y_3) = \mathbf{0.858934}$$
   Objective:
   $$\mathbb{E}[r] = 3.769239, \quad D_{\text{KL}} = 1.689001 \implies \mathcal{J} = 3.769239 - 0.5(1.689001) = \mathbf{2.924739}$$
   Formula check: $0.5 \times \ln(347.053181) = 0.5 \times 5.849478 = \mathbf{2.924739} \quad \checkmark$

4. **Comparison with Unconstrained Greedy Policy:**
   Consider the greedy policy that always picks the maximum reward response: $\pi_{\text{greedy}} = [0, 0, 1]^T$ ($y_3$ with probability $1.0$).
   - Expected reward: $\mathbb{E}[r] = 4.000000$.
   - KL divergence:
     $$D_{\text{KL}}(\pi_{\text{greedy}} \parallel \pi_{\text{ref}}) = 1.0 \times \ln\left(\frac{1.0000}{0.1000}\right) = \ln(10) \approx \mathbf{2.302585}$$
   - Net objective under $\beta = 1.0000$:
     $$\mathcal{J}(\pi_{\text{greedy}}) = 4.000000 - 1.0000 \times 2.302585 = \mathbf{1.697415} < \mathbf{2.374490}$$
   - Net objective under $\beta = 0.5000$:
     $$\mathcal{J}(\pi_{\text{greedy}}) = 4.000000 - 0.5000 \times 2.302585 = \mathbf{2.848708} < \mathbf{2.924739}$$
   **Key Insight:** While the greedy policy achieves the maximal nominal reward ($4.0000 > 3.0345$), its severe deviation from the reference model incurs an overwhelming KL penalty of $2.3026$, dragging its net objective down to $1.6974$. The mathematically optimal policy balances exploitation and linguistic fidelity, achieving a $40\%$ higher net objective ($2.3745$). $\blacksquare$

---

### Illustration 5: Full 4-Model RLHF Architecture Forward Pass and Multi-Task PPO Loss Computation
**Problem:**
Coordinate the complete 4-model RLHF training engine over a generated 2-token completion $y = (y_1, y_2)$ given prompt $x$:
- **Actor Model (Rollout Policy $\pi_{\text{old}}$ vs. Updated Policy $\pi_\theta$):**
  - Rollout token probabilities: $\pi_{\text{old}}(y_1 \mid x) = 0.5000, \quad \pi_{\text{old}}(y_2 \mid x, y_1) = 0.4000$
  - Updated actor probabilities: $\pi_\theta(y_1 \mid x) = 0.6000, \quad \pi_\theta(y_2 \mid x, y_1) = 0.5200$
- **Reference Model (Frozen Base $\pi_{\text{ref}}$):**
  - Reference token probabilities: $\pi_{\text{ref}}(y_1 \mid x) = 0.4500, \quad \pi_{\text{ref}}(y_2 \mid x, y_1) = 0.3500$
- **Critic Model (Value Network $V_\psi$):**
  - Rollout baseline estimates: $V_{\text{old}}(s_1) = 1.8000, \quad V_{\text{old}}(s_2) = 2.2000, \quad V_{\text{old}}(s_3) \equiv 0.0000$
  - Current critic predictions during optimization: $V_\psi(s_1) = 1.8500, \quad V_\psi(s_2) = 2.1800$
- **Reward Model (Frozen Preference Model $r_\phi$):**
  - Terminal EOS reward: $r_\phi(x, y) = 2.8000$
- **Hyperparameters:**
  - KL penalty coefficient: $\beta = 0.1000$
  - PPO clipping threshold: $\epsilon = 0.2000 \implies [1 - \epsilon, 1 + \epsilon] = [0.8000, 1.2000]$
  - Discount and GAE decay: $\gamma = 1.0000, \quad \lambda = 0.9500$
  - Critic loss weight: $c_1 = 0.5000$

1. Compute the token-level KL penalties and rewards $R_1, R_2$.
2. Compute the Critic TD residuals $\delta_1, \delta_2$ and GAE advantages $\hat{A}_1, \hat{A}_2$.
3. Compute the importance sampling ratios $\rho_1, \rho_2$ and evaluate the clipped PPO actor loss $\mathcal{L}_{\text{actor}}(\theta)$.
4. Compute the value regression targets $V_t^{\text{targ}}$ and the Critic loss $\mathcal{L}_{\text{critic}}(\psi)$.
5. Compute the total unified loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} + c_1 \mathcal{L}_{\text{critic}}$.

**Solution:**
1. **Rollout Token KL Penalties & Rewards:**
   - **Token 1 ($t = 1$):**
     $$\Delta \ln \pi_{\text{old}, 1} = \ln\left(\frac{0.5000}{0.4500}\right) = \ln(1.111111) \approx \mathbf{+0.105361}$$
     $$R_1 = -\beta \Delta \ln \pi_{\text{old}, 1} = -0.1000 \times 0.105361 = \mathbf{-0.010536}$$
   - **Token 2 ($t = 2$, Terminal):**
     $$\Delta \ln \pi_{\text{old}, 2} = \ln\left(\frac{0.4000}{0.3500}\right) = \ln(1.142857) \approx \mathbf{+0.133531}$$
     $$R_2 = r_\phi(x, y) - \beta \Delta \ln \pi_{\text{old}, 2} = 2.8000 - 0.1000 \times 0.133531 = 2.8000 - 0.013353 = \mathbf{+2.786647}$$

2. **Critic TD Errors & GAE Advantages:**
   - **TD Residuals:**
     $$\delta_1 = R_1 + \gamma V_{\text{old}}(s_2) - V_{\text{old}}(s_1) = -0.010536 + 2.200000 - 1.800000 = \mathbf{+0.389464}$$
     $$\delta_2 = R_2 + \gamma V_{\text{old}}(s_3) - V_{\text{old}}(s_2) = +2.786647 + 0.000000 - 2.200000 = \mathbf{+0.586647}$$
   - **GAE Advantages:**
     $$\hat{A}_2 = \delta_2 = \mathbf{+0.586647}$$
     $$\hat{A}_1 = \delta_1 + (\gamma \lambda) \hat{A}_2 = 0.389464 + (1.0 \times 0.95) \times 0.586647 = 0.389464 + 0.557315 = \mathbf{+0.946778}$$

3. **PPO Actor Ratios & Clipped Objective:**
   - **Token 1:**
     $$\rho_1 = \frac{\pi_\theta(y_1)}{\pi_{\text{old}}(y_1)} = \frac{0.6000}{0.5000} = \mathbf{1.200000}$$
     Clipping interval is $[0.8000, 1.2000]$. Since $\rho_1 = 1.200000 \le 1.200000$:
     $$\operatorname{clip}(\rho_1, 0.8, 1.2) = 1.200000$$
     $$\text{surr1}_1 = \rho_1 \hat{A}_1 = 1.200000 \times 0.946778 = \mathbf{1.136134}$$
     $$\text{surr2}_1 = \operatorname{clip}(\rho_1, 0.8, 1.2) \hat{A}_1 = 1.200000 \times 0.946778 = \mathbf{1.136134}$$
     $$\text{obj}_1 = \min(\text{surr1}_1, \text{surr2}_1) = \mathbf{1.136134}$$
   - **Token 2:**
     $$\rho_2 = \frac{\pi_\theta(y_2)}{\pi_{\text{old}}(y_2)} = \frac{0.5200}{0.4000} = \mathbf{1.300000}$$
     Since $\rho_2 = 1.300000 > 1.200000$, clipping activates:
     $$\operatorname{clip}(\rho_2, 0.8, 1.2) = 1.200000$$
     $$\text{surr1}_2 = \rho_2 \hat{A}_2 = 1.300000 \times 0.586647 = \mathbf{0.762641}$$
     $$\text{surr2}_2 = \operatorname{clip}(\rho_2, 0.8, 1.2) \hat{A}_2 = 1.200000 \times 0.586647 = \mathbf{0.703976}$$
     $$\text{obj}_2 = \min(\text{surr1}_2, \text{surr2}_2) = \mathbf{0.703976}$$
     *(Clipping actively truncates the surrogate from $0.762641$ down to $0.703976$, setting the policy gradient $\frac{\partial}{\partial \rho_2} = 0$ to prevent overconfidence!)*
   - **Actor Loss:**
     $$\mathcal{L}_{\text{actor}}(\theta) = -\frac{1}{T} \sum_{t=1}^2 \text{obj}_t = -\frac{1.136134 + 0.703976}{2} = -\frac{1.840110}{2} = \mathbf{-0.920055}$$

4. **Value Targets & Critic Loss:**
   Target values for the Critic:
   $$V_1^{\text{targ}} = V_{\text{old}}(s_1) + \hat{A}_1 = 1.800000 + 0.946778 = \mathbf{2.746778}$$
   $$V_2^{\text{targ}} = V_{\text{old}}(s_2) + \hat{A}_2 = 2.200000 + 0.586647 = \mathbf{2.786647}$$
   Critic prediction squared errors under $V_\psi$:
   $$(V_\psi(s_1) - V_1^{\text{targ}})^2 = (1.850000 - 2.746778)^2 = (-0.896778)^2 \approx \mathbf{0.804212}$$
   $$(V_\psi(s_2) - V_2^{\text{targ}})^2 = (2.180000 - 2.786647)^2 = (-0.606647)^2 \approx \mathbf{0.368020}$$
   Mean Squared Value Loss:
   $$\mathcal{L}_{\text{critic}}(\psi) = \frac{1}{2 T} \sum_{t=1}^2 (V_\psi(s_t) - V_t^{\text{targ}})^2 = \frac{0.804212 + 0.368020}{2 \times 2} = \frac{1.172232}{4} = \mathbf{0.293058}$$

5. **Unified Multi-Task PPO Loss:**
   $$\begin{aligned}
   \mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{actor}}(\theta) + c_1 \mathcal{L}_{\text{critic}}(\psi) \\
   &= -0.920055 + 0.5000 \times 0.293058 \\
   &= -0.920055 + 0.146529 = \mathbf{-0.773526} \quad \blacksquare
   \end{aligned}$$

**Full 4-Model System Ledger:**
| Component | Token $t=1$ | Token $t=2$ (Terminal) | Function & Mathematical Role |
| :--- | :---: | :---: | :--- |
| **Actor $\pi_{\text{old}}$ (Rollout)** | $0.5000$ | $0.4000$ | Generates tokens during environment interaction |
| **Actor $\pi_\theta$ (Optimized)** | $0.6000$ | $0.5200$ | Evaluated policy undergoing gradient ascent |
| **Reference $\pi_{\text{ref}}$ (Frozen)** | $0.4500$ | $0.3500$ | Prior anchor preventing linguistic collapse |
| **Reward Model $r_\phi$ (Frozen)** | — | $2.8000$ | Scalar evaluator of complete sequence |
| **Critic $V_{\text{old}}$ (Rollout Baseline)** | $1.8000$ | $2.2000$ | Estimates baseline value for variance reduction |
| **Token Reward $R_t$** | $-0.0105$ | $+2.7866$ | Decomposed step reward: $-\beta \Delta \ln \pi_t + \delta_{t, T} r_\phi$ |
| **TD Error $\delta_t$** | $+0.3895$ | $+0.5866$ | One-step Bellman prediction residual |
| **GAE Advantage $\hat{A}_t$** | $+0.9468$ | $+0.5866$ | Multi-step credit assignment |
| **Ratio $\rho_t$** | $1.2000$ | $1.3000$ | Importance sampling ratio $\pi_\theta / \pi_{\text{old}}$ |
| **PPO Clipped Surrogate** | $1.1361$ | $0.7040$ | Pessimistic clipped objective bound |
| **Critic Target $V_t^{\text{targ}}$** | $2.7468$ | $2.7866$ | Regression target $V_{\text{old}} + \hat{A}_t$ |
| **Critic $V_\psi$ (Current)** | $1.8500$ | $2.1800$ | Current value prediction |
| **Total Loss $\mathcal{L}_{\text{total}}$** | \multicolumn{2}{c|}{$\mathbf{-0.773526}$} | Joint Actor-Critic objective |

---

## 7. Deep Learning Connection & Modern Applications

### 1. InstructGPT & ChatGPT: RLHF Transforms the AI Industry (OpenAI, 2022)
RLHF-PPO was the technique that made LLMs practically useful:
- **InstructGPT (Ouyang et al., NeurIPS 2022):** A 1.3B RLHF-aligned model was preferred over the unaligned 175B GPT-3 in 71% of human evaluations — a 100× parameter advantage erased by alignment. The critical finding: raw next-token prediction does not optimize for helpfulness.
- **ChatGPT (2022):** Applied RLHF-PPO at scale to GPT-3.5, creating the fastest-growing consumer product in history (100M users in 2 months). The reward model was trained on human comparisons of response quality on 20,000+ prompts.
- **GPT-4 Technical Report (2023):** Uses RLHF-PPO combined with "Superalignment" techniques and process-based reward models, achieving unprecedented alignment scores on TruthfulQA, MMLU, and human preference benchmarks.

### 2. Direct Preference Optimization: Bypassing the PPO Overhead (Rafailov et al., NeurIPS 2023)
DPO derives a closed-form alignment loss by substituting the optimal RLHF policy:
$$\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\!\left(\frac{r(x,y)}{\beta}\right) \implies r(x,y) = \beta \ln \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \ln Z(x)$$
Substituting into the Bradley-Terry preference model and canceling $Z(x)$ yields:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \ln\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \ln\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
This trains the policy directly on preference pairs without a separate reward model or PPO actor-critic loop, reducing alignment compute by 40–60% while matching PPO performance on Alpaca Eval.

### 3. Constitutional AI & RLAIF: Scalable Alignment Beyond Human Labels (Anthropic, 2022)
Anthropic's Constitutional AI extends RLHF to reduce dependence on expensive human preference labels:
- **AI Feedback (RLAIF):** Instead of human raters comparing responses, Claude is prompted to evaluate its own outputs against a written "constitution" of principles (harmlessness, honesty, helpfulness). The AI's preference judgments replace human labels for reward model training.
- **Self-Critique & Revision Loop:** A critique model identifies problematic aspects of an initial response; a revision model rewrites it to satisfy constitutional principles. This loop produces preference pairs automatically at zero marginal cost per sample.
- **Scale Advantage:** RLAIF generates 1,000× more preference labels than human RLHF budgets allow, enabling Claude 2/3 to maintain alignment across a much wider distribution of adversarial prompts than PPO-only baselines.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Bradley-Terry loss & gradients matching $-0.268941, +0.268941$.
   - Token-level KL rewards: $R_1 = 0.022314, R_2 = 2.930685$.
   - TD residuals $\delta_1 = 0.322314, \delta_2 = 0.130685$ and GAE advantages $\hat{A}_1 = 0.446465, \hat{A}_2 = 0.130685$ matching to $< 10^{-6}$.
2. **Section 6 Illustrations 1–5 Analytical Verification Suite:**
   - Illustration 1: Bradley-Terry loss gradient ($-0.268941, +0.268941$).
   - Illustration 2: Reward model parameter gradient update ($[-0.158253, +0.079126]^T$, monotonic loss reduction $0.220417 \to 0.217307$).
   - Illustration 3: 4-token sequence KL reward decomposition and GAE advantages ($\hat{A}_1 = 0.234652, \hat{A}_2 = 0.163080, \hat{A}_3 = 0.029918, \hat{A}_4 = 0.015343$).
   - Illustration 4: Non-parametric optimal policy distribution ($\pi^* = [0.151781, 0.340118, 0.508101]^T$, matching $\beta \ln Z(x) = 2.374490$).
   - Illustration 5: Complete 4-model forward pass and PPO loss ($\mathcal{L}_{\text{actor}} = -0.920055, \mathcal{L}_{\text{critic}} = 0.293058, \mathcal{L}_{\text{total}} = -0.773526$).
3. **Complete PyTorch Bradley-Terry Preference Reward Model Trainer.**
4. **Token-Level PPO RLHF Engine:**
   - Vectorized computation of token KL penalties, per-token GAE, and clipped surrogate loss.

See implementation in:
[`11_reinforcement_learning/code/27_rlhf_ppo.py`](./code/27_rlhf_ppo.py)

