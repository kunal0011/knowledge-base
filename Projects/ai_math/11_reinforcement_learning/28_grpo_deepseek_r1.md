# Chapter 28: Group Relative Policy Optimization (GRPO) & Frontier Reasoning (DeepSeek-R1)

---

## 1. Intuition & 101 Motivation

In the previous chapter on **RLHF with PPO**, we saw how large language models can be aligned with human intent. However, applying classical PPO to complex mathematical reasoning, competitive programming, and multi-step scientific problem-solving reveals severe bottlenecks:

1. **The Four-Model Memory Wall:** PPO requires keeping four massive copies of the language model in GPU memory simultaneously:
   - The **Actor** $\pi_\theta$ (trainable)
   - The **Critic** $V_\phi$ (trainable)
   - The **Reference Model** $\pi_{\text{ref}}$ (frozen)
   - The **Reward Model** $r_\psi$ (frozen)
   For a 70B or 671B MoE model, the critic network alone consumes hundreds of gigabytes of VRAM for parameters, Adam optimizer states, and forward activations.
2. **The Brittleness of Learned Reward Models:** Neural reward models are vulnerable to reward hacking, sycophancy, and length bias. In mathematics and coding, we do not need a subjective neural reward model—we have **deterministic truth verifiers** (a Python compiler or symbolic math engine)!
3. **The Value Function Bottleneck:** Training a token-level value critic $V_\phi$ to accurately predict whether a 10,000-token mathematical proof will succeed is notoriously noisy and unstable.

Enter **Group Relative Policy Optimization (GRPO)** (Shao et al., DeepSeekMath 2024; DeepSeek-AI, DeepSeek-R1 2025):
- **Eliminate the Critic Network entirely!** GRPO requires no value network $V_\phi$, cutting GPU memory requirements by more than **$50\%$**.
- For each question/prompt $q$, the model samples a **group of $G$ candidate reasoning paths**:
  $$\mathcal{G} = \{o_1, o_2, \dots, o_G\}$$
- Each completion is evaluated by **rule-based deterministic verifiers** (mathematical correctness + XML tag structure).
- The baseline is computed dynamically **from the group itself** by normalizing rewards:
  $$\hat{A}_i = \frac{r_i - \operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})}$$
- Completions that outperform their peers receive positive advantages; completions that fail receive negative advantages.

In January 2025, DeepSeek released **DeepSeek-R1-Zero** and **DeepSeek-R1**: trained entirely via GRPO without any human supervised warm-start data, the model autonomously developed **long Chain-of-Thought (CoT)**, self-reflection, and backtracking, rivaling OpenAI's proprietary o1 on competitive benchmarks (AIME, MATH-500, Codeforces)!

```
Prompt q: "Solve 2 + 3 * 4"
       |
       +---> Output 1: "... = 14" (Correct) ------> Reward r_1 = 1.0  \
       +---> Output 2: "... = 14" (Correct) ------> Reward r_2 = 1.0   |--> Group Mean mu = 0.525
       +---> Output 3: "... = 20" (Arithmetic Err) > Reward r_3 = 0.1   |--> Group Std  sigma = 0.476
       +---> Output 4: "broken xml..." -----------> Reward r_4 = 0.0  /
                                                         |
                                                         v
                                  Advantage A_1 = +1.00  (Reinforce!)
                                  Advantage A_2 = +1.00  (Reinforce!)
                                  Advantage A_3 = -0.89  (Suppress!)
                                  Advantage A_4 = -1.10  (Suppress!)
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 Group Sampling & Verifier Reward Functions

For a given training prompt $q \sim P(Q)$, the old policy $\pi_{\theta_{\text{old}}}$ generates a group of $G$ candidate completions:
$$\mathcal{G} = \{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$$

Each candidate completion $o_i$ is evaluated by two deterministic, rule-based reward functions:

$$r_i = r_{\text{acc}}(q, o_i) + r_{\text{format}}(o_i)$$

1. **Accuracy Verifier ($r_{\text{acc}}$):**
   - **Math:** Extracts the final answer inside `\boxed{...}` and compares it symbolically to ground truth:
     $$r_{\text{acc}}(q, o_i) = \begin{cases} 1.0, & \text{if answer is correct} \\ 0.0, & \text{if answer is incorrect} \end{cases}$$
   - **Code:** Executes generated code against unit test testbeds (compiler execution).
2. **Format Verifier ($r_{\text{format}}$):**
   - Enforces that the model places its reasoning process inside `<think> ... </think>` tags and its final conclusion inside `<answer> ... </answer>` tags:
     $$r_{\text{format}}(o_i) = \begin{cases} 0.1, & \text{if XML tags are properly formatted} \\ 0.0, & \text{otherwise} \end{cases}$$

---

### 2.2 Group Relative Advantage Estimation

Unlike PPO which requires a parametric value network $V_\phi(s)$, GRPO normalizes rewards across the group $\mathcal{G}$:

$$\mu_{\mathcal{G}} = \frac{1}{G} \sum_{j=1}^G r_j$$

$$\sigma_{\mathcal{G}} = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_j - \mu_{\mathcal{G}})^2 + \epsilon}$$

The advantage of completion $o_i$ is defined as the standardized score:

$$\hat{A}_i = \frac{r_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}}$$

#### Mathematical Properties of GRPO Advantages:
1. **Zero-Sum Group Property:**
   $$\sum_{i=1}^G \hat{A}_i = \frac{\sum_{i=1}^G (r_i - \mu_{\mathcal{G}})}{\sigma_{\mathcal{G}}} = \frac{G \mu_{\mathcal{G}} - G \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}} \equiv 0$$
   The group provides its own self-centering baseline!
2. **Dynamic Task Difficulty Adaptation:**
   - On an **easy question** where all $G$ outputs succeed ($r_i = 1.0$), $\sigma_{\mathcal{G}} \to 0$, and no spurious gradient updates are taken.
   - On an **impossible question** where all outputs fail ($r_i = 0.0$), the policy is not destabilized.
   - Updates occur primarily on **frontier questions** with high variance (some successes, some failures), focusing learning where the policy has the highest information gain!

---

### 2.3 The GRPO Objective Function

The policy parameters $\theta$ are trained to maximize the clipped surrogate objective with an analytical token-level KL divergence penalty:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( \rho_{i, t}(\theta) \hat{A}_i, \; \operatorname{clip}(\rho_{i, t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) \right) \right]$$

where the importance sampling probability ratio is:
$$\rho_{i, t}(\theta) = \frac{\pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})}$$

---

### 2.4 Analytical Unbiased KL Divergence Estimator

Instead of computing a sample-based log-ratio approximation $(\ln \pi_\theta - \ln \pi_{\text{ref}})$, DeepSeek-R1 employs the **Schulman unbiased non-negative KL estimator**:

$$D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} - \ln \left( \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} \right) - 1$$

Letting $u = \frac{\pi_{\text{ref}}}{\pi_\theta}$:
$$k(u) = u - \ln u - 1$$

#### Properties:
1. Since $u - 1 \ge \ln u$ for all $u > 0$, $k(u) \ge 0$ strictly.
2. $k(u) = 0$ if and only if $u = 1$ (i.e. $\pi_\theta \equiv \pi_{\text{ref}}$).
3. The estimator has **zero variance** when the policy matches reference, preventing gradient noise from destabilizing training.

---

### 2.5 First-Principles Mathematical Derivations

#### Derivation 11.28.1: Group Relative Advantage Formulation and Unbiased Policy Gradient Estimation

```
====================================================================================================
DERIVATION 11.28.1: Group Relative Advantage Formulation and Unbiased Policy Gradient Estimation
====================================================================================================
Problem Statement:
Given a prompt q ~ P(Q) and an old policy π_{θ_old}, sample a group of G completions 
𝒢 = {o_1, o_2, ..., o_G} ~ π_{θ_old}(· | q) evaluated with scalar verifier rewards {r_1, ..., r_G}.
Prove that:
  1. The group relative advantage Â_i = (r_i - μ_𝒢) / σ_𝒢 acts as an asymptotically unbiased 
     estimator of the relative advantage with uniform contraction factor (G - 1) / G.
  2. The group advantage satisfies the Zero-Sum Group Identity: ∑_{i=1}^G Â_i ≡ 0.
  3. Â_i is identically equal to the normalized all-to-all tournament win margin: 
     Â_i = (1 / (G σ_𝒢)) ∑_{j=1}^G (r_i - r_j).
  4. Â_i is strictly invariant under arbitrary positive affine reward transformations r̃ = a · r + c.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In policy gradient reinforcement learning, the expected gradient of the expected trajectory return $J(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_\theta}[R(q, o)]$ is given by the Policy Gradient Theorem:
$$\nabla_\theta J(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_\theta} \left[ \nabla_\theta \ln \pi_\theta(o \mid q) (R(q, o) - b(q)) \right]$$
where $b(q)$ is any baseline function independent of the sampled completion $o$.

In Group Relative Policy Optimization (GRPO; Shao et al., DeepSeekMath 2024; DeepSeek-R1 2025), no parametric value network critic $V_\phi(q)$ is trained. Instead, for each query $q$, a group of $G$ candidate completions is drawn i.i.d. from the proposal policy $\pi_{\theta_{\text{old}}}(\cdot \mid q)$:
$$\mathcal{G} = \{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$$
Each completion $o_i$ receives a deterministic rule-based scalar reward $r_i = R(q, o_i)$. The empirical group mean and empirical group standard deviation are computed as:
$$\mu_{\mathcal{G}} = \frac{1}{G} \sum_{j=1}^G r_j, \quad \sigma_{\mathcal{G}} = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_j - \mu_{\mathcal{G}})^2 + \epsilon_{\text{std}}}$$
where $\epsilon_{\text{std}} > 0$ is a small numerical regularizer. The group relative advantage for completion $o_i$ is defined as:
$$\hat{A}_i = \frac{r_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}}$$

Our mathematical goals are:
1. Prove that the leave-one-out baseline $b_{-i}(q) \triangleq \frac{1}{G-1} \sum_{j \ne i} r_j$ yields a strictly unbiased estimator of the policy gradient $\nabla_\theta J(\theta)$, and that the group-mean baseline $r_i - \mu_{\mathcal{G}} = \frac{G-1}{G} (r_i - b_{-i}(q))$ contracts the policy gradient estimator by an exact deterministic factor of $1 - \frac{1}{G}$, preserving the exact directional alignment of the true gradient.
2. Prove the Zero-Sum Group Identity: $\sum_{i=1}^G \hat{A}_i = 0$ identically for every sampled group realization.
3. Prove that group standardization is algebraically equivalent to an all-pairs round-robin tournament comparison: $\hat{A}_i = \frac{1}{G \sigma_{\mathcal{G}}} \sum_{j=1}^G (r_i - r_j)$, computing all pairwise margins in $O(G)$ time instead of $O(G^2)$.
4. Prove that $\hat{A}_i$ is invariant to arbitrary positive affine transformations of the reward signal $R \mapsto a R + c$ for $a > 0$ and $c \in \mathbb{R}$.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Conditional I.I.D. Generation:** Given query $q \sim P(Q)$, candidate completions $o_1, \dots, o_G$ are conditionally independent and identically distributed draws from $\pi_{\theta_{\text{old}}}(\cdot \mid q)$:
   $$P(o_1, o_2, \dots, o_G \mid q) = \prod_{i=1}^G \pi_{\theta_{\text{old}}}(o_i \mid q)$$
2. **Bounded Reward Support:** The scalar reward function $R: \mathcal{Q} \times \mathcal{Y}^* \to [r_{\min}, r_{\max}]$ is bounded with $-\infty < r_{\min} \le r_{\max} < \infty$, ensuring the existence of finite conditional expectation $\mu(q) \triangleq \mathbb{E}_{o \sim \pi}[R(q, o) \mid q]$ and finite variance $\sigma^2(q) \triangleq \operatorname{Var}_{o \sim \pi}(R(q, o) \mid q) < \infty$.
3. **Leibniz Regularity:** The parameterized policy $\pi_\theta(o \mid q)$ is strictly positive and continuously differentiable in parameter vector $\theta \in \mathbb{R}^d$ across the output space $\mathcal{Y}^*$, and satisfies dominated convergence conditions permitting the interchange of differentiation and integration:
   $$\nabla_\theta \sum_{o \in \mathcal{Y}^*} \pi_\theta(o \mid q) = \sum_{o \in \mathcal{Y}^*} \nabla_\theta \pi_\theta(o \mid q) = \nabla_\theta (1) = \mathbf{0}$$
4. **Non-Degenerate Regularization:** The variance regularizer satisfies $\epsilon_{\text{std}} > 0$, ensuring $\sigma_{\mathcal{G}} \ge \sqrt{\epsilon_{\text{std}}} > 0$ strictly, preventing numerical division-by-zero singularities when all candidate completions receive identical rewards.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
- **Orthogonal Projection onto the Zero-Mean Hyperplane:** In the $G$-dimensional reward space $\mathbb{R}^G$, let $\mathbf{r} = [r_1, r_2, \dots, r_G]^T$. The empirical group mean defines a projection onto the all-ones vector $\mathbf{1} = [1, 1, \dots, 1]^T$:
  $$\operatorname{proj}_{\mathbf{1}}(\mathbf{r}) = \frac{\mathbf{r}^T \mathbf{1}}{\mathbf{1}^T \mathbf{1}} \mathbf{1} = \left( \frac{1}{G} \sum_{j=1}^G r_j \right) \mathbf{1} = \mu_{\mathcal{G}} \mathbf{1}$$
  The deviation vector $\mathbf{r} - \mu_{\mathcal{G}} \mathbf{1}$ is the exact orthogonal projection of $\mathbf{r}$ onto the hyperplane $\mathbf{1}^\perp = \{\mathbf{v} \in \mathbb{R}^G : \sum_{i=1}^G v_i = 0\}$. By removing the common mode $\mu_{\mathcal{G}} \mathbf{1}$, GRPO eliminates the ambient question difficulty and isolates pure relative performance differences.
- **Center-of-Mass Reference Frame:** In Newtonian mechanics, transforming equations of motion into the center-of-mass reference frame eliminates external net motion and isolates internal relative interactions. GRPO places the origin at the center of mass of the candidate outputs, ensuring that the model is only reinforced or penalized relative to its own current competence frontier.
- **Automatic Dynamic Curricula:** On trivial prompts (where all $G$ completions succeed, $r_i = 1$) or impossible prompts (where all $G$ completions fail, $r_i = 0$), $r_i - \mu_{\mathcal{G}} = 0$ for all $i$. Hence $\hat{A}_i = 0$ and the policy gradient update is identically zero! Updates are automatically focused exclusively on "frontier questions" where the policy displays high epistemic variance (some completions succeed while others fail).

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: First-principles proof of baseline invariance in policy gradients.*
Let $b(q)$ be any arbitrary scalar baseline independent of candidate completion $o$. Consider the expected baseline gradient term:
$$\mathbb{E}_{o \sim \pi_\theta} \left[ \nabla_\theta \ln \pi_\theta(o \mid q) \cdot b(q) \right] = \sum_{o \in \mathcal{Y}^*} \pi_\theta(o \mid q) \left( \frac{\nabla_\theta \pi_\theta(o \mid q)}{\pi_\theta(o \mid q)} \right) b(q)$$
Canceling $\pi_\theta(o \mid q)$ in the numerator and denominator:
$$= b(q) \sum_{o \in \mathcal{Y}^*} \nabla_\theta \pi_\theta(o \mid q)$$
By the Leibniz rule (Assumption 3):
$$= b(q) \nabla_\theta \left( \sum_{o \in \mathcal{Y}^*} \pi_\theta(o \mid q) \right) = b(q) \nabla_\theta (1) = b(q) \cdot \mathbf{0} = \mathbf{0}$$
Therefore:
$$\mathbb{E}_{q, o \sim \pi_\theta} \left[ \nabla_\theta \ln \pi_\theta(o \mid q) (R(q, o) - b(q)) \right] = \mathbb{E}_{q, o \sim \pi_\theta} \left[ \nabla_\theta \ln \pi_\theta(o \mid q) R(q, o) \right] = \nabla_\theta J(\theta)$$

*Step 2: Unbiasedness of the leave-one-out baseline estimator.*
Define the leave-one-out baseline for candidate $i$:
$$b_{-i}(q) \triangleq \frac{1}{G - 1} \sum_{j \ne i} r_j = \frac{1}{G - 1} \sum_{j \ne i} R(q, o_j)$$
Under conditional independence (Assumption 1), $o_j$ for all $j \ne i$ are statistically independent of $o_i$ given $q$. Taking the conditional expectation of $b_{-i}(q)$ given $q$:
$$\mathbb{E}_{\mathcal{G} \setminus \{o_i\} \mid q} [b_{-i}(q)] = \frac{1}{G - 1} \sum_{j \ne i} \mathbb{E}_{o_j \sim \pi_{\theta}}[R(q, o_j) \mid q] = \frac{1}{G - 1} (G - 1) \mu(q) = \mu(q)$$
Because $b_{-i}(q)$ is conditionally independent of $o_i$, we evaluate the inner expectation over $o_i$:
$$\mathbb{E}_{o_i \sim \pi_\theta} \left[ \nabla_\theta \ln \pi_\theta(o_i \mid q) \cdot b_{-i}(q) \;\Big|\; q, \{o_j\}_{j \ne i} \right] = b_{-i}(q) \cdot \mathbb{E}_{o_i \sim \pi_\theta} \left[ \nabla_\theta \ln \pi_\theta(o_i \mid q) \;\Big|\; q \right] = b_{-i}(q) \cdot \mathbf{0} = \mathbf{0}$$
Applying the tower property of conditional expectation:
$$\mathbb{E}_{\mathcal{G} \mid q} \left[ \nabla_\theta \ln \pi_\theta(o_i \mid q) (r_i - b_{-i}(q)) \right] = \mathbb{E}_{o_i \mid q} \left[ \nabla_\theta \ln \pi_\theta(o_i \mid q) r_i \right]$$
Taking the empirical average over all $G$ completions in the group:
$$\hat{g}_{\text{LOO}} \triangleq \frac{1}{G} \sum_{i=1}^G \nabla_\theta \ln \pi_\theta(o_i \mid q) (r_i - b_{-i}(q))$$
$$\mathbb{E}_{q, \mathcal{G}} [\hat{g}_{\text{LOO}}] = \frac{1}{G} \sum_{i=1}^G \mathbb{E}_{q, o_i} \left[ \nabla_\theta \ln \pi_\theta(o_i \mid q) r_i \right] = \nabla_\theta J(\theta)$$
Hence, $\hat{g}_{\text{LOO}}$ is a strictly unbiased estimator of the true policy gradient!

*Step 3: Relationship between group-mean baseline and leave-one-out baseline.*
Now consider the empirical group mean $\mu_{\mathcal{G}} = \frac{1}{G} \sum_{j=1}^G r_j$. We decompose $\mu_{\mathcal{G}}$ by isolating the $i$-th term:
$$\mu_{\mathcal{G}} = \frac{1}{G} r_i + \frac{1}{G} \sum_{j \ne i} r_j = \frac{1}{G} r_i + \frac{G - 1}{G} \left( \frac{1}{G - 1} \sum_{j \ne i} r_j \right) = \frac{1}{G} r_i + \frac{G - 1}{G} b_{-i}(q)$$
Subtracting $\mu_{\mathcal{G}}$ from $r_i$:
$$r_i - \mu_{\mathcal{G}} = r_i - \left( \frac{1}{G} r_i + \frac{G - 1}{G} b_{-i}(q) \right) = \left( 1 - \frac{1}{G} \right) r_i - \frac{G - 1}{G} b_{-i}(q) = \frac{G - 1}{G} (r_i - b_{-i}(q))$$
Taking the expectation of the group-centered policy gradient estimator:
$$\hat{g}_{\text{GRPO}} \triangleq \frac{1}{G} \sum_{i=1}^G \nabla_\theta \ln \pi_\theta(o_i \mid q) (r_i - \mu_{\mathcal{G}})$$
$$= \frac{1}{G} \sum_{i=1}^G \nabla_\theta \ln \pi_\theta(o_i \mid q) \left[ \frac{G - 1}{G} (r_i - b_{-i}(q)) \right] = \frac{G - 1}{G} \hat{g}_{\text{LOO}}$$
Taking expectations over the sampling distribution:
$$\mathbb{E}_{q, \mathcal{G}} [\hat{g}_{\text{GRPO}}] = \left( 1 - \frac{1}{G} \right) \mathbb{E}_{q, \mathcal{G}} [\hat{g}_{\text{LOO}}] = \left( 1 - \frac{1}{G} \right) \nabla_\theta J(\theta)$$
This proves that the group-mean baseline estimator points in the exact direction of the true policy gradient, with an exact scalar shrinkage factor of $\frac{G - 1}{G} = 1 - O(1/G)$. For typical group sizes ($G = 6 \implies 0.833$; $G = 16 \implies 0.938$), this scale factor is absorbed into the learning rate without distorting the gradient manifold.

*Step 4: Proof of the Zero-Sum Group Identity.*
Summing the standardized advantages $\hat{A}_i$ over the entire group $\mathcal{G}$:
$$\sum_{i=1}^G \hat{A}_i = \sum_{i=1}^G \frac{r_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}} = \frac{1}{\sigma_{\mathcal{G}}} \sum_{i=1}^G (r_i - \mu_{\mathcal{G}}) = \frac{1}{\sigma_{\mathcal{G}}} \left( \sum_{i=1}^G r_i - \sum_{i=1}^G \mu_{\mathcal{G}} \right)$$
Substitute $\sum_{i=1}^G r_i = G \mu_{\mathcal{G}}$ and $\sum_{i=1}^G \mu_{\mathcal{G}} = G \mu_{\mathcal{G}}$:
$$= \frac{1}{\sigma_{\mathcal{G}}} \left( G \mu_{\mathcal{G}} - G \mu_{\mathcal{G}} \right) = \frac{0}{\sigma_{\mathcal{G}}} \equiv 0 \quad \blacksquare$$

*Step 5: Equivalence to all-pairs tournament advantage.*
Expand $r_i - \mu_{\mathcal{G}}$:
$$r_i - \mu_{\mathcal{G}} = r_i - \frac{1}{G} \sum_{j=1}^G r_j = \frac{1}{G} \sum_{j=1}^G r_i - \frac{1}{G} \sum_{j=1}^G r_j = \frac{1}{G} \sum_{j=1}^G (r_i - r_j)$$
Dividing by $\sigma_{\mathcal{G}}$:
$$\hat{A}_i = \frac{1}{G \sigma_{\mathcal{G}}} \sum_{j=1}^G (r_i - r_j) \quad \blacksquare$$
Each completion $o_i$'s advantage is strictly equal to its average win-loss margin against all other completions in the group, proving that GRPO performs an all-pairs preference tournament in $O(G)$ complexity instead of $O(G^2)$.

*Step 6: Proof of positive affine invariance.*
Let $\tilde{r}_i = a r_i + c$ with $a > 0$ and $c \in \mathbb{R}$.
The transformed group mean is:
$$\tilde{\mu}_{\mathcal{G}} = \frac{1}{G} \sum_{j=1}^G (a r_j + c) = a \left( \frac{1}{G} \sum_{j=1}^G r_j \right) + c = a \mu_{\mathcal{G}} + c$$
The transformed deviation is:
$$\tilde{r}_i - \tilde{\mu}_{\mathcal{G}} = (a r_i + c) - (a \mu_{\mathcal{G}} + c) = a (r_i - \mu_{\mathcal{G}})$$
The transformed variance is:
$$\tilde{\sigma}_{\mathcal{G}}^2 = \frac{1}{G} \sum_{j=1}^G (\tilde{r}_j - \tilde{\mu}_{\mathcal{G}})^2 + \epsilon_{\text{std}} = \frac{1}{G} \sum_{j=1}^G a^2 (r_j - \mu_{\mathcal{G}})^2 + \epsilon_{\text{std}} = a^2 \sigma_{\mathcal{G}}^2 + \mathcal{O}(\epsilon_{\text{std}})$$
For $\epsilon_{\text{std}} \to 0$:
$$\tilde{\sigma}_{\mathcal{G}} = a \sigma_{\mathcal{G}}$$
Substituting into the advantage:
$$\hat{\tilde{A}}_i = \frac{\tilde{r}_i - \tilde{\mu}_{\mathcal{G}}}{\tilde{\sigma}_{\mathcal{G}}} = \frac{a (r_i - \mu_{\mathcal{G}})}{a \sigma_{\mathcal{G}}} = \frac{r_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}} = \hat{A}_i \quad \blacksquare$$

---

#### Derivation 11.28.2: Elimination of Critic Network in GRPO and Memory Complexity Reduction

```
====================================================================================================
DERIVATION 11.28.2: Elimination of Critic Network and VRAM Memory Complexity Reduction
====================================================================================================
Problem Statement:
In classical LLM PPO, four distinct models must reside in GPU memory:
  Actor π_θ (trainable), Critic V_ϕ (trainable), Reference π_ref (frozen), Reward Model r_ψ (frozen).
Prove that GRPO eliminates the Critic network V_ϕ and replaces neural reward models with rule-based 
verifiers (r_ψ on CPU), reducing static VRAM memory consumption from:
  M_static^{PPO} = 36 N bytes  to  M_static^{GRPO} = 18 N bytes  (an exact 50.0% reduction)
under 16-bit mixed precision AdamW training with ZeRO-3 / FSDP distributed parallelism, and cuts 
training computational complexity (FLOPs per token) by 42.86%.
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In Reinforcement Learning from Human Feedback (RLHF) via Proximal Policy Optimization (PPO), learning a stable policy over long language sequences requires maintaining four models in GPU memory:
1. **Trainable Actor Policy $\pi_\theta$** ($N_{\text{actor}}$ parameters).
2. **Trainable Value Critic $V_\phi$** ($N_{\text{critic}}$ parameters), predicting scalar expected future return $V_\phi(s_t) \in \mathbb{R}$ for every prefix.
3. **Frozen Reference Policy $\pi_{\text{ref}}$** ($N_{\text{ref}}$ parameters), preventing excessive policy drift.
4. **Frozen Reward Model $r_\psi$** ($N_{\text{rm}}$ parameters), scoring generated completions.

Our mathematical goals are:
1. Formally derive the exact memory footprint in bytes per parameter for trainable models under 16-bit mixed-precision AdamW and frozen models.
2. Prove that under homogeneous model architectures ($N_{\text{actor}} = N_{\text{critic}} = N_{\text{ref}} = N_{\text{rm}} = N$), PPO requires $36 N$ bytes of static model memory across the cluster, whereas GRPO requires exactly $18 N$ bytes, achieving an exact $50.0\%$ reduction in static VRAM footprint.
3. Prove that backward activation memory for the value network drops from $O(B \cdot T \cdot L \cdot d_{\text{model}})$ to $0$.
4. Prove that training compute per token drops from $14 N$ FLOPs to $8 N$ FLOPs (a $42.86\%$ reduction).

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Transformer Parameter Parity:** Actor, Critic, Reference, and Reward models share identical decoder-only Transformer architectures with $L$ layers, hidden dimension $d$, attention heads $H$, and total parameter count $N$:
   $$N_{\text{actor}} = N_{\text{critic}} = N_{\text{ref}} = N_{\text{rm}} = N$$
2. **Mixed-Precision AdamW Optimizer Layout:** Trainable models ($\pi_\theta, V_\phi$) use 16-bit (bf16/fp16, 2 bytes/param) weights and gradients, and 32-bit (fp32, 4 bytes/param) AdamW optimizer states:
   - FP32 master weights: $4$ bytes/param
   - FP32 first moment vector $m_t$: $4$ bytes/param
   - FP32 second moment vector $v_t$: $4$ bytes/param
   Total memory per trainable parameter: $2 + 2 + 4 + 4 + 4 = 16$ bytes/parameter.
3. **Frozen Model Layout:** Frozen models ($\pi_{\text{ref}}, r_\psi$) require only 16-bit parameter weights ($2$ bytes/parameter), with zero gradients and zero optimizer states.
4. **Deterministic Rule-Based Verifiers:** In mathematical reasoning and code generation tasks (e.g. DeepSeek-R1), rewards are evaluated using deterministic compilers and symbolic execution engines running entirely on CPU host sandboxes. Hence, $M_{\text{rm}}^{\text{GPU}} = 0$ bytes in GRPO.
5. **Distributed Sharding (ZeRO Stage 3 / FSDP):** Model parameters, gradients, and optimizer states are uniformly sharded across a cluster of $K$ homogeneous GPUs.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
- **The Value Network Manifold Bottleneck:** In multi-step mathematical reasoning, verifying whether step 47 of a 2,000-token proof is logically sound is computationally as challenging as generating the proof itself. Therefore, a value network $V_\phi$ cannot be downscaled; it must possess the full capacity of the base language model ($N_{\text{critic}} = N_{\text{actor}}$).
- **Surrogate Manifold vs. Local Empirical Bundle:** PPO attempts to learn a global parametric regression surface $V_\phi: \mathcal{V}^{\le T} \to \mathbb{R}$ across the entire exponential state space. GRPO completely discards this parametric regression manifold, replacing it with an empirical Monte Carlo bundle sampled locally around the prompt $q$.
- **Shifting from Memory-Bound to Compute-Bound:** Training 70B+ parameter models in PPO is fundamentally memory-bound (fitting $36 N$ bytes plus backward activations). By eliminating the critic, GRPO frees up tens of gigabytes of VRAM per GPU, enabling massive rollout batches and long reasoning contexts ($> 32\text{k}$ tokens) that fully saturate modern Tensor Cores.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: Byte breakdown per parameter.*
For an arbitrary model with $N$ parameters:
- Model weights in 16-bit precision:
  $$M_{\text{weight}} = 2 N \text{ bytes}$$
- Gradients in 16-bit precision:
  $$M_{\text{grad}} = 2 N \text{ bytes}$$
- AdamW optimizer state in 32-bit precision:
  $$M_{\text{adamw}} = (4_{\text{master}} + 4_{m} + 4_{v}) N = 12 N \text{ bytes}$$
Total memory for a trainable model:
$$M_{\text{trainable}}(N) = M_{\text{weight}} + M_{\text{grad}} + M_{\text{adamw}} = 2 N + 2 N + 12 N = 16 N \text{ bytes}$$
For a frozen model (inference only):
$$M_{\text{frozen}}(N) = M_{\text{weight}} = 2 N \text{ bytes}$$

*Step 2: Total static memory in PPO.*
In classical LLM PPO:
- Actor $\pi_\theta$: Trainable $\implies M_{\text{actor}} = 16 N$ bytes
- Critic $V_\phi$: Trainable $\implies M_{\text{critic}} = 16 N$ bytes
- Reference $\pi_{\text{ref}}$: Frozen $\implies M_{\text{ref}} = 2 N$ bytes
- Reward Model $r_\psi$: Frozen $\implies M_{\text{rm}} = 2 N$ bytes
Summing all static components:
$$M_{\text{static}}^{\text{PPO}} = M_{\text{actor}} + M_{\text{critic}} + M_{\text{ref}} + M_{\text{rm}} = 16 N + 16 N + 2 N + 2 N = 36 N \text{ bytes}$$
Under ZeRO Stage 3 sharding across $K$ GPUs:
$$M_{\text{static, per-GPU}}^{\text{PPO}} = \frac{36 N}{K} \text{ bytes}$$

*Step 3: Total static memory in GRPO.*
In GRPO:
- Actor $\pi_\theta$: Trainable $\implies M_{\text{actor}} = 16 N$ bytes
- Critic $V_\phi$: **Eliminated** $\implies M_{\text{critic}} = 0$ bytes
- Reference $\pi_{\text{ref}}$: Frozen $\implies M_{\text{ref}} = 2 N$ bytes
- Reward Model: **CPU Verifier** $\implies M_{\text{rm}} = 0$ bytes
Summing all static components:
$$M_{\text{static}}^{\text{GRPO}} = M_{\text{actor}} + M_{\text{critic}} + M_{\text{ref}} + M_{\text{rm}} = 16 N + 0 + 2 N + 0 = 18 N \text{ bytes}$$
Under ZeRO Stage 3 sharding across $K$ GPUs:
$$M_{\text{static, per-GPU}}^{\text{GRPO}} = \frac{18 N}{K} \text{ bytes}$$

*Step 4: Derivation of the exact memory reduction ratio.*
Compute the ratio of GRPO to PPO static memory:
$$\frac{M_{\text{static}}^{\text{GRPO}}}{M_{\text{static}}^{\text{PPO}}} = \frac{18 N}{36 N} = \frac{1}{2} = 50.0\%$$
The absolute reduction in static GPU memory across the cluster is:
$$\Delta M_{\text{static}} = M_{\text{static}}^{\text{PPO}} - M_{\text{static}}^{\text{GRPO}} = 36 N - 18 N = 18 N \text{ bytes}$$
This constitutes an exact $50.0\%$ savings in static VRAM consumption.
*(Note: Even if PPO were executed with a CPU reward model, PPO requires $16 N + 16 N + 2 N = 34 N$ bytes; GRPO still saves $16 N$ bytes, yielding $\frac{18 N}{34 N} = 52.94\%$, an absolute reduction of $47.06\%$).*

*Step 5: Activation memory elimination.*
For a forward-backward pass through a Transformer with micro-batch size $B$, sequence length $T$, hidden dimension $d$, and layer count $L$:
In PPO:
- Actor backward pass requires storing activations $\mathcal{A}_{\text{actor}}$ for computing $\nabla_\theta \ln \pi_\theta$.
- Critic backward pass requires storing activations $\mathcal{A}_{\text{critic}}$ for computing $\nabla_\phi \frac{1}{2}(V_\phi(s_t) - V_t^{\text{targ}})^2$.
In GRPO:
- Because no value network exists, $\mathcal{A}_{\text{critic}} \equiv 0$ identically! Backward activations are retained solely for the actor policy, cutting training activation memory by $50\%$.

*Step 6: Computational complexity (FLOPs) reduction.*
A standard Transformer forward pass requires $2 N$ FLOPs per token. A backward pass requires $4 N$ FLOPs per token ($2 N$ for activation gradients, $2 N$ for parameter gradients).
Per training token:
- **PPO Training FLOPs:**
  $$\text{FLOPs}_{\text{PPO}} = \underbrace{(2N + 4N)}_{\text{Actor forward + backward}} + \underbrace{(2N + 4N)}_{\text{Critic forward + backward}} + \underbrace{2N}_{\text{Reference forward}} = 6N + 6N + 2N = 14 N \text{ FLOPs/token}$$
- **GRPO Training FLOPs:**
  $$\text{FLOPs}_{\text{GRPO}} = \underbrace{(2N + 4N)}_{\text{Actor forward + backward}} + \underbrace{2N}_{\text{Reference forward}} + \underbrace{0}_{\text{Critic}} = 6N + 2N = 8 N \text{ FLOPs/token}$$
Compute the reduction in training computational FLOPs:
$$\text{Reduction} = \frac{\text{FLOPs}_{\text{PPO}} - \text{FLOPs}_{\text{GRPO}}}{\text{FLOPs}_{\text{PPO}}} = \frac{14 N - 8 N}{14 N} = \frac{6 N}{14 N} = \frac{3}{7} \approx 42.86\% \quad \blacksquare$$

---

#### Derivation 11.28.3: GRPO Clipped Objective with KL Regularization

```
====================================================================================================
DERIVATION 11.28.3: GRPO Clipped Surrogate Objective and Schulman Non-Negative KL Divergence
====================================================================================================
Problem Statement:
Given prompt q, completions {o_i}_{i=1}^G, and standardized group advantages Â_i, prove that:
  1. The sequence-level surrogate objective J(θ) = 𝔼[ (π_θ(o|q) / π_{θ_old}(o|q)) Â ] matches the 
     token-level averaged surrogate loss at first order around θ = θ_old.
  2. The clipping operator min(ρ Â, clip(ρ, 1-ε, 1+ε) Â) forms a pointwise pessimistic lower bound 
     on the unclipped surrogate for all ratios ρ > 0 and advantages Â ∈ ℝ.
  3. The Schulman KL estimator k(u) = u - ln(u) - 1 where u = π_ref / π_θ is strictly non-negative, 
     has unique minimum k(1) = 0, and has zero variance when π_θ ≡ π_ref.
  4. Derive the exact first-order analytical parameter gradient ∇_θ ℒ_{GRPO}(θ).
====================================================================================================
```

**Part 1: Problem Statement & Mathematical Goal**
In GRPO, the policy parameters $\theta$ are optimized to maximize the token-averaged clipped surrogate objective regularized by the reference policy KL divergence:
$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( \rho_{i, t}(\theta) \hat{A}_i, \; \operatorname{clip}(\rho_{i, t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) \right) \right]$$
where the token probability ratio is:
$$\rho_{i, t}(\theta) \triangleq \frac{\pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})}$$
and the token-level KL divergence is parameterized via the analytical Schulman non-negative estimator:
$$D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) \triangleq \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} - \ln \left( \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} \right) - 1$$

Our mathematical goals are:
1. Prove that replacing the sequence-level importance sampling ratio $\prod_{t=1}^{|o|} \rho_t$ with the token-averaged ratio $\frac{1}{|o|} \sum_{t=1}^{|o|} \rho_t$ prevents exponential variance collapse while preserving exact first-order gradient equivalence at $\theta = \theta_{\text{old}}$.
2. Prove that the clipping function $f(\rho; \hat{A}) = \min(\rho \hat{A}, \operatorname{clip}(\rho, 1-\epsilon, 1+\epsilon) \hat{A})$ is point-wise a pessimistic lower bound on $\rho \hat{A}$ for all $\rho > 0$ and all $\hat{A} \in \mathbb{R}$.
3. Prove that the Schulman kernel $k(u) = u - \ln u - 1$ satisfies $k(u) \ge 0$ for all $u > 0$, with equality if and only if $u = 1$, and exhibits zero sample variance when $\pi_\theta \equiv \pi_{\text{ref}}$.
4. Derive the exact closed-form parameter gradient $\nabla_\theta \mathcal{J}_{\text{GRPO}}(\theta)$, characterizing the active gradient regime for both positive and negative advantages.

**Part 2: Explicit Assumptions & Regularity Conditions**
1. **Autoregressive Factorization:** The policy $\pi_\theta$ defines a valid conditional probability distribution over output sequences $o = (y_1, \dots, y_T)$ via the chain rule of probability:
   $$\pi_\theta(o \mid q) = \prod_{t=1}^T \pi_\theta(y_t \mid q, y_{<t})$$
2. **Absolute Continuity / Common Support:** For all tokens $y \in \mathcal{V}$ and contexts $s = (q, y_{<t})$, if $\pi_\theta(y \mid s) > 0$, then $\pi_{\theta_{\text{old}}}(y \mid s) > 0$ and $\pi_{\text{ref}}(y \mid s) > 0$, ensuring that likelihood ratios $\rho = \frac{\pi_\theta}{\pi_{\text{old}}}$ and $u = \frac{\pi_{\text{ref}}}{\pi_\theta}$ are well-defined on $(0, \infty)$.
3. **Clipping Trust Region:** The clipping parameter satisfies $\epsilon \in (0, 1)$, defining the compact symmetric trust region $\mathcal{C}_\epsilon \triangleq [1 - \epsilon, 1 + \epsilon]$.
4. **Smooth Differentiability:** Token logits $z_\theta(y \mid s)$ and softmax probabilities $\pi_\theta(y \mid s) = \frac{e^{z_\theta(y \mid s)}}{\sum_{y'} e^{z_\theta(y' \mid s)}}$ are continuously differentiable with respect to $\theta$.

**Part 3: Underlying Intuition & Geometric / Physical Interpretation**
- **The Variance Catastrophe of Sequence Likelihood Ratios:** In frontier reasoning models (e.g. DeepSeek-R1), mathematical solutions routinely span $T = 4,096$ to $16,384$ tokens. The full sequence importance weight is $\rho_{\text{seq}} = \prod_{t=1}^T \rho_t$. Even if each token ratio deviates from unity by merely $0.5\%$, $(1.005)^{4096} \approx 7.8 \times 10^8$ and $(0.995)^{4096} \approx 1.2 \times 10^{-9}$. Sequence-level importance sampling collapses catastrophically to zero or explodes to infinity. By assigning the trajectory advantage $\hat{A}_i$ to each token and applying token-level clipping, GRPO stabilizes gradient updates across arbitrary sequence lengths!
- **Pessimistic Bounding in Logit Space:** When $\hat{A}_i > 0$ (a correct mathematical proof), we wish to increase token probabilities, but we refuse to credit probability ratios beyond $1 + \epsilon$. Once $\rho > 1 + \epsilon$, the gradient drops to zero, preventing the policy from excessively exploiting a single lucky sample. When $\hat{A}_i < 0$ (a flawed proof), we penalize the tokens, but stop penalizing once $\rho < 1 - \epsilon$, preventing policy collapse and token suppression cascades.
- **The Schulman Asymmetric Bregman Geometry:** A naive sample estimate of reverse KL divergence is $\ln \frac{\pi_\theta}{\pi_{\text{ref}}} = -\ln u$. If an individual token has $\pi_\theta < \pi_{\text{ref}}$ (i.e. $u > 1$), $-\ln u < 0$, creating a negative penalty that mistakenly rewards the model for diverging! The Schulman function $k(u) = u - \ln u - 1$ is strictly convex with minimum $k(1) = 0$, guaranteeing that every token update penalizes divergence.

**Part 4: End-to-End Step-by-Step Algebraic Proof**

*Step 1: First-order equivalence of token-level surrogate.*
Consider the sequence-level importance sampling objective:
$$J_{\text{seq}}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(o \mid q)}{\pi_{\theta_{\text{old}}}(o \mid q)} \hat{A}(q, o) \right]$$
Differentiating with respect to $\theta$ and evaluating at $\theta = \theta_{\text{old}}$:
$$\left. \nabla_\theta \left( \frac{\pi_\theta(o \mid q)}{\pi_{\theta_{\text{old}}}(o \mid q)} \right) \right|_{\theta = \theta_{\text{old}}} = \frac{\nabla_\theta \pi_\theta(o \mid q)}{\pi_{\theta_{\text{old}}}(o \mid q)} = \nabla_\theta \ln \pi_\theta(o \mid q)$$
Using the autoregressive factorization $\ln \pi_\theta(o \mid q) = \sum_{t=1}^{|o|} \ln \pi_\theta(o_t \mid q, o_{<t})$:
$$= \sum_{t=1}^{|o|} \nabla_\theta \ln \pi_\theta(o_t \mid q, o_{<t}) = \sum_{t=1}^{|o|} \left. \nabla_\theta \left( \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})} \right) \right|_{\theta = \theta_{\text{old}}}$$
Now consider the token-averaged surrogate loss scaled by $\frac{1}{|o|}$:
$$\mathcal{L}_{\text{tok}}(\theta) \triangleq \frac{1}{|o|} \sum_{t=1}^{|o|} \rho_t(\theta) \hat{A}(q, o), \quad \rho_t(\theta) = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})}$$
Differentiating at $\theta = \theta_{\text{old}}$:
$$\left. \nabla_\theta \mathcal{L}_{\text{tok}}(\theta) \right|_{\theta = \theta_{\text{old}}} = \frac{1}{|o|} \sum_{t=1}^{|o|} \nabla_\theta \ln \pi_\theta(o_t \mid q, o_{<t}) \hat{A}(q, o) = \frac{1}{|o|} \nabla_\theta \ln \pi_\theta(o \mid q) \hat{A}(q, o)$$
Thus, $\nabla_\theta \mathcal{L}_{\text{tok}}(\theta_{\text{old}}) = \frac{1}{|o|} \nabla_\theta J_{\text{seq}}(\theta_{\text{old}})$. The token-averaged surrogate preserves the exact direction of the true policy gradient while scaling gradient variance inversely with sequence length $|o|$.

*Step 2: Pointwise proof of the pessimistic clipping bound.*
Define the clipped surrogate function:
$$f(\rho; \hat{A}) \triangleq \min \left( \rho \hat{A}, \; \operatorname{clip}(\rho, 1 - \epsilon, 1 + \epsilon) \hat{A} \right)$$
We prove that $f(\rho; \hat{A}) \le \rho \hat{A}$ for all $\rho > 0$ and $\hat{A} \in \mathbb{R}$ by analyzing both advantage cases:

- **Case 1: Positive Advantage ($\hat{A} > 0$).**
  Because $\hat{A} > 0$, multiplying by $\hat{A}$ preserves inequalities:
  $$\operatorname{clip}(\rho, 1 - \epsilon, 1 + \epsilon) \hat{A} = \begin{cases} (1 - \epsilon) \hat{A}, & \rho < 1 - \epsilon \\ \rho \hat{A}, & 1 - \epsilon \le \rho \le 1 + \epsilon \\ (1 + \epsilon) \hat{A}, & \rho > 1 + \epsilon \end{cases}$$
  Evaluating $f(\rho; \hat{A}) = \min(\rho \hat{A}, \operatorname{clip}(\rho) \hat{A})$ across all three regions:
  1. If $\rho < 1 - \epsilon$: $\rho \hat{A} < (1 - \epsilon) \hat{A} \implies \min = \rho \hat{A}$.
  2. If $1 - \epsilon \le \rho \le 1 + \epsilon$: both arguments equal $\rho \hat{A} \implies \min = \rho \hat{A}$.
  3. If $\rho > 1 + \epsilon$: $\rho \hat{A} > (1 + \epsilon) \hat{A} \implies \min = (1 + \epsilon) \hat{A} < \rho \hat{A}$.
  Combining:
  $$f(\rho; \hat{A}) = \min(\rho, 1 + \epsilon) \hat{A} \le \rho \hat{A}$$

- **Case 2: Negative Advantage ($\hat{A} < 0$).**
  Because $\hat{A} < 0$, multiplying by $\hat{A}$ reverses inequalities:
  $$\operatorname{clip}(\rho, 1 - \epsilon, 1 + \epsilon) \hat{A} = \begin{cases} (1 - \epsilon) \hat{A}, & \rho < 1 - \epsilon \\ \rho \hat{A}, & 1 - \epsilon \le \rho \le 1 + \epsilon \\ (1 + \epsilon) \hat{A}, & \rho > 1 + \epsilon \end{cases}$$
  Evaluating $f(\rho; \hat{A}) = \min(\rho \hat{A}, \operatorname{clip}(\rho) \hat{A})$ across all three regions:
  1. If $\rho < 1 - \epsilon$: since $\rho < 1 - \epsilon$ and $\hat{A} < 0$, we have $\rho \hat{A} > (1 - \epsilon) \hat{A} \implies \min = (1 - \epsilon) \hat{A} < \rho \hat{A}$.
  2. If $1 - \epsilon \le \rho \le 1 + \epsilon$: both arguments equal $\rho \hat{A} \implies \min = \rho \hat{A}$.
  3. If $\rho > 1 + \epsilon$: since $\rho > 1 + \epsilon$ and $\hat{A} < 0$, we have $\rho \hat{A} < (1 + \epsilon) \hat{A} \implies \min = \rho \hat{A}$.
  Combining:
  $$f(\rho; \hat{A}) = \begin{cases} (1 - \epsilon) \hat{A}, & \rho < 1 - \epsilon \\ \rho \hat{A}, & \rho \ge 1 - \epsilon \end{cases} \le \rho \hat{A}$$
In both cases, $f(\rho; \hat{A}) \le \rho \hat{A}$ identically everywhere. Thus, the clipped surrogate is strictly a pessimistic lower bound.

*Step 3: First-principles derivation of the Schulman KL estimator.*
The true KL divergence between $\pi_\theta$ and $\pi_{\text{ref}}$ over the next-token distribution is:
$$D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) = \sum_{y \in \mathcal{V}} \pi_\theta(y) \ln \left( \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)} \right)$$
Expressed as an expectation over samples $y \sim \pi_\theta$:
$$D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} \left[ -\ln \left( \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} \right) \right]$$
Let $u(y) \triangleq \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)}$. Compute the expectation of $u(y)$ under $y \sim \pi_\theta$:
$$\mathbb{E}_{y \sim \pi_\theta}[u(y)] = \sum_{y \in \mathcal{V}} \pi_\theta(y) \left( \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} \right) = \sum_{y \in \mathcal{V}} \pi_{\text{ref}}(y) = 1$$
Because $\mathbb{E}_{y \sim \pi_\theta}[u(y) - 1] = 1 - 1 = 0$, we can construct an exact zero-expectation control variate:
$$D_{\text{KL}}(\pi_\theta \;\Vert\; \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} [-\ln u(y)] + \mathbb{E}_{y \sim \pi_\theta} [u(y) - 1] = \mathbb{E}_{y \sim \pi_\theta} [u(y) - \ln u(y) - 1]$$
This yields the Schulman unbiased estimator:
$$\hat{D}_{\text{KL}}^{\text{Schulman}} = \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} - \ln \left( \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} \right) - 1$$

*Step 4: Proof of strict convexity, non-negativity, and zero variance.*
Define the kernel function $k(u) \triangleq u - \ln u - 1$ for $u \in (0, \infty)$.
1. First derivative:
   $$k'(u) = 1 - \frac{1}{u}$$
   Setting $k'(u) = 0 \implies 1 = \frac{1}{u} \implies u = 1$.
2. Second derivative:
   $$k''(u) = \frac{1}{u^2} > 0 \quad \forall u > 0$$
   Because $k''(u) > 0$ strictly on $(0, \infty)$, $k(u)$ is strictly convex with a unique global minimum at $u = 1$.
3. Value at minimum:
   $$k(1) = 1 - \ln(1) - 1 = 0$$
   Therefore:
   $$k(u) \ge 0 \quad \forall u > 0, \quad \text{with } k(u) = 0 \iff u = 1$$
4. Zero-variance property:
   When the trained policy matches the reference policy ($\pi_\theta(y) \equiv \pi_{\text{ref}}(y)$ for all $y \in \mathcal{V}$), $u(y) = \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} = 1$ identically for every token. Thus:
   $$k(u(y)) = k(1) = 0 \quad \forall y \in \mathcal{V}$$
   The sample variance across realizations is:
   $$\operatorname{Var}_{y \sim \pi_\theta}[k(u)] = \mathbb{E}[(0 - 0)^2] \equiv 0$$
   In stark contrast, the naive estimator $-\ln u(y) = \ln \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}$ evaluates to $0$ on average, but has non-zero sample variance $\operatorname{Var}[-\ln u] > 0$ unless all token probabilities are uniform.

*Step 5: Analytical parameter gradient of the full GRPO objective.*
The complete GRPO training objective is:
$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ f(\rho_{i, t}(\theta); \hat{A}_i) - \beta \left( \frac{\pi_{\text{ref}}(o_{i, t})}{\pi_\theta(o_{i, t})} - \ln \frac{\pi_{\text{ref}}(o_{i, t})}{\pi_\theta(o_{i, t})} - 1 \right) \right]$$
Differentiating with respect to $\theta$:
1. For the surrogate term $f(\rho_{i, t}(\theta); \hat{A}_i)$:
   $$\nabla_\theta f(\rho_{i, t}(\theta); \hat{A}_i) = \mathbb{I}_{\text{active}}(i, t) \hat{A}_i \nabla_\theta \rho_{i, t}(\theta) = \mathbb{I}_{\text{active}}(i, t) \hat{A}_i \frac{\nabla_\theta \pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})}$$
   where the indicator of active gradient propagation is:
   $$\mathbb{I}_{\text{active}}(i, t) = \begin{cases} 1, & (\hat{A}_i > 0 \text{ and } \rho_{i, t} < 1 + \epsilon) \lor (\hat{A}_i < 0 \text{ and } \rho_{i, t} > 1 - \epsilon) \\ 0, & \text{otherwise} \end{cases}$$
2. For the Schulman KL penalty with $u = \frac{\pi_{\text{ref}}}{\pi_\theta}$:
   $$\nabla_\theta [u - \ln u - 1] = \left( 1 - \frac{1}{u} \right) \nabla_\theta u = \left( 1 - \frac{\pi_\theta}{\pi_{\text{ref}}} \right) \left( -\frac{\pi_{\text{ref}}}{\pi_\theta^2} \nabla_\theta \pi_\theta \right)$$
   $$= -\frac{\pi_{\text{ref}}}{\pi_\theta^2} \nabla_\theta \pi_\theta + \frac{1}{\pi_\theta} \nabla_\theta \pi_\theta = \left( 1 - \frac{\pi_{\text{ref}}}{\pi_\theta} \right) \frac{\nabla_\theta \pi_\theta}{\pi_\theta} = \left( 1 - \frac{\pi_{\text{ref}}}{\pi_\theta} \right) \nabla_\theta \ln \pi_\theta$$
Combining both parts:
$$\nabla_\theta \mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \mathbb{I}_{\text{active}}(i, t) \hat{A}_i \frac{\nabla_\theta \pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})} - \beta \left( 1 - \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} \right) \nabla_\theta \ln \pi_\theta(o_{i, t} \mid q, o_{i, <t}) \right] \quad \blacksquare$$

---

## 3. Geometric & Physical Interpretation

### 3.1 The Self-Centering Point Cloud
In reward space, the group of completions $\{o_1, \dots, o_G\}$ forms a constellation of outcomes:
```
Reward Axis:
 0.0 ---------- 0.1 ----------------------- 0.525 ----------------------- 1.0
  |              |                            |                            |
[o_4]          [o_3]                        [Mean mu]                 [o_1, o_2]
Broken XML     Wrong Math                                             Correct Proofs
  |              |                                                         |
  v              v                                                         v
A_4 = -1.10    A_3 = -0.89                                               A_1,2 = +1.00
<-- Suppressed Tokens                                       Reinforced Tokens -->
```
By subtracting $\mu_{\mathcal{G}}$, the centroid of the constellation is shifted to zero. The gradient acts as an electrostatic dipole: it attracts the policy distribution toward the successful reasoning trajectory cluster ($+1.00$) while repelling it from the faulty dead ends ($-0.89, -1.10$).

---

## 4. Real-World Analogy: The Math Olympiad Study Group

Imagine a team of 4 students working together to solve an unsolved Olympiad geometry problem:
- **No Teacher / Critic:** There is no omniscient professor hovering over them predicting the probability of success at every step.
- **Trial and Verification:** All 4 students independently draft a proof. At the end, they check the final answer against an official answer key.
- **Relative Comparison:**
  - Alice and Bob find the correct proof.
  - Charlie makes an algebraic error on line 5.
  - David writes an illegible mess.
- The group discusses: *"Alice and Bob's lemma worked ($A > 0$); let us all study and adopt that technique! Charlie's false assumption ($A < 0$) should never be used again."*

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace a complete numerical calculation of **GRPO Group Statistics, Advantages, and Token PPO Gradients** by hand.

---

### 5.1 System & Group Parameters
- **Prompt $q$:** *"Evaluate $2 + 3 \times 4$."*
- **Ground Truth Answer:** $14$
- **Group Size:** $G = 4$ candidate completions
- **Hyperparameters:**
  - PPO clipping threshold: $\epsilon = 0.2000$
  - KL penalty coefficient: $\beta = 0.0500$
- **Candidate Outputs & Evaluated Rewards:**
  - $o_1$: Correct reasoning & answer ($14$), valid format $\implies r_1 = \mathbf{1.0000}$
  - $o_2$: Correct reasoning & answer ($14$), valid format $\implies r_2 = \mathbf{1.0000}$
  - $o_3$: Arithmetic error (answer $20$), valid format $\implies r_3 = \mathbf{0.1000}$ (format only)
  - $o_4$: Wrong answer, broken XML format $\implies r_4 = \mathbf{0.0000}$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $r_i$ | `rewards[i]` | Deterministic scalar reward assigned to candidate completion $o_i$ |
| $\mu_{\mathcal{G}}$ | `group_mean` | Empirical mean reward across the group of $G = 4$ completions |
| $\sigma_{\mathcal{G}}$ | `group_std` | Empirical standard deviation across group completions |
| $\hat{A}_i$ | `group_adv[i]` | Standardized advantage: $(r_i - \mu_{\mathcal{G}}) / \sigma_{\mathcal{G}}$ |
| $\rho_{i, t}$ | `prob_ratio` | Importance sampling token probability ratio: $\pi_\theta / \pi_{\text{old}}$ |
| $D_{\text{KL}}$ | `kl_term` | Schulman unbiased non-negative KL divergence: $u - \ln u - 1$ |

---

### 5.3 Step-by-Step Hand Calculations: Group Advantages

#### Step 1: Compute Group Mean $\mu_{\mathcal{G}}$
$$\mu_{\mathcal{G}} = \frac{r_1 + r_2 + r_3 + r_4}{4} = \frac{1.0000 + 1.0000 + 0.1000 + 0.0000}{4} = \frac{2.1000}{4} = \mathbf{0.525000}$$

#### Step 2: Compute Group Variance & Standard Deviation $\sigma_{\mathcal{G}}$
1. Squared deviations from mean:
   $$(r_1 - \mu)^2 = (1.0000 - 0.5250)^2 = (0.4750)^2 = \mathbf{0.225625}$$
   $$(r_2 - \mu)^2 = (1.0000 - 0.5250)^2 = (0.4750)^2 = \mathbf{0.225625}$$
   $$(r_3 - \mu)^2 = (0.1000 - 0.5250)^2 = (-0.4250)^2 = \mathbf{0.180625}$$
   $$(r_4 - \mu)^2 = (0.0000 - 0.5250)^2 = (-0.5250)^2 = \mathbf{0.275625}$$
2. Sum of squared deviations:
   $$\sum = 0.225625 + 0.225625 + 0.180625 + 0.275625 = \mathbf{0.907500}$$
3. Variance:
   $$\sigma^2 = \frac{0.907500}{4} = \mathbf{0.226875}$$
4. Standard Deviation:
   $$\sigma_{\mathcal{G}} = \sqrt{0.226875} \approx \mathbf{0.47631397}$$

#### Step 3: Compute Standardized Group Advantages $\hat{A}_i$
$$\hat{A}_1 = \frac{1.0000 - 0.5250}{0.47631397} = \frac{+0.4750}{0.47631397} \approx \mathbf{+0.997241}$$
$$\hat{A}_2 = \frac{1.0000 - 0.5250}{0.47631397} = \frac{+0.4750}{0.47631397} \approx \mathbf{+0.997241}$$
$$\hat{A}_3 = \frac{0.1000 - 0.5250}{0.47631397} = \frac{-0.4250}{0.47631397} \approx \mathbf{-0.892269}$$
$$\hat{A}_4 = \frac{0.0000 - 0.5250}{0.47631397} = \frac{-0.5250}{0.47631397} \approx \mathbf{-1.102214}$$

#### Verification of Zero-Sum Property:
$$\sum_{i=1}^4 \hat{A}_i = +0.997241 + 0.997241 - 0.892269 - 1.102214 = \mathbf{0.000000} \quad \checkmark$$

---

### 5.4 Step-by-Step Hand Calculations: Token PPO & KL on Completion $o_1$

Consider a specific token in completion $o_1$:
- Probability under new policy: $\pi_\theta = 0.5500$
- Probability under old policy: $\pi_{\theta_{\text{old}}} = 0.5000$
- Probability under reference policy: $\pi_{\text{ref}} = 0.5000$
- Group Advantage: $\hat{A}_1 = +0.997241$

#### 1. Probability Ratio:
$$\rho = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} = \frac{0.5500}{0.5000} = \mathbf{1.100000}$$

#### 2. PPO Clipped Surrogate:
With $\epsilon = 0.2000$, the clipping interval is $[1 - \epsilon, 1 + \epsilon] = [0.80, 1.20]$.
- Since $\rho = 1.10 \in [0.80, 1.20]$, the ratio is unclipped!
$$\text{Surrogate} = \rho \cdot \hat{A}_1 = 1.100000 \times 0.997241 = \mathbf{1.096965}$$

#### 3. Schulman Analytical KL Penalty:
$$u = \frac{\pi_{\text{ref}}}{\pi_\theta} = \frac{0.5000}{0.5500} = \frac{10}{11} \approx \mathbf{0.909091}$$
$$\ln u = \ln(0.909091) \approx \mathbf{-0.095310}$$
$$D_{\text{KL}} = u - \ln u - 1 = 0.909091 - (-0.095310) - 1.0 = \mathbf{0.004401}$$

#### 4. Net Token Objective:
$$\text{Token Objective} = \text{Surrogate} - \beta \cdot D_{\text{KL}}$$
$$= 1.096965 - 0.0500 \times 0.004401 = 1.096965 - 0.000220 = \mathbf{1.096745}$$

---

### 5.5 Summary Visual Grid: GRPO Optimization Ledger

| Completion $o_i$ | Verifier Acc | Format Tag | Total $r_i$ | Deviation $(r_i - \mu)$ | Advantage $\hat{A}_i$ | Direction | Policy Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$o_1$** | $1.0$ | $0.1$ | **$1.0000$** | $+0.4750$ | $\mathbf{+0.9972}$ | Positive | $\mathbf{\uparrow}$ Strongly Boost Logits |
| **$o_2$** | $1.0$ | $0.1$ | **$1.0000$** | $+0.4750$ | $\mathbf{+0.9972}$ | Positive | $\mathbf{\uparrow}$ Strongly Boost Logits |
| **$o_3$** | $0.0$ | $0.1$ | **$0.1000$** | $-0.4250$ | $\mathbf{-0.8923}$ | Negative | $\mathbf{\downarrow}$ Suppress Wrong Steps |
| **$o_4$** | $0.0$ | $0.0$ | **$0.0000$** | $-0.5250$ | $\mathbf{-1.1022}$ | Negative | $\mathbf{\downarrow}$ Suppress Broken Output |
| **Group Sum** | — | — | $\mu = 0.5250$ | $\sigma = 0.4763$ | $\mathbf{\sum A_i \equiv 0.0000}$ | Centered | **Zero Critic Memory Needed!** |

---

## 6. Solved Illustrations

### Illustration 1: Pure Verification Reward Functions (Symbolic Math & XML Verifiers)

**Problem:**
Implement a deterministic rule-based reward suite for mathematical reasoning and XML tagging. Given the ground truth answer `14` to the problem *"Evaluate $2 + 3 \times 4$."*, evaluate the exact numerical reward for the following four candidate model completions:
1. Completion $o_1$: `"<think>2 + 3 * 4 = 2 + 12 = 14</think>\n<answer>\\boxed{14}</answer>"`
2. Completion $o_2$: `"<think>2 + 3 * 4 = 5 * 4 = 20</think>\n<answer>\\boxed{20}</answer>"`
3. Completion $o_3$: `"The answer is \\boxed{14}."` (No `<think>` or `<answer>` tags)
4. Completion $o_4$: `"Broken attempt with no answer."`

**Solution:**

```python
import re

def math_accuracy_verifier(completion: str, ground_truth: str) -> float:
    """Extracts the answer from \\boxed{...} and compares symbolically to ground truth."""
    match = re.search(r'\\boxed\{([^}]+)\}', completion)
    if not match:
        return 0.0
    extracted_answer = match.group(1).strip()
    return 1.0 if extracted_answer == ground_truth.strip() else 0.0

def format_verifier(completion: str) -> float:
    """Validates presence of <think> ... </think> and <answer> ... </answer> tags."""
    has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL))
    return 0.1 if (has_think and has_answer) else 0.0

def total_verifier_reward(completion: str, ground_truth: str) -> float:
    return math_accuracy_verifier(completion, ground_truth) + format_verifier(completion)
```

**Step-by-Step Numerical Evaluation:**

1. **Completion $o_1$:**
   - Math Accuracy: $\boxed{14}$ matches ground truth `14` $\implies r_{\text{acc}} = \mathbf{1.000000}$.
   - XML Format: Contains valid `<think>` and `<answer>` blocks $\implies r_{\text{format}} = \mathbf{0.100000}$.
   - Total Reward: $r_1 = 1.000000 + 0.100000 = \mathbf{1.100000}$.

2. **Completion $o_2$:**
   - Math Accuracy: $\boxed{20} \ne 14 \implies r_{\text{acc}} = \mathbf{0.000000}$.
   - XML Format: Contains valid `<think>` and `<answer>` blocks $\implies r_{\text{format}} = \mathbf{0.100000}$.
   - Total Reward: $r_2 = 0.000000 + 0.100000 = \mathbf{0.100000}$.

3. **Completion $o_3$:**
   - Math Accuracy: $\boxed{14}$ matches ground truth `14` $\implies r_{\text{acc}} = \mathbf{1.000000}$.
   - XML Format: Missing `<think>` and `<answer>` tags $\implies r_{\text{format}} = \mathbf{0.000000}$.
   - Total Reward: $r_3 = 1.000000 + 0.000000 = \mathbf{1.000000}$.

4. **Completion $o_4$:**
   - Math Accuracy: No $\boxed{...}$ block $\implies r_{\text{acc}} = \mathbf{0.000000}$.
   - XML Format: Missing XML tags $\implies r_{\text{format}} = \mathbf{0.000000}$.
   - Total Reward: $r_4 = 0.000000 + 0.000000 = \mathbf{0.000000}$.

$\blacksquare$

---

### Illustration 2: GRPO Group Advantage Calculation on a Group of $G = 6$ Completions

**Problem:**
A frontier reasoning model samples a group of $G = 6$ candidate completions for an AIME contest problem. Evaluating the completions with a deterministic rule-based correctness verifier produces the binary reward vector:
$$\mathbf{r} = [r_1, r_2, r_3, r_4, r_5, r_6]^T = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]^T$$
Using the GRPO group advantage formulation with variance regularizer $\epsilon_{\text{std}} = 0$:
1. Calculate the empirical group mean $\mu_{\mathcal{G}}$ and empirical standard deviation $\sigma_{\mathcal{G}}$.
2. Compute the standardized group relative advantages $\hat{A}_1, \dots, \hat{A}_6$.
3. Verify the Zero-Sum Group Property $\sum_{i=1}^6 \hat{A}_i = 0$.

**Solution:**

#### Step 1: Compute Empirical Group Mean $\mu_{\mathcal{G}}$
$$\sum_{i=1}^6 r_i = 1.0 + 0.0 + 1.0 + 1.0 + 0.0 + 0.0 = 3.000000$$
$$\mu_{\mathcal{G}} = \frac{1}{G} \sum_{i=1}^G r_i = \frac{3.000000}{6} = \mathbf{0.500000}$$

#### Step 2: Compute Empirical Group Variance & Standard Deviation $\sigma_{\mathcal{G}}$
Compute squared deviations from the group mean $(r_i - \mu_{\mathcal{G}})^2$:
- For $r_1 = 1.0$: $(1.000000 - 0.500000)^2 = (+0.500000)^2 = \mathbf{0.250000}$
- For $r_2 = 0.0$: $(0.000000 - 0.500000)^2 = (-0.500000)^2 = \mathbf{0.250000}$
- For $r_3 = 1.0$: $(1.000000 - 0.500000)^2 = (+0.500000)^2 = \mathbf{0.250000}$
- For $r_4 = 1.0$: $(1.000000 - 0.500000)^2 = (+0.500000)^2 = \mathbf{0.250000}$
- For $r_5 = 0.0$: $(0.000000 - 0.500000)^2 = (-0.500000)^2 = \mathbf{0.250000}$
- For $r_6 = 0.0$: $(0.000000 - 0.500000)^2 = (-0.500000)^2 = \mathbf{0.250000}$

Sum of squared deviations:
$$\sum_{i=1}^6 (r_i - \mu_{\mathcal{G}})^2 = 6 \times 0.250000 = \mathbf{1.500000}$$
Group variance:
$$\sigma_{\mathcal{G}}^2 = \frac{1}{G} \sum_{i=1}^G (r_i - \mu_{\mathcal{G}})^2 = \frac{1.500000}{6} = \mathbf{0.250000}$$
Group standard deviation:
$$\sigma_{\mathcal{G}} = \sqrt{0.250000} = \mathbf{0.500000}$$

#### Step 3: Compute Standardized Group Relative Advantages $\hat{A}_i = \frac{r_i - \mu_{\mathcal{G}}}{\sigma_{\mathcal{G}}}$
- Completion $o_1$ ($r_1 = 1.0$):
  $$\hat{A}_1 = \frac{1.000000 - 0.500000}{0.500000} = \frac{+0.500000}{0.500000} = \mathbf{+1.000000}$$
- Completion $o_2$ ($r_2 = 0.0$):
  $$\hat{A}_2 = \frac{0.000000 - 0.500000}{0.500000} = \frac{-0.500000}{0.500000} = \mathbf{-1.000000}$$
- Completion $o_3$ ($r_3 = 1.0$):
  $$\hat{A}_3 = \frac{1.000000 - 0.500000}{0.500000} = \frac{+0.500000}{0.500000} = \mathbf{+1.000000}$$
- Completion $o_4$ ($r_4 = 1.0$):
  $$\hat{A}_4 = \frac{1.000000 - 0.500000}{0.500000} = \frac{+0.500000}{0.500000} = \mathbf{+1.000000}$$
- Completion $o_5$ ($r_5 = 0.0$):
  $$\hat{A}_5 = \frac{0.000000 - 0.500000}{0.500000} = \frac{-0.500000}{0.500000} = \mathbf{-1.000000}$$
- Completion $o_6$ ($r_6 = 0.0$):
  $$\hat{A}_6 = \frac{0.000000 - 0.500000}{0.500000} = \frac{-0.500000}{0.500000} = \mathbf{-1.000000}$$

#### Step 4: Verification of the Zero-Sum Property
$$\sum_{i=1}^6 \hat{A}_i = (+1.000000) + (-1.000000) + (+1.000000) + (+1.000000) + (-1.000000) + (-1.000000)$$
$$= 3.000000 - 3.000000 = \mathbf{0.000000} \quad \checkmark$$

| Candidate Completion | Verifier Reward $r_i$ | Deviation $(r_i - \mu_{\mathcal{G}})$ | Group Advantage $\hat{A}_i$ | Gradient Role |
| :---: | :---: | :---: | :---: | :---: |
| **$o_1$** | $1.0$ | $+0.500000$ | $\mathbf{+1.000000}$ | Reinforce tokens ($\uparrow$ log-probabilities) |
| **$o_2$** | $0.0$ | $-0.500000$ | $\mathbf{-1.000000}$ | Suppress tokens ($\downarrow$ log-probabilities) |
| **$o_3$** | $1.0$ | $+0.500000$ | $\mathbf{+1.000000}$ | Reinforce tokens ($\uparrow$ log-probabilities) |
| **$o_4$** | $1.0$ | $+0.500000$ | $\mathbf{+1.000000}$ | Reinforce tokens ($\uparrow$ log-probabilities) |
| **$o_5$** | $0.0$ | $-0.500000$ | $\mathbf{-1.000000}$ | Suppress tokens ($\downarrow$ log-probabilities) |
| **$o_6$** | $0.0$ | $-0.500000$ | $\mathbf{-1.000000}$ | Suppress tokens ($\downarrow$ log-probabilities) |
| **Group Total** | $\sum = 3.0, \; \mu = 0.5$ | $\sum \Delta = 0.000000$ | $\mathbf{\sum \hat{A}_i \equiv 0.000000}$ | Self-Centering Baseline (No Critic Needed) |

$\blacksquare$

---

### Illustration 3: Token-Level Clipped Surrogate Loss Forward Pass for Positive and Negative Advantages

**Problem:**
Consider two completions from the group in Illustration 2 evaluated with PPO clipping threshold $\epsilon = 0.20$ (trust region $[1 - \epsilon, 1 + \epsilon] = [0.80, 1.20]$):
- **Completion $o_1$** with positive advantage $\hat{A}_1 = +1.000000$, length $T_1 = 3$ tokens.
- **Completion $o_2$** with negative advantage $\hat{A}_2 = -1.000000$, length $T_2 = 3$ tokens.

Given the token probabilities under the proposal rollout policy $\pi_{\text{old}}$ and updated policy $\pi_\theta$:
- For $o_1$:
  - Token 1: $\pi_{\text{old}} = 0.40, \pi_\theta = 0.44$
  - Token 2: $\pi_{\text{old}} = 0.50, \pi_\theta = 0.65$
  - Token 3: $\pi_{\text{old}} = 0.80, \pi_\theta = 0.72$
- For $o_2$:
  - Token 1: $\pi_{\text{old}} = 0.50, \pi_\theta = 0.60$
  - Token 2: $\pi_{\text{old}} = 0.50, \pi_\theta = 0.35$
  - Token 3: $\pi_{\text{old}} = 0.40, \pi_\theta = 0.36$

Compute:
1. Token probability ratios $\rho_{i, t} = \frac{\pi_\theta(t)}{\pi_{\text{old}}(t)}$.
2. Per-token clipped surrogate values $f(\rho_{i, t}; \hat{A}_i) = \min(\rho_{i, t} \hat{A}_i, \operatorname{clip}(\rho_{i, t}, 0.80, 1.20) \hat{A}_i)$.
3. Mean surrogate objective for each completion: $\mathcal{L}_{\text{surr}}^{(i)} = \frac{1}{T_i} \sum_{t=1}^{T_i} f(\rho_{i, t}; \hat{A}_i)$.

**Solution:**

#### Part A: Forward Pass for Completion $o_1$ ($\hat{A}_1 = +1.000000$)
- **Token $t = 1$:**
  $$\rho_{1, 1} = \frac{0.440000}{0.400000} = \mathbf{1.100000}$$
  Unclipped term: $\rho_{1, 1} \hat{A}_1 = 1.100000 \times (+1.000000) = +1.100000$
  Clipped term: $\operatorname{clip}(1.100000, 0.80, 1.20) \hat{A}_1 = 1.100000 \times (+1.000000) = +1.100000$
  $$\text{Surrogate}_{1, 1} = \min(+1.100000, +1.100000) = \mathbf{+1.100000} \quad (\text{Unclipped})$$

- **Token $t = 2$:**
  $$\rho_{1, 2} = \frac{0.650000}{0.500000} = \mathbf{1.300000}$$
  Unclipped term: $\rho_{1, 2} \hat{A}_1 = 1.300000 \times (+1.000000) = +1.300000$
  Clipped term: $\operatorname{clip}(1.300000, 0.80, 1.20) \hat{A}_1 = 1.200000 \times (+1.000000) = +1.200000$
  $$\text{Surrogate}_{1, 2} = \min(+1.300000, +1.200000) = \mathbf{+1.200000} \quad (\text{Clipped at upper bound } 1 + \epsilon)$$

- **Token $t = 3$:**
  $$\rho_{1, 3} = \frac{0.720000}{0.800000} = \mathbf{0.900000}$$
  Unclipped term: $\rho_{1, 3} \hat{A}_1 = 0.900000 \times (+1.000000) = +0.900000$
  Clipped term: $\operatorname{clip}(0.900000, 0.80, 1.20) \hat{A}_1 = 0.900000 \times (+1.000000) = +0.900000$
  $$\text{Surrogate}_{1, 3} = \min(+0.900000, +0.900000) = \mathbf{+0.900000} \quad (\text{Unclipped})$$

Mean surrogate for Completion $o_1$:
$$\mathcal{L}_{\text{surr}}^{(1)} = \frac{1}{3} (1.100000 + 1.200000 + 0.900000) = \frac{3.200000}{3} \approx \mathbf{+1.066667}$$

#### Part B: Forward Pass for Completion $o_2$ ($\hat{A}_2 = -1.000000$)
- **Token $t = 1$:**
  $$\rho_{2, 1} = \frac{0.600000}{0.500000} = \mathbf{1.200000}$$
  Unclipped term: $\rho_{2, 1} \hat{A}_2 = 1.200000 \times (-1.000000) = -1.200000$
  Clipped term: $\operatorname{clip}(1.200000, 0.80, 1.20) \hat{A}_2 = 1.200000 \times (-1.000000) = -1.200000$
  $$\text{Surrogate}_{2, 1} = \min(-1.200000, -1.200000) = \mathbf{-1.200000} \quad (\text{Unclipped boundary})$$

- **Token $t = 2$:**
  $$\rho_{2, 2} = \frac{0.350000}{0.500000} = \mathbf{0.700000}$$
  Unclipped term: $\rho_{2, 2} \hat{A}_2 = 0.700000 \times (-1.000000) = -0.700000$
  Clipped term: $\operatorname{clip}(0.700000, 0.80, 1.20) \hat{A}_2 = 0.800000 \times (-1.000000) = -0.800000$
  $$\text{Surrogate}_{2, 2} = \min(-0.700000, -0.800000) = \mathbf{-0.800000} \quad (\text{Clipped at lower bound } 1 - \epsilon)$$
  *(Note: Because $-0.80 < -0.70$, the pessimistic $\min$ selects $-0.80$, bounding the negative surrogate).*

- **Token $t = 3$:**
  $$\rho_{2, 3} = \frac{0.360000}{0.400000} = \mathbf{0.900000}$$
  Unclipped term: $\rho_{2, 3} \hat{A}_2 = 0.900000 \times (-1.000000) = -0.900000$
  Clipped term: $\operatorname{clip}(0.900000, 0.80, 1.20) \hat{A}_2 = 0.900000 \times (-1.000000) = -0.900000$
  $$\text{Surrogate}_{2, 3} = \min(-0.900000, -0.900000) = \mathbf{-0.900000} \quad (\text{Unclipped})$$

Mean surrogate for Completion $o_2$:
$$\mathcal{L}_{\text{surr}}^{(2)} = \frac{1}{3} (-1.200000 + (-0.800000) + (-0.900000)) = \frac{-2.900000}{3} \approx \mathbf{-0.966667}$$

Combined pair objective:
$$\mathcal{L}_{\text{surr}} = \frac{1}{2} \left( \mathcal{L}_{\text{surr}}^{(1)} + \mathcal{L}_{\text{surr}}^{(2)} \right) = \frac{1.066667 + (-0.966667)}{2} = \frac{0.100000}{2} = \mathbf{+0.050000}$$

$\blacksquare$

---

### Illustration 4: Unbiased Reference KL Penalty Approximation via Schulman Estimator

**Problem:**
Calculate the analytical Schulman non-negative KL divergence estimator:
$$D_{\text{KL}}^{\text{Schulman}} = u - \ln u - 1, \quad \text{where } u = \frac{\pi_{\text{ref}}}{\pi_\theta}$$
and compare it to the naive log-ratio estimator $D_{\text{KL}}^{\text{naive}} = \ln \left( \frac{\pi_\theta}{\pi_{\text{ref}}} \right) = -\ln u$ across four canonical token probability transitions:
1. **Case 1 (Exact Identity):** $\pi_\theta = 0.50, \pi_{\text{ref}} = 0.50$
2. **Case 2 (Confidence Increase):** $\pi_\theta = 0.60, \pi_{\text{ref}} = 0.40$
3. **Case 3 (Confidence Drop):** $\pi_\theta = 0.30, \pi_{\text{ref}} = 0.60$
4. **Case 4 (Extreme Divergence):** $\pi_\theta = 0.80, \pi_{\text{ref}} = 0.20$

**Solution:**

#### Step 1: Case 1 (Exact Identity: $\pi_\theta = 0.50, \pi_{\text{ref}} = 0.50$)
$$u = \frac{0.500000}{0.500000} = 1.000000$$
$$\ln u = \ln(1.000000) = 0.000000$$
$$D_{\text{KL}}^{\text{Schulman}} = 1.000000 - 0.000000 - 1.0 = \mathbf{0.000000}$$
$$D_{\text{KL}}^{\text{naive}} = -\ln(1.000000) = \mathbf{0.000000}$$
*Outcome:* Both estimators produce exactly zero; Schulman guarantees zero sample variance.

#### Step 2: Case 2 (Confidence Increase: $\pi_\theta = 0.60, \pi_{\text{ref}} = 0.40$)
$$u = \frac{0.400000}{0.600000} = \frac{2}{3} \approx 0.666667$$
$$\ln u = \ln(0.666667) \approx -0.405465$$
$$D_{\text{KL}}^{\text{Schulman}} = 0.666667 - (-0.405465) - 1.0 = 0.666667 + 0.405465 - 1.0 = \mathbf{0.072132}$$
$$D_{\text{KL}}^{\text{naive}} = -\ln(0.666667) \approx \mathbf{+0.405465}$$
*Outcome:* Schulman applies a smooth, quadratic-like penalty $(0.072132)$ rather than the aggressive linear log-penalty.

#### Step 3: Case 3 (Confidence Drop: $\pi_\theta = 0.30, \pi_{\text{ref}} = 0.60$)
$$u = \frac{0.600000}{0.300000} = 2.000000$$
$$\ln u = \ln(2.000000) \approx 0.693147$$
$$D_{\text{KL}}^{\text{Schulman}} = 2.000000 - 0.693147 - 1.0 = 1.000000 - 0.693147 = \mathbf{0.306853} \ge 0$$
$$D_{\text{KL}}^{\text{naive}} = -\ln(2.000000) \approx \mathbf{-0.693147} < 0$$
*Critical Analysis:* The naive estimator evaluates to a **negative value** ($-0.693147$). In an RL objective $\mathcal{L} = \text{Surrogate} - \beta D_{\text{KL}}$, a negative KL penalty transforms into an artificial **reward bonus** $(-\beta(-0.693) = +0.693\beta)$, actively incentivizing the model to drift further away from the reference policy! The Schulman estimator strictly enforces a positive penalty ($+0.306853$), guaranteeing mathematical stability.

#### Step 4: Case 4 (Extreme Divergence: $\pi_\theta = 0.80, \pi_{\text{ref}} = 0.20$)
$$u = \frac{0.200000}{0.800000} = 0.250000$$
$$\ln u = \ln(0.250000) \approx -1.386294$$
$$D_{\text{KL}}^{\text{Schulman}} = 0.250000 - (-1.386294) - 1.0 = 0.250000 + 1.386294 - 1.0 = \mathbf{0.636294}$$
$$D_{\text{KL}}^{\text{naive}} = -\ln(0.250000) \approx \mathbf{+1.386294}$$

| Scenario | Policy $\pi_\theta$ | Ref $\pi_{\text{ref}}$ | Ratio $u = \frac{\pi_{\text{ref}}}{\pi_\theta}$ | Schulman KL $u - \ln u - 1$ | Naive KL $\ln \frac{\pi_\theta}{\pi_{\text{ref}}}$ | Regularization Behavior |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Match** | $0.50$ | $0.50$ | $1.000000$ | $\mathbf{0.000000}$ | $0.000000$ | Perfect equilibrium (zero variance) |
| **Probability Boost** | $0.60$ | $0.40$ | $0.666667$ | $\mathbf{0.072132}$ | $+0.405465$ | Stable convex penalty |
| **Probability Drop** | $0.30$ | $0.60$ | $2.000000$ | $\mathbf{0.306853}$ | $\mathbf{-0.693147}$ | **Schulman penalizes; Naive erroneously rewards!** |
| **Severe Divergence** | $0.80$ | $0.20$ | $0.250000$ | $\mathbf{0.636294}$ | $+1.386294$ | Strong non-negative restorative force |

$\blacksquare$

---

### Illustration 5: Memory Footprint Comparison: PPO (4 Models in VRAM) vs. GRPO on an 8x H100 GPU Cluster

**Problem:**
An AI research lab trains a 14-billion parameter reasoning model (e.g. DeepSeek-R1-Distill-Qwen-14B, $N = 14 \times 10^9$ parameters) on a single 8-GPU node equipped with **$8 \times$ NVIDIA H100 SXM5 GPUs** ($80\text{ GB}$ HBM3 VRAM per GPU; total cluster VRAM = $8 \times 80 = 640\text{ GB}$).
Training uses 16-bit mixed-precision AdamW and ZeRO Stage 3 / FSDP distributed parameter sharding.
1. Compute the exact static VRAM footprint across the cluster and per GPU for standard **PPO** (Actor, Critic, Reference, and Reward Model).
2. Compute the exact static VRAM footprint across the cluster and per GPU for **GRPO** (Actor and Reference only; Critic eliminated; Reward Model on CPU).
3. Compute the available dynamic headroom remaining for activations and rollout KV-cache per GPU.
4. Scale the analysis to a **32-billion parameter model** ($N = 32 \times 10^9$) to determine if PPO or GRPO can fit.

**Solution:**

#### Memory Layout Constants (Mixed Precision AdamW):
- Trainable Model ($\pi_\theta, V_\phi$):
  - Weights (bf16): $2\text{ bytes/param}$
  - Gradients (bf16): $2\text{ bytes/param}$
  - AdamW Optimizer (fp32 master weights, momentum $m$, variance $v$): $4 + 4 + 4 = 12\text{ bytes/param}$
  - Total trainable memory: $2 + 2 + 12 = \mathbf{16\text{ bytes/param}}$
- Frozen Model ($\pi_{\text{ref}}, r_\psi$):
  - Weights (bf16): $\mathbf{2\text{ bytes/param}}$ (no gradients, no optimizer states)

#### Part 1: 14B Parameter Model under Classical PPO ($N = 14 \times 10^9$)
1. **Cluster Static Memory Breakdown:**
   - Trainable Actor $\pi_\theta$: $16\text{ bytes} \times 14 \times 10^9 = 224.0\text{ GB}$
   - Trainable Critic $V_\phi$: $16\text{ bytes} \times 14 \times 10^9 = 224.0\text{ GB}$
   - Frozen Reference $\pi_{\text{ref}}$: $2\text{ bytes} \times 14 \times 10^9 = 28.0\text{ GB}$
   - Frozen Reward Model $r_\psi$: $2\text{ bytes} \times 14 \times 10^9 = 28.0\text{ GB}$
   - Total Cluster Static Memory:
     $$M_{\text{static}}^{\text{PPO}} = 224.0 + 224.0 + 28.0 + 28.0 = \mathbf{504.0\text{ GB}}$$
2. **Per-GPU Static Memory (Sharded across $K = 8$ GPUs):**
   $$M_{\text{per-GPU}}^{\text{PPO}} = \frac{504.0\text{ GB}}{8} = \mathbf{63.00\text{ GB/GPU}}$$
3. **Dynamic Headroom per GPU:**
   $$\text{Headroom}_{\text{PPO}} = 80.00\text{ GB} - 63.00\text{ GB} = \mathbf{17.00\text{ GB/GPU}}$$
   *Failure Mode:* Storing forward-backward activations for both Actor and Critic ($> 12\text{ GB}$) and maintaining rollout KV caches for long sequences ($T = 8,192$, $\sim 10\text{ GB}$) exceeds the $17.00\text{ GB}$ headroom ($12 + 10 = 22\text{ GB} > 17\text{ GB}$), triggering **CUDA Out of Memory (OOM)**!

#### Part 2: 14B Parameter Model under GRPO ($N = 14 \times 10^9$)
1. **Cluster Static Memory Breakdown:**
   - Trainable Actor $\pi_\theta$: $16\text{ bytes} \times 14 \times 10^9 = 224.0\text{ GB}$
   - Frozen Reference $\pi_{\text{ref}}$: $2\text{ bytes} \times 14 \times 10^9 = 28.0\text{ GB}$
   - Trainable Critic $V_\phi$: **Eliminated!** ($0.0\text{ GB}$)
   - Reward Model $r_\psi$: **Rule-based verifier on CPU!** ($0.0\text{ GB}$)
   - Total Cluster Static Memory:
     $$M_{\text{static}}^{\text{GRPO}} = 224.0 + 28.0 + 0.0 + 0.0 = \mathbf{252.0\text{ GB}}$$
2. **Per-GPU Static Memory (Sharded across $K = 8$ GPUs):**
   $$M_{\text{per-GPU}}^{\text{GRPO}} = \frac{252.0\text{ GB}}{8} = \mathbf{31.50\text{ GB/GPU}}$$
3. **Dynamic Headroom per GPU:**
   $$\text{Headroom}_{\text{GRPO}} = 80.00\text{ GB} - 31.50\text{ GB} = \mathbf{48.50\text{ GB/GPU}}$$
4. **VRAM Savings & Headroom Gain:**
   $$\Delta M_{\text{cluster}} = 504.0\text{ GB} - 252.0\text{ GB} = \mathbf{252.0\text{ GB}} \quad \left( \frac{252.0}{504.0} = \mathbf{50.00\%} \text{ savings} \right)$$
   $$\text{Headroom Multiplier} = \frac{48.50\text{ GB}}{17.00\text{ GB}} \approx \mathbf{2.85\times \text{ increase in usable dynamic VRAM!}}$$
   With $48.50\text{ GB}$ of free memory per GPU, the system easily accommodates long Chain-of-Thought rollouts up to $16,384$ tokens and group sizes $G = 16$ without memory swapping.

#### Part 3: Scaling to a 32B Parameter Model ($N = 32 \times 10^9$)
- **PPO Cluster Static Footprint:**
  $$M_{\text{static}}^{\text{PPO}} = (16 + 16 + 2 + 2) \times 32 = 36 \times 32 = \mathbf{1,152.0\text{ GB}}$$
  Per GPU: $\frac{1152.0}{8} = \mathbf{144.00\text{ GB/GPU}} \gg 80.0\text{ GB}$ (**Impossible on 8x H100**; requires minimum 16 to 32 GPUs).
- **GRPO Cluster Static Footprint:**
  $$M_{\text{static}}^{\text{GRPO}} = (16 + 2) \times 32 = 18 \times 32 = \mathbf{576.0\text{ GB}}$$
  Per GPU: $\frac{576.0}{8} = \mathbf{72.00\text{ GB/GPU}} \le 80.0\text{ GB}$ (**Fits on 8x H100 node!**).

| Metric / Configuration | PPO (14B Model) | GRPO (14B Model) | PPO (32B Model) | GRPO (32B Model) |
| :--- | :---: | :---: | :---: | :---: |
| **Actor $\pi_\theta$ Memory** | $224.0\text{ GB}$ | $224.0\text{ GB}$ | $512.0\text{ GB}$ | $512.0\text{ GB}$ |
| **Critic $V_\phi$ Memory** | $224.0\text{ GB}$ | **$0.0\text{ GB}$ (Eliminated)** | $512.0\text{ GB}$ | **$0.0\text{ GB}$ (Eliminated)** |
| **Reference $\pi_{\text{ref}}$ Memory** | $28.0\text{ GB}$ | $28.0\text{ GB}$ | $64.0\text{ GB}$ | $64.0\text{ GB}$ |
| **Reward Model $r_\psi$ Memory** | $28.0\text{ GB}$ | **$0.0\text{ GB}$ (CPU)** | $64.0\text{ GB}$ | **$0.0\text{ GB}$ (CPU)** |
| **Total Cluster Static VRAM** | $\mathbf{504.0\text{ GB}}$ | $\mathbf{252.0\text{ GB}}$ | $\mathbf{1,152.0\text{ GB}}$ | $\mathbf{576.0\text{ GB}}$ |
| **Per-GPU Static VRAM ($8\times$ H100)** | $63.00\text{ GB}$ | $\mathbf{31.50\text{ GB}}$ | $144.00\text{ GB}$ | $\mathbf{72.00\text{ GB}}$ |
| **Remaining Headroom / GPU** | $17.00\text{ GB}$ | $\mathbf{48.50\text{ GB}}$ | $-64.00\text{ GB}$ (OOM) | $\mathbf{+8.00\text{ GB}}$ |
| **Hardware Feasibility (8x H100)** | **Fails (OOM during training)** | **Trains Smoothly ($G=16$)** | **Impossible ($> 2\times$ VRAM)** | **Feasible ($T=4k$)** |

$\blacksquare$

---

### Illustration 6: Frontier Reasoning Dynamics: The Autonomous Emergence of the "Aha Moment"

In the landmark DeepSeek-R1 investigations (DeepSeek-AI, 2025), training a base language model with pure rule-based GRPO (without supervised fine-tuning warm-start data, termed **DeepSeek-R1-Zero**) yielded a historic emergent behavior: the autonomous development of internal self-reflection, backtracking, and algorithmic search.

During training on mathematical olympiad problems, intermediate checkpoint rollouts revealed spontaneous chain-of-thought phrases such as:
> *"Wait, let me double check this equation... Ah, wait, that was a mistake! If $x < 0$, the square root is undefined. Let me re-evaluate from line 3..."*

#### Why GRPO Spontaneously Drives Self-Correction:
1. **All-or-Nothing Verifier Sparsity:** Under binary verifier rewards ($r \in \{0, 1\}$), partial proofs receive exactly $0.0$ reward. If the model commits an arithmetic blunder on token 50 of a 5,000-token proof, the entire completion fails ($r = 0.0$).
2. **Dynamic Exploration Incentive:** Candidate completions that spend an additional 500 tokens conducting internal sanity checks, exploring alternative lemmas, and re-deriving intermediate equations have a significantly higher statistical probability of arriving at the correct verified boxed answer ($r = 1.0 \implies \hat{A}_i = +1.0$).
3. **Autonomous Length Scaling:** GRPO does not impose an artificial token penalty; the policy gradient autonomously learns that allocating inference-time compute to self-verification maximizes expected group advantage, validating the modern **Inference-Time Compute Scaling Law**.

$\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. DeepSeek-R1 & R1-Zero: Pure RL Reasoning Breakthrough (DeepSeek-AI, January 2025)
GRPO produced the most impactful open-weights reasoning model to date:
- **DeepSeek-R1-Zero:** Trained with GRPO from a base language model checkpoint with **zero supervised fine-tuning** — no curated chain-of-thought data, no demonstration rationales. The model spontaneously developed self-verification (re-reading answers), reflection (noting mistakes), and extended reasoning traces purely from outcome reward signals. Achieved 71.0% pass@1 on AIME 2024.
- **DeepSeek-R1:** After cold-start SFT on 1,000 curated long-CoT examples, GRPO training on 600K math/coding problems produced a model achieving **97.3% on MATH-500** and **79.8% pass@1 on AIME 2024**, matching OpenAI o1 at $\sim$3% of the compute cost.
- **Emergent "Aha Moment":** During GRPO training, the model discovered that re-reading its own answer and checking it character-by-character (using \<think\>\</think\> tags) dramatically improved accuracy — a behavior never demonstrated in training data.

### 2. OpenAI o1 & o3-mini: Inference-Time Compute Scaling Laws (2024–2025)
GRPO-related techniques underpin the new frontier of inference scaling:
- **o1 (September 2024):** Achieved 83.3% on AIME 2024 (vs. GPT-4o's 13.4%) through extended reasoning token generation. The critical insight: **FLOPs spent at inference (thinking) substitute for FLOPs spent at training**, following a log-linear scaling law.
- **o3-mini (January 2025):** Achieves 86.5% on AIME 2025 with "high" compute budget setting ($\sim$5× more thinking tokens than "medium"). Outperforms human 90th-percentile scorers on AIME for the first time.
- **Compute-Optimal Inference:** The relationship $\text{accuracy} \propto \log(\text{inference FLOPs})$ was empirically validated — doubling thinking token budget consistently improves hard mathematical reasoning by ~3–5 percentile points.

### 3. Qwen-Math, QwQ & the Open Reasoning Ecosystem (Alibaba, 2025)
GRPO's open publication enabled a wave of competitive reasoning models:
- **QwQ-32B-Preview:** Qwen's open-weights reasoning model using GRPO-style training, achieving 50% pass@1 on AIME 2024 — competitive with o1-preview at 3% of the parameter count and fully open weights.
- **Qwen2.5-Math-72B:** GRPO fine-tuning on mathematical reasoning datasets achieves 90.6% on MATH-500, establishing a new open-weights SOTA.
- **GRPO vs. PPO in Math RL:** Empirical comparison across 8 models shows GRPO consistently outperforms PPO on mathematical reasoning benchmarks when the reward signal is dense (every solution is verifiable) — the critic-free group baseline eliminates the value function's tendency to overfit to common problem patterns.

---

## 8. Code Implementation & Verification

The accompanying Python script implements and rigorously validates every mathematical formulation in this chapter:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact numerical calculation of group mean $\mu = 0.525000$, standard deviation $\sigma = 0.476314$.
   - Group advantages $\hat{A} = [+0.997241, +0.997241, -0.892269, -1.102214]$ verifying zero-sum property $\sum \hat{A}_i = 0.000000$ to $< 10^{-12}$.
   - Schulman analytical non-negative KL divergence $D_{\text{KL}} = 0.004401$ and net token objective $1.096745$.
2. **Deterministic Rule-Based Verifiers (Illustration 1):**
   - Regex-based accuracy verifier extracting `\boxed{...}` and comparing against ground truth.
   - XML tag syntax verifier validating `<think> ... </think>` and `<answer> ... </answer>` blocks.
3. **Group Advantage on $G=6$ Completions (Illustration 2):**
   - Group rewards $r = [1, 0, 1, 1, 0, 0] \implies \mu = 0.500000, \sigma = 0.500000$.
   - Normalized advantages $\hat{A} = [+1.0, -1.0, +1.0, +1.0, -1.0, -1.0]$ with zero-sum check $\sum \hat{A}_i \equiv 0.000000$.
4. **Token-Level Clipped Surrogate Forward Pass (Illustration 3):**
   - Exact computation of token ratios and clipping bounds for positive ($\hat{A} = +1.0$, surrogate $= +1.066667$) and negative ($\hat{A} = -1.0$, surrogate $= -0.966667$) advantages.
5. **Schulman Non-Negative KL Divergence Suite (Illustration 4):**
   - Multi-case validation proving strict non-negativity and demonstrating pathological negative penalties in naive log-ratio estimators.
6. **VRAM Memory Footprint Audit (Illustration 5):**
   - Verification of 50.0% static VRAM memory reduction ($504\text{ GB} \to 252\text{ GB}$) and $2.85\times$ dynamic headroom expansion on an 8x H100 cluster.
7. **PyTorch Vectorized GRPO Engine:**
   - Full implementation of Critic-free group-relative advantage estimation, vectorized Schulman KL divergence regularizer, and end-to-end backpropagation gradient verification.

See implementation in:
[`11_reinforcement_learning/code/28_grpo_deepseek_r1.py`](./code/28_grpo_deepseek_r1.py)

