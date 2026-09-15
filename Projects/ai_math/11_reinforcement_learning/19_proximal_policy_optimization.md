# Module 11.19: Proximal Policy Optimization (PPO)

---

## 1. Intuition & 101 Motivation

In Chapter 11.18, we studied **Trust Region Policy Optimization (TRPO)**, which provided rigorous mathematical guarantees of monotonic policy improvement by enforcing an average KL divergence constraint $\bar{D}_{\text{KL}}(\pi_{\text{old}} \parallel \pi) \le \delta$.

However, TRPO's heavy reliance on second-order optimization introduces severe practical drawbacks:
- It requires **Conjugate Gradient (CG)** loops and **Hessian-Vector Products**, adding architectural complexity.
- It cannot easily share parameters between the Actor and Critic trunks.
- It is incompatible with modern deep architectures like Transformers, LSTMs, and complex multi-head networks trained via standard backpropagation.

In 2017, John Schulman et al. at OpenAI introduced **Proximal Policy Optimization (PPO)**. PPO asked a simple question: *Can we achieve the stability and sample efficiency of TRPO using simple, first-order stochastic gradient descent (Adam)?*

The answer is the **PPO Clipped Surrogate Objective**: an elegant mathematical formulation that clips the probability ratio $r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$ to the interval $[1 - \epsilon, 1 + \epsilon]$, creating a zero-gradient safety plateau that physically prevents destructive policy updates.

Today, PPO is the undisputed **workhorse algorithm of modern deep reinforcement learning**, powering breakthroughs from OpenAI Five (Dota 2) to **Reinforcement Learning from Human Feedback (RLHF)** in ChatGPT and frontier reasoning models.

```
+-----------------------------------------------------------------------------------------+
|                                    PPO-CLIP OBJECTIVE                                   |
|                                                                                         |
|       Objective L^CLIP                                                                  |
|             ^                                                                           |
|             |          Positive Advantage (A > 0)                                       |
|  (1+eps)*A  +                 /-------------------------  (Zero gradient plateau!)      |
|             |                /                                                          |
|             |               /                                                           |
|             |              /                                                            |
|          0  +-------------+--------+--------+-------------> Probability Ratio r         |
|             |            1-eps    1.0      1+eps                                        |
|             |                                                                           |
|             |          Negative Advantage (A < 0)                                       |
|             |    --------------\                                                        |
|  (1-eps)*A  +  (Zero grad)      \                                                       |
|             |                    \                                                      |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Probability Ratio

Let $\pi_{\boldsymbol{\theta}_{\text{old}}}$ be the policy parameters used to collect rollout experience.
For any candidate parameter vector $\boldsymbol{\theta}$, the **probability ratio** at time step $t$ is:

$$r_t(\boldsymbol{\theta}) \triangleq \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$$

Notice that when $\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{old}}$, the ratio is identically unity: $r_t(\boldsymbol{\theta}_{\text{old}}) = 1.0$.

The unconstrained surrogate objective is:
$$L^{\text{CPI}}(\boldsymbol{\theta}) \triangleq \hat{\mathbb{E}}_t \left[ \frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\boldsymbol{\theta}) \hat{A}_t \right]$$
where CPI stands for *Conservative Policy Iteration*. Without constraints, maximizing $L^{\text{CPI}}$ leads to excessively large steps and catastrophic policy collapse.

---

### 2.2 The Clipped Surrogate Objective ($L^{\text{CLIP}}$)

To penalize moves that take $r_t(\boldsymbol{\theta})$ far away from $1$, PPO modifies the objective using a **min-clip** operator:

$$L^{\text{CLIP}}(\boldsymbol{\theta}) \triangleq \hat{\mathbb{E}}_t \left[ \min\left( r_t(\boldsymbol{\theta}) \hat{A}_t, \quad \operatorname{clip}(r_t(\boldsymbol{\theta}), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
where $\epsilon$ is a hyperparameter (typically $\epsilon = 0.20$).

#### Mathematical Case Analysis:

1. **Case 1: Positive Advantage ($\hat{A}_t > 0$):**
   The chosen action yielded a higher return than average. We want to *increase* its probability ($\pi_{\boldsymbol{\theta}} \uparrow \implies r_t > 1$).
   - When $1 \le r_t \le 1 + \epsilon$: The step is within the trust region. The objective is $r_t \hat{A}_t$, providing a positive gradient pushing $\pi_{\boldsymbol{\theta}}$ upward.
   - When $r_t > 1 + \epsilon$: The update has already made the action sufficiently more likely. The objective is clipped to $(1 + \epsilon) \hat{A}_t$.
     $$\left. \nabla_{\boldsymbol{\theta}} L^{\text{CLIP}} \right|_{r_t > 1 + \epsilon} = \mathbf{0}$$
     The gradient vanishes! The optimizer cannot push the ratio any further, preventing overconfidence.

2. **Case 2: Negative Advantage ($\hat{A}_t < 0$):**
   The chosen action performed worse than expected. We want to *decrease* its probability ($\pi_{\boldsymbol{\theta}} \downarrow \implies r_t < 1$).
   - When $1 - \epsilon \le r_t \le 1$: The objective is $r_t \hat{A}_t$, providing a gradient that decreases $\pi_{\boldsymbol{\theta}}$.
   - When $r_t < 1 - \epsilon$: The action has already been sufficiently suppressed. Because $\hat{A}_t < 0$, multiplying by $(1 - \epsilon)$ gives a larger (less negative) number than multiplying by $r_t < 1 - \epsilon$. The $\min$ operator selects the lower bound $(1 - \epsilon) \hat{A}_t$.
     $$\left. \nabla_{\boldsymbol{\theta}} L^{\text{CLIP}} \right|_{r_t < 1 - \epsilon} = \mathbf{0}$$
     The gradient vanishes! The policy avoids over-correcting and destroying exploratory entropy.

#### The Pessimistic Bound:
Taking the minimum between the unclipped and clipped terms ensures that $L^{\text{CLIP}}(\boldsymbol{\theta})$ forms a **pessimistic lower bound**:
$$L^{\text{CLIP}}(\boldsymbol{\theta}) \le r_t(\boldsymbol{\theta}) \hat{A}_t$$
We only ignore the change when the objective would have become *better* than the bound, never when it becomes worse!

---

### 2.3 The Full Multi-Task Objective

In deep networks, Actor and Critic heads share underlying representation layers. The unified objective maximized via Adam is:

$$\mathcal{L}^{\text{PPO}}(\boldsymbol{\theta}, \boldsymbol{\phi}) \triangleq \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\boldsymbol{\theta}) - c_1 L_t^{\text{VF}}(\boldsymbol{\phi}) + c_2 \mathcal{H}(\pi_{\boldsymbol{\theta}}(\cdot \mid S_t)) \right]$$

where:
1. **Clipped Value Function Loss ($L^{\text{VF}}$):**
   Similar to the policy, value targets can also be clipped around the old value $V_{\boldsymbol{\phi}_{\text{old}}}$:
   $$L_t^{\text{VF}}(\boldsymbol{\phi}) = \max \left( (V_{\boldsymbol{\phi}}(S_t) - V_t^{\text{targ}})^2, \quad (\operatorname{clip}(V_{\boldsymbol{\phi}}(S_t), V_{\boldsymbol{\phi}_{\text{old}}}(S_t) - \epsilon_v, V_{\boldsymbol{\phi}_{\text{old}}}(S_t) + \epsilon_v) - V_t^{\text{targ}})^2 \right)$$
2. **Entropy Bonus ($\mathcal{H}$):**
   $$\mathcal{H}(\pi_{\boldsymbol{\theta}}(\cdot \mid S_t)) = - \sum_{a \in \mathcal{A}} \pi_{\boldsymbol{\theta}}(a \mid S_t) \log \pi_{\boldsymbol{\theta}}(a \mid S_t)$$
3. **Coefficients:** Typically $c_1 \in [0.5, 1.0]$ and $c_2 \in [0.01, 0.05]$.

---

### 2.4 Multi-Epoch Mini-Batch Optimization Loop

Unlike standard policy gradients (which can only perform 1 gradient step per rollout), PPO can perform **multiple epochs of mini-batch gradient descent** on the same collected data:

```
Algorithm: PPO with Clipped Objective
Parameters: Rollout horizon T, Actors N, Epochs K, Mini-batch size M, Clip eps=0.2

Loop forever:
    1. For actor = 1 to N:
         Run policy pi_theta_old for T timesteps in environment
         Compute GAE advantages A_1, ..., A_T using lambda and gamma
    2. Collect all N * T samples into dataset D
    3. Normalize advantages: A_hat = (A - mean(A)) / (std(A) + 1e-8)
    4. For epoch = 1 to K (e.g., K = 4 or 10):
         Shuffle dataset D into mini-batches of size M
         For each mini-batch:
             Compute ratio r_t(theta) = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)
             Compute L^CLIP(theta)
             Compute Value Loss L^VF(phi)
             Compute Entropy H
             Perform Adam gradient update on -L^PPO(theta, phi)
    5. Set theta_old <-- theta
```

---

## 3. Geometric & Physical Interpretation: The Safety Plateaus

In the 1D slice along any parameter direction:
- Unconstrained policy gradients form an unbounded linear slope: $\nabla_\theta J$ pulls the parameters uphill forever.
- PPO's clipping operator introduces **two horizontal plateaus** that bracket the objective:
  - An upper ceiling at $(1 + \epsilon)\hat{A}$ for good actions.
  - A lower floor at $(1 - \epsilon)\hat{A}$ for bad actions.
- As soon as the parameter vector reaches the edge of the trust region ($|r_t - 1| > \epsilon$), the objective flattens completely. The directional derivative vanishes ($\nabla L = \mathbf{0}$), acting like a **frictionless brake** that stops the optimizer from drifting outside the trust zone.

```
       Surrogate Objective
              ^
              |       /-------------- Upper Plateau (grad = 0)
              |      /
              |     /   Active Slope (grad > 0)
              |    /
  ------------+---/------------------> r_t
   (grad = 0) |  1-eps   1.0   1+eps
```

---

## 4. Real-World Analogy: The Speed Governor on a Performance Car

Imagine driving a supercar equipped with an intelligent speed governor:
- You are trying to maintain the optimal racing speed on a dangerous mountain track.
- If you are moving at $90\%$ of optimal speed, the throttle responds linearly to your foot, helping you accelerate.
- If you press the accelerator too aggressively and hit $121\%$ of optimal speed ($r > 1.20$), the governor smoothly cuts engine power to zero ($\nabla_\theta = 0$). You cannot accelerate into a fatal crash, no matter how hard you press the pedal!
- Conversely, if you hit a patch of ice and slide below $80\%$ speed, the system regulates brake pressure to prevent the wheels from locking.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

### 5.1 Problem Setup: The Four Canonical Cases of PPO-Clip
To understand every branch of the min-clip operator down to the exact decimal digit, let us evaluate a mini-batch of 4 transitions with clip parameter $\epsilon = 0.2000 \implies [1 - \epsilon, 1 + \epsilon] = [0.8000, 1.2000]$:

- **Sample 1 (Positive Advantage, Moderate Ratio):**
  Advantage $\hat{A}_1 = +2.0000$, Ratio $r_1 = 1.1000$ (Action improved moderately).
- **Sample 2 (Positive Advantage, Excessive Ratio):**
  Advantage $\hat{A}_2 = +2.0000$, Ratio $r_2 = 1.3500$ (Action probability exploded!).
- **Sample 3 (Negative Advantage, Moderate Ratio):**
  Advantage $\hat{A}_3 = -1.5000$, Ratio $r_3 = 0.9000$ (Action suppressed moderately).
- **Sample 4 (Negative Advantage, Excessive Ratio):**
  Advantage $\hat{A}_4 = -1.5000$, Ratio $r_4 = 0.6500$ (Action probability collapsed!).

We will compute:
1. The unclipped objective: $T_{\text{unclip}} = r \hat{A}$
2. The clipped ratio: $\tilde{r} = \operatorname{clip}(r, 0.8, 1.2)$
3. The clipped objective: $T_{\text{clip}} = \tilde{r} \hat{A}$
4. The final PPO objective: $L^{\text{CLIP}} = \min(T_{\text{unclip}}, T_{\text{clip}})$
5. The effective gradient signal $\frac{\partial L^{\text{CLIP}}}{\partial r}$ (Active vs Clipped).

---

### 5.2 "What Refers to What" Legend Protocol

| Symbol | Mathematical Entity | Hand Walkthrough Meaning |
| :--- | :--- | :--- |
| $\epsilon$ | Clipping Threshold | $\epsilon = 0.2000 \implies [0.8000, 1.2000]$ |
| $r_t$ | Probability Ratio | $\frac{\pi_{\boldsymbol{\theta}}(a_t \mid s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t \mid s_t)}$ |
| $\hat{A}_t$ | Estimated Advantage | $\hat{A} > 0$ (good action), $\hat{A} < 0$ (bad action) |
| $T_{\text{unclip}}$ | Unclipped Surrogate | $r_t \cdot \hat{A}_t$ |
| $\tilde{r}_t$ | Clipped Ratio | $\operatorname{clip}(r_t, 1 - \epsilon, 1 + \epsilon)$ |
| $T_{\text{clip}}$ | Clipped Surrogate | $\tilde{r}_t \cdot \hat{A}_t$ |
| $L^{\text{CLIP}}$ | PPO-Clip Value | $\min(T_{\text{unclip}}, T_{\text{clip}})$ |
| Gradient | Status of $\frac{\partial L}{\partial r}$ | $\hat{A}$ if unclipped; $0.0$ if on plateau |

---

### 5.3 Step-by-Step Hand Calculations: All 4 Cases

#### Sample 1: $\hat{A}_1 = +2.0000, \quad r_1 = 1.1000$
1. Unclipped term:
   $$T_{\text{unclip}} = r_1 \hat{A}_1 = 1.1000 \times (+2.0000) = \mathbf{+2.2000}$$
2. Clipped ratio:
   $$\tilde{r}_1 = \operatorname{clip}(1.1000, 0.8000, 1.2000) = \mathbf{1.1000} \quad (\text{Not clipped})$$
3. Clipped term:
   $$T_{\text{clip}} = 1.1000 \times (+2.0000) = \mathbf{+2.2000}$$
4. Final Objective:
   $$L_1^{\text{CLIP}} = \min(+2.2000, +2.2000) = \mathbf{+2.2000}$$
5. Gradient: $\frac{\partial L}{\partial r_1} = \hat{A}_1 = \mathbf{+2.0000}$ (**ACTIVE GRADIENT** $\checkmark$).

#### Sample 2: $\hat{A}_2 = +2.0000, \quad r_2 = 1.3500$
1. Unclipped term:
   $$T_{\text{unclip}} = r_2 \hat{A}_2 = 1.3500 \times (+2.0000) = \mathbf{+2.7000}$$
2. Clipped ratio:
   $$\tilde{r}_2 = \operatorname{clip}(1.3500, 0.8000, 1.2000) = \mathbf{1.2000} \quad (\text{CLIPPED!})$$
3. Clipped term:
   $$T_{\text{clip}} = 1.2000 \times (+2.0000) = \mathbf{+2.4000}$$
4. Final Objective:
   $$L_2^{\text{CLIP}} = \min(+2.7000, +2.4000) = \mathbf{+2.4000}$$
5. Gradient: The $\min$ chose $T_{\text{clip}}$, which is constant with respect to $r$ ($r > 1.2$).
   $$\frac{\partial L}{\partial r_2} = \mathbf{0.0000} \quad (\textbf{CLIPPED TO ZERO! Safe plateau reached})$$

#### Sample 3: $\hat{A}_3 = -1.5000, \quad r_3 = 0.9000$
1. Unclipped term:
   $$T_{\text{unclip}} = r_3 \hat{A}_3 = 0.9000 \times (-1.5000) = \mathbf{-1.3500}$$
2. Clipped ratio:
   $$\tilde{r}_3 = \operatorname{clip}(0.9000, 0.8000, 1.2000) = \mathbf{0.9000} \quad (\text{Not clipped})$$
3. Clipped term:
   $$T_{\text{clip}} = 0.9000 \times (-1.5000) = \mathbf{-1.3500}$$
4. Final Objective:
   $$L_3^{\text{CLIP}} = \min(-1.3500, -1.3500) = \mathbf{-1.3500}$$
5. Gradient: $\frac{\partial L}{\partial r_3} = \hat{A}_3 = \mathbf{-1.5000}$ (**ACTIVE GRADIENT** $\checkmark$).

#### Sample 4: $\hat{A}_4 = -1.5000, \quad r_4 = 0.6500$
1. Unclipped term:
   $$T_{\text{unclip}} = r_4 \hat{A}_4 = 0.6500 \times (-1.5000) = \mathbf{-0.9750}$$
2. Clipped ratio:
   $$\tilde{r}_4 = \operatorname{clip}(0.6500, 0.8000, 1.2000) = \mathbf{0.8000} \quad (\text{CLIPPED!})$$
3. Clipped term:
   $$T_{\text{clip}} = 0.8000 \times (-1.5000) = \mathbf{-1.2000}$$
4. Final Objective:
   $$L_4^{\text{CLIP}} = \min(-0.9750, -1.2000) = \mathbf{-1.2000}$$
   *Crucial Observation:* Because $\hat{A}_4 < 0$, $-1.2000 < -0.9750$, so the $\min$ correctly picks the lower (more pessimistic) value $-1.2000$!
5. Gradient: Because the $\min$ selected $T_{\text{clip}}$ where $\tilde{r} = 0.80$ is fixed:
   $$\frac{\partial L}{\partial r_4} = \mathbf{0.0000} \quad (\textbf{CLIPPED TO ZERO! Avoids excessive penalty})$$

---

### 5.4 Summary Visual Grid: PPO-Clip Decision Matrix

| Sample | Advantage $\hat{A}$ | Ratio $r$ | Unclipped $r\hat{A}$ | Clipped $\tilde{r}\hat{A}$ | Final $L^{\text{CLIP}}$ | Selected by Min | Gradient $\frac{\partial L}{\partial r}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $+2.0000$ | $1.1000$ | $+2.2000$ | $+2.2000$ | $\mathbf{+2.2000}$ | Both | $\mathbf{+2.0000}$ (Active) |
| **2** | $+2.0000$ | $1.3500$ | $+2.7000$ | $+2.4000$ | $\mathbf{+2.4000}$ | Clipped | $\mathbf{0.0000}$ (Clipped) |
| **3** | $-1.5000$ | $0.9000$ | $-1.3500$ | $-1.3500$ | $\mathbf{-1.3500}$ | Both | $\mathbf{-1.5000}$ (Active) |
| **4** | $-1.5000$ | $0.6500$ | $-0.9750$ | $-1.2000$ | $\mathbf{-1.2000}$ | Clipped | $\mathbf{0.0000}$ (Clipped) |

**Batch Mean PPO-Clip Objective:**
$$\bar{L}^{\text{CLIP}} = \frac{2.2000 + 2.4000 - 1.3500 - 1.2000}{4} = \frac{2.0500}{4} = \mathbf{+0.5125}$$

Every single arithmetic operation matches PyTorch execution to exact machine precision!

---

## 6. Solved Illustrations

### Illustration 1: Why the $\min$ Operator is Mandatory
**Problem:**
Why can't we simply use $\tilde{L}(\boldsymbol{\theta}) = \operatorname{clip}(r_t(\boldsymbol{\theta}), 1 - \epsilon, 1 + \epsilon) \hat{A}_t$ without the outer $\min$?
**Solution:**
Consider Sample 4 ($\hat{A} = -1.5$, $r = 0.65$):
- If the policy took a disastrous action, decreasing its probability to $r = 0.65$ made the policy significantly *better* (return increased from $-1.5$ to $-0.975$).
- If we only used $\operatorname{clip}(r) \hat{A}$, the objective would be $(0.80)(-1.5) = -1.20$.
- More dangerously, consider an update where $r$ becomes very large ($r = 2.0$) on an action with negative advantage ($\hat{A} = -10.0$).
  Under $\operatorname{clip}(r) \hat{A}$, the objective would be $(1.2)(-10.0) = -12.0$.
  Under unclipped $r \hat{A}$, the objective is $(2.0)(-10.0) = -20.0$.
  The outer $\min$ chooses $\min(-20.0, -12.0) = -20.0$!
  The $\min$ ensures that if a policy change produces a **catastrophic collapse**, the full unclipped penalty $-20.0$ is maintained, allowing large negative gradients to rapidly steer the policy back to safety! $\blacksquare$

---

## 7. Deep Learning Connection & Modern Applications

### 1. The Core of ChatGPT / RLHF (InstructGPT - Ouyang et al., NeurIPS 2022)
In Reinforcement Learning from Human Feedback (RLHF), an LLM generates a response $y$ to prompt $x$. The reward model outputs scalar score $R(x, y)$.
The PPO objective aligns the LLM while adding a token-level KL penalty from the base SFT model:
$$\mathcal{L}_{\text{RLHF}}(\boldsymbol{\theta}) = \mathbb{E}_{(x, y)} \left[ r_t(\boldsymbol{\theta}) \hat{A}_t - \beta D_{\text{KL}}(\pi_{\boldsymbol{\theta}} \parallel \pi_{\text{ref}}) \right]$$
PPO-Clip ensures the LLM's token distribution does not drift into hallucination or gibberish.

### 2. DeepSeek-R1 Predecessor: PPO vs GRPO
Before inventing Group Relative Policy Optimization (GRPO - Chapter 11.28), DeepSeek initially aligned its reasoning models using PPO with GAE. GRPO maintains the exact same PPO-Clip surrogate loss:
$$\min\left( \frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}, \operatorname{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) \hat{A} \right)$$
proving that the PPO-Clip operator remains foundational even in frontier reasoning models!

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. Exact numerical verification of the Part 5 hand calculations:
   - Evaluates all 4 cases of PPO-Clip
   - Verifies clipped values $L_1 = 2.20, L_2 = 2.40, L_3 = -1.35, L_4 = -1.20$
   - Computes batch mean $+0.5125$ and verifies autograd gradients matching machine precision to $< 10^{-14}$.
2. Complete, production-grade PPO Actor-Critic agent with GAE, multi-epoch mini-batch SGD, and entropy regularization tested on Inverted Pendulum.

See implementation in:
[`11_reinforcement_learning/code/19_proximal_policy_optimization.py`](./code/19_proximal_policy_optimization.py)
