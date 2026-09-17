## 1. Intuition & 101 Motivation



Historically, Large Language Models (LLMs) were trained almost exclusively on the **Next-Token Prediction Objective** (autoregressive log-likelihood). Given a prompt $x$ and a sequence of tokens $y = (y_1, \dots, y_T)$, the model maximized $\sum_{t} \log P(y_t \mid x, y_{<t})$.

However, there is a fundamental misalignment: **predicting the next token on the internet does not automatically make an AI helpful, honest, or harmless**. The internet contains toxic text, contradictory statements, and unhelpful noise. An LLM trained purely on internet text is an uncontrollable mimic.



The core insight of **Reinforcement Learning from Human Feedback (RLHF)** is that **humans can RANK much more easily than they can AUTHOR**. It takes a human expert 30 minutes to write a perfect essay from scratch, but it only takes them 1 minute to read two candidate essays and confidently say, "Essay A is better than Essay B."



RLHF harnesses this asymmetry by converting human preferences into a mathematical reward signal, which is then used to optimize the LLM via Proximal Policy Optimization (PPO).



### The Three Phases of RLHF



1.  **Phase 1: Supervised Fine-Tuning (SFT)**

    *   **Data Collection:** Human experts manually write perfect demonstrations (responses) to a curated set of prompts.

    *   **Training Objective:** The base pre-trained LLM is fine-tuned on this dataset using standard Cross-Entropy Loss (autoregressive next-token prediction).

    *   **Outcome:** The model learns to follow instructions, adopt a conversational tone, and format its outputs correctly. It transitions from a base completion model to an instruction-following model. However, it is not yet perfectly aligned, as it may still hallucinate or generate unsafe content.



2.  **Phase 2: Reward Model (RM) Training**

    *   **Data Collection:** The SFT model generates multiple responses (e.g., $y_1$ and $y_2$) to a single prompt $x$.

    *   **Human Annotation:** Human rankers evaluate the responses and indicate their preference (e.g., $y_w \succ y_l$, meaning $y_w$ is better than $y_l$).

    *   **Training Objective:** A separate neural network, the Reward Model (initialized from the SFT model but with a scalar output head), is trained to predict these preferences. The training uses the Bradley-Terry model to convert scalar reward predictions into probabilities.

    *   **Outcome:** We obtain a proxy for human judgment—a differentiable function $r_\psi(x, y)$ that assigns a high scalar score to responses humans would like, and a low score to responses humans would dislike.



3.  **Phase 3: Reinforcement Learning (RL) via PPO**

    *   **Setup:** We formulate the text generation process as a Markov Decision Process (MDP). The environment provides no intermediate rewards; the only reward comes at the end of generation from the frozen Reward Model.

    *   **Optimization:** We train an Actor model $\pi_\theta$ (initialized from the SFT model) to maximize the expected reward using Proximal Policy Optimization (PPO).

    *   **Regularization:** To prevent the Actor from exploiting the Reward Model (reward hacking), we heavily penalize deviations from the original SFT model using a per-token Kullback-Leibler (KL) divergence penalty.



```text

+----------------------------------------------------------------------------------------------------+

|                                    THE 3-PHASE RLHF PIPELINE                                       |

|                                                                                                    |

|  PHASE 1: Supervised Fine-Tuning (SFT)                                                             |

|  [Human writes perfect demonstrations] -> [Train LLM via standard cross-entropy]                   |

|  * Goal: Teach the model to follow basic instructions and format.                                  |

|                                                                                                    |

|  PHASE 2: Reward Model (RM) Training                                                               |

|  [LLM generates Pairs (y_1, y_2)] -> [Human ranks y_w > y_l] -> [Train RM via Bradley-Terry]       |

|  * Goal: Distill human intuition into a differentiable scalar reward function.                     |

|                                                                                                    |

|  PHASE 3: RL via PPO                                                                               |

|  [Train Actor against frozen RM, regularized by frozen Ref Model to prevent reward hacking]        |

|  * Goal: Optimize the language model to maximize the scalar reward while staying coherent.         |

+----------------------------------------------------------------------------------------------------+



+----------------------------------------------------------------------------------------------------+

|                                    THE 4-MODEL PPO RL LOOP                                         |

|                                                                                                    |

|       (Prompt x)                                                                                   |

|           |                                                                                        |

|           v                                                                                        |

|    +-------------+      (y_t)      +-------------------+                                           |

|    | ACTOR (π_θ) | --------------->| REWARD MODEL (r_ψ)| --> Reward r(x,y) (Only at t=T)           |

|    | (Trainable) |                 | (Frozen)          |                                           |

|    +-------------+                 +-------------------+                                           |

|           |                                                                                        |

|           | KL Penalty δ_t                                                                         |

|           v                                                                                        |

|    +-------------+                 +-------------------+                                           |

|    | REF (π_ref) |                 | CRITIC (V_φ)      | --> Value Estimate V(s_t)                 |

|    | (Frozen)    |                 | (Trainable)       |                                           |

|    +-------------+                 +-------------------+                                           |

+----------------------------------------------------------------------------------------------------+

```

---



## 2. Rigorous Mathematical Formulation



The transition from Phase 2 (Reward Model Training) to Phase 3 (PPO Optimization) requires a robust mathematical framework that links human preferences to reinforcement learning objectives. We will build this up step-by-step.



### 2.1 The Bradley-Terry Preference Model



In Phase 2, we train a Reward Model $r_{\psi}$ parameterised by weights $\psi$ that maps a prompt $x$ and response $y$ to a scalar reward $r \in \mathbb{R}$. We collect a dataset of human preferences $\mathcal{D} = \{(x, y_w, y_l)\}$, where $y_w$ is the winning response and $y_l$ is the losing response.



To map scalar rewards into a probability distribution over preferences, we use the **Bradley-Terry (BT) model**:

$$P(y_w \succ y_l \mid x) = \frac{\exp(r_{\psi}(x, y_w))}{\exp(r_{\psi}(x, y_w)) + \exp(r_{\psi}(x, y_l))}$$



Dividing the numerator and denominator by $\exp(r_{\psi}(x, y_l))$, this simplifies to the standard logistic sigmoid $\sigma(\cdot)$ over the reward difference:

$$P(y_w \succ y_l \mid x) = \sigma(r_{\psi}(x, y_w) - r_{\psi}(x, y_l))$$





This formulation elegantly connects arbitrary scalar reward values to a bounded probability space $[0, 1]$, enabling the use of binary cross-entropy loss for training.



### 2.2 The RLHF Policy Optimization Objective



Once the Reward Model $r_{\psi}$ is frozen, we enter Phase 3. Our goal is to find an optimal policy $\pi_{\theta}$ that maximizes the expected reward while remaining close to the original SFT model (Reference Policy $\pi_{\text{ref}}$) to prevent degenerate outputs.



The unconstrained reward objective would be $\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x)} [r_{\psi}(x, y)]$.

However, optimizing this directly leads to **reward hacking**—the LLM outputs gibberish that exploits blind spots in the reward model. We regularize this using a KL-divergence penalty with coefficient $\beta$:



$$\max_{\theta} J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x)} \left[ r_{\psi}(x, y) - \beta \mathbb{D}_{\text{KL}}(\pi_{\theta}(\cdot \mid x) \parallel \pi_{\text{ref}}(\cdot \mid x)) \right]$$



The coefficient $\beta$ dictates the strength of the regularization. A very high $\beta$ forces the Actor to perfectly mimic the Reference Model, ignoring the reward. A very low $\beta$ allows the Actor to maximize the reward at the cost of generating incoherent text.



### 2.3 Token-Level Reward Decomposition



While $r_{\psi}(x, y)$ is only computed at the *end* of the sequence (t=T), the KL penalty can be decomposed per-token.

For a sequence $y = (y_1, \dots, y_T)$, the KL divergence expands via the chain rule of probability.

We define the token-level KL penalty as $\delta_t = \log \pi_{\theta}(y_t \mid \dots) - \log \pi_{\text{ref}}(y_t \mid \dots)$.

The augmented per-token reward $\tilde{r}_t$ becomes:

$$\tilde{r}_t = \begin{cases} - \beta \delta_t & \text{if } t < T \ r_{\psi}(x, y) - \beta \delta_T & \text{if } t = T \end{cases}$$





This transforms the problem from a single-step bandit problem into a sequential Markov Decision Process (MDP).



### 2.4 Token-Level GAE & PPO-Clipped Update



With the per-token augmented rewards $\tilde{r}_t$, we train a Critic network $V_{\phi}(x, y_{<t})$ to estimate the expected future return.

We compute the Generalized Advantage Estimation (GAE) at each token $t$:

$$\delta_t^V = \tilde{r}_t + \gamma V_{\phi}(x, y_{\le t}) - V_{\phi}(x, y_{<t})$$



$$\hat{A}_t = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}^V$$





We then apply the standard PPO clipped surrogate objective to update the Actor parameters $\theta$:

$$L^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \min\left( \frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}} \hat{A}_t, \text{clip}\left( \frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon \right) \hat{A}_t \right) \right]$$





### 2.7 The PPO Value Function (Critic) Loss Derivation

The Actor network optimization in PPO requires a robust baseline to reduce the variance of the policy gradient. This is provided by the Critic network, parameterized by $\phi$, which estimates the value function $V_{\phi}(s_t)$.

In the RLHF setting, the state $s_t$ at time $t$ encapsulates the prompt $x$ and the generated tokens up to that point $y_{<t}$. The Critic's objective is to minimize the Mean Squared Error (MSE) between its prediction and the empirical discounted return.



Let $G_t$ be the exact discounted return from timestep $t$:

$$ G_t = \sum_{k=0}^{T-t} \gamma^k \tilde{r}_{t+k} $$



The unclipped value function loss is simply the MSE:

$$ L^{\text{VF}}_{\text{unclipped}}(\phi) = \mathbb{E} \left[ \frac{1}{2} (V_{\phi}(s_t) - G_t)^2 \right] $$



However, standard PPO implementations (such as the one utilized in OpenAI's InstructGPT) often apply a clipping mechanism to the value function update as well, to prevent the Critic from taking steps that are too large, which could destabilize the GAE calculation for the Actor.

Let $V_{\phi_{\text{old}}}(s_t)$ be the value prediction from the previous epoch. The clipped value prediction is:

$$ V_{\text{clipped}}(s_t) = V_{\phi_{\text{old}}}(s_t) + \text{clip}\left( V_{\phi}(s_t) - V_{\phi_{\text{old}}}(s_t), -\epsilon, \epsilon \right) $$



The clipped value function loss then becomes the maximum of the unclipped and clipped MSE:

$$ L^{\text{VF}}(\phi) = \mathbb{E} \left[ \max \left( \frac{1}{2} (V_{\phi}(s_t) - G_t)^2, \frac{1}{2} (V_{\text{clipped}}(s_t) - G_t)^2 \right) \right] $$



This ensures that if the Critic's prediction moves further away from the target $G_t$ by more than $\epsilon$, the gradient is heavily penalized or zeroed out, maintaining stability in the coupled Actor-Critic system.



### 2.8 The Entropy Bonus

To encourage exploration and prevent the Actor policy from prematurely collapsing into a deterministic argmax generator (which degrades text diversity and quality), an entropy bonus is added to the PPO objective.

The Shannon entropy of the policy distribution at step $t$ is:

$$ \mathcal{H}(\pi_{\theta}(\cdot \mid s_t)) = - \sum_{y_t \in \mathcal{V}} \pi_{\theta}(y_t \mid s_t) \log \pi_{\theta}(y_t \mid s_t) $$

where $\mathcal{V}$ is the vocabulary size.

The total PPO objective maximized by Adam is thus:

$$ \mathcal{L}^{\text{PPO}}(\theta, \phi) = \mathbb{E} \left[ L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\phi) + c_2 \mathcal{H}(\pi_{\theta}(\cdot \mid s_t)) \right] $$

where $c_1$ is the value coefficient (typically $0.5$) and $c_2$ is the entropy coefficient (typically $0.01$).





### 2.5 First-Principles Mathematical Derivations



### 2.6 Advanced Mathematical Corollaries



In addition to the primary derivations above, it is crucial to understand the Fisher Information Matrix (FIM) implications of the KL penalty.

When we constrain the policy update using $\mathbb{D}_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{ref}})$, we are effectively transforming the Euclidean gradient descent into Natural Gradient Descent (NGD).

The KL divergence can be Taylor-expanded around the current policy parameters $\theta$:

$$ \mathbb{D}_{\text{KL}}(\pi_{\theta} \parallel \pi_{\theta + \Delta \theta}) \approx \frac{1}{2} \Delta \theta^	op \mathbf{F}(\theta) \Delta \theta $$

where $\mathbf{F}(\theta)$ is the Fisher Information Matrix:

$$ \mathbf{F}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}} \left[ 



abla_{\theta} \log \pi_{\theta}(y \mid x) 

abla_{\theta} \log \pi_{\theta}(y \mid x)^	op \right] $$

By regularizing against the reference model, PPO implicitly approximates this natural gradient trust region. This ensures that a small step in parameter space does not cause a catastrophic shift in the actual probability distribution of the text generated.

If $\beta$ is too small, the Fisher Information constraint is weak, and the model takes steps that are too large in the distribution space, destroying the linguistic coherence learned during SFT.

If $\beta$ is too large, the constraint is overwhelmingly strong, and the model cannot update its parameters enough to maximize the reward.

This is why adaptive $\beta$ scaling (or using the clipped surrogate objective of PPO instead of a pure penalty) is favored in modern RLHF systems.



Furthermore, consider the variance of the GAE advantage estimator:

$$ \hat{A}_t = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}^V $$



The bias-variance tradeoff is heavily controlled by $\lambda$. In RLHF, episodes (generations) are typically short (e.g., 512 or 1024 tokens). The reward is entirely delayed to the end.

If $\lambda = 1.0$, the estimator is equivalent to Monte Carlo returns. It is unbiased but has massive variance due to the stochasticity of the entire sequence generation.

If $\lambda = 0.0$, the estimator is a 1-step TD error. It has low variance but high bias, relying heavily on the Critic network $V_{\phi}$ which might be inaccurate early in training.

Empirically, $\lambda = 0.95$ is used to exponentially decay the weight of distant future rewards, smoothing out the credit assignment across tokens.



The following derivations provide the rigorous theoretical bedrock for the equations presented above. We will meticulously unpack the Bradley-Terry update, the optimal closed-form policy (which leads to DPO), and the per-token KL decomposition.



#### Derivation 11.27.1: The Bradley-Terry Reward Model Update

====================================================================================================



Problem Statement:

Given human preference data $y_w \succ y_l$, we model the probability of preference using the Bradley-Terry-Luce model parameterized by a neural network reward model $r_{\psi}$:

$P(y_w \succ y_l \mid x) = \sigma(r_{\psi}(x, y_w) - r_{\psi}(x, y_l))$

Derive the Binary Cross-Entropy (BCE) loss for the Reward Model, and compute its gradients with respect to the individual scalar reward predictions to show how it pushes $y_w$ up and $y_l$ down symmetrically.



Step 1: Formulate the Negative Log-Likelihood (BCE Loss)

The likelihood of the observed preference dataset $\mathcal{D}$ under our parameterized model is the product of individual probabilities. To ensure numerical stability and transform the product into a sum, we take the negative logarithm, yielding the Binary Cross-Entropy (BCE) loss:

$$L_{\text{RM}}(\psi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log P(y_w \succ y_l \mid x) \right]$$

Substitute the Bradley-Terry probability model into the objective:

$$L_{\text{RM}}(\psi) = - \mathbb{E} \left[ \log \left( \sigma(r_{\psi}(x, y_w) - r_{\psi}(x, y_l)) \right) \right]$$



Step 2: Simplify via the Margin Formulation

Let us define the reward margin $\Delta r$ as the difference between the predicted reward for the winning response and the predicted reward for the losing response:

$$\Delta r = r_{\psi}(x, y_w) - r_{\psi}(x, y_l)$$



The loss function for a single training sample gracefully simplifies to:

$$L_{\text{RM}} = - \log \sigma(\Delta r)$$





Step 3: Differentiate with respect to the Margin $\Delta r$

We require the derivative of the logarithm of the logistic sigmoid function.

Recall the fundamental derivative property of $\sigma(z)$:

$$ \frac{d \sigma(z)}{dz} = \sigma(z)(1 - \sigma(z))$$

Therefore, applying the chain rule to the natural logarithm:

$$ \frac{d}{dz} \log \sigma(z) = \frac{1}{\sigma(z)} \frac{d \sigma(z)}{dz} = \frac{1}{\sigma(z)} \left[ \sigma(z)(1 - \sigma(z)) \right] = 1 - \sigma(z)$$

Applying this identity to our loss function:

$$ \frac{\partial L_{\text{RM}}}{\partial \Delta r} = \frac{\partial}{\partial \Delta r} (- \log \sigma(\Delta r)) = - (1 - \sigma(\Delta r))$$



Step 4: Gradient with respect to the Winning Response $r_w$

Let $r_w = r_{\psi}(x, y_w)$. We want to evaluate how the loss changes when the raw score of the winning response changes.

First, compute the partial derivative of the margin with respect to $r_w$:

$$ \frac{\partial \Delta r}{\partial r_w} = \frac{\partial}{\partial r_w} (r_w - r_l) = 1$$

Next, employ the chain rule to find the gradient of the loss:

$$ \frac{\partial L_{\text{RM}}}{\partial r_w} = \frac{\partial L_{\text{RM}}}{\partial \Delta r} \frac{\partial \Delta r}{\partial r_w} = - (1 - \sigma(\Delta r)) \cdot 1 = - (1 - \sigma(\Delta r))$$

Because the output of the sigmoid function $\sigma(\Delta r)$ is strictly bounded in $(0, 1)$, the term $(1 - \sigma(\Delta r))$ is always strictly positive. Consequently, the gradient is strictly negative.

During gradient descent, we update the parameters by subtracting a fraction of the gradient: $\psi \leftarrow \psi - \alpha \nabla L$. Subtracting a negative number means we ADD to $r_w$. Thus, the optimization correctly pushes the score of the winning response UP.



Step 5: Gradient with respect to the Losing Response $r_l$

Let $r_l = r_{\psi}(x, y_l)$. Evaluate the partial derivative of the margin:

$$ \frac{\partial \Delta r}{\partial r_l} = \frac{\partial}{\partial r_l} (r_w - r_l) = -1$$

Employ the chain rule:

$$ \frac{\partial L_{\text{RM}}}{\partial r_l} = \frac{\partial L_{\text{RM}}}{\partial \Delta r} \frac{\partial \Delta r}{\partial r_l} = - (1 - \sigma(\Delta r)) \cdot (-1) = + (1 - \sigma(\Delta r))$$

This gradient is strictly positive. During gradient descent, subtracting a positive gradient means we SUBTRACT from $r_l$. Thus, the optimization correctly pushes the score of the losing response DOWN.



Step 6: Analysis of Symmetry and Logistic Saturation

Observe that the gradients are perfectly equal in magnitude and precisely opposite in sign:

$$ \frac{\partial L_{\text{RM}}}{\partial r_w} = - \frac{\partial L_{\text{RM}}}{\partial r_l}$$

This establishes that the reward model operates symmetrically: for every unit it pushes the winning score up, it pushes the losing score down by the exact same amount.

Furthermore, analyze the asymptotic behavior. As the margin $\Delta r \to \infty$ (meaning the model is highly confident in its correct prediction), $\sigma(\Delta r) \to 1$. Consequently, the gradient magnitude $(1 - \sigma(\Delta r)) \to 0$. This logistic saturation is a critical safety mechanism: it prevents the model from endlessly increasing the magnitude of its weights (exploding gradients) once a clear, decisive margin has been established.

====================================================================================================





#### Derivation 11.27.2: Closed-Form Optimal RLHF Policy (DPO Foundation)

====================================================================================================



Problem Statement:

Find the globally optimal policy $\pi^*$ that maximizes the regularized RLHF objective analytically.

$$\max_{\pi} \mathbb{E}_{y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$$

subject to the valid probability distribution constraint $\sum_y \pi(y \mid x) = 1$.

Show how this derivation leads to the Direct Preference Optimization (DPO) objective by canceling the intractable partition function.



Step 1: Formulate the Constrained Optimization Lagrangian

To optimize a functional subject to an equality constraint, we rely on the method of Lagrange multipliers. We introduce a multiplier $\lambda$ corresponding to the requirement that the probabilities must sum to 1.

$$L(\pi, \lambda) = \sum_y \pi(y \mid x) \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] + \lambda \left( 1 - \sum_y \pi(y \mid x) \right)$$

Here, we have explicitly expanded the expectation $\mathbb{E}_{y \sim \pi}$ into a sum over all possible sequences $y$.



Step 2: Functional Derivative with respect to $\pi(y \mid x)$

We take the partial derivative of the Lagrangian with respect to the probability of a specific, arbitrary sequence $y$, and set it to zero to find the critical points.

We will require the product rule and chain rule for the derivative of $x \log x$:

$$ \frac{d}{dx} (x \log x) = \log x + x \cdot \frac{1}{x} = \log x + 1$$

Applying this to our Lagrangian:

$$ \frac{\partial L}{\partial \pi(y \mid x)} = r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \beta \pi(y \mid x) \left( \frac{\pi_{\text{ref}}(y \mid x)}{\pi(y \mid x)} \right) \left( \frac{1}{\pi_{\text{ref}}(y \mid x)} \right) - \lambda = 0$$

The complex third term elegantly simplifies:

$$- \beta \pi(y \mid x) \cdot \frac{1}{\pi(y \mid x)} = - \beta$$

Substituting this back, we obtain the simplified stationarity condition:

$$r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \beta - \lambda = 0$$



Step 3: Algebraic Isolation of the Optimal Policy Distribution

We rearrange the stationarity condition to isolate the logarithm term:

$$\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = r(x, y) - \beta - \lambda$$

Divide the entire equation by the regularization coefficient $\beta$:

$$\log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \frac{r(x, y)}{\beta} - \frac{\beta + \lambda}{\beta}$$

Exponentiate both sides of the equation to eliminate the logarithm:

$$ \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \exp\left( \frac{r(x, y)}{\beta} \right) \exp\left( - \frac{\beta + \lambda}{\beta} \right)$$

Multiply by $\pi_{\text{ref}}(y \mid x)$ to solve for $\pi^*$:

$$\pi^*(y \mid x) = \pi_{\text{ref}}(y \mid x) \exp\left( \frac{r(x, y)}{\beta} \right) \exp\left( - \frac{\beta + \lambda}{\beta} \right)$$



Step 4: Resolution of the Lagrange Multiplier via Normalization

We must ensure that our solution satisfies the initial constraint: $\sum_y \pi^*(y \mid x) = 1$.

Summing our expression over all possible sequences $y$:

$$1 = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{r(x, y)}{\beta} \right) \exp\left( - \frac{\beta + \lambda}{\beta} \right)$$

Crucially, the term $\exp\left( - \frac{\beta + \lambda}{\beta} \right)$ acts as a constant with respect to $y$, allowing us to factor it out of the summation:

$$1 = \exp\left( - \frac{\beta + \lambda}{\beta} \right) \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{r(x, y)}{\beta} \right)$$

Let us define the partition function $Z(x)$ as the sum over all possible responses, weighted by their exponentially scaled reward under the reference model:

$$Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{r(x, y)}{\beta} \right)$$

Substituting $Z(x)$ back into the normalization equation:

$$1 = \exp\left( - \frac{\beta + \lambda}{\beta} \right) Z(x) \implies \exp\left( - \frac{\beta + \lambda}{\beta} \right) = \frac{1}{Z(x)}$$



Step 5: The Final Form of the Optimal Policy

Substitute the derived normalization constant back into the equation for $\pi^*$:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{r(x, y)}{\beta} \right)$$

This represents a profound theoretical result: the optimal RLHF policy is simply the original SFT reference policy, exponentially re-weighted by the reward landscape, and normalized by the partition function $Z(x)$.



Step 6: The Foundational Derivation of DPO

We can invert the relationship derived above to express the reward $r(x,y)$ entirely as a function of policy probabilities.

$$\exp\left( \frac{r(x, y)}{\beta} \right) = \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} Z(x)$$

Taking the natural logarithm and multiplying by $\beta$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

If we substitute this optimal implicit reward formulation into the Bradley-Terry preference loss $\sigma(r_w - r_l)$, we compute the reward margin:

$$r_w - r_l = \left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right)$$

The intractable partition function $Z(x)$, which is computationally impossible to evaluate directly as it requires a sum over all valid text sequences in the universe, perfectly cancels out!

$$r_w - r_l = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

This beautiful cancellation forms the absolute bedrock of Direct Preference Optimization (DPO), allowing engineers to bypass the explicit reward model phase entirely and optimize the policy directly on human preferences.

====================================================================================================





#### Derivation 11.27.3: Per-Token KL Divergence Decomposition

====================================================================================================



Problem Statement:

Prove mathematically that the sequence-level KL divergence between the Actor $\pi_{\theta}$ and Reference $\pi_{\text{ref}}$ can be decomposed into an exact sum of expected per-token log-probability ratios. Show how this integrates into an augmented reward MDP, enabling token-by-token reinforcement learning.



Step 1: Define the Sequence-Level KL Divergence

Consider a full trajectory (a complete sequence of text) $y = (y_1, \dots, y_T)$.

By the standard definition of relative entropy, the KL divergence is the expectation over the policy distribution of the log-ratio of the probability distributions:

$$\mathbb{D}_{\text{KL}}(\pi_{\theta}(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)) = \mathbb{E}_{y \sim \pi_{\theta}} \left[ \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$$



Step 2: Factorize via the Autoregressive Chain Rule

Large language models generate text autoregressively, token by token. The joint probability of generating a complete sequence is strictly equivalent to the product of the conditional probabilities of generating each individual token, given all preceding tokens.

For the Actor policy:

$$\pi_{\theta}(y \mid x) = \prod_{t=1}^T \pi_{\theta}(y_t \mid x, y_{<t})$$



For the Reference policy:

$$\pi_{\text{ref}}(y \mid x) = \prod_{t=1}^T \pi_{\text{ref}}(y_t \mid x, y_{<t})$$





Step 3: Substitute the Factorizations into the Log-Ratio

We replace the sequence-level probabilities with their token-level autoregressive products:

$$\log \frac{\pi_{\theta}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \log \left( \frac{\prod_{t=1}^T \pi_{\theta}(y_t \mid x, y_{<t})}{\prod_{t=1}^T \pi_{\text{ref}}(y_t \mid x, y_{<t})} \right)$$

Group the products into a single product of ratios:

$$= \log \left( \prod_{t=1}^T \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} \right)$$



Step 4: Apply Fundamental Logarithmic Properties

A core property of logarithms is that the logarithm of a product is exactly equivalent to the sum of the logarithms: $\log(A \cdot B) = \log A + \log B$.

Applying this to our sequence:

$$\log \left( \prod_{t=1}^T \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} \right) = \sum_{t=1}^T \log \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$$



Step 5: Formulate the Token-Level KL Penalty

Let us define $\delta_t$ to represent the precise per-token logarithmic deviation of the Actor from the Reference model.

$$\delta_t = \log \pi_{\theta}(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t})$$



Substituting this definition back into our expectation, we conclude that the full sequence KL divergence is simply the expectation over the sum of these individual $\delta_t$ terms:

$$\mathbb{D}_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_{\theta}} \left[ \sum_{t=1}^T \delta_t \right]$$



Step 6: Construction of the Augmented Reward MDP

In reinforcement learning, the agent receives an environmental reward $r_t^{\text{env}}$. In standard RLHF, this environmental reward is entirely sparse: it is $0$ for all intermediate tokens $t < T$, and is equal to the Bradley-Terry Reward Model score $r_{\psi}(x, y)$ only at the terminal token $t = T$.

However, optimizing a sparse reward is highly inefficient and prone to reward hacking. To rectify this, we integrate the token-level KL penalty $\delta_t$ directly into a dense, step-by-step **augmented reward** signal:

$$\tilde{r}_t = r_t^{\text{env}} - \beta \delta_t$$



Explicitly evaluating this for different timesteps:

For intermediate tokens $t < T$: $\tilde{r}_t = 0 - \beta \delta_t = - \beta \delta_t$

For the terminal token $t = T$: $\tilde{r}_t = r_{\psi}(x, y) - \beta \delta_T$



This mathematical reformulation is profoundly significant: it transforms the sequence-level RLHF objective from a sparse, single-step contextual bandit problem into a standard, dense token-level Markov Decision Process (MDP). This transformation is exactly what permits the deployment of standard temporal-difference actor-critic algorithms, such as Proximal Policy Optimization (PPO), to train language models.

====================================================================================================





---



## 3. Geometric & Physical Interpretation



Imagine a highly complex, rugged mountain landscape where the elevation represents the **Reward Model's score**. The LLM (the Actor) wants to climb to the highest possible peak to maximize its return.

- **Unconstrained RL:** Without a Reference Model (no KL penalty), the LLM will strap on a jetpack and blast off towards bizarre, alien coordinates far off the map. This is "reward hacking." The neural network finds inputs that mathematically yield infinite reward due to adversarial artifacts, but these inputs are actually complete gibberish text in reality.

- **The Anchor (Reference Policy):** The SFT Reference Policy $\pi_{\text{ref}}$ acts like a heavy anchor permanently bolted to a known safe basecamp (the manifold of human-readable text).

- **The Bungee Cord (KL Penalty):** The KL Penalty coefficient $\beta$ acts as a massive elastic bungee cord connecting the Actor to the anchor.

- **Physical Equilibrium:** The optimization process seeks a physical equilibrium. The Actor hikes uphill, pulled by the upward force of the reward gradient. However, as it moves further away from the basecamp, the elastic tension of the KL bungee cord increases. The Actor will eventually stop at a point where the upward pull of the reward exactly balances the downward snap of the bungee cord.



```text

       (High Reward, Gibberish) 

               ^ 

              /|\  <-- Reward Gradient Force

             / | \ 

            /  |  \

           /   O   \  <-- Actor LLM

          /    |    \

         /     |     \ <-- KL Bungee Cord (Tension = β * D_KL)

        /      |      \

-------+-------+-------+-------

       |   Basecamp    |

       |  (Ref Model)  |

       +---------------+

```



---



## 4. Real-World Analogy: Speechwriter & Public Opinion Poll



Imagine a politician (the **Actor**) trying to give a great speech.

- **Phase 1 (SFT):** The politician learns to speak English properly by reading dictionaries and encyclopedias (the Reference Model). They learn grammar, structure, and basic facts.

- **Phase 2 (RM):** The politician tests out multiple draft speeches in front of a focus group. The focus group doesn't write speeches; they just vote on which one sounds better (Bradley-Terry preferences). We train a "Pollster AI" (the Reward Model) to predict how the focus group will vote based on the text of the speech.

- **Phase 3 (PPO):** The politician writes new speeches and consults the internal Pollster AI for a score. However, to ensure the politician doesn't just start repeatedly shouting "FREE ICE CREAM!" (which might temporarily break the pollster's model and score highly), a team of strict grammar teachers (the KL Penalty) stands by. They dock points every time the speech deviates too far from normal, coherent English (the Reference Model). The politician must learn to maximize the pollster's score while keeping the grammar teachers happy.



---



## 5. Prof. Tom Yeh "AI by Hand" Visual Grids



Let's execute the fundamental mechanics of RLHF manually with concrete numerical grids. We will compute the Bradley-Terry Reward Model training for a batch, and then walk through the per-token PPO step for an LLM response.



### Walkthrough 1: Bradley-Terry Reward Model Training



We have a batch of 3 preference pairs. For each, we have a prompt $x$, a winning response $y_w$, and a losing response $y_l$. The frozen LLM backbone outputs scalar logits which are the raw rewards. Let's compute the margin, the predicted probability, the loss, and the gradient multiplier.



Assume our current Reward Model weights $\psi$ produce the following scores:

- Pair 1: $r(y_w) = 2.3$, $r(y_l) = -0.8$

- Pair 2: $r(y_w) = 1.1$, $r(y_l) = 1.5$  (Model gets this wrong initially, assigns higher score to loser)

- Pair 3: $r(y_w) = 0.5$, $r(y_l) = 0.4$



**Computations:**

For Pair 1:

- Margin $\Delta r = 2.3 - (-0.8) = 3.1$

- $\sigma(\Delta r) = \sigma(3.1) = \frac{1}{1 + \exp(-3.1)} = \frac{1}{1 + 0.0450} \approx 0.9569$

- Loss $= -\log(0.9569) \approx 0.0441$

- Grad multiplier for $r_w$: $-(1 - 0.9569) = -0.0431$ (push up slightly)

- Grad multiplier for $r_l$: $+0.0431$ (push down slightly)



For Pair 2 (Model predicts incorrectly):

- Margin $\Delta r = 1.1 - 1.5 = -0.4$

- $\sigma(-0.4) = \frac{1}{1 + \exp(0.4)} = \frac{1}{1 + 1.4918} \approx 0.4013$

- Loss $= -\log(0.4013) \approx 0.9130$

- Grad multiplier for $r_w$: $-(1 - 0.4013) = -0.5987$ (push up strongly)

- Grad multiplier for $r_l$: $+0.5987$ (push down strongly)



For Pair 3:

- Margin $\Delta r = 0.5 - 0.4 = 0.1$

- $\sigma(0.1) = \frac{1}{1 + \exp(-0.1)} \approx 0.5250$

- Loss $= -\log(0.5250) \approx 0.6444$

- Grad multiplier for $r_w$: $-(1 - 0.5250) = -0.4750$

- Grad multiplier for $r_l$: $+0.4750$



```text

========================================================================



| Pair | r(y_w) | r(y_l) | Margin | P(w > l) | Loss   | Grad_w | Grad_l|

========================================================================



|  1   |   2.3  |  -0.8  |   3.1  |  0.9569  | 0.0441 | -0.043 | +0.043|

|  2   |   1.1  |   1.5  |  -0.4  |  0.4013  | 0.9130 | -0.599 | +0.599|

|  3   |   0.5  |   0.4  |   0.1  |  0.5250  | 0.6444 | -0.475 | +0.475|

========================================================================



```

Observe that the model gets the strongest gradient signal (-0.599) from the pair it gets wrong (Pair 2), forcing a large update. It gets a tiny gradient signal (-0.043) from the pair it gets right with high confidence (Pair 1).



### Walkthrough 2: RLHF PPO Token Step & KL Penalty



Prompt: "What is the capital of France?"

The Actor generates 4 tokens: `["The", "capital", "is", "Paris"]`.

We have a KL penalty coefficient $\beta = 0.1$.

Let's compute the token-level KL penalties, the augmented rewards, and the PPO probability ratios.



Assume the Actor $\pi_{\theta}$ and Ref Model $\pi_{\text{ref}}$ assign the following log-probabilities to the generated tokens during the rollout:

- Token 1 ("The"): Actor $\log \pi_{\theta} = -0.2$, Ref $\log \pi_{\text{ref}} = -0.5$

- Token 2 ("capital"): Actor $\log \pi_{\theta} = -0.1$, Ref $\log \pi_{\text{ref}} = -0.1$

- Token 3 ("is"): Actor $\log \pi_{\theta} = -0.4$, Ref $\log \pi_{\text{ref}} = -0.2$

- Token 4 ("Paris"): Actor $\log \pi_{\theta} = -0.1$, Ref $\log \pi_{\text{ref}} = -0.8$



At the end of the sequence, the frozen Reward Model gives a scalar score: $r_{\text{RM}} = 5.0$.



**KL Penalty ($\delta_t = \log \pi_{\theta} - \log \pi_{\text{ref}}$) and Augmented Reward ($\tilde{r}_t = r_t^{\text{env}} - \beta \delta_t$):**

- t=1 ("The"): $\delta_1 = -0.2 - (-0.5) = 0.3$. 

  $\tilde{r}_1 = 0 - 0.1(0.3) = -0.03$

- t=2 ("capital"): $\delta_2 = -0.1 - (-0.1) = 0.0$. 

  $\tilde{r}_2 = 0 - 0.1(0.0) = 0.00$

- t=3 ("is"): $\delta_3 = -0.4 - (-0.2) = -0.2$. 

  $\tilde{r}_3 = 0 - 0.1(-0.2) = +0.02$

- t=4 ("Paris"): $\delta_4 = -0.1 - (-0.8) = 0.7$. 

  $\tilde{r}_4 = 5.0 - 0.1(0.7) = 5.0 - 0.07 = 4.93$



```text

=============================================================================



| t | Token   | Log π_θ | Log π_ref | KL(δ_t) | RM Reward | Aug Reward r_t|

=============================================================================



| 1 | The     |  -0.2   |   -0.5    |   0.3   |    0.0    |    -0.03      |

| 2 | capital |  -0.1   |   -0.1    |   0.0   |    0.0    |     0.00      |

| 3 | is      |  -0.4   |   -0.2    |  -0.2   |    0.0    |    +0.02      |

| 4 | Paris   |  -0.1   |   -0.8    |   0.7   |    5.0    |    +4.93      |

=============================================================================



```

Notice how Token 3 receives a small POSITIVE reward (+0.02) because the Actor was actually less confident than the Reference model ($\log(-0.4) < \log(-0.2)$). The KL penalty technically "rewards" the actor for moving back towards the reference model in this case!



---



## 6. Solved Illustrations



### Illustration 1: Bradley-Terry Gradient Update for a Batch



### Extended Commentary on Reward Scaling and Normalization

In all the numerical illustrations above, we used raw scalar outputs from the neural networks. However, in production systems, these values must be strictly normalized.

1. **Reward Normalization:** Before passing the Reward Model score $r_{\psi}(x, y)$ to the PPO algorithm, it is standard practice to maintain a running mean $\mu_R$ and variance $\sigma_R^2$ of the rewards, and normalize them:

   $$ r_{\text{norm}} = \frac{r - \mu_R}{\sqrt{\sigma_R^2 + \epsilon}} $$

   This ensures that the environmental reward remains on a consistent scale (usually unit variance) regardless of the prompt difficulty, preventing the PPO gradients from exploding.

2. **Advantage Normalization:** Similarly, the GAE advantages $\hat{A}_t$ computed in Illustration 5 are normalized across the mini-batch before the PPO clip objective is applied.

   $$ \hat{A}_t^{\text{norm}} = \frac{\hat{A}_t - \mu_A}{\sigma_A} $$

   This mini-batch normalization provides a consistent learning rate scale and improves the condition number of the optimization landscape.

3. **KL Penalty Clipping:** The KL penalty $\delta_t = \log \pi_{\theta} - \log \pi_{\text{ref}}$ can occasionally become massively positive or negative. To prevent a single catastrophic token prediction from dominating the augmented reward, production implementations often clip the KL penalty:

   $$ \delta_t^{\text{clipped}} = \text{clip}(\delta_t, -10.0, 10.0) $$

   This acts as an additional layer of defensive regularization.



By meticulously handling these numerical scaling factors, engineers bridge the gap between the beautiful theoretical derivations (like the exact DPO equivalence) and the harsh realities of training 70-billion-parameter neural networks using 16-bit floating-point arithmetic.

**Problem:**

A Reward Model with a single scalar weight $w = 2.0$ maps input features $x$ to scalar rewards via the simple linear transformation $r = w \cdot x$. We have collected one preference pair from our human feedback dataset: the human preferred the winning feature $x_w = 1.5$ over the losing feature $x_l = 0.5$.

Your task is to compute the forward pass loss, evaluate the exact analytical gradient $ \frac{\partial L}{\partial w}$, and explicitly calculate the new weight $w'$ after one step of Stochastic Gradient Descent (SGD) with learning rate $\eta = 0.5$. Show all intermediate arithmetic steps clearly and verify that the weight update aligns with the human preference.



**Step-by-Step Solution:**

1. Compute the scalar rewards for the winning and losing responses based on the current parameter weight $w = 2.0$:

   For the winning response: $r_w = w \cdot x_w = 2.0 \cdot 1.5 = \mathbf{3.0}$

   For the losing response: $r_l = w \cdot x_l = 2.0 \cdot 0.5 = \mathbf{1.0}$

2. Compute the reward margin $\Delta r$, which represents the raw scalar difference in scores:

   $\Delta r = r_w - r_l = 3.0 - 1.0 = \mathbf{2.0}$

3. Map this scalar margin to a normalized probability using the Bradley-Terry logistic sigmoid function:

   $\sigma(\Delta r) = \frac{1}{1 + \exp(-2.0)}$

   Calculate the exponential: $\exp(-2.0) \approx 0.135335$

   Calculate the sigmoid probability: $\sigma(2.0) = \frac{1}{1 + 0.135335} = \frac{1}{1.135335} \approx \mathbf{0.880797}$

   The model currently possesses an 88.08% confidence that $x_w$ is superior to $x_l$.

4. Compute the Binary Cross-Entropy Loss to penalize uncertainty:

   $L = - \log(P(w \succ l)) = - \log(0.880797)$

   Evaluate the natural logarithm: $L \approx \mathbf{0.126928}$

5. Compute the gradient of the loss with respect to the reward margin:

   As derived in Section 2, the gradient of the log-sigmoid loss is strictly:

   $ \frac{\partial L}{\partial \Delta r} = - (1 - \sigma(\Delta r))$

   $ \frac{\partial L}{\partial \Delta r} = - (1 - 0.880797) = \mathbf{-0.119203}$

6. Apply the multivariate chain rule to find the gradient with respect to the underlying model weight $w$:

   First, establish the margin function: $\Delta r = w \cdot x_w - w \cdot x_l = w(x_w - x_l)$

   Compute the local derivative: $\frac{\partial \Delta r}{\partial w} = x_w - x_l = 1.5 - 0.5 = \mathbf{1.0}$

   Multiply using the chain rule: $ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \Delta r} \cdot \frac{\partial \Delta r}{\partial w} = -0.119203 \cdot 1.0 = \mathbf{-0.119203}$

7. Execute the SGD weight update step to modify the neural network parameters:

   The update rule is: $w' = w - \eta \frac{\partial L}{\partial w}$

   Substitute the values: $w' = 2.0 - (0.5) \cdot (-0.119203)$

   Perform the multiplication: $w' = 2.0 + 0.0596015$

   Final updated weight: $w' \approx \mathbf{2.0596}$

Conclusion: The model's weight increased from 2.0 to 2.0596. Because the winning feature $x_w = 1.5$ is strictly larger than the losing feature $x_l = 0.5$, increasing the weight $w$ will further amplify the margin $\Delta r$ in future forward passes. This mathematically confirms that the gradient descent step has correctly aligned the model's internal representation with the human annotator's preference.

$\blacksquare$



---



### Illustration 2: RM Forward Pass and Logistic Saturation (Gradient Vanishing)

**Problem:**

Consider a scenario where three distinct candidate preference pairs are evaluated by a trained Reward Model during a training epoch.

- Pair A: $r_w = 5.0, r_l = 4.9$ (A very hard pair, the model is highly unsure)

- Pair B: $r_w = 10.0, r_l = 8.0$ (A medium pair, the model is fairly confident)

- Pair C: $r_w = 20.0, r_l = 10.0$ (An extremely easy pair, the model is absolutely confident)

Your task is to compute the precise Bradley-Terry loss and the gradient multiplier magnitude $|1 - \sigma|$ for each pair. You must demonstrate mathematically how the gradient signal saturates and effectively vanishes for "easy" pairs, preventing destructive over-optimization.



**Step-by-Step Solution:**

1. **Analyze Pair A (Small Margin / Hard Example):**

   Compute margin: $\Delta r = r_w - r_l = 5.0 - 4.9 = \mathbf{0.1}$

   Compute probability: $\sigma(0.1) = \frac{1}{1 + \exp(-0.1)} = \frac{1}{1 + 0.904837} \approx \mathbf{0.524979}$

   Compute loss: $L_A = -\log(0.524979) \approx \mathbf{0.644400}$

   Compute gradient multiplier magnitude: $|1 - \sigma| = 1 - 0.524979 = \mathbf{0.475021}$

   Interpretation: The model is barely better than a random coin flip (52.49% confidence). The gradient multiplier is massively large (~0.475), providing a very strong learning signal. The neural network will aggressively update its weights to separate these two indistinguishable responses further.



2. **Analyze Pair B (Medium Margin / Typical Example):**

   Compute margin: $\Delta r = r_w - r_l = 10.0 - 8.0 = \mathbf{2.0}$

   Compute probability: $\sigma(2.0) = \frac{1}{1 + \exp(-2.0)} = \frac{1}{1 + 0.135335} \approx \mathbf{0.880797}$

   Compute loss: $L_B = -\log(0.880797) \approx \mathbf{0.126928}$

   Compute gradient multiplier magnitude: $|1 - \sigma| = 1 - 0.880797 = \mathbf{0.119203}$

   Interpretation: The model exhibits a solid 88% confidence. The gradient magnitude drops significantly to ~0.119. It continues to learn, but with much less aggression than Pair A, refining the boundary rather than shifting it drastically.



3. **Analyze Pair C (Huge Margin / Trivial Example):**

   Compute margin: $\Delta r = r_w - r_l = 20.0 - 10.0 = \mathbf{10.0}$

   Compute exponential: $\exp(-10.0) \approx 0.0000453999$

   Compute probability: $\sigma(10.0) = \frac{1}{1 + 0.0000453999} \approx \mathbf{0.99995460}$

   Compute loss: $L_C = -\log(0.99995460) \approx \mathbf{0.00004540}$

   Compute gradient multiplier magnitude: $|1 - \sigma| = 1 - 0.99995460 = \mathbf{0.00004540}$

   Interpretation: The model is virtually 100% confident (99.995%). The corresponding gradient is microscopically close to zero (0.000045). 

   

Conclusion: The model learns absolutely nothing from Pair C because it already perfectly ranks it. The vast majority of the strong learning signal in RLHF comes from hard, ambiguous pairs (Pair A) where the margin is near zero. This logistic saturation is a critical architectural feature, not a bug—it elegantly prevents the reward model from blowing up its parameter weights to infinity on trivial, easy examples, maintaining stable convergence.

$\blacksquare$



---



### Illustration 3: Full Per-Token KL Penalty Computation in an Episode

**Problem:**

A response generated by the Actor model during a PPO rollout consists of exactly 3 tokens. The KL regularization penalty coefficient is strictly set to $\beta = 0.2$.

During generation, the Actor and Reference models assign the following raw, un-logged probabilities to the sequentially generated tokens:

- Token 1: Actor prob = 0.8, Ref prob = 0.8

- Token 2: Actor prob = 0.9, Ref prob = 0.1

- Token 3: Actor prob = 0.1, Ref prob = 0.5

Upon sequence completion, the final sequence receives a global score from the frozen Reward Model of $10.0$.

Your task is to compute the augmented per-token rewards $\tilde{r}_t$ for the entire sequence, explicitly calculating the per-token KL divergence penalties along the way.



**Step-by-Step Solution:**

1. **Analyze Token 1 (No deviation):**

   Compute Actor log-prob: $\log \pi_\theta(y_1) = \log(0.8) \approx -0.223143$

   Compute Ref log-prob: $\log \pi_{\text{ref}}(y_1) = \log(0.8) \approx -0.223143$

   Compute local KL penalty: $\delta_1 = \log \pi_\theta(y_1) - \log \pi_{\text{ref}}(y_1) = -0.223143 - (-0.223143) = \mathbf{0.0}$

   Since $t=1 < T$, the environmental reward is 0.

   Compute augmented reward: $\tilde{r}_1 = 0 - \beta \delta_1 = 0 - 0.2 \cdot 0.0 = \mathbf{0.0}$

   Interpretation: The actor perfectly mimics the reference model, so it suffers absolutely no penalty.



2. **Analyze Token 2 (Massive overconfidence deviation):**

   Compute Actor log-prob: $\log \pi_\theta(y_2) = \log(0.9) \approx -0.105360$

   Compute Ref log-prob: $\log \pi_{\text{ref}}(y_2) = \log(0.1) \approx -2.302585$

   Compute local KL penalty: $\delta_2 = \log \pi_\theta(y_2) - \log \pi_{\text{ref}}(y_2) = -0.105360 - (-2.302585) = \mathbf{2.197225}$

   Since $t=2 < T$, the environmental reward is 0.

   Compute augmented reward: $\tilde{r}_2 = 0 - \beta \delta_2 = 0 - 0.2 \cdot 2.197225 = \mathbf{-0.439445}$

   Interpretation: The actor heavily over-indexes on this token (90%) compared to the reference model (10%). This massive deviation triggers a significant negative penalty (-0.439), actively discouraging the model from reward-hacking at this token.



3. **Analyze Token 3 (Underconfidence deviation & Terminal Final Reward):**

   Compute Actor log-prob: $\log \pi_\theta(y_3) = \log(0.1) \approx -2.302585$

   Compute Ref log-prob: $\log \pi_{\text{ref}}(y_3) = \log(0.5) \approx -0.693147$

   Compute local KL penalty: $\delta_3 = \log \pi_\theta(y_3) - \log \pi_{\text{ref}}(y_3) = -2.302585 - (-0.693147) = \mathbf{-1.609438}$

   Since $t=3 = T$, this is the terminal state. The environmental reward is the RM score $10.0$.

   Compute augmented reward: $\tilde{r}_3 = 10.0 - \beta \delta_3 = 10.0 - 0.2 \cdot (-1.609438) = 10.0 + 0.321887 = \mathbf{10.321887}$

   Interpretation: The actor is significantly under-confident compared to the reference model. Because $\delta_3$ is negative, the KL "penalty" actually functions as a small positive bonus (+0.321). This bonus is added directly to the terminal Reward Model score of 10.0, yielding a final massive reward step.

Conclusion: The terminal token absorbs the massive scalar RM reward, but intermediate generative tokens face continuous, dense, and rigorously evaluated penalties or bonuses depending precisely on how far they stray from the reference policy distribution.

$\blacksquare$



---



### Illustration 4: Optimal Policy Distribution $\pi^*$ vs Greedy Policy Selection

**Problem:**

A language model prompt possesses exactly 3 valid candidate complete responses: $y_1, y_2, y_3$.

The frozen SFT Reference Model assigns the following normalized probabilities to these responses: $\pi_{\text{ref}}(y_1) = 0.5, \pi_{\text{ref}}(y_2) = 0.3, \pi_{\text{ref}}(y_3) = 0.2$.

The Bradley-Terry Reward Model assigns the following scalar scores to each response: $r(y_1) = 1.0, r(y_2) = 5.0, r(y_3) = 2.0$.

Given a strict KL penalty regularization coefficient $\beta = 2.0$, you must compute the exact analytical optimal policy distribution $\pi^*(y)$. Afterwards, contrast this outcome with a purely greedy policy that solely maximizes reward.



**Step-by-Step Solution:**

From Derivation 11.27.2, we established that the closed-form optimal policy takes the analytical form: $\pi^*(y) = \frac{1}{Z(x)} \pi_{\text{ref}}(y) \exp\left( \frac{r(y)}{\beta} \right)$.

1. Compute the unnormalized exponential reward weights $\exp\left( \frac{r(y)}{\beta} \right)$ for each response:

   For response $y_1$: $\exp(r(y_1) / \beta) = \exp(1.0 / 2.0) = \exp(0.5) \approx \mathbf{1.648721}$

   For response $y_2$: $\exp(r(y_2) / \beta) = \exp(5.0 / 2.0) = \exp(2.5) \approx \mathbf{12.182494}$

   For response $y_3$: $\exp(r(y_3) / \beta) = \exp(2.0 / 2.0) = \exp(1.0) \approx \mathbf{2.718281}$

2. Compute the composite unnormalized optimal weights $w(y) = \pi_{\text{ref}}(y) \exp(r(y) / \beta)$ by incorporating the reference priors:

   Weight for $y_1$: $w(y_1) = 0.5 \cdot 1.648721 = \mathbf{0.824360}$

   Weight for $y_2$: $w(y_2) = 0.3 \cdot 12.182494 = \mathbf{3.654748}$

   Weight for $y_3$: $w(y_3) = 0.2 \cdot 2.718281 = \mathbf{0.543656}$

3. Compute the partition function scalar $Z(x)$ by summing all unnormalized weights across the domain:

   $Z(x) = \sum w(y) = 0.824360 + 3.654748 + 0.543656 = \mathbf{5.022764}$

4. Normalize the weights to obtain the final, exact optimal probability distribution $\pi^*(y) = w(y) / Z(x)$:

   Optimal prob for $y_1$: $\pi^*(y_1) = 0.824360 / 5.022764 \approx \mathbf{0.164124}$ (or 16.41%)

   Optimal prob for $y_2$: $\pi^*(y_2) = 3.654748 / 5.022764 \approx \mathbf{0.727636}$ (or 72.76%)

   Optimal prob for $y_3$: $\pi^*(y_3) = 0.543656 / 5.022764 \approx \mathbf{0.108238}$ (or 10.82%)

   Verify constraint sum: $0.164124 + 0.727636 + 0.108238 = 0.999998 \approx 1.0$.

5. Contrast analysis with Greedy Policy:

   A purely greedy optimization policy (equivalent to removing the KL penalty, i.e., $\beta \to 0$) would assign 100% probability to response $y_2$ entirely because it possesses the highest absolute reward (5.0).

   However, our optimal, regularized policy carefully assigns $\pi^*(y_2) = 72.76\%$. Its probability is heavily capped because the baseline reference model only gave it a 30% probability initially. The substantial KL penalty $\beta = 2.0$ acts as a massive anchor, heavily penalizing moving too far away from the safe reference distribution. Astonishingly, the model retains a non-trivial 16.41% probability mass on response $y_1$. Although its reward is comparatively very low, it is considered exceptionally "safe" according to the reference model (50% prior probability).

$\blacksquare$



---



### Illustration 5: Unified 4-Model PPO Forward Pass & Surrogate Advantage Computation

**Problem:**

At generation step $t=1$, we evaluate a specific prompt across the complex 4-model PPO architecture. The vocabulary space is tightly constrained to only two tokens: $[A, B]$.

- **Actor (trainable $\theta$)** outputs raw logits: $[2.0, -1.0]$.

- **Ref (frozen)** outputs raw logits: $[1.0, 1.0]$.

- **Critic (trainable $\phi$)** predicts the temporal state value $V(s_1) = 3.5$.

The Actor stochastically samples token $A$. The environment transitions to step $t=2$, the sequence ends, and the Reward Model outputs a terminal score $r_{\text{RM}} = 5.0$.

Your task is to compute the exact PPO surrogate advantage $\hat{A}_1$ for updating the Actor at step $t=1$. Assume a temporal discount factor of $\gamma=1.0$ and a KL penalty coefficient of $\beta=0.1$.



**Step-by-Step Solution:**

1. **Compute Actor Probabilities for Sampled Token A:**

   The unnormalized Actor logits are $[2.0, -1.0]$. We apply the standard softmax function to extract probabilities.

   Calculate exponentials: $\exp(2.0) \approx 7.389056$, $\exp(-1.0) \approx 0.367879$

   Denominator sum: $7.389056 + 0.367879 = 7.756935$

   Actor probability for A: $\pi_{\theta}(A) = \frac{7.389056}{7.756935} \approx \mathbf{0.952574}$

   Log-probability for A: $\log \pi_{\theta}(A) = \log(0.952574) \approx \mathbf{-0.048587}$

2. **Compute Reference Probabilities for Sampled Token A:**

   The unnormalized Ref logits are $[1.0, 1.0]$. Softmax makes this a perfectly uniform distribution.

   Ref probability for A: $\pi_{\text{ref}}(A) = \frac{\exp(1.0)}{\exp(1.0) + \exp(1.0)} = \mathbf{0.500000}$

   Log-probability for A: $\log \pi_{\text{ref}}(A) = \log(0.5) \approx \mathbf{-0.693147}$

3. **Evaluate the Step 1 KL Penalty and Local Reward:**

   Compute local KL penalty: $\delta_1 = \log \pi_{\theta}(A) - \log \pi_{\text{ref}}(A) = -0.048587 - (-0.693147) = \mathbf{0.644560}$

   The augmented step reward at $t=1$ is strictly the penalty, since intermediate environmental reward is 0:

   Augmented reward $\tilde{r}_1 = -\beta \delta_1 = -0.1 \cdot 0.644560 = \mathbf{-0.064456}$

4. **Compute the Final Trajectory Return $G_1$:**

   Assume that at step $t=2$, the Actor outputs token B, the sequence deterministically ends, and after conducting identical mathematical operations for $t=2$, we extract a final augmented reward $\tilde{r}_2 = 4.9$. The Critic correctly evaluates the terminal state as $V(s_2) = 0$.

   The true empirical return from step $t=1$ is the discounted sum of future rewards (with $\gamma=1.0$):

   $G_1 = \tilde{r}_1 + \gamma \tilde{r}_2 = -0.064456 + (1.0 \cdot 4.9) = \mathbf{4.835544}$

5. **Advantage Calculation for the Actor Update:**

   The advantage $\hat{A}_1$ is strictly defined as the actual return minus the Critic's predicted temporal baseline value.

   $\hat{A}_1 = G_1 - V(s_1) = 4.835544 - 3.5 = \mathbf{1.335544}$

6. **Formulate the PPO Surrogate Objective:**

   The Actor network will be updated using the clipped surrogate objective function: $L^{\text{CLIP}} = \min(r_t(\theta)\cdot \hat{A}_1, \dots)$. 

   Since the computed advantage $\hat{A}_1 = 1.335544 > 0$ is highly positive, the gradient will dynamically attempt to push the logit for token A even higher during the backward pass, increasing its probability up to the strict PPO clip limit threshold. Concurrently, the Critic network will be updated using an MSE loss to shift its internal prediction $V(s_1) = 3.5$ much closer to the true empirical return of $4.835544$.

$\blacksquare$



---





### Illustration 6: PPO Value Function Clipping and Gradient Update

**Problem:**

At step $t=5$, the empirical discounted return from the environment is $G_5 = 12.0$.

During the previous epoch, the old Critic network predicted $V_{\phi_{\text{old}}}(s_5) = 10.0$.

During the current epoch, the updated Critic network predicts $V_{\phi}(s_5) = 11.5$.

The PPO value clipping hyperparameter is $\epsilon = 0.5$.

Compute the unclipped value loss, the clipped value prediction, the clipped value loss, and the final $L^{\text{VF}}$. Determine if the clipping mechanism alters the gradient in this scenario.



**Step-by-Step Solution:**

1. **Compute Unclipped Loss:**

   The error is: $e_{\text{unclipped}} = V_{\phi}(s_5) - G_5 = 11.5 - 12.0 = \mathbf{-0.5}$

   The MSE loss is: $L_{\text{unclipped}} = \frac{1}{2} (-0.5)^2 = \frac{1}{2} (0.25) = \mathbf{0.125}$

2. **Compute Clipped Value Prediction:**

   The parameter change is: $\Delta V = V_{\phi}(s_5) - V_{\phi_{\text{old}}}(s_5) = 11.5 - 10.0 = \mathbf{1.5}$

   We apply the clip function to this delta: $\text{clip}(1.5, -0.5, 0.5) = \mathbf{0.5}$

   The clipped value prediction is: $V_{\text{clipped}} = V_{\phi_{\text{old}}}(s_5) + 0.5 = 10.0 + 0.5 = \mathbf{10.5}$

3. **Compute Clipped Loss:**

   The error using the clipped prediction is: $e_{\text{clipped}} = V_{\text{clipped}} - G_5 = 10.5 - 12.0 = \mathbf{-1.5}$

   The MSE loss is: $L_{\text{clipped}} = \frac{1}{2} (-1.5)^2 = \frac{1}{2} (2.25) = \mathbf{1.125}$

4. **Determine Final Loss:**

   The final PPO value loss is the maximum of the unclipped and clipped losses.

   $L^{\text{VF}} = \max(L_{\text{unclipped}}, L_{\text{clipped}}) = \max(0.125, 1.125) = \mathbf{1.125}$

5. **Gradient Analysis:**

   Because the clipped loss (1.125) is strictly greater than the unclipped loss (0.125), the maximum operator selects the clipped term.

   The derivative of the clipped term with respect to the network weights $\phi$ involves the derivative of $V_{\text{clipped}}$. However, because the prediction $V_{\phi}(s_5)$ was clipped, it lies in the flat region of the clip function, meaning its derivative with respect to $\phi$ is identically $\mathbf{0}$.

Conclusion: The Critic network attempted to update its prediction from 10.0 to 11.5 in a single epoch, exceeding the allowed trust region of $\epsilon = 0.5$. Consequently, the PPO value clipping mechanism engaged, selecting the pessimistic upper bound on the loss, and effectively zeroed out the gradient to prevent further destabilization on this specific data point.

$\blacksquare$



### Illustration 7: Entropy Bonus Calculation over a Vocabulary

**Problem:**

A language model vocabulary has been drastically simplified to $\mathcal{V} = 4$ tokens: $[A, B, C, D]$.

The Actor policy outputs the following probability distribution at step $t$:

$\pi_{\theta}(A) = 0.70$

$\pi_{\theta}(B) = 0.15$

$\pi_{\theta}(C) = 0.10$

$\pi_{\theta}(D) = 0.05$

Compute the exact Shannon entropy $\mathcal{H}$ of this distribution, and calculate the gradient contribution to the logit of token A if the entropy bonus coefficient is $c_2 = 0.01$.



**Step-by-Step Solution:**

1. **Compute Log Probabilities:**

   $\log(0.70) \approx \mathbf{-0.3567}$

   $\log(0.15) \approx \mathbf{-1.8971}$

   $\log(0.10) \approx \mathbf{-2.3026}$

   $\log(0.05) \approx \mathbf{-2.9957}$

2. **Compute Individual Entropy Terms ($p \log p$):**

   $0.70 \cdot (-0.3567) = \mathbf{-0.2497}$

   $0.15 \cdot (-1.8971) = \mathbf{-0.2846}$

   $0.10 \cdot (-2.3026) = \mathbf{-0.2303}$

   $0.05 \cdot (-2.9957) = \mathbf{-0.1498}$

3. **Compute Total Shannon Entropy:**

   $\mathcal{H} = - (-0.2497 - 0.2846 - 0.2303 - 0.1498) = - (-0.9144) = \mathbf{0.9144}$ nats.

4. **Gradient Contribution:**

   The objective is to maximize $J = \dots + c_2 \mathcal{H}$.

   The derivative of entropy with respect to the pre-softmax logit $z_A$ is given by the standard identity:

   $\frac{\partial \mathcal{H}}{\partial z_A} = - \pi_{\theta}(A) (\log \pi_{\theta}(A) + 1 - \mathcal{H})$

   Let us plug in the values:

   $\log \pi_{\theta}(A) = -0.3567$

   $\log \pi_{\theta}(A) + 1 - \mathcal{H} = -0.3567 + 1.0 - 0.9144 = \mathbf{-0.2711}$

   Multiply by the probability and negate:

   $\frac{\partial \mathcal{H}}{\partial z_A} = - (0.70) \cdot (-0.2711) = \mathbf{0.1898}$

5. **Final Entropy Bonus Gradient:**

   Multiply by the coefficient $c_2 = 0.01$:

   Gradient = $0.01 \cdot 0.1898 = \mathbf{0.001898}$

Conclusion: Because token A is highly dominant (70%), the entropy gradient is positive, gently pushing the logit of token A down (or rather, allowing other tokens to rise) in order to increase the uncertainty and diversity of the distribution, thereby preventing premature convergence to a deterministic policy.

$\blacksquare$





## 7. Deep Learning Connection & Modern Applications



### 7.1 InstructGPT and ChatGPT Breakthroughs

The fundamental RLHF pipeline detailed here was popularized by OpenAI's **InstructGPT** paper (Ouyang et al., 2022), which served as the direct precursor to ChatGPT. A massive empirical finding from that paper was that a **1.3B parameter model fine-tuned with RLHF was preferred by humans over a 175B parameter base model** trained only on next-token prediction. This proved that alignment is radically more sample-efficient and parameter-efficient than raw next-token pretraining. The 4-model PPO architecture (Actor, Critic, RM, Ref) remains the industrial standard for post-training frontier models today. It allows models to be explicitly trained to be helpful, honest, and harmless rather than just mimicking internet text.



### 7.2 Direct Preference Optimization (DPO)

As derived in Derivation 11.27.2, the partition function $Z(x)$ perfectly cancels out when the analytical optimal policy $\pi^*$ is substituted into the Bradley-Terry loss. This mathematical quirk birthed **Direct Preference Optimization (DPO)** (Rafailov et al., 2023). DPO completely skips Phase 2 (training an explicit Reward Model) and Phase 3 (PPO RL). Instead, it trains the Actor directly on the human preference pairs using a modified cross-entropy loss that uses the Reference model as a dynamic baseline. While DPO is much cheaper computationally (only 2 models in memory instead of 4) and avoids the instability of reinforcement learning, PPO often achieves higher asymptotic performance on complex reasoning tasks due to its ability to explore the environment generated by the Reward Model.



### 7.3 Constitutional AI and RLAIF

Gathering human preference data ($y_w \succ y_l$) is astronomically expensive and difficult to scale, requiring thousands of hours of expert annotator time. Anthropic introduced **Constitutional AI** and **RL from AI Feedback (RLAIF)** to solve this bottleneck. Instead of humans reading and ranking responses, a separate powerful "Judge" LLM is given a list of constitutional principles (e.g., "Choose the response that is less harmful", "Choose the response that is more logically sound") and automatically generates the preference dataset by grading the Actor's outputs. The underlying mathematics of the Reward Model and PPO remain exactly identical; the only difference is the source of the labels.



---



## 8. Code Implementation & Verification



See companion code: [code/27_rlhf_ppo.py](code/27_rlhf_ppo.py)
