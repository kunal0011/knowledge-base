# Module 11.26: Decision Transformer

---

## 1. Intuition & 101 Motivation



For decades, reinforcement learning was framed exclusively through the lens of **Dynamic Programming and the Bellman Equation**:



$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$



This formulation requires temporal-difference bootstrapping, actor-critic architectures, discount factors, target networks, and replay buffers. When scaled to large neural networks, this paradigm often suffers from numerical instabilities—most notoriously the **Deadly Triad** (bootstrapping + function approximation + off-policy distribution shift).



In 2021, a revolutionary paradigm emerged: **Can we throw away the Bellman equation entirely and solve Reinforcement Learning as generic autoregressive Sequence Modeling?**



Enter the **Decision Transformer (DT)** (Chen et al., NeurIPS 2021). Instead of predicting future values, an entire RL episode is transcribed into a sequence of tokens. The key insight is to condition the autoregressive generation on the **Return-to-Go (RTG)**, effectively transforming RL into conditional imitation learning.



```text

    +-------------------------------------------------------------------+

    |                   AUTOREGRESSIVE RTG CONDITIONING LOOP            |

    |                                                                   |

    |  [ Target RTG R_1 ] ----> [ State s_1 ] ----> [ Action a_1 ]      |

    |                                                     |             |

    |                                                     v (Env)       |

    |                                                [ Reward r_1 ]     |

    |                                                [ Next State s_2 ] |

    |                                                     |             |

    |                                                     v (Decrement) |

    |  [ Target RTG R_2 ] ----> [ State s_2 ] ----> [ Action a_2 ]      |

    |   (R_2 = R_1 - r_1)                                               |

    |                                                     |             |

    |                                                     v (Env)       |

    |                                                [ Reward r_2 ]     |

    |                                                [ Next State s_3 ] |

    |                                                     |             |

    |                                                     v (Decrement) |

    |  [ Target RTG R_3 ] ----> [ State s_3 ] ----> [ Action a_3 ]      |

    |   (R_3 = R_2 - r_2)                                               |

    +-------------------------------------------------------------------+

```



Why does this work? By conditioning on the desired return, the Decision Transformer slices the offline dataset manifold. If you prompt it with a high return, it simply imitates the sub-population of trajectories that achieved that high return. There is no TD error, no bootstrapping, and no deadly triad.



---



## 2. Rigorous Mathematical Formulation



### 2.1 Trajectory Representation & Return-to-Go



Let a collected episode of length $T$ consist of states $s_t \in \mathbb{R}^{d_s}$, actions $a_t \in \mathbb{R}^{d_a}$, and scalar stage rewards $r_t \in \mathbb{R}$.



The **Return-to-Go (RTG)** at timestep $t$ is the exact empirical sum of future rewards:



$$\hat{R}_t \triangleq \sum_{t'=t}^T r_{t'}$$



It satisfies the deterministic backward recurrence:



$$\hat{R}_{t} = r_t + \hat{R}_{t+1} \iff \hat{R}_{t+1} = \hat{R}_t - r_t$$



The trajectory is ordered chronologically into a sequence of triplets:



$$\tau = \left( \hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T \right)$$



### 2.2 Token Embedding & Modality Encodings



Because the three modalities (returns, states, actions) have different physical dimensions, each modality is passed through a distinct linear projection matrix into a shared hidden dimension $d$:



$$e_t^R = W_R \hat{R}_t + e_t^{\text{pos}}$$



$$e_t^s = W_s s_t + e_t^{\text{pos}}$$



$$e_t^a = W_a a_t + e_t^{\text{pos}}$$



where $W_R \in \mathbb{R}^{d \times 1}$, $W_s \in \mathbb{R}^{d \times d_s}$, $W_a \in \mathbb{R}^{d \times d_a}$ and $e_t^{\text{pos}} \in \mathbb{R}^d$ is the timestep positional embedding.



### 2.3 Causal Masking & Training Objective



The Transformer blocks use standard **multi-head causal self-attention**:



$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$



where $M$ is the causal mask ($M_{i, j} = -\infty$ for $j > i$).



Crucially, the mask prevents the model from attending to future tokens or the action it is currently trying to predict.



The model is trained via standard supervised regression or cross-entropy:



$$\mathcal{L}_{\text{DT}}(\theta) = \frac{1}{T} \sum_{t=1}^T \left\| a_t - \hat{a}_t \left( \hat{R}_{1:t}, s_{1:t}, a_{1:t-1} \right) \right\|_2^2$$



### 2.4 Test-Time Autoregressive Rollout Protocol



To deploy a trained Decision Transformer:

1. Choose a target return $\hat{R}_1$ (e.g., maximum training return).

2. Receive $s_1$ from the environment.

3. Pass $[\hat{R}_1, s_1]$ to the Transformer to predict $\hat{a}_1$.

4. Execute $\hat{a}_1$, observe reward $r_1$ and next state $s_2$.

5. Decrement the return-to-go: $\hat{R}_2 \leftarrow \hat{R}_1 - r_1$.

6. Append $(\hat{a}_1, \hat{R}_2, s_2)$ to context and repeat.



### 2.5 First-Principles Mathematical Derivations



#### Derivation 11.26.1: Equivalence between Return-Conditioned BC and In-Context RL

====================================================================================================

**Part 1: The Optimal Policy Definition**



In a standard Markov Decision Process (MDP) defined by $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$, the optimal action-value function satisfies the Bellman Optimality Equation:



$$Q^*(s, a) = \max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \mid s_0=s, a_0=a \right]$$



The optimal policy is defined deterministically as:



$$\pi^*(a \mid s) = \delta\left(a - \arg\max_{a' \in \mathcal{A}} Q^*(s, a')\right)$$



where $\delta$ is the Dirac delta function.



**Part 2: Return-Conditioned Behavior Cloning (RCBC)**



The Decision Transformer frames policy learning as conditional sequence modeling. Given an offline dataset $\mathcal{D}$ of trajectories $\tau$, we model the conditional probability $P_{\theta}(a_t \mid s_t, \hat{R}_t)$.



Assuming infinite model capacity and full data coverage (i.e., the dataset contains trajectories exploring all possible returns from all states), the model learns the exact empirical distribution:



$$P_{\theta}(a \mid s, \hat{R}) \to P_{\mathcal{D}}(a \mid s, \hat{R})$$



By Bayes' Theorem, this conditional probability can be inverted:



$$P(a \mid s, \hat{R}) = \frac{P(\hat{R} \mid s, a) P(a \mid s)}{P(\hat{R} \mid s)}$$



Here, $P(\hat{R} \mid s, a)$ is the probability of achieving a total return $\hat{R}$ given that we take action $a$ in state $s$.



**Part 3: Conditioning on the Optimal Return**



Let $\hat{R}^* = V^*(s) = \max_a Q^*(s, a)$ be the optimal return achievable from state $s$.



At inference time, the user prompts the Decision Transformer with $\hat{R} = V^*(s)$.



Substituting this into our Bayes expansion:



$$P(a \mid s, V^*(s)) = \frac{P(\hat{R}=V^*(s) \mid s, a) P(a \mid s)}{P(\hat{R}=V^*(s) \mid s)}$$



For any suboptimal action $a_{sub} \notin \arg\max_a Q^*(s, a)$, the maximum achievable return is $Q^*(s, a_{sub}) < V^*(s)$. Assuming a deterministic environment, it is strictly impossible to achieve a return of $V^*(s)$ after taking $a_{sub}$.



Therefore, $P(\hat{R}=V^*(s) \mid s, a_{sub}) = 0$.



Consequently, $P(a_{sub} \mid s, V^*(s)) = 0$.



For an optimal action $a^* \in \arg\max_a Q^*(s, a)$, the dataset contains trajectories (by our full coverage assumption) that achieve $V^*(s)$ after taking $a^*$.



Thus, $P(\hat{R}=V^*(s) \mid s, a^*) > 0$.



Normalizing the probabilities over all actions, the conditional probability collapses into a Dirac delta distribution centered entirely on the optimal actions:



$$P(a \mid s, V^*(s)) = \begin{cases} \frac{P(a \mid s)}{\sum_{a'} P(a' \mid s)} & \text{if } a \in \arg\max Q^*(s, a) \\ 0 & \text{otherwise} \end{cases}$$



This exactly matches the optimal policy $\pi^*(a \mid s)$. Thus, return-conditioned BC perfectly imitates optimal RL.



**Part 4: Conclusion**



The Decision Transformer, by conditioning purely on the empirical highest return, perfectly mimics the output of dynamic programming without ever bootstrapping or suffering from off-policy function approximation issues.

====================================================================================================



#### Derivation 11.26.2: RTG Bellman-like Backward Recursion

====================================================================================================

**Part 1: The Goal**



Prove that the simple subtraction rule used during inference ($\hat{R}_{t+1} = \hat{R}_t - r_t$) is mathematically equivalent to propagating the optimal value function forward through the episode.



**Part 2: The Initial Condition**



At timestep $t=1$, the operator sets the initial return-to-go token to the optimal value of the starting state:



$$\hat{R}_1 = V^*(s_1)$$



By the equivalence proven in Derivation 11.26.1, conditioning the Decision Transformer on $\hat{R}_1 = V^*(s_1)$ forces it to output an optimal action $a_1 = a_1^*$.



**Part 3: Execution and Transition**



The optimal action $a_1^*$ is executed in the environment.



The environment yields a deterministic reward $r_1 = r(s_1, a_1^*)$ and transitions to state $s_2 = f(s_1, a_1^*)$.



The Bellman Optimality Equation states that the value of a state is the sum of the immediate reward and the optimal value of the next state, assuming the optimal action is taken:



$$V^*(s_1) = r(s_1, a_1^*) + V^*(s_2)$$



Substituting $r_1$:



$$V^*(s_1) = r_1 + V^*(s_2)$$



**Part 4: The RTG Decrement Rule**



The standard Decision Transformer inference protocol updates the Return-to-Go for the next timestep using simple subtraction:



$$\hat{R}_2 = \hat{R}_1 - r_1$$



Substitute our initial condition $\hat{R}_1 = V^*(s_1)$:



$$\hat{R}_2 = V^*(s_1) - r_1$$



Now, substitute the Bellman expansion $V^*(s_1) = r_1 + V^*(s_2)$:



$$\hat{R}_2 = (r_1 + V^*(s_2)) - r_1$$



$$\hat{R}_2 = V^*(s_2)$$



**Part 5: Mathematical Induction**



By mathematical induction, if $\hat{R}_t = V^*(s_t)$, then $\hat{R}_{t+1} = V^*(s_{t+1})$.



This proves that the simple subtraction rule magically tracks the exact optimal Bellman value of the current state at every single timestep, ensuring the transformer is continually conditioned to act optimally without ever computing a Temporal Difference error.

====================================================================================================



#### Derivation 11.26.3: Cross-Entropy Causal LM Loss Derivation

====================================================================================================

**Part 1: The Decision Transformer Training Objective**



The Decision Transformer is trained using standard sequence modeling loss. For discrete actions, this is the Cross-Entropy loss over the dataset $\mathcal{D}$:



$$\mathcal{L}_{DT}(\theta) = - \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_{t=1}^T \log \pi_\theta(a_t \mid \hat{R}_{1:t}, s_{1:t}, a_{1:t-1}) \right]$$



**Part 2: Decomposing the Expectation**



The expectation over trajectories $\tau \sim \mathcal{D}$ can be rewritten by marginalizing over the total return $R(\tau)$ of the trajectory.



Let $P_{\mathcal{D}}(R)$ be the marginal distribution of total returns in the dataset.



Let $P_{\mathcal{D}}(\tau \mid R)$ be the conditional distribution of trajectories that achieve exactly return $R$.



$$\mathcal{L}_{DT}(\theta) = - \int_R P_{\mathcal{D}}(R) \mathbb{E}_{\tau \sim P_{\mathcal{D}}(\tau \mid R)} \left[ \sum_{t=1}^T \log \pi_\theta(a_t \mid \hat{R}_{1:t}, s_{1:t}, a_{1:t-1}) \right] dR$$



**Part 3: The Gradient of the Loss**



To optimize the parameters $\theta$, we compute the gradient $\nabla_\theta \mathcal{L}_{DT}(\theta)$:



$$\nabla_\theta \mathcal{L}_{DT}(\theta) = - \int_R P_{\mathcal{D}}(R) \mathbb{E}_{\tau \mid R} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid \hat{R}_{1:t}, s_{1:t}, a_{1:t-1}) \right] dR$$



Notice that $\nabla_\theta \log \pi_\theta(a_t \mid \cdot)$ is the score function, identical to the standard Policy Gradient (REINFORCE) step direction.



**Part 4: Physical Interpretation of the Gradient**



In standard Policy Gradient algorithms, the gradient is weighted by the advantage $A_t$:



$$\nabla_\theta \mathcal{L}_{PG} = - \mathbb{E} \left[ A_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$



In the Decision Transformer, there is no explicit advantage weighting. Instead, the weighting is **implicit** through the conditioning variable $\hat{R}$.



The network learns a vast family of policies, one for every possible value of $R$.



When updating the network for a trajectory with $R=100$, the gradient perfectly mimics Behavior Cloning (weight=1) for the policy conditioned on $R=100$.



When updating for a trajectory with $R=10$, it mimics Behavior Cloning for the policy conditioned on $R=10$.



At inference time, we simply discard all policies where $R$ is low by fixing the input token $\hat{R}_1$ to the maximum value. This elegantly bypasses the need to compute value functions or advantages during training!

====================================================================================================



---



## 3. Geometric & Physical Interpretation



Imagine the space of all possible trajectories as a sprawling, tangled web of paths on a 2D plane. 



- A random policy diffuses in all directions.

- Behavior Cloning averages all vectors, creating a blurry vector field that often points into walls.

- The **Decision Transformer** groups trajectories by their length (Return). 



```text

Return = 100 Manifold (Direct, fast paths)

==============================================> (Goal)

    ^

    | (Conditioning parameter R effectively shifts you vertically between manifolds)

    v

Return = 10 Manifold (Wandering, cyclic paths)

~~~~~~~~~~~~~~~~~~~O~~~~~~~~~~O~~~~~~~~~O~~~~~>

```



Conditioning on $\hat{R} = 100$ physically lifts the model's attention entirely off the messy, suboptimal paths and restricts generation strictly to the high-return manifold.



---



## 4. Real-World Analogy



Consider asking a taxi driver for a ride.



- **Q-Learning / TD:** The driver constantly re-evaluates the expected time to destination at every intersection, bootstrapping from adjacent intersections.

- **Decision Transformer:** You tell the driver upfront: "I will pay you $100 if we get there in 15 minutes." The driver simply accesses their memory of past drives that took 15 minutes and executes that precise sequence of turns. The goal condition shapes the behavior directly from sequence memory.



---



## 5. Prof. Tom Yeh "AI by Hand" Visual Grids



### Walkthrough 1: 3-Step CartPole Token Sequence Construction



**Context & Scenario:**



Consider an offline episode of a simplified 3-step CartPole environment.



- Rewards: $r_1=1, r_2=1, r_3=1$.

- Return-to-Go (RTG): $\hat{R}_t = \sum_{t'=t}^3 r_{t'}$.

  - $\hat{R}_3 = 1$

  - $\hat{R}_2 = 1 + 1 = 2$

  - $\hat{R}_1 = 1 + 1 + 1 = 3$

- States $s_t \in \mathbb{R}^4$:

  $s_1 = [0.1, -0.2, 0.3, 0.0]$

  $s_2 = [0.15, -0.1, 0.28, 0.05]$

  $s_3 = [0.2, 0.0, 0.25, 0.1]$

- Actions $a_t \in \{0, 1\}$ (represented as continuous embeddings for simplicity, e.g., $a_1=[1.0], a_2=[0.0], a_3=[1.0]$).



**Embedding Linear Projections:**



Let the hidden dimension $d_{\text{model}} = 128$.



For hand computation, we trace just the **first element (index 0)** of the 128-d vector.



- $W_R[0, 0] = 0.5$

- $W_s[0, :] = [0.1, 0.2, -0.1, 0.0]$

- $W_a[0, 0] = 0.8$



**Step 1: Compute Modality Embeddings ($t=1$)**



- $e_1^R[0] = W_R[0,0] \times \hat{R}_1 = 0.5 \times 3.0 = \mathbf{1.5000}$

- $e_1^s[0] = W_s[0,:] \cdot s_1 = 0.1(0.1) + 0.2(-0.2) - 0.1(0.3) + 0.0(0.0) = 0.01 - 0.04 - 0.03 = \mathbf{-0.0600}$

- $e_1^a[0] = W_a[0,0] \times a_1 = 0.8 \times 1.0 = \mathbf{0.8000}$



**Step 2: Compute Modality Embeddings ($t=2$)**



- $e_2^R[0] = W_R[0,0] \times \hat{R}_2 = 0.5 \times 2.0 = \mathbf{1.0000}$

- $e_2^s[0] = W_s[0,:] \cdot s_2 = 0.1(0.15) + 0.2(-0.1) - 0.1(0.28) + 0 = 0.015 - 0.02 - 0.028 = \mathbf{-0.0330}$

- $e_2^a[0] = W_a[0,0] \times a_2 = 0.8 \times 0.0 = \mathbf{0.0000}$



**Step 3: Compute Modality Embeddings ($t=3$)**



- $e_3^R[0] = W_R[0,0] \times \hat{R}_3 = 0.5 \times 1.0 = \mathbf{0.5000}$

- $e_3^s[0] = W_s[0,:] \cdot s_3 = 0.1(0.2) + 0.2(0.0) - 0.1(0.25) + 0 = 0.02 + 0 - 0.025 = \mathbf{-0.0050}$

- $e_3^a[0] = W_a[0,0] \times a_3 = 0.8 \times 1.0 = \mathbf{0.8000}$



**Summary Matrix (first dim only):**



| Token Index | Type | Modality | Output $h[0]$ |

|---|---|---|---|

| 1 | $\hat{R}_1$ | [3.0] | 1.5000 |

| 2 | $s_1$ | [0.1, -0.2, 0.3, 0.0] | -0.0600 |

| 3 | $a_1$ | [1.0] | 0.8000 |

| 4 | $\hat{R}_2$ | [2.0] | 1.0000 |

| 5 | $s_2$ | [0.15, -0.1, 0.28, 0.05] | -0.0330 |

| 6 | $a_2$ | [0.0] | 0.0000 |

| 7 | $\hat{R}_3$ | [1.0] | 0.5000 |

| 8 | $s_3$ | [0.2, 0.0, 0.25, 0.1] | -0.0050 |

| 9 | $a_3$ | [1.0] | 0.8000 |



### Walkthrough 2: Causal Masking Logic



The 9 tokens are fed sequentially. The action prediction at time $t$ occurs at the output of the state token $s_t$.



The causal mask $M \in \mathbb{R}^{9 \times 9}$ where $1$ allows attention (0 in logit space) and $0$ blocks attention ($-\infty$ in logit space):



```text

Idx Token | R1 s1 a1 R2 s2 a2 R3 s3 a3

--------------------------------------

1   R1    | 1  0  0  0  0  0  0  0  0

2   s1    | 1  1  0  0  0  0  0  0  0  <- Predicts a1. Cannot see a1!

3   a1    | 1  1  1  0  0  0  0  0  0

4   R2    | 1  1  1  1  0  0  0  0  0

5   s2    | 1  1  1  1  1  0  0  0  0  <- Predicts a2. Sees past + R2, s2.

6   a2    | 1  1  1  1  1  1  0  0  0

7   R3    | 1  1  1  1  1  1  1  0  0

8   s3    | 1  1  1  1  1  1  1  1  0  <- Predicts a3. Cannot see a3!

9   a3    | 1  1  1  1  1  1  1  1  1

```



Notice how row $s_1$ (index 2) has `0` for $a_1$ (index 3). This perfectly prevents leakage!



---



## 6. Solved Illustrations



### Illustration 1: Return Extrapolation Beyond Training Distribution



**Problem:**

A Decision Transformer is trained on an offline dataset where the maximum observed Return-to-Go is $R_{\max} = 300$. 

(a) Compute the maximum training return embedding $e_R^{train\_max}$, assuming a linear projection weight $W_R = [0.2, -0.1]^T \in \mathbb{R}^2$.

(b) At test time, an operator prompts the model with $\hat{R}_0 = 500$ to demand "super-expert" performance. Show exactly what tokens the model receives and calculate the Euclidean extrapolation gap.

(c) Quantify the consequences: if the model learned $\mathbb{E}[a \mid s, \hat{R}=300] = a_{\text{expert}}$, what does it output at $\hat{R}=500$? Analyze why extrapolation may or may not work.

(d) Compare to Q-learning which uses $\arg\max_a Q(s, a)$ regardless of any return conditioning.



**Step-by-Step Solution:**



**(Part a) Training Maximum Embedding:**

The highest return seen during training is 300.

The embedding is linearly projected:

$$e_R^{train\_max} = W_R \times 300 = \begin{bmatrix} 0.2 \\ -0.1 \end{bmatrix} \times 300 = \begin{bmatrix} 60.0 \\ -30.0 \end{bmatrix}$$

This vector $\begin{bmatrix} 60.0 \\ -30.0 \end{bmatrix}$ defines the boundary of the training manifold in the transformer's hidden space.



**(Part b) Test Prompt Embedding and Tokens Received:**

The operator prompts with $\hat{R}_0 = 500$.

The initial sequence fed to the model is the pair of tokens: $[e_R^{test}, e_s^1]$.

The return embedding for 500 is:

$$e_R^{test} = W_R \times 500 = \begin{bmatrix} 0.2 \\ -0.1 \end{bmatrix} \times 500 = \begin{bmatrix} 100.0 \\ -50.0 \end{bmatrix}$$

The Euclidean extrapolation gap is the exact distance between the vectors:

$$\Delta = \|e_R^{test} - e_R^{train\_max}\|_2 = \sqrt{(100.0 - 60.0)^2 + (-50.0 - (-30.0))^2}$$

$$\Delta = \sqrt{40.0^2 + (-20.0)^2} = \sqrt{1600 + 400} = \sqrt{2000} \approx \mathbf{44.7214}$$



**(Part c) Consequences of OOD Shift:**

The prompt embedding is shifted by a massive distance of $\sim 44.7$ beyond the convex hull of the training data.

The attention mechanism computes raw logits as $S = Q K^T / \sqrt{d}$.

Since the return embedding vector has vastly larger magnitude ($\|\cdot\|_2 = 111.8$) compared to training data ($\|\cdot\|_2 = 67.0$), any dot product aligning with this vector will explode multiplicatively.

For example, if a query vector aligns with $e_R$, the logit might be $S = +1000$.

The softmax operation will yield:

$$\frac{e^{1000}}{e^{1000} + \dots} \to 1.0$$

This extreme attention saturation causes the state token to completely ignore its own state (attention weight $\to 0$) and obsess entirely over the return token. Without state information, the action prediction completely collapses. Thus, the model fails to output $a_{\text{expert}}$ and degrades to random or repeating actions. Extrapolation catastrophically fails.



**(Part d) Comparison to Q-learning:**

Standard Q-learning evaluates $\arg\max_a Q(s, a)$. It does not take a requested return as input. It simply queries the learned value surface and greedily picks the action with the highest expected value. Q-learning naturally attempts to achieve the maximum possible return without suffering from embedding-space OOD saturation. It implicitly targets the optimal return rather than explicitly conditioning on an arbitrary scalar, allowing it to perform optimally at the ceiling without failure.

$\blacksquare$



---



### Illustration 2: Full Trajectory Tokenization for a 4-Step CartPole Episode



**Problem:**

Consider an episode with 4 timesteps.

(a) Episode parameters: states $s_t \in \mathbb{R}^4$, actions $a_t \in \{0, 1\}$, rewards $[1, 1, 1, 0]$. Compute the exact RTG sequence.

(b) For each $t=1 \dots 4$, compute the token dimensions assuming $d_{\text{model}} = 128$. The RTG is projected by a linear layer ($d_r=1 \to 128$), the state is projected by a linear layer ($d_s=4 \to 128$), and the discrete action uses an embedding table ($d_a=2 \to 128$).

(c) Concatenate the tokens: show that each triplet $[e_R; e_s; e_a] \in \mathbb{R}^{3 \times 128}$.

(d) Compute the full sequence shape and detail the addition of positional embeddings. Show the complete tensor shape at each stage.



**Step-by-Step Solution:**



**(Part a) RTG Sequence Computation:**

Using the backward sum definition $\hat{R}_t = \sum_{k=t}^4 r_k$:

- Step 4: $\hat{R}_4 = r_4 = \mathbf{0}$

- Step 3: $\hat{R}_3 = r_3 + \hat{R}_4 = 1 + 0 = \mathbf{1}$

- Step 2: $\hat{R}_2 = r_2 + \hat{R}_3 = 1 + 1 = \mathbf{2}$

- Step 1: $\hat{R}_1 = r_1 + \hat{R}_2 = 1 + 2 = \mathbf{3}$

The final RTG sequence is explicitly: $\mathbf{[3, 2, 1, 0]}$.



**(Part b) Token Dimensions and Projection:**

The model requires all tokens to share a common hidden dimension $d_{\text{model}} = 128$.

1. **RTG Embedding:** The scalar RTG $\hat{R}_t \in \mathbb{R}^{1}$ is multiplied by $W_R \in \mathbb{R}^{128 \times 1}$. The output is a continuous vector $e_R \in \mathbb{R}^{128}$.

2. **State Embedding:** The continuous state vector $s_t \in \mathbb{R}^4$ is multiplied by $W_s \in \mathbb{R}^{128 \times 4}$. The output is a continuous vector $e_s \in \mathbb{R}^{128}$.

3. **Action Embedding:** The discrete action $a_t \in \{0, 1\}$ is used to index an embedding matrix $W_a \in \mathbb{R}^{2 \times 128}$. The output is a continuous vector $e_a \in \mathbb{R}^{128}$.

Thus, for every single timestep $t$, we produce exactly three $128$-dimensional vectors.



**(Part c) Concatenation of Triplets:**

For a given timestep $t$, the modalities are chronologically ordered as:

$$\text{Triplet}_t = \left[ e_R^{(t)}; \; e_s^{(t)}; \; e_a^{(t)} \right]$$

Since each vector is of shape $(1, 128)$, stacking them sequentially yields a tensor of shape $\mathbf{3 \times 128}$.



**(Part d) Full Sequence Shape and Positional Embedding:**

Over the full $T=4$ steps, we concatenate the 4 triplets sequentially:

$$\text{Total Tokens} = 4 \text{ steps} \times 3 \text{ tokens/step} = \mathbf{12} \text{ tokens}$$

The complete un-encoded sequence tensor $\mathbf{X}$ has shape:

$$\mathbf{X} \in \mathbb{R}^{\mathbf{12 \times 128}}$$

To inject temporal awareness, a learned timestep positional embedding is added. A common approach is to map the timestep integer $t$ to a 128-dimensional vector $p_t \in \mathbb{R}^{128}$.

Crucially, all three tokens belonging to timestep $t$ receive the **exact same** positional embedding $p_t$:

$$X'_{3t-2} = e_R^{(t)} + p_t$$

$$X'_{3t-1} = e_s^{(t)} + p_t$$

$$X'_{3t} = e_a^{(t)} + p_t$$

The addition is element-wise, so the final tensor $\mathbf{X}'$ maintains the shape $\mathbf{12 \times 128}$. This tensor is now perfectly formatted and ready for the causal self-attention layers.

$\blacksquare$



---



### Illustration 3: Causal Self-Attention Forward Pass for T=3 (9 tokens)



**Problem:**

Consider a length 3 sequence (9 tokens) ordered $[\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \hat{R}_3, s_3, a_3]$. Let $d_{\text{model}} = 4$, with 1 attention head.

(a) Show the $9 \times 9$ causal mask, explicitly defining which entries are $0$ (attend) vs $-\infty$ (masked). State the critical rule.

(b) Given concrete $Q$ and $K$ matrices for tokens 2 ($s_1$) and 3 ($a_1$), compute the dot products.

(c) Compute attention scores, apply the mask, softmax, and weighted value. Show full arithmetic for these 2 query tokens.



**Step-by-Step Solution:**



**(Part a) Causal Mask Matrix:**

Let $M_{i, j}$ be the mask for Query token $i$ and Key token $j$ (1-indexed).

$M_{i, j} = 0$ if $j \le i$ (allowed).

$M_{i, j} = -\infty$ if $j > i$ (blocked).

**Critical Rule:** The action prediction head sits on top of the state token $s_t$. Therefore, $s_t$ must NOT be allowed to attend to its own ground truth action $a_t$, nor to any future tokens.

For $T=3$ (9 tokens), the matrix visually looks like a lower triangle of zeros, and an upper triangle of $-\infty$.

For row $2$ (which is $s_1$), the valid keys are $j \in \{1, 2\}$ ($\hat{R}_1, s_1$). Thus $M_{2, 1} = 0, M_{2, 2} = 0$, and $M_{2, 3} = -\infty$.



**(Part b) Dot Products (Unmasked Scores):**

Let $Q_{2:3} = \begin{bmatrix} 1.0 & 0.0 & 1.0 & 0.0 \\ 0.0 & 2.0 & 0.0 & 1.0 \end{bmatrix}$ (Query rows for $s_1$ and $a_1$).

Let $K_{1:3} = \begin{bmatrix} 1.0 & 0.0 & 0.0 & 1.0 \\ 1.0 & 1.0 & 1.0 & 0.0 \\ 0.0 & 2.0 & 0.0 & 1.0 \end{bmatrix}$ (Key rows for $\hat{R}_1, s_1, a_1$).

Compute the unmasked scores $S = Q_{2:3} K_{1:3}^T$:

- **Query 2 ($s_1$) dot Key 1 ($\hat{R}_1$):**

  $S_{2,1} = (1.0)(1.0) + (0.0)(0.0) + (1.0)(0.0) + (0.0)(1.0) = \mathbf{1.0}$

- **Query 2 ($s_1$) dot Key 2 ($s_1$):**

  $S_{2,2} = (1.0)(1.0) + (0.0)(1.0) + (1.0)(1.0) + (0.0)(0.0) = \mathbf{2.0}$

- **Query 2 ($s_1$) dot Key 3 ($a_1$):**

  $S_{2,3} = (1.0)(0.0) + (0.0)(2.0) + (1.0)(0.0) + (0.0)(1.0) = \mathbf{0.0}$

- **Query 3 ($a_1$) dot Key 1 ($\hat{R}_1$):**

  $S_{3,1} = (0.0)(1.0) + (2.0)(0.0) + (0.0)(0.0) + (1.0)(1.0) = \mathbf{1.0}$

- **Query 3 ($a_1$) dot Key 2 ($s_1$):**

  $S_{3,2} = (0.0)(1.0) + (2.0)(1.0) + (0.0)(1.0) + (1.0)(0.0) = \mathbf{2.0}$

- **Query 3 ($a_1$) dot Key 3 ($a_1$):**

  $S_{3,3} = (0.0)(0.0) + (2.0)(2.0) + (0.0)(0.0) + (1.0)(1.0) = \mathbf{5.0}$

Unmasked Scores $S = \begin{bmatrix} 1.0 & 2.0 & 0.0 \\ 1.0 & 2.0 & 5.0 \end{bmatrix}$



**(Part c) Masking, Softmax, and Value Aggregation:**

Scale by $\sqrt{d_{\text{model}}} = \sqrt{4} = 2.0$:

$S_{scaled} = \begin{bmatrix} 0.5 & 1.0 & 0.0 \\ 0.5 & 1.0 & 2.5 \end{bmatrix}$

Apply the Causal Mask (Row 2 cannot see col 3):

$S_{masked} = \begin{bmatrix} 0.5 & 1.0 & -\infty \\ 0.5 & 1.0 & 2.5 \end{bmatrix}$

Apply Softmax (row-wise):

- **Row 2 ($s_1$):**

  $e^{0.5} \approx 1.6487, \quad e^{1.0} \approx 2.7183, \quad e^{-\infty} = 0.0$

  $\Sigma_2 = 1.6487 + 2.7183 = 4.3670$

  Attention Weights $W_2 = \left[ \frac{1.6487}{4.3670}, \frac{2.7183}{4.3670}, 0 \right] = \mathbf{[0.3775, 0.6225, 0.0000]}$

- **Row 3 ($a_1$):**

  $e^{0.5} \approx 1.6487, \quad e^{1.0} \approx 2.7183, \quad e^{2.5} \approx 12.1825$

  $\Sigma_3 = 1.6487 + 2.7183 + 12.1825 = 16.5495$

  Attention Weights $W_3 = \left[ \frac{1.6487}{16.5495}, \frac{2.7183}{16.5495}, \frac{12.1825}{16.5495} \right] = \mathbf{[0.0996, 0.1643, 0.7361]}$



If $V_{1:3} = K_{1:3}$, the context vector for $s_1$ is computed as:

$Y_2 = 0.3775 V_1 + 0.6225 V_2 = 0.3775 \begin{bmatrix} 1.0 \\ 0.0 \\ 0.0 \\ 1.0 \end{bmatrix} + 0.6225 \begin{bmatrix} 1.0 \\ 1.0 \\ 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} \mathbf{1.0000} \\ \mathbf{0.6225} \\ \mathbf{0.6225} \\ \mathbf{0.3775} \end{bmatrix}$

This vector $Y_2$ is perfectly causally insulated from $a_1$ and is safely used to predict $a_1$.

$\blacksquare$



---



### Illustration 4: 5-Step Inference Rollout Protocol



**Problem:**

Walk through a 5-step interactive inference rollout of a deployed Decision Transformer.

Start condition: User prompts with initial target $\hat{R}_0 = 10$. Initial state from environment $s_1 = [0.1, -0.2, 0.3, 0.0]$.

(a) Step 1: Form the context, query the model, get action, execute, and compute the RTG update.

(b) Repeat for steps 2-5 assuming the environment yields rewards $r_2=2, r_3=0, r_4=-1, r_5=4$. Show the full context window growth and the exact RTG decrement sequence explicitly.



**Step-by-Step Solution:**



**(Part a) Step 1 Rollout:**

- **Context Sequence:** The model receives the initial 2 tokens: $[\hat{R}_0=10, \quad s_1=[0.1, -0.2, 0.3, 0.0]]$

- **Prediction:** The output head on the $s_1$ token predicts the required action $a_1$. Suppose it predicts $a_1 = \text{right}$ (action 1).

- **Execution:** We send $a_1$ to the environment. The environment steps forward, yielding reward $r_1 = 1$ and next state $s_2 = [0.15, -0.1, 0.28, 0.05]$.

- **RTG Update:** We decrement the Return-to-Go by the consumed reward:

  $$\hat{R}_1 = \hat{R}_0 - r_1 = 10 - 1 = \mathbf{9}$$



**(Part b) Steps 2-5 Rollout and Context Growth:**

- **Step 2:**

  - Append $(a_1, \hat{R}_1, s_2)$.

  - Context: $[\hat{R}_0, s_1, a_1, \hat{R}_1=9, s_2]$ (Total: 5 tokens)

  - Predict $a_2$. Execute $a_2 \implies$ environment yields $r_2 = 2, s_3$.

  - RTG Update: $\hat{R}_2 = 9 - 2 = \mathbf{7}$.

- **Step 3:**

  - Append $(a_2, \hat{R}_2, s_3)$.

  - Context: $[\hat{R}_0, s_1, a_1, \hat{R}_1, s_2, a_2, \hat{R}_2=7, s_3]$ (Total: 8 tokens)

  - Predict $a_3$. Execute $a_3 \implies$ environment yields $r_3 = 0, s_4$.

  - RTG Update: $\hat{R}_3 = 7 - 0 = \mathbf{7}$.

- **Step 4:**

  - Append $(a_3, \hat{R}_3, s_4)$.

  - Context: $[\dots \dots, \hat{R}_3=7, s_4]$ (Total: 11 tokens)

  - Predict $a_4$. Execute $a_4 \implies$ environment yields $r_4 = -1, s_5$.

  - RTG Update: $\hat{R}_4 = 7 - (-1) = \mathbf{8}$.

- **Step 5:**

  - Append $(a_4, \hat{R}_4, s_5)$.

  - Context: $[\dots \dots, \hat{R}_4=8, s_5]$ (Total: 14 tokens)

  - Predict $a_5$. Execute $a_5 \implies$ environment yields $r_5 = 4, s_6$.

  - RTG Update: $\hat{R}_5 = 8 - 4 = \mathbf{4}$.



**Summary of Rollout Dynamics:**

At the end of step 5, the model's context window contains precisely $14$ tokens, representing the full historical trajectory up to the latest state.

The explicit sequence of RTG decrements passed to the model over time was: $\mathbf{[10, 9, 7, 7, 8]}$.

Notice that at step 4, the agent received a negative reward (penalty), which caused the Return-to-Go budget to *increase* from 7 to 8, signaling to the model that it must now work harder to achieve the remaining target!

$\blacksquare$



---



### Illustration 5: Stitching Failure Mode — DT vs Q-Learning



**Problem:**

A major capability of RL algorithms is "stitching" sub-optimal trajectories to discover optimal behavior.

Consider a deterministic MDP and a dataset with two offline trajectories:

- **Trajectory A:** Starts at $s_1$. Action $A_{good}$ transitions to $s_2$ with reward 5. Action $A_{bad}$ transitions to $s_3$ with reward -2. Total Return = $3$.

- **Trajectory B:** Starts at $s_4$. Action $B_{bad}$ transitions to $s_2$ with reward -3. Action $B_{good}$ transitions to $s_3$ (an excellent state) with reward 8. Total Return = $5$.

The true optimal policy is to take the good first half from A ($s_1 \to s_2$) and the excellent second half from B ($s_2 \to s_3$).

Total Return of A = 3. Total Return of B = 5.

Optimal stitched path: $s_1 \xrightarrow{A_{good}} s_2 \xrightarrow{B_{good}} s_3$. Total Return = $5 + 8 = 13$.

(a) Show why DT CANNOT stitch these trajectories, reverting to distribution average.

(b) Show why Q-learning CAN stitch via Bellman backups.

(c) Quantify the expected return for DT under $\hat{R}=13$ vs Q-learning.



**Step-by-Step Solution:**



**(Part a) Why Decision Transformer CANNOT Stitch:**

Decision Transformers operate purely by conditional density estimation $P(a \mid s, \hat{R})$.

To force the model to execute the stitched optimal trajectory ($Return = 13$), the user must prompt it at $s_1$ with $\hat{R} = 13$.

However, the maximum return ever achieved from $s_1$ in the training dataset is 3 (from Trajectory A).

There is zero support in the training data for the joint probability $P(s_1, \hat{R}=13)$. The condition is entirely out-of-distribution.

Because DT lacks any dynamic programming mechanism (it does not perform backwards temporal value propagation across different trajectory episodes), it cannot recognize that $s_2$ contains a high-value escape route. When prompted with $\hat{R}=13$, it reverts to a noisy average of the actions observed at $s_1$ (e.g., averaging $A_{good}$ and whatever else it saw), fundamentally failing to confidently output $A_{good}$.



**(Part b) Why Q-learning CAN Stitch:**

Q-learning leverages the Bellman equation $Q(s, a) = r + \max_{a'} Q(s', a')$ to propagate information backwards.

1. **Backup from Trajectory B:**

   The transition $s_2 \xrightarrow{B_{good}} s_3$ (reward 8) is processed.

   $Q(s_2, B_{good}) = 8 + 0 = \mathbf{8}$.

   The maximum value at $s_2$ is now updated to $V^*(s_2) = 8$.

2. **Backup from Trajectory A:**

   The transition $s_1 \xrightarrow{A_{good}} s_2$ (reward 5) is processed.

   $Q(s_1, A_{good}) = r + V^*(s_2) = 5 + 8 = \mathbf{13}$.

By breaking trajectories into local atomic transitions $(s, a, r, s')$ and bootstrapping, Q-learning correctly identifies that $A_{good}$ leads to a state $s_2$ worth 8 points, successfully stitching the two disparate trajectories together without ever having seen the full combined path in the dataset.



**(Part c) Quantifying Expected Return:**

- **DT Expected Return:** Prompted with $\hat{R}=13$, the model is guessing OOD. It likely outputs a weighted distribution of actions seen at $s_1$. If it randomly takes $A_{good}$ (reward 5) but fails to take $B_{good}$ at $s_2$ (taking $A_{bad}$ instead), it scores 3. The expected return reverts to the training mean, $\approx \mathbf{4.0}$ (weighted average of (3+5)/2).

- **Q-learning Expected Return:** The agent greedily selects $\arg\max_a Q(s_1, a) = A_{good}$, then at $s_2$ selects $\arg\max_a Q(s_2, a) = B_{good}$, achieving the theoretical maximum of $\mathbf{13.0}$.

This limitation highlights why Decision Transformers often require massive, highly diverse datasets to rival the sample efficiency of TD-based algorithms.

$\blacksquare$



---



## 7. Deep Learning Connection & Modern Applications



### 7.1 RT-1, RT-2, and OpenVLA

Google's Robotics Transformer (RT) series applies the sequence modeling paradigm to physical robot control. RT-1 tokenizes camera images and language instructions alongside robotic actions. RT-2 expands this by co-tuning vision-language models (VLMs) directly on robot action tokens, allowing the model to generalize reasoning (e.g., "pick up the extinct animal" $\to$ grabs dinosaur toy) to low-level motor control. OpenVLA further democratizes this by open-sourcing a 7B parameter vision-language-action model.



### 7.2 Multi-Game Decision Transformers

Unlike Q-learning which often struggles with catastrophic forgetting and scale, transformers excel at multi-task representation. Multi-Game Decision Transformers train a single unified model across dozens of Atari games simultaneously. By feeding the game observation and a generic target return, a single set of weights can master a wide distribution of disjoint MDPs, a feat historically difficult for DQN or PPO.



### 7.3 Gato: A Generalist Agent

DeepMind's Gato takes the sequence formulation to its logical extreme. It tokenizes text, images, robot joint torques, and Atari button presses into a single uniform vocabulary. A single transformer is trained on this massive multimodal offline dataset. During inference, Gato acts as a Decision Transformer when prompted with RL tokens, a language model when prompted with text, and an image captioner when prompted with pixels.



---

## 8. Code Implementation & Verification



For a complete PyTorch implementation of the Decision Transformer, including causal masking and trajectory batching, see:

`code/26_decision_transformer.py`

