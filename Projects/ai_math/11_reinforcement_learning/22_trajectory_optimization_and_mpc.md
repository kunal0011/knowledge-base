# Chapter 22: Trajectory Optimization & Model Predictive Control (MPC)

---

## 1. Intuition & 101 Motivation

In the previous chapter on **Dyna-Q**, we saw how learning an explicit transition model allows an agent to mentally rehearse experiences and update its value function offline. However, Dyna-Q still relied on a global value table $Q(s, a)$ or parameterized policy to select actions.

What if we face a high-dimensional continuous control task—such as landing a SpaceX rocket booster on an autonomous droneship or maneuvering an autonomous vehicle through dense traffic—where learning a global value function across all possible states is intractable?

Enter **Trajectory Optimization** and **Model Predictive Control (MPC)**:
- Instead of learning what to do across the *entire* universe beforehand, we solve an optimization problem *locally* and *on the fly* for the specific state we currently occupy!
- We look ahead over a finite time horizon $H$ (e.g., the next 30 timesteps), optimize a sequence of control actions $(a_0, a_1, \dots, a_{H-1})$ to minimize cumulative costs, and execute the plan.

### The Receding Horizon Principle (Why MPC?)
If we simply plan $H$ steps into the future and blindly execute all $H$ actions, we are performing **open-loop control**. In the real world, two fatal flaws destroy open-loop execution:
1. **Model Inaccuracy:** Any slight discrepancy between our internal transition model $\hat{f}(s, a)$ and true physics $f(s, a)$ compounds exponentially over time.
2. **Stochastic Disturbances:** Unexpected gusts of wind, slippery road patches, or external obstacles knock the system off its predicted trajectory.

To conquer this, **Model Predictive Control (MPC)** adopts the **receding-horizon principle**:
1. At current state $s_t$, solve the trajectory optimization problem over horizon $H$:
   $$U^* = (a_t^*, a_{t+1}^*, \dots, a_{t+H-1}^*)$$
2. **Execute ONLY the first action $a_t^*$ in the real environment.**
3. Advance to the next real state $s_{t+1}$, discard the rest of the plan, observe the true state $s_{t+1}$, and **re-solve the entire optimization problem** over the new receding horizon $[t+1, t+1+H]$.

By constantly re-anchoring the plan to real sensory feedback, MPC converts an open-loop trajectory optimizer into an extraordinarily robust **closed-loop feedback controller**!

```
     t=0: [ a_0* ] ---> Execute in Real World!
          [ a_1*, a_2*, ..., a_H* ] ---> Discarded!
           |
           v
     t=1: Measure true state s_1 (absorbing wind gusts, drift, sensor noise)
          [ a_1** ] ---> Execute in Real World!
          [ a_2**, a_3**, ..., a_{H+1}** ] ---> Discarded!
```

---

## 2. Rigorous Mathematical Formulation

### 2.1 The Finite-Horizon Optimal Control Problem

Let the state at time $t$ be $s_t \in \mathbb{R}^{d_s}$ and control action be $a_t \in \mathbb{R}^{d_a}$. The system evolves according to deterministic (or mean) dynamics:
$$s_{t+1} = f(s_t, a_t)$$

Over a planning horizon $H$, given initial state $s_0$, we wish to find an action sequence $U = (a_0, a_1, \dots, a_{H-1})$ minimizing the objective functional:

$$J(s_0, U) = \ell_f(s_H) + \sum_{t=0}^{H-1} \ell(s_t, a_t)$$

$$\text{subject to} \quad s_{t+1} = f(s_t, a_t), \quad a_t \in \mathcal{A}, \quad s_t \in \mathcal{S}$$

where:
- $\ell(s_t, a_t)$ is the running stage cost (e.g., tracking error + control effort).
- $\ell_f(s_H)$ is the terminal cost (approximating the cost-to-go beyond horizon $H$).
- $\mathcal{A}, \mathcal{S}$ define control and state constraints (e.g., actuator torque saturation, obstacle avoidance).

---

### 2.2 Shooting Methods vs. Direct Collocation

How do we transcribe this continuous-time or discrete-time problem into numerical optimization?

#### 1. Single Shooting
The decision variables are strictly the control inputs $U = (a_0, a_1, \dots, a_{H-1}) \in \mathbb{R}^{H \cdot d_a}$.
The state trajectory is computed sequentially by forward simulation:
$$s_1 = f(s_0, a_0), \quad s_2 = f(s_1, a_1), \quad \dots, \quad s_H = f(s_{H-1}, a_{H-1})$$
- **Pros:** Smallest number of decision variables; constraints are solely on actions $a_t$.
- **Cons:** Highly non-linear and ill-conditioned. A tiny gradient step in $a_0$ causes massive, chaotic changes in $s_H$ (exploding/vanishing sensitivities).

#### 2. Multiple Shooting
Both states and actions are decision variables: $Z = (s_0, a_0, s_1, a_1, \dots, s_{H-1}, a_{H-1}, s_H)$.
Dynamics are enforced via explicit equality constraints:
$$s_{t+1} - f(s_t, a_t) = 0 \quad \forall t \in \{0, \dots, H-1\}$$
- **Pros:** Decouples sensitivity across time steps; robust convergence on highly unstable or chaotic non-linear systems.
- **Cons:** High-dimensional sparse constrained optimization problem requiring specialized interior-point or SQP solvers.

---

### 2.3 The Linear-Quadratic Regulator (LQR)

When dynamics are linear and costs are quadratic, the optimal control problem can be solved **analytically and globally in closed form** via dynamic programming!

#### System Definition:
$$s_{t+1} = A_t s_t + B_t a_t$$
$$\ell(s_t, a_t) = \frac{1}{2} s_t^T Q_t s_t + \frac{1}{2} a_t^T R_t a_t, \quad \ell_f(s_H) = \frac{1}{2} s_H^T Q_f s_H$$

where $Q_t \succeq 0$ (positive semi-definite state penalty), $R_t \succ 0$ (strictly positive definite action penalty), and $Q_f \succeq 0$.

#### Theorem 22.1 (Discrete-Time Dynamic Programming & Riccati Recursion):
The optimal cost-to-go (value function) at every time step $t$ is an exact quadratic function of state:
$$V_t(s) = \frac{1}{2} s^T P_t s$$
with terminal condition:
$$P_H = Q_f$$

The optimal control law is linear in state:
$$a_t^* = K_t s_t$$

The feedback gain matrix $K_t$ and cost matrix $P_t$ are computed via the **Backward Riccati Difference Equations**:
$$K_t = -\left( R_t + B_t^T P_{t+1} B_t \right)^{-1} B_t^T P_{t+1} A_t$$
$$P_t = Q_t + A_t^T P_{t+1} A_t - A_t^T P_{t+1} B_t \left( R_t + B_t^T P_{t+1} B_t \right)^{-1} B_t^T P_{t+1} A_t$$
$$= Q_t + A_t^T P_{t+1} (A_t + B_t K_t)$$

#### Proof:
By induction from $t = H-1$ down to $0$. At terminal step $H$:
$$V_H(s_H) = \frac{1}{2} s_H^T Q_f s_H \implies P_H = Q_f$$

Assume $V_{t+1}(s') = \frac{1}{2} s'^T P_{t+1} s'$. By Bellman's principle of optimality:
$$V_t(s_t) = \min_{a_t} \left[ \frac{1}{2} s_t^T Q_t s_t + \frac{1}{2} a_t^T R_t a_t + V_{t+1}(A_t s_t + B_t a_t) \right]$$
Substitute $V_{t+1}$:
$$Q(s_t, a_t) = \frac{1}{2} s_t^T Q_t s_t + \frac{1}{2} a_t^T R_t a_t + \frac{1}{2} (A_t s_t + B_t a_t)^T P_{t+1} (A_t s_t + B_t a_t)$$

Differentiating with respect to $a_t$ and setting to zero:
$$\nabla_{a_t} Q(s_t, a_t) = R_t a_t + B_t^T P_{t+1} (A_t s_t + B_t a_t) = 0$$
$$(R_t + B_t^T P_{t+1} B_t) a_t = -B_t^T P_{t+1} A_t s_t$$

Since $R_t \succ 0$ and $P_{t+1} \succeq 0$, the matrix $(R_t + B_t^T P_{t+1} B_t)$ is strictly positive definite and invertible:
$$a_t^* = -\left( R_t + B_t^T P_{t+1} B_t \right)^{-1} B_t^T P_{t+1} A_t s_t = K_t s_t$$

Substituting $a_t^*$ back into $Q(s_t, a_t^*)$ yields:
$$V_t(s_t) = \frac{1}{2} s_t^T \left[ Q_t + K_t^T R_t K_t + (A_t + B_t K_t)^T P_{t+1} (A_t + B_t K_t) \right] s_t$$
Using $R_t K_t + B_t^T P_{t+1} (A_t + B_t K_t) = 0$, this simplifies to:
$$P_t = Q_t + A_t^T P_{t+1} (A_t + B_t K_t) \quad \blacksquare$$

---

### 2.4 Iterative LQR (iLQR) and Differential Dynamic Programming (DDP)

For non-linear dynamics $s_{t+1} = f(s_t, a_t)$ and general non-linear costs $\ell(s_t, a_t)$:
1. Linearize dynamics: $A_t \approx \nabla_{s} f(\bar{s}_t, \bar{a}_t)$, $B_t \approx \nabla_{a} f(\bar{s}_t, \bar{a}_t)$ around nominal trajectory $(\bar{s}, \bar{a})$.
2. Quadraticize costs: expand $\ell(s_t, a_t)$ to second order (Taylor series).
3. Solve backward pass for feedback gain $K_t$ and feedforward term $k_t$:
   $$\delta a_t^* = K_t \delta s_t + k_t$$
4. Roll forward with line search to update the nominal trajectory. Iterate until convergence.

---

### 2.5 Sampling-Based Trajectory Optimization (CEM & Random Shooting)

When gradients $\nabla_s f$ or $\nabla_a f$ are unavailable, discontinuous, or non-differentiable (e.g., rigid contact physics, collisions):

#### Cross-Entropy Method (CEM):
CEM iteratively fits an exponential family distribution (typically Gaussian) over the sequence of control inputs $U \in \mathbb{R}^{H \cdot d_a}$:

1. **Initialize:** Gaussian prior $\mathcal{N}(\boldsymbol{\mu}^{(0)}, \boldsymbol{\Sigma}^{(0)})$, where $\boldsymbol{\mu}^{(0)} = \mathbf{0}, \boldsymbol{\Sigma}^{(0)} = \sigma_0^2 \mathbf{I}$.
2. **For iteration $m = 1, \dots, M$:**
   - Sample $K$ candidate trajectories:
     $$U^{(k)} \sim \mathcal{N}(\boldsymbol{\mu}^{(m-1)}, \boldsymbol{\Sigma}^{(m-1)}), \quad k \in \{1, \dots, K\}$$
   - Roll out model forward to compute objective cost for each candidate:
     $$J^{(k)} = J(s_0, U^{(k)})$$
   - Sort candidates by cost in ascending order: $J^{(1)} \le J^{(2)} \le \dots \le J^{(K)}$.
   - Select top $K_{\text{elite}} = \lceil \alpha K \rceil$ elite candidates ($\mathcal{E}$).
   - Update Gaussian parameters using sample mean and variance of the elite set:
     $$\boldsymbol{\mu}_{\text{elite}} = \frac{1}{K_{\text{elite}}} \sum_{k \in \mathcal{E}} U^{(k)}$$
     $$\boldsymbol{\Sigma}_{\text{elite}} = \frac{1}{K_{\text{elite}}} \sum_{k \in \mathcal{E}} (U^{(k)} - \boldsymbol{\mu}_{\text{elite}})(U^{(k)} - \boldsymbol{\mu}_{\text{elite}})^T$$
   - Apply soft Polyak momentum updating:
     $$\boldsymbol{\mu}^{(m)} = \beta \boldsymbol{\mu}_{\text{elite}} + (1 - \beta) \boldsymbol{\mu}^{(m-1)}$$
     $$\boldsymbol{\Sigma}^{(m)} = \beta \boldsymbol{\Sigma}_{\text{elite}} + (1 - \beta) \boldsymbol{\Sigma}^{(m-1)}$$
3. Return the final mean $\boldsymbol{\mu}^{(M)}$ as the optimal plan $U^*$.

---

## 3. Geometric & Physical Interpretation

### 3.1 The Diverging Funnel of Open-Loop vs. Closed-Loop Tube
In dynamical systems theory, non-linear trajectories exhibit positive Lyapunov exponents: small initial errors or disturbances $\delta s_0$ grow exponentially as $\|\delta s_t\| \sim e^{\lambda t} \|\delta s_0\|$.

```
State Space
 ^
 |         /--- Open-Loop Plan Divergence (Drift + Gusts)
 |        /
 |       /    /=== MPC Closed-Loop Tube (Contracting Envelope) ===
 |      /    /
 |-----*----*----*----*----*----*----*----*----*----*----*------> Time t
 |      \    \
 |       \    \================================================
 |        \
```
- **Open-Loop:** Error tube expands exponentially without bound.
- **MPC Receding Horizon:** At every step $t$, measuring true $s_t$ collapses the error tube back to the origin, creating an **asymptotically stable invariant control tube** around the reference trajectory!

---

## 4. Real-World Analogy: Driving at Night in Thick Fog

Imagine driving a car down a winding mountain road at midnight in dense fog:
- Your headlights can only illuminate **50 meters ahead** (the planning horizon $H$).
- You evaluate candidate steering angles and brake pressures to negotiate the upcoming 50 meters safely.
- **Do you close your eyes and drive all 50 meters blindly?** Absolutely not! You tap the gas and steer for **the immediate 0.1 seconds** ($a_0^*$).
- As your car rolls forward, you see the next 50 meters from your new location, notice a pothole that wasn't visible before, and **instantly recalculate the plan**.
- That is **Model Predictive Control**: short-horizon local foresight combined with relentless closed-loop feedback!

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace the complete mathematical mechanics of **Backward Riccati LQR Optimization** and **Forward Closed-Loop Rollout** by hand on a concrete 1D dynamical system.

---

### 5.1 Environment & System Parameters
Consider a 1D scalar system:
- **State:** $s_t \in \mathbb{R}$ (position error relative to origin)
- **Control:** $u_t \in \mathbb{R}$ (acceleration control input)
- **Dynamics:** $s_{t+1} = a s_t + b u_t$, where $a = 1.0, b = 1.0$
- **Horizon:** $H = 2$ (planning over steps $t = 0, 1, 2$)
- **Cost Weights:**
  - Running state cost: $q = 2.0 \implies \ell(s) = \frac{1}{2} q s^2 = 1.0 s^2$
  - Running action cost: $r = 1.0 \implies \ell(u) = \frac{1}{2} r u^2 = 0.5 u^2$
  - Terminal state cost: $p_H = 4.0 \implies \ell_f(s_2) = \frac{1}{2} p_H s_2^2 = 2.0 s_2^2$
- **Initial State:** $s_0 = 10.0$

---

### 5.2 What Refers to What: Legend Protocol Table

| Mathematical Symbol | Computational Variable | Concrete Role in Hand Trace |
| :--- | :--- | :--- |
| $s_t$ | `state_t` | Scalar state at timestep $t$ |
| $u_t$ | `action_t` | Scalar control input at timestep $t$ |
| $a, b$ | `A, B` | Dynamics coefficients ($a = 1.0, b = 1.0$) |
| $q, r$ | `Q, R` | Running stage cost penalties ($q = 2.0, r = 1.0$) |
| $P_t$ | `riccati_P[t]` | Riccati cost-to-go quadratic curvature $V_t(s) = \frac{1}{2} P_t s^2$ |
| $K_t$ | `gain_K[t]` | Optimal linear state feedback gain: $u_t^* = K_t s_t$ |
| $J^*$ | `total_cost` | Total cumulative cost accumulated across horizon |

---

### 5.3 Backward Riccati Pass (Timesteps $t = 2 \to 1 \to 0$)

#### Terminal Step $t = 2$:
$$P_2 = p_H = \mathbf{4.0000}$$

#### Timestep $t = 1$:
1. **Compute Feedback Gain $K_1$:**
   $$K_1 = -\frac{b \cdot P_2 \cdot a}{r + b^2 P_2} = -\frac{1.0 \times 4.0000 \times 1.0}{1.0 + (1.0)^2 \times 4.0000} = -\frac{4.0}{5.0} = \mathbf{-0.8000}$$

2. **Compute Riccati Matrix $P_1$:**
   $$P_1 = q + a^2 P_2 + a P_2 b K_1 = 2.0 + (1.0)^2 \times 4.0000 + 1.0 \times 4.0000 \times 1.0 \times (-0.8000)$$
   $$= 2.0 + 4.0000 - 3.2000 = \mathbf{2.8000}$$

#### Timestep $t = 0$:
1. **Compute Feedback Gain $K_0$:**
   $$K_0 = -\frac{b \cdot P_1 \cdot a}{r + b^2 P_1} = -\frac{1.0 \times 2.8000 \times 1.0}{1.0 + (1.0)^2 \times 2.8000} = -\frac{2.8000}{3.8000} = -\frac{14}{19} \approx \mathbf{-0.736842}$$

2. **Compute Riccati Matrix $P_0$:**
   $$P_0 = q + a^2 P_1 + a P_1 b K_0 = 2.0 + 2.8000 + 2.8000 \times \left( -\frac{14}{19} \right)$$
   $$= 4.8000 - \frac{39.2}{19} = \frac{91.2 - 39.2}{19} = \frac{52}{19} \approx \mathbf{2.736842}$$

---

### 5.4 Forward Optimal Trajectory Rollout (Initial $s_0 = 10.0$)

#### Timestep $t = 0$:
- State: $s_0 = \mathbf{10.0000}$
- Optimal Action:
  $$u_0^* = K_0 s_0 = -\frac{14}{19} \times 10.0 = -\frac{140}{19} \approx \mathbf{-7.368421}$$
- Running Cost $\ell_0$:
  $$\ell_0 = \frac{1}{2} q s_0^2 + \frac{1}{2} r (u_0^*)^2 = \frac{1}{2}(2.0)(10.0)^2 + \frac{1}{2}(1.0)(-7.368421)^2$$
  $$= 100.0000 + 0.5 \times 54.293629 = 100.0000 + 27.146814 = \mathbf{127.146814}$$
- Next State $s_1$:
  $$s_1 = a s_0 + b u_0^* = 10.0000 - 7.368421 = \frac{50}{19} \approx \mathbf{2.631579}$$

#### Timestep $t = 1$:
- State: $s_1 = \mathbf{2.631579}$
- Optimal Action:
  $$u_1^* = K_1 s_1 = -0.8000 \times \frac{50}{19} = -\frac{40}{19} \approx \mathbf{-2.105263}$$
- Running Cost $\ell_1$:
  $$\ell_1 = \frac{1}{2}(2.0)(2.631579)^2 + \frac{1}{2}(1.0)(-2.105263)^2$$
  $$= 6.925208 + 0.5 \times 4.432133 = 6.925208 + 2.216066 = \mathbf{9.141274}$$
- Next State $s_2$ (Terminal State):
  $$s_2 = a s_1 + b u_1^* = \frac{50}{19} - \frac{40}{19} = \frac{10}{19} \approx \mathbf{0.526316}$$

#### Timestep $t = 2$ (Terminal):
- Terminal State: $s_2 = \mathbf{0.526316}$
- Terminal Cost $\ell_f$:
  $$\ell_f = \frac{1}{2} p_H s_2^2 = \frac{1}{2}(4.0)\left( \frac{10}{19} \right)^2 = 2.0 \times \frac{100}{361} = \frac{200}{361} \approx \mathbf{0.554017}$$

---

### 5.5 Total Optimal Cost Verification via Bellman Equivalence

Let us sum the running costs and terminal cost along the forward rollout:
$$J_{\text{rollout}} = \ell_0 + \ell_1 + \ell_f = 127.146814 + 9.141274 + 0.554017 = \mathbf{136.842105}$$

Now evaluate the Riccati value function at the initial state $s_0 = 10.0$:
$$V_0(s_0) = \frac{1}{2} P_0 s_0^2 = \frac{1}{2} \left( \frac{52}{19} \right) (10.0)^2 = \frac{2600}{19} \approx \mathbf{136.842105}$$

$$\mathbf{J_{\text{rollout}} \equiv V_0(s_0) = 136.842105} \quad \text{(Exact Match to 15 decimal places!)}$$

---

### 5.6 Summary Visual Grid: Backward Riccati & Forward Rollout Ledger

| Step $t$ | Phase | State $s_t$ | Gain $K_t$ | Riccati $P_t$ | Action $u_t^*$ | Next $s_{t+1}$ | Step Cost $\ell_t$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t = 2$** | Backward | — | — | **$4.0000$** | — | — | $\ell_f = 0.5540$ |
| **$t = 1$** | Backward | — | **$-0.8000$** | **$2.8000$** | — | — | — |
| **$t = 0$** | Backward | — | **$-0.7368$** | **$2.7368$** | — | — | — |
| **$t = 0$** | Forward | **$10.0000$** | $-0.7368$ | $2.7368$ | **$-7.3684$** | **$2.6316$** | $\ell_0 = 127.1468$ |
| **$t = 1$** | Forward | **$2.6316$** | $-0.8000$ | $2.8000$ | **$-2.1053$** | **$0.5263$** | $\ell_1 = 9.1413$ |
| **$t = 2$** | Forward | **$0.5263$** | — | $4.0000$ | — | — | $\ell_f = 0.5540$ |
| **Total** | — | — | — | — | — | — | **$J^* = 136.8421$** |

---

## 6. Solved Illustrations

### Illustration 1: Disturbance Rejection in MPC vs. Open-Loop Failure
**Problem:**
Suppose at $t = 1$, an unexpected physical wind gust injects a disturbance $\delta = +3.0$ into the state:
$$s_1^{\text{true}} = s_1 + \delta = 2.6316 + 3.0000 = 5.6316$$
Compare the behavior of:
1. **Open-Loop Execution:** The controller blindly executes the pre-planned action $u_1^{\text{open}} = -2.1053$.
2. **Closed-Loop MPC:** The controller senses $s_1^{\text{true}} = 5.6316$ and computes the feedback action $u_1^{\text{mpc}} = K_1 s_1^{\text{true}}$.

**Solution:**
1. **Open-Loop Controller:**
   $$s_2^{\text{open}} = a s_1^{\text{true}} + b u_1^{\text{open}} = 5.6316 - 2.1053 = \mathbf{3.5263}$$
   The terminal error is huge ($3.5263$), and terminal cost explodes:
   $$\ell_f^{\text{open}} = 2.0 \times (3.5263)^2 = \mathbf{24.870}$$

2. **Closed-Loop MPC Controller:**
   $$u_1^{\text{mpc}} = K_1 s_1^{\text{true}} = -0.8000 \times 5.6316 = \mathbf{-4.5053}$$
   The controller actively counter-steers against the wind gust!
   $$s_2^{\text{mpc}} = a s_1^{\text{true}} + b u_1^{\text{mpc}} = 5.6316 - 4.5053 = \mathbf{1.1263}$$
   Terminal cost:
   $$\ell_f^{\text{mpc}} = 2.0 \times (1.1263)^2 = \mathbf{2.537}$$
   **Result:** Closed-loop MPC achieves a **$10\times$ reduction in terminal error cost** by sensing and rejecting the disturbance in real time! $\blacksquare$

---

## 7. Deep RL Connection & Modern Applications

### 1. Probabilistic Ensembles with Trajectory Sampling (PETS, Chua et al., NeurIPS 2018)
Instead of a single deterministic neural network, PETS trains an ensemble of bootstrapped probabilistic neural networks $p_\theta(s_{t+1} \mid s_t, a_t) = \mathcal{N}(\mu_\theta, \sigma_\theta^2)$ to capture both aleatoric and epistemic uncertainty. At every timestep, CEM plans action trajectories by propagating particles across the ensemble, achieving human-level sample efficiency on MuJoCo benchmarks in under 100 episodes!

### 2. Model-Based Policy Optimization (MBPO, Janner et al., 2019)
Uses short model rollouts (branched rollouts of length $k = 1 \dots 15$) starting from real environment states stored in a replay buffer. The imagined transitions train a model-free actor-critic (SAC), avoiding long-horizon model compounding errors while retaining $20\times$ higher sample efficiency than pure model-free SAC.

### 3. Falcon 9 Rocket Guidance (SpaceX Autonomous Drone-Ship Landing)
SpaceX uses Real-Time Convex Optimization (Successive Convexification) and MPC running at 50 Hz to calculate thruster gimbal angles and throttle commands. The receding-horizon controller continuously rejects atmospheric wind turbulence, pinpointing the booster landing pad within centimeters.

---

## 8. Code Implementation & Verification

The accompanying Python script implements:
1. **Part 5 'AI by Hand' Numerical Verification:**
   - Exact verification of backward Riccati gains $K_1 = -0.8000, K_0 \approx -0.736842$.
   - Forward rollout matching cumulative cost $J^* = 136.842105$ against Riccati cost $V_0(s_0)$ down to $< 10^{-14}$.
2. **General Multi-Dimensional LQR Solver:**
   - Full matrix Riccati recursion for arbitrary $A, B, Q, R, Q_f$.
3. **Cross-Entropy Method (CEM) Non-Linear Optimizer:**
   - Sampling-based trajectory optimizer for continuous control tasks.
4. **Closed-Loop MPC vs. Open-Loop Disturbance Rejection Benchmark:**
   - Empirical proof of MPC robustness under stochastic wind gusts.

See implementation in:
[`11_reinforcement_learning/code/22_trajectory_optimization_and_mpc.py`](./code/22_trajectory_optimization_and_mpc.py)
