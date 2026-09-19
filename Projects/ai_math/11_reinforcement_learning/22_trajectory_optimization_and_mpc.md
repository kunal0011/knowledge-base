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

1. **Initialize:** Gaussian prior $\mathcal{N}(\mu^{(0)}, \Sigma^{(0)})$, where $\mu^{(0)} = \mathbf{0}, \Sigma^{(0)} = \sigma_0^2 \mathbf{I}$.
2. **For iteration $m = 1, \dots, M$:**
   - Sample $K$ candidate trajectories:
     $$U^{(k)} \sim \mathcal{N}(\mu^{(m-1)}, \Sigma^{(m-1)}), \quad k \in \{1, \dots, K\}$$
   - Roll out model forward to compute objective cost for each candidate:
     $$J^{(k)} = J(s_0, U^{(k)})$$
   - Sort candidates by cost in ascending order: $J^{(1)} \le J^{(2)} \le \dots \le J^{(K)}$.
   - Select top $K_{\text{elite}} = \lceil \alpha K \rceil$ elite candidates ($\mathcal{E}$).
   - Update Gaussian parameters using sample mean and variance of the elite set:
     $$\mu_{\text{elite}} = \frac{1}{K_{\text{elite}}} \sum_{k \in \mathcal{E}} U^{(k)}$$
     $$\Sigma_{\text{elite}} = \frac{1}{K_{\text{elite}}} \sum_{k \in \mathcal{E}} (U^{(k)} - \mu_{\text{elite}})(U^{(k)} - \mu_{\text{elite}})^T$$
   - Apply soft Polyak momentum updating:
     $$\mu^{(m)} = \beta \mu_{\text{elite}} + (1 - \beta) \mu^{(m-1)}$$
     $$\Sigma^{(m)} = \beta \Sigma_{\text{elite}} + (1 - \beta) \Sigma^{(m-1)}$$
3. Return the final mean $\mu^{(M)}$ as the optimal plan $U^*$.

---

### 2.6 Derivation 11.22.1: Dynamic Programming Principle for Linear Quadratic Regulators (LQR)

#### Part 1: Problem Statement & Mathematical Goal
Consider a discrete-time linear dynamical system evolving over a finite planning horizon $t \in \{0, 1, \dots, T\}$:
$$x_{t+1} = A_t x_t + B_t u_t, \quad x_0 \text{ given}$$
where $x_t \in \mathbb{R}^n$ denotes the state vector, $u_t \in \mathbb{R}^m$ denotes the control input vector, $A_t \in \mathbb{R}^{n \times n}$ is the state transition matrix, and $B_t \in \mathbb{R}^{n \times m}$ is the control input matrix.

The finite-horizon objective functional to be minimized is the quadratic performance index:
$$J(x_0, U) = \frac{1}{2} x_T^\top Q_T x_T + \sum_{t=0}^{T-1} \left( \frac{1}{2} x_t^\top Q_t x_t + \frac{1}{2} u_t^\top R_t u_t \right)$$
where $U = (u_0, u_1, \dots, u_{T-1})$ is the sequence of control inputs.

The optimal cost-to-go (value function) at any time step $t \in \{0, 1, \dots, T\}$ from state $x$ is defined by:
$$V_t(x) = \min_{u_t, \dots, u_{T-1}} \left\{ \frac{1}{2} x_T^\top Q_T x_T + \sum_{\tau=t}^{T-1} \left( \frac{1}{2} x_\tau^\top Q_\tau x_\tau + \frac{1}{2} u_\tau^\top R_\tau u_\tau \right) \right\} \quad \text{subject to } x_{\tau+1} = A_\tau x_\tau + B_\tau u_\tau$$

**Mathematical Goal:** Prove by backward mathematical induction from $t = T$ down to $t = 0$ that:
1. The optimal value function at every time step $t$ is an exact quadratic form in the state:
   $$V_t(x) = \frac{1}{2} x^\top P_t x$$
   with terminal boundary condition $P_T = Q_T$.
2. The optimal control policy is a linear state feedback control law:
   $$u_t^*(x) = K_t x$$
3. The feedback gain matrix $K_t \in \mathbb{R}^{m \times n}$ and the symmetric cost-to-go matrix $P_t \in \mathbb{R}^{n \times n}$ satisfy the backward discrete-time Riccati difference equations:
   $$K_t = -\left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} B_t^\top P_{t+1} A_t$$
   $$P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t \left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} B_t^\top P_{t+1} A_t = Q_t + A_t^\top P_{t+1} (A_t + B_t K_t)$$
4. The Riccati matrix $P_t$ remains symmetric and positive semi-definite ($P_t = P_t^\top \succeq 0$) for all $t$.

#### Part 2: Explicit Assumptions & Regularity Conditions
1. **Linear Dynamics:** The state transition map is strictly linear and deterministic: $x_{t+1} = A_t x_t + B_t u_t$ with known system matrices $A_t, B_t$.
2. **Symmetric Positive Semi-Definite State Penalties:** The stage state penalty matrices $Q_t = Q_t^\top \in \mathbb{R}^{n \times n}$ and terminal penalty matrix $Q_T = Q_T^\top \in \mathbb{R}^{n \times n}$ are symmetric positive semi-definite:
   $$x^\top Q_t x \ge 0, \quad x^\top Q_T x \ge 0 \quad \forall x \in \mathbb{R}^n \quad (Q_t \succeq 0, Q_T \succeq 0)$$
3. **Symmetric Strictly Positive Definite Control Penalties:** The control penalty matrices $R_t = R_t^\top \in \mathbb{R}^{m \times m}$ are symmetric strictly positive definite:
   $$u^\top R_t u > 0 \quad \forall u \in \mathbb{R}^m \setminus \{0\} \quad (R_t \succ 0)$$
   This implies $\lambda_{\min}(R_t) \ge \rho > 0$, guaranteeing that any non-zero actuator effort incurs a strictly positive cost.
4. **Unconstrained State and Control Spaces:** The state $x_t \in \mathbb{R}^n$ and control $u_t \in \mathbb{R}^m$ are unconstrained ($\mathcal{S} = \mathbb{R}^n, \mathcal{A} = \mathbb{R}^m$). Hence, unconstrained first-order necessary conditions for optimality are both necessary and sufficient for global optimality.
5. **Strict Positive Definiteness and Invertibility of the Control Curvature:** Since $P_{t+1} \succeq 0$ and $R_t \succ 0$, for any non-zero vector $v \in \mathbb{R}^m \setminus \{0\}$:
   $$v^\top (R_t + B_t^\top P_{t+1} B_t) v = v^\top R_t v + (B_t v)^\top P_{t+1} (B_t v) \ge v^\top R_t v > 0$$
   Consequently, $(R_t + B_t^\top P_{t+1} B_t) \succ 0$ is strictly positive definite and always invertible.

#### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- **Dynamic Programming Decomposition:** Joint optimization over the entire trajectory sequence $(u_0, \dots, u_{T-1}) \in \mathbb{R}^{m T}$ requires solving a coupled $mT$-dimensional linear system. Bellman's Principle of Optimality decouples this joint problem into $T$ independent single-step subproblems solved backward from the terminal time $T$ to $0$.
- **Geometric Curvature of the Cost Basin:** In the product space $(x_t, u_t)$, the stage cost is an elliptic paraboloid centered at the origin. Composing a quadratic terminal cost through a linear dynamical map preserves the quadratic structure. Slicing this paraboloid along the control subspace for any fixed state $x_t$ yields a strictly convex quadratic bowl in $u_t$. The vertex of this bowl represents the optimal control $u_t^*(x_t)$, which depends linearly on $x_t$.
- **Curvature Projection (Riccati Step):** Evaluating the action-value paraboloid at its vertex $u_t^*$ projects out the control dimensions, yielding a lower-dimensional quadratic bowl in $x_t$ alone. The curvature matrix of this projected bowl is precisely $P_t$. The Riccati difference equation algebraically captures this recursive projection of curvature backward through time.

#### Part 4: End-to-End Step-by-Step Algebraic Proof
The proof proceeds by backward mathematical induction on $t$.

**Base Case ($t = T$):**
At the terminal time $T$, no control actions remain to be selected. The terminal cost-to-go is:
$$V_T(x) = \frac{1}{2} x^\top Q_T x$$
Comparing this with the quadratic ansatz $V_T(x) = \frac{1}{2} x^\top P_T x$, we obtain:
$$P_T = Q_T$$
By Assumption 2, $Q_T = Q_T^\top \succeq 0$, so $P_T$ is symmetric and positive semi-definite. The base case holds.

**Induction Hypothesis:**
Assume that at time step $t+1$, the optimal value function is quadratic:
$$V_{t+1}(x) = \frac{1}{2} x^\top P_{t+1} x$$
where $P_{t+1} = P_{t+1}^\top \succeq 0$.

**Induction Step (Step $t$):**
By Bellman's Dynamic Programming Principle of Optimality, the value function at time $t$ satisfies:
$$V_t(x_t) = \min_{u_t \in \mathbb{R}^m} \left\{ \frac{1}{2} x_t^\top Q_t x_t + \frac{1}{2} u_t^\top R_t u_t + V_{t+1}(x_{t+1}) \right\}$$
Substitute the dynamic transition equation $x_{t+1} = A_t x_t + B_t u_t$ and the induction hypothesis for $V_{t+1}$:
$$V_t(x_t) = \min_{u_t \in \mathbb{R}^m} Q_t(x_t, u_t)$$
where the action-value function $Q_t(x_t, u_t)$ is defined as:
$$Q_t(x_t, u_t) \triangleq \frac{1}{2} x_t^\top Q_t x_t + \frac{1}{2} u_t^\top R_t u_t + \frac{1}{2} (A_t x_t + B_t u_t)^\top P_{t+1} (A_t x_t + B_t u_t)$$

**Step 1: Algebraic expansion of the quadratic form $(A_t x_t + B_t u_t)^\top P_{t+1} (A_t x_t + B_t u_t)$:**
Using the distributive property of matrix multiplication:
$$(A_t x_t + B_t u_t)^\top P_{t+1} (A_t x_t + B_t u_t) = (x_t^\top A_t^\top + u_t^\top B_t^\top) P_{t+1} (A_t x_t + B_t u_t)$$
$$= x_t^\top A_t^\top P_{t+1} A_t x_t + x_t^\top A_t^\top P_{t+1} B_t u_t + u_t^\top B_t^\top P_{t+1} A_t x_t + u_t^\top B_t^\top P_{t+1} B_t u_t$$

Because $P_{t+1} = P_{t+1}^\top$, the cross-term $x_t^\top A_t^\top P_{t+1} B_t u_t$ is a scalar, meaning it equals its own transpose:
$$x_t^\top A_t^\top P_{t+1} B_t u_t = (x_t^\top A_t^\top P_{t+1} B_t u_t)^\top = u_t^\top B_t^\top P_{t+1}^\top A_t x_t = u_t^\top B_t^\top P_{t+1} A_t x_t$$
Therefore, the cross-terms combine:
$$x_t^\top A_t^\top P_{t+1} B_t u_t + u_t^\top B_t^\top P_{t+1} A_t x_t = 2 u_t^\top B_t^\top P_{t+1} A_t x_t$$
Substituting this back into $Q_t(x_t, u_t)$:
$$Q_t(x_t, u_t) = \frac{1}{2} x_t^\top Q_t x_t + \frac{1}{2} u_t^\top R_t u_t + \frac{1}{2} x_t^\top A_t^\top P_{t+1} A_t x_t + u_t^\top B_t^\top P_{t+1} A_t x_t + \frac{1}{2} u_t^\top B_t^\top P_{t+1} B_t u_t$$
Rearranging terms by powers of $u_t$:
$$Q_t(x_t, u_t) = \frac{1}{2} x_t^\top \left( Q_t + A_t^\top P_{t+1} A_t \right) x_t + u_t^\top \left( B_t^\top P_{t+1} A_t x_t \right) + \frac{1}{2} u_t^\top \left( R_t + B_t^\top P_{t+1} B_t \right) u_t$$

**Step 2: First-order and second-order optimality conditions:**
To minimize $Q_t(x_t, u_t)$ with respect to $u_t$, we compute the gradient $\nabla_{u_t} Q_t(x_t, u_t)$:
Using the matrix calculus identities $\nabla_u (u^\top M x) = M x$ and $\nabla_u (\frac{1}{2} u^\top S u) = S u$ for symmetric $S$:
$$\nabla_{u_t} Q_t(x_t, u_t) = B_t^\top P_{t+1} A_t x_t + \left( R_t + B_t^\top P_{t+1} B_t \right) u_t$$
The Hessian with respect to $u_t$ is:
$$\nabla_{u_t u_t}^2 Q_t(x_t, u_t) = R_t + B_t^\top P_{t+1} B_t$$
By Assumption 5, $R_t \succ 0$ and $P_{t+1} \succeq 0$, so $\nabla_{u_t u_t}^2 Q_t \succ 0$ is strictly positive definite everywhere.
Thus, $Q_t(x_t, u_t)$ is strictly convex in $u_t$, and the stationary point where $\nabla_{u_t} Q_t(x_t, u_t) = 0$ is the unique global minimizer.

Setting the gradient to zero:
$$\left( R_t + B_t^\top P_{t+1} B_t \right) u_t + B_t^\top P_{t+1} A_t x_t = 0$$
$$\left( R_t + B_t^\top P_{t+1} B_t \right) u_t = - B_t^\top P_{t+1} A_t x_t$$
Multiplying both sides by the inverse $\left( R_t + B_t^\top P_{t+1} B_t \right)^{-1}$:
$$u_t^*(x_t) = -\left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} B_t^\top P_{t+1} A_t x_t$$
Defining the feedback gain matrix:
$$K_t \triangleq -\left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} B_t^\top P_{t+1} A_t$$
we obtain the optimal linear control law:
$$u_t^*(x_t) = K_t x_t$$

**Step 3: Substituting $u_t^*$ back into $V_t(x_t) = Q_t(x_t, u_t^*(x_t))$:**
Now evaluate $Q_t(x_t, K_t x_t)$:
$$V_t(x_t) = \frac{1}{2} x_t^\top Q_t x_t + \frac{1}{2} (K_t x_t)^\top R_t (K_t x_t) + \frac{1}{2} (A_t x_t + B_t K_t x_t)^\top P_{t+1} (A_t x_t + B_t K_t x_t)$$
Factoring out $x_t^\top$ on the left and $x_t$ on the right:
$$V_t(x_t) = \frac{1}{2} x_t^\top \left[ Q_t + K_t^\top R_t K_t + (A_t + B_t K_t)^\top P_{t+1} (A_t + B_t K_t) \right] x_t$$
This demonstrates that $V_t(x_t)$ is indeed a pure quadratic form:
$$V_t(x_t) = \frac{1}{2} x_t^\top P_t x_t$$
where the matrix $P_t$ is given in **Joseph stabilized form**:
$$P_t = Q_t + K_t^\top R_t K_t + (A_t + B_t K_t)^\top P_{t+1} (A_t + B_t K_t)$$

**Step 4: Simplifying $P_t$ to the standard Riccati difference equation:**
Expand $(A_t + B_t K_t)^\top P_{t+1} (A_t + B_t K_t)$:
$$(A_t + B_t K_t)^\top P_{t+1} (A_t + B_t K_t) = A_t^\top P_{t+1} A_t + A_t^\top P_{t+1} B_t K_t + K_t^\top B_t^\top P_{t+1} A_t + K_t^\top B_t^\top P_{t+1} B_t K_t$$
Group all terms containing $K_t$:
$$P_t = Q_t + A_t^\top P_{t+1} A_t + A_t^\top P_{t+1} B_t K_t + K_t^\top B_t^\top P_{t+1} A_t + K_t^\top \left( R_t + B_t^\top P_{t+1} B_t \right) K_t$$
Now substitute the explicit definition $K_t = -\left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} B_t^\top P_{t+1} A_t$ into the last term:
$$K_t^\top \left( R_t + B_t^\top P_{t+1} B_t \right) K_t = \left[ -A_t^\top P_{t+1} B_t \left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} \right] \left( R_t + B_t^\top P_{t+1} B_t \right) K_t$$
Since $M^{-1} M = I$:
$$K_t^\top \left( R_t + B_t^\top P_{t+1} B_t \right) K_t = -A_t^\top P_{t+1} B_t K_t$$
Substitute this cancellation directly back into the expression for $P_t$:
$$P_t = Q_t + A_t^\top P_{t+1} A_t + A_t^\top P_{t+1} B_t K_t + K_t^\top B_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t K_t$$
The third and fifth terms cancel exactly:
$$P_t = Q_t + A_t^\top P_{t+1} A_t + K_t^\top B_t^\top P_{t+1} A_t$$
Note that $K_t^\top B_t^\top P_{t+1} A_t$ is symmetric:
$$K_t^\top B_t^\top P_{t+1} A_t = - A_t^\top P_{t+1} B_t \left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} B_t^\top P_{t+1} A_t = A_t^\top P_{t+1} B_t K_t$$
Hence:
$$P_t = Q_t + A_t^\top P_{t+1} A_t + A_t^\top P_{t+1} B_t K_t = Q_t + A_t^\top P_{t+1} (A_t + B_t K_t)$$
Substituting $K_t$:
$$P_t = Q_t + A_t^\top P_{t+1} A_t - A_t^\top P_{t+1} B_t \left( R_t + B_t^\top P_{t+1} B_t \right)^{-1} B_t^\top P_{t+1} A_t$$
This is the discrete-time backward Riccati difference equation.

**Step 5: Symmetry and Positive Semi-Definiteness of $P_t$:**
From the Joseph form:
$$P_t = Q_t + K_t^\top R_t K_t + (A_t + B_t K_t)^\top P_{t+1} (A_t + B_t K_t)$$
Let $z \in \mathbb{R}^n$ be any arbitrary vector. Then:
$$z^\top P_t z = z^\top Q_t z + (K_t z)^\top R_t (K_t z) + \left( (A_t + B_t K_t) z \right)^\top P_{t+1} \left( (A_t + B_t K_t) z \right)$$
- $z^\top Q_t z \ge 0$ because $Q_t \succeq 0$.
- $(K_t z)^\top R_t (K_t z) \ge 0$ because $R_t \succ 0$.
- $\left( (A_t + B_t K_t) z \right)^\top P_{t+1} \left( (A_t + B_t K_t) z \right) \ge 0$ because $P_{t+1} \succeq 0$ by the induction hypothesis.
Since each of the three terms is non-negative, $z^\top P_t z \ge 0$ for all $z \in \mathbb{R}^n$, meaning $P_t \succeq 0$. Furthermore, taking the transpose of the Joseph form:
$$P_t^\top = Q_t^\top + K_t^\top R_t^\top K_t + (A_t + B_t K_t)^\top P_{t+1}^\top (A_t + B_t K_t) = P_t$$
Thus $P_t = P_t^\top \succeq 0$.

This completes the induction step for all $t \in \{T-1, \dots, 0\}$. $\blacksquare$

---

### 2.7 Derivation 11.22.2: Iterative LQR (iLQR) / DDP Quadratic Expansion

#### Part 1: Problem Statement & Mathematical Goal
Consider a general continuous nonlinear dynamical system in discrete time:
$$x_{t+1} = f(x_t, u_t)$$
where $x_t \in \mathbb{R}^n$, $u_t \in \mathbb{R}^m$, with finite-horizon cumulative cost functional:
$$J(x_0, U) = \ell_f(x_T) + \sum_{t=0}^{T-1} \ell(x_t, u_t)$$
where $\ell(x, u)$ is the running stage cost and $\ell_f(x)$ is the terminal cost.

Suppose we are given a nominal state-action trajectory:
$$\bar{\tau} = (\bar{x}_0, \bar{u}_0, \bar{x}_1, \bar{u}_1, \dots, \bar{x}_{T-1}, \bar{u}_{T-1}, \bar{x}_T)$$
satisfying the nominal dynamic recursion $\bar{x}_{t+1} = f(\bar{x}_t, \bar{u}_t)$. Let the local state and control perturbations around this nominal trajectory be:
$$\delta x_t \triangleq x_t - \bar{x}_t, \quad \delta u_t \triangleq u_t - \bar{u}_t$$

**Mathematical Goal:**
1. Derive the second-order Taylor series expansion of the perturbed action-value function:
   $$\Delta Q(\delta x_t, \delta u_t) \triangleq Q(\bar{x}_t + \delta x_t, \bar{u}_t + \delta u_t) - Q(\bar{x}_t, \bar{u}_t)$$
   around the nominal pair $(\bar{x}_t, \bar{u}_t)$.
2. Explicitly compute the gradient vectors and Hessian matrices:
   $$Q_x, Q_u, Q_{xx}, Q_{uu}, Q_{ux}$$
   in terms of stage cost derivatives $(\ell_x, \ell_u, \ell_{xx}, \ell_{uu}, \ell_{ux})$, dynamic Jacobians $(f_x, f_u)$, dynamic Hessians (for Differential Dynamic Programming) and their Gauss-Newton omission (for Iterative LQR), and the next-step value function derivatives $(V_x', V_{xx}')$.
3. Solve the unconstrained local quadratic minimization problem to find the optimal control perturbation:
   $$\delta u_t^*(\delta x_t) = \arg\min_{\delta u_t} \Delta Q(\delta x_t, \delta u_t) = k_t + K_t \delta x_t$$
   and derive exact expressions for the feedforward offset $k_t \in \mathbb{R}^m$ and feedback gain $K_t \in \mathbb{R}^{m \times n}$.
4. Derive the backward pass value function update for the current step:
   $$V_x(t) \in \mathbb{R}^n, \quad V_{xx}(t) \in \mathbb{R}^{n \times n}$$
   and the expected step cost reduction $\Delta V_t$.

#### Part 2: Explicit Assumptions & Regularity Conditions
1. **$C^2$ Differentiability:** The dynamics $f: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^n$, running cost $\ell: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$, and terminal cost $\ell_f: \mathbb{R}^n \to \mathbb{R}$ are twice continuously differentiable with Lipschitz continuous second derivatives.
2. **Strict Positive Definiteness of Control Curvature:** The action-space Hessian matrix:
   $$Q_{uu} \triangleq \ell_{uu} + f_u^\top V_{xx}' f_u + \sum_{i=1}^n [V_x']_i [f_{uu}]_i$$
   satisfies $Q_{uu} \succ 0$. In regions where $Q_{uu}$ has non-positive eigenvalues (due to non-convex dynamics or cost), a Levenberg-Marquardt regularization is applied: $\tilde{Q}_{uu} = Q_{uu} + \mu I_m$ with $\mu > 0$ chosen sufficiently large so that $\tilde{Q}_{uu} \succ 0$ and is invertible.
3. **Local Perturbation Validity:** The state and control deviations remain sufficiently small ($\|\delta x_t\| \le \epsilon, \|\delta u_t\| \le \epsilon$) so that Taylor expansion residuals of order $\mathcal{O}(\|\delta x\|^3 + \|\delta u\|^3)$ are negligible.
4. **iLQR Approximation vs. Full DDP:** In Iterative LQR (iLQR), the second-order derivatives of the system dynamics with respect to state and control are neglected:
   $$[f_{xx}]_i \approx \mathbf{0}_{n \times n}, \quad [f_{uu}]_i \approx \mathbf{0}_{m \times m}, \quad [f_{ux}]_i \approx \mathbf{0}_{m \times n} \quad \forall i \in \{1, \dots, n\}$$
   This represents a Gauss-Newton approximation of the full Differential Dynamic Programming (DDP) equations, requiring only the dynamic Jacobians $f_x$ and $f_u$.

#### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- **Iterative Local Quadratic Models:** Finding global optimal trajectories on general nonlinear manifolds is NP-hard. iLQR sidesteps global intractability by fitting a sequence of osculating quadratic paraboloids to the cost-to-go hypersurface along the current rollout.
- **Physical Interpretation of $(k_t, K_t)$:**
  - *Feedforward control $k_t = -Q_{uu}^{-1} Q_u$:* Represents a steepest-descent Newton step along the nominal trajectory. When $\delta x_t = 0$, $k_t$ pushes the control inputs directly downhill toward the bottom of the cost basin.
  - *Feedback gain $K_t = -Q_{uu}^{-1} Q_{ux}$:* Represents a stabilizing linear controller. When unmodeled physics or initial condition shifts cause the system to deviate by $\delta x_t \neq 0$, $K_t \delta x_t$ steers the system back onto the optimal ridge.
- **Forward Rollout with Backtracking Line Search:** After backward recursions compute $(k_t, K_t)_{t=0}^{T-1}$, a forward simulation is executed with a step size parameter $\alpha \in (0, 1]$:
  $$u_t^{\text{new}} = \bar{u}_t + \alpha k_t + K_t (x_t^{\text{new}} - \bar{x}_t)$$
  guaranteeing monotonic cost reduction via Armijo line search.

#### Part 4: End-to-End Step-by-Step Algebraic Proof
Let the optimal value function at step $t+1$ around $\bar{x}_{t+1}$ be expanded to second order:
$$V_{t+1}(\bar{x}_{t+1} + \delta x_{t+1}) \approx V_{t+1}(\bar{x}_{t+1}) + V_x'^\top \delta x_{t+1} + \frac{1}{2} \delta x_{t+1}^\top V_{xx}' \delta x_{t+1}$$
where $V_x' \triangleq \nabla_x V_{t+1}(\bar{x}_{t+1}) \in \mathbb{R}^n$ and $V_{xx}' \triangleq \nabla^2_{xx} V_{t+1}(\bar{x}_{t+1}) \in \mathbb{R}^{n \times n}$.

**Step 1: Taylor Expansion of the Dynamics:**
The true next state is $x_{t+1} = f(\bar{x}_t + \delta x_t, \bar{u}_t + \delta u_t) = \bar{x}_{t+1} + \delta x_{t+1}$.
Expanding each coordinate $f_i(x_t, u_t)$ around $(\bar{x}_t, \bar{u}_t)$:
$$\delta x_{t+1} = f_x \delta x_t + f_u \delta u_t + \frac{1}{2} \sum_{i=1}^n e_i \left( \delta x_t^\top [f_{xx}]_i \delta x_t + 2 \delta u_t^\top [f_{ux}]_i \delta x_t + \delta u_t^\top [f_{uu}]_i \delta u_t \right) + \mathcal{O}(3)$$
where:
$$[f_x]_{ij} = \frac{\partial f_i}{\partial x_j}, \quad [f_u]_{ij} = \frac{\partial f_i}{\partial u_j}, \quad [f_{xx}]_i = \nabla_{xx}^2 f_i, \quad [f_{uu}]_i = \nabla_{uu}^2 f_i, \quad [f_{ux}]_i = \nabla_{ux}^2 f_i$$

**Step 2: Taylor Expansion of the Future Value Function:**
Substitute $\delta x_{t+1}$ into the linear term $V_x'^\top \delta x_{t+1}$:
$$V_x'^\top \delta x_{t+1} = V_x'^\top \left( f_x \delta x_t + f_u \delta u_t \right) + \frac{1}{2} \sum_{i=1}^n [V_x']_i \left( \delta x_t^\top [f_{xx}]_i \delta x_t + 2 \delta u_t^\top [f_{ux}]_i \delta x_t + \delta u_t^\top [f_{uu}]_i \delta u_t \right)$$
For the quadratic term $\frac{1}{2} \delta x_{t+1}^\top V_{xx}' \delta x_{t+1}$, higher-order terms in $\delta x_{t+1}$ yield terms of order $\mathcal{O}(3)$ or higher, so we retain only the first-order dynamics expansion $\delta x_{t+1} \approx f_x \delta x_t + f_u \delta u_t$:
$$\frac{1}{2} \delta x_{t+1}^\top V_{xx}' \delta x_{t+1} = \frac{1}{2} (f_x \delta x_t + f_u \delta u_t)^\top V_{xx}' (f_x \delta x_t + f_u \delta u_t)$$
$$= \frac{1}{2} \delta x_t^\top f_x^\top V_{xx}' f_x \delta x_t + \delta u_t^\top f_u^\top V_{xx}' f_x \delta x_t + \frac{1}{2} \delta u_t^\top f_u^\top V_{xx}' f_u \delta u_t$$

**Step 3: Taylor Expansion of the Stage Cost:**
Expand $\ell(\bar{x}_t + \delta x_t, \bar{u}_t + \delta u_t)$ to second order:
$$\ell(\bar{x}_t + \delta x_t, \bar{u}_t + \delta u_t) \approx \ell(\bar{x}_t, \bar{u}_t) + \ell_x^\top \delta x_t + \ell_u^\top \delta u_t + \frac{1}{2} \delta x_t^\top \ell_{xx} \delta x_t + \frac{1}{2} \delta u_t^\top \ell_{uu} \delta u_t + \delta u_t^\top \ell_{ux} \delta x_t$$

**Step 4: Assembling the Action-Value Function Expansion $\Delta Q(\delta x_t, \delta u_t)$:**
Recall $Q(x_t, u_t) = \ell(x_t, u_t) + V_{t+1}(f(x_t, u_t))$. Define:
$$\Delta Q(\delta x_t, \delta u_t) \triangleq Q(\bar{x}_t + \delta x_t, \bar{u}_t + \delta u_t) - Q(\bar{x}_t, \bar{u}_t)$$
Summing the expansions from Steps 2 and 3:
$$\Delta Q(\delta x_t, \delta u_t) \approx Q_x^\top \delta x_t + Q_u^\top \delta u_t + \frac{1}{2} \delta x_t^\top Q_{xx} \delta x_t + \frac{1}{2} \delta u_t^\top Q_{uu} \delta u_t + \delta u_t^\top Q_{ux} \delta x_t$$
Equating terms order-by-order:
- **First-order gradient vectors:**
  $$Q_x = \ell_x + f_x^\top V_x'$$
  $$Q_u = \ell_u + f_u^\top V_x'$$
- **Second-order Hessian matrices (Differential Dynamic Programming):**
  $$Q_{xx} = \ell_{xx} + f_x^\top V_{xx}' f_x + \sum_{i=1}^n [V_x']_i [f_{xx}]_i$$
  $$Q_{uu} = \ell_{uu} + f_u^\top V_{xx}' f_u + \sum_{i=1}^n [V_x']_i [f_{uu}]_i$$
  $$Q_{ux} = \ell_{ux} + f_u^\top V_{xx}' f_x + \sum_{i=1}^n [V_x']_i [f_{ux}]_i$$
- **Iterative LQR (iLQR) Gauss-Newton Simplification:**
  Dropping the second-order dynamic tensor terms according to Assumption 4:
  $$Q_{xx} = \ell_{xx} + f_x^\top V_{xx}' f_x$$
  $$Q_{uu} = \ell_{uu} + f_u^\top V_{xx}' f_u$$
  $$Q_{ux} = \ell_{ux} + f_u^\top V_{xx}' f_x$$

**Step 5: Solving for Optimal Perturbation $\delta u_t^*(\delta x_t)$:**
For any given state perturbation $\delta x_t$, we find $\delta u_t$ that minimizes $\Delta Q$:
$$\nabla_{\delta u_t} \Delta Q(\delta x_t, \delta u_t) = Q_u + Q_{uu} \delta u_t + Q_{ux} \delta x_t = 0$$
Since $Q_{uu} \succ 0$ by Assumption 2, we solve directly:
$$Q_{uu} \delta u_t = - Q_u - Q_{ux} \delta x_t$$
$$\delta u_t^*(\delta x_t) = - Q_{uu}^{-1} Q_u - Q_{uu}^{-1} Q_{ux} \delta x_t$$
We define:
$$k_t \triangleq - Q_{uu}^{-1} Q_u \quad (\text{feedforward term})$$
$$K_t \triangleq - Q_{uu}^{-1} Q_{ux} \quad (\text{feedback gain matrix})$$
yielding the optimal control perturbation:
$$\delta u_t^*(\delta x_t) = k_t + K_t \delta x_t$$

**Step 6: Backward Update of the Value Function:**
Substitute $\delta u_t^* = k_t + K_t \delta x_t$ back into $\Delta Q(\delta x_t, \delta u_t)$ to obtain $\Delta V_t(\delta x_t) \equiv \Delta Q(\delta x_t, \delta u_t^*(\delta x_t))$:
$$\Delta V_t(\delta x_t) = Q_x^\top \delta x_t + Q_u^\top (k_t + K_t \delta x_t) + \frac{1}{2} \delta x_t^\top Q_{xx} \delta x_t + \frac{1}{2} (k_t + K_t \delta x_t)^\top Q_{uu} (k_t + K_t \delta x_t) + (k_t + K_t \delta x_t)^\top Q_{ux} \delta x_t$$

Expand each term and collect by powers of $\delta x_t$:
1. **Constant Term ($\Delta V_t$, independent of $\delta x_t$):**
   $$\Delta V_t = Q_u^\top k_t + \frac{1}{2} k_t^\top Q_{uu} k_t$$
   Since $k_t = - Q_{uu}^{-1} Q_u \implies Q_{uu} k_t = - Q_u$, we have:
   $$Q_u^\top k_t = -(Q_{uu} k_t)^\top k_t = - k_t^\top Q_{uu} k_t$$
   Thus:
   $$\Delta V_t = - k_t^\top Q_{uu} k_t + \frac{1}{2} k_t^\top Q_{uu} k_t = -\frac{1}{2} k_t^\top Q_{uu} k_t = \frac{1}{2} k_t^\top Q_u$$
   Because $Q_{uu} \succ 0$, $\Delta V_t \le 0$ represents the guaranteed non-positive expected cost reduction.

2. **Linear Term in $\delta x_t$ ($V_x(t)^\top \delta x_t$):**
   $$V_x(t)^\top \delta x_t = \left[ Q_x^\top + Q_u^\top K_t + k_t^\top Q_{uu} K_t + k_t^\top Q_{ux} \right] \delta x_t$$
   Notice that:
   $$Q_u^\top K_t + k_t^\top Q_{uu} K_t = (Q_u + Q_{uu} k_t)^\top K_t = \mathbf{0}^\top K_t = 0$$
   Furthermore, using $K_t = - Q_{uu}^{-1} Q_{ux} \implies Q_{ux}^\top = - K_t^\top Q_{uu}$:
   $$k_t^\top Q_{ux} = (Q_{ux}^\top k_t)^\top = (-K_t^\top Q_{uu} k_t)^\top = (K_t^\top Q_u)^\top = Q_u^\top K_t$$
   Therefore:
   $$V_x(t) = Q_x + Q_{ux}^\top k_t = Q_x + K_t^\top Q_u$$

3. **Quadratic Term in $\delta x_t$ ($\frac{1}{2} \delta x_t^\top V_{xx}(t) \delta x_t$):**
   $$\frac{1}{2} \delta x_t^\top V_{xx}(t) \delta x_t = \frac{1}{2} \delta x_t^\top \left[ Q_{xx} + K_t^\top Q_{uu} K_t + 2 K_t^\top Q_{ux} \right] \delta x_t$$
   Using $Q_{ux} = - Q_{uu} K_t$:
   $$2 K_t^\top Q_{ux} = - 2 K_t^\top Q_{uu} K_t$$
   Substituting this into the bracket:
   $$V_{xx}(t) = Q_{xx} + K_t^\top Q_{uu} K_t - 2 K_t^\top Q_{uu} K_t = Q_{xx} - K_t^\top Q_{uu} K_t$$
   Equivalently, substituting $K_t = - Q_{uu}^{-1} Q_{ux}$:
   $$V_{xx}(t) = Q_{xx} - Q_{ux}^\top Q_{uu}^{-1} Q_{ux} = Q_{xx} + K_t^\top Q_{ux}$$

**Step 7: Terminal Boundary Condition:**
At the final step $t = T$, the value function is simply the terminal cost:
$$V_T(x_T) = \ell_f(x_T) \implies V_x(T) = \nabla_x \ell_f(\bar{x}_T), \quad V_{xx}(T) = \nabla^2_{xx} \ell_f(\bar{x}_T)$$
This establishes the complete backward recursion for iLQR/DDP. $\blacksquare$

---

### 2.8 Derivation 11.22.3: Model Predictive Control (MPC) Receding Horizon Stability Theorem

#### Part 1: Problem Statement & Mathematical Goal
Consider a discrete-time autonomous nonlinear dynamical system:
$$x_{t+1} = f(x_t, u_t)$$
where $x_t \in \mathbb{X} \subseteq \mathbb{R}^n$ is the state and $u_t \in \mathbb{U} \subseteq \mathbb{R}^m$ is the control input. The origin $x = 0, u = 0$ is an equilibrium point: $f(0, 0) = 0$.

At each sampling time $t$, given current measurement $x_t = x$, Model Predictive Control solves a constrained finite-horizon optimal control problem over horizon $N \ge 1$:
$$V_N^*(x) \triangleq \min_{U_N} \left\{ \sum_{k=0}^{N-1} \ell(x_k, u_k) + V_f(x_N) \right\}$$
$$\text{subject to:} \quad x_{k+1} = f(x_k, u_k), \quad x_0 = x$$
$$x_k \in \mathbb{X}, \quad u_k \in \mathbb{U} \quad \forall k \in \{0, 1, \dots, N-1\}$$
$$x_N \in \mathbb{X}_f$$
where:
- $U_N = (u_0, u_1, \dots, u_{N-1})$ is the sequence of control inputs over the horizon.
- $\ell(x, u)$ is the running stage cost with $\ell(0, 0) = 0$.
- $V_f(x)$ is the terminal cost functional with $V_f(0) = 0$.
- $\mathbb{X}_f \subseteq \mathbb{X}$ is the terminal constraint set.

Let the set of feasible states for which this problem admits a solution be:
$$\mathcal{X}_N \triangleq \{ x \in \mathbb{X} \mid \exists U_N \in \mathbb{U}^N \text{ such that } x_k \in \mathbb{X} \ \forall k \in \{0, \dots, N-1\} \text{ and } x_N \in \mathbb{X}_f \}$$
Let $U_N^*(x) = (u_0^*(x), u_1^*(x), \dots, u_{N-1}^*(x))$ be the optimal control sequence. The **Receding Horizon Control Policy** executes strictly the first control input:
$$\kappa_{\text{mpc}}(x) \triangleq u_0^*(x)$$
The resulting closed-loop autonomous system is:
$$x_{t+1} = f(x_t, \kappa_{\text{mpc}}(x_t))$$

**Mathematical Goal:**
Prove that under standard terminal Lyapunov design conditions (existence of a local invariant terminal set $\mathbb{X}_f$ and a local controller $\kappa_f(x)$ satisfying $V_f(f(x, \kappa_f(x))) - V_f(x) \le -\ell(x, \kappa_f(x))$):
1. **Recursive Feasibility:** The feasible set $\mathcal{X}_N$ is positively invariant under closed-loop MPC feedback: if $x_0 \in \mathcal{X}_N$, then $x_t \in \mathcal{X}_N$ for all $t \ge 0$.
2. **Lyapunov Decrease Property:** The optimal value function $V_N^*(x)$ serves as a strict Lyapunov function for the closed-loop system, satisfying:
   $$V_N^*(f(x, \kappa_{\text{mpc}}(x))) - V_N^*(x) \le -\ell(x, \kappa_{\text{mpc}}(x)) \quad \forall x \in \mathcal{X}_N$$
3. **Asymptotic Stability:** The closed-loop equilibrium at the origin $x = 0$ is asymptotically stable with region of attraction $\mathcal{X}_N$.

#### Part 2: Explicit Assumptions & Regularity Conditions
1. **Continuity and Equilibrium:** The vector field $f: \mathbb{X} \times \mathbb{U} \to \mathbb{R}^n$ is continuous on $\mathbb{X} \times \mathbb{U}$, and $f(0, 0) = 0$.
2. **Constraint Sets:** The state constraint set $\mathbb{X}$ and control constraint set $\mathbb{U}$ are closed, with $0 \in \operatorname{int}(\mathbb{X})$ and $0 \in \operatorname{int}(\mathbb{U})$. The terminal set $\mathbb{X}_f \subseteq \mathbb{X}$ is closed and contains the origin in its interior: $0 \in \operatorname{int}(\mathbb{X}_f)$.
3. **Strictly Positive Stage Cost:** The stage cost $\ell: \mathbb{X} \times \mathbb{U} \to \mathbb{R}_{\ge 0}$ is continuous, satisfies $\ell(0, 0) = 0$, and there exists a class $\mathcal{K}_\infty$ function $\alpha_1$ such that:
   $$\ell(x, u) \ge \alpha_1(\|x\|) \quad \forall x \in \mathbb{X}, \ u \in \mathbb{U}$$
   Hence, $\ell(x, u) > 0$ for all $x \neq 0$.
4. **Terminal Controller and Invariance:** There exists a continuous local terminal control law $\kappa_f: \mathbb{X}_f \to \mathbb{U}$ such that for all $x \in \mathbb{X}_f$:
   $$f(x, \kappa_f(x)) \in \mathbb{X}_f$$
   That is, the terminal set $\mathbb{X}_f$ is forward (positively) invariant under $\kappa_f$.
5. **Terminal Lyapunov Decrease Condition:** The terminal cost functional $V_f: \mathbb{X}_f \to \mathbb{R}_{\ge 0}$ is continuous, $V_f(0) = 0$, and satisfies the discrete Lyapunov decrease inequality:
   $$V_f(f(x, \kappa_f(x))) - V_f(x) \le -\ell(x, \kappa_f(x)) \quad \forall x \in \mathbb{X}_f$$
6. **Local Boundedness of Value Function:** There exists a class $\mathcal{K}_\infty$ function $\alpha_2$ such that $V_N^*(x) \le \alpha_2(\|x\|)$ for all $x$ in a neighborhood of the origin.

#### Part 3: Underlying Intuition & Geometric / Physical Interpretation
- **The Finite-Horizon Myopia Trap:** When an agent optimizes over a finite horizon $N$, it does not inherently care about what happens at step $N+1$. Left unconstrained, the optimizer might drive the system at maximum velocity toward an obstacle or boundary at step $N$ to minimize costs over the interval $[0, N-1]$, causing catastrophic instability when executed in a receding horizon fashion.
- **Terminal Invariant Set as an Infinite-Horizon Bridge:** The terminal conditions $(\mathbb{X}_f, V_f, \kappa_f)$ eliminate this myopia. By requiring $x_N \in \mathbb{X}_f$, we guarantee that once the planned trajectory enters $\mathbb{X}_f$, the auxiliary controller $\kappa_f$ can steer the system forward for infinite time without violating constraints.
- **Lyapunov Cost-to-Go Surrogate:** Summing the terminal Lyapunov condition $V_f(x_{k+1}) - V_f(x_k) \le -\ell(x_k, \kappa_f(x_k))$ from $k = N$ to $\infty$ yields:
  $$V_f(x_N) \ge \sum_{k=N}^\infty \ell(x_k, \kappa_f(x_k))$$
  Thus, $V_f(x_N)$ acts as an exact upper bound on the infinite-horizon tail cost.
- **The Shifted Candidate Plan:** At the next sampling instant, the controller constructs a feasible candidate sequence by shifting the previous optimal plan by one step and appending the terminal control $\kappa_f(x_N^*)$. Because the new plan can only improve upon this candidate, the optimal cost $V_N^*$ must strictly decrease from step to step, forcing the real system to converge to the origin.

#### Part 4: End-to-End Step-by-Step Algebraic Proof
Let $x \in \mathcal{X}_N$ be the measured state at the current time step.

**Step 1: The Optimal Trajectory at Current State $x$:**
By definition of feasibility, there exists an optimal control sequence $U_N^*(x)$:
$$U_N^*(x) = \left( u_0^*, u_1^*, \dots, u_{N-1}^* \right)$$
and an associated optimal state sequence $X_N^*(x) = \left( x_0^*, x_1^*, \dots, x_N^* \right)$ satisfying:
$$x_0^* = x$$
$$x_{k+1}^* = f(x_k^*, u_k^*) \quad \forall k \in \{0, 1, \dots, N-1\}$$
$$x_k^* \in \mathbb{X}, \quad u_k^* \in \mathbb{U} \quad \forall k \in \{0, 1, \dots, N-1\}$$
$$x_N^* \in \mathbb{X}_f$$

The optimal objective value is:
$$V_N^*(x) = \sum_{k=0}^{N-1} \ell(x_k^*, u_k^*) + V_f(x_N^*) = \ell(x, u_0^*) + \sum_{k=1}^{N-1} \ell(x_k^*, u_k^*) + V_f(x_N^*)$$

**Step 2: Receding Horizon Transition to Next State $x^+$:**
The MPC controller applies the first action $\kappa_{\text{mpc}}(x) = u_0^*$. The true system transitions to:
$$x^+ = f(x, \kappa_{\text{mpc}}(x)) = f(x, u_0^*) = x_1^*$$

**Step 3: Construction of a Feasible Candidate Sequence $\tilde{U}_N$ at State $x^+$:**
To bound $V_N^*(x^+)$, construct a warm-start candidate control sequence $\tilde{U}_N$ of length $N$ starting from $x^+$:
$$\tilde{U}_N \triangleq \left( \tilde{u}_0, \tilde{u}_1, \dots, \tilde{u}_{N-2}, \tilde{u}_{N-1} \right) = \left( u_1^*, u_2^*, \dots, u_{N-1}^*, \kappa_f(x_N^*) \right)$$
Let the predicted state trajectory resulting from applying $\tilde{U}_N$ starting from $\tilde{x}_0 = x^+ = x_1^*$ be $(\tilde{x}_0, \tilde{x}_1, \dots, \tilde{x}_N)$:
- For $k \in \{0, 1, \dots, N-2\}$:
  $$\tilde{u}_k = u_{k+1}^* \in \mathbb{U}$$
  $$\tilde{x}_{k+1} = f(\tilde{x}_k, \tilde{u}_k) = f(x_{k+1}^*, u_{k+1}^*) = x_{k+2}^* \in \mathbb{X}$$
- At step $k = N-1$:
  $$\tilde{x}_{N-1} = x_N^* \in \mathbb{X}_f \subseteq \mathbb{X}$$
  $$\tilde{u}_{N-1} = \kappa_f(x_N^*) \in \mathbb{U} \quad (\text{by Assumption 4})$$
  $$\tilde{x}_N = f(\tilde{x}_{N-1}, \tilde{u}_{N-1}) = f(x_N^*, \kappa_f(x_N^*)) \in \mathbb{X}_f \quad (\text{by positive invariance of } \mathbb{X}_f \text{ under } \kappa_f)$$

Every element of $\tilde{U}_N$ satisfies the control constraints $\mathbb{U}$, all intermediate states $\tilde{x}_k$ satisfy the state constraints $\mathbb{X}$, and the terminal state satisfies $\tilde{x}_N \in \mathbb{X}_f$.
Therefore, $\tilde{U}_N$ is a strictly admissible, feasible control sequence for state $x^+$.
This proves that $x^+ \in \mathcal{X}_N$, establishing **Recursive Feasibility** for all $t \ge 0$.

**Step 4: Objective Value of the Candidate Sequence:**
The cost associated with candidate sequence $\tilde{U}_N$ evaluated from $x^+$ is:
$$J_N(x^+, \tilde{U}_N) = \sum_{k=0}^{N-1} \ell(\tilde{x}_k, \tilde{u}_k) + V_f(\tilde{x}_N)$$
$$= \sum_{k=0}^{N-2} \ell(\tilde{x}_k, \tilde{u}_k) + \ell(\tilde{x}_{N-1}, \tilde{u}_{N-1}) + V_f(\tilde{x}_N)$$
Substitute the trajectories:
$$J_N(x^+, \tilde{U}_N) = \sum_{k=0}^{N-2} \ell(x_{k+1}^*, u_{k+1}^*) + \ell(x_N^*, \kappa_f(x_N^*)) + V_f(f(x_N^*, \kappa_f(x_N^*)))$$
Perform index substitution $j = k+1$ on the summation:
$$\sum_{k=0}^{N-2} \ell(x_{k+1}^*, u_{k+1}^*) = \sum_{j=1}^{N-1} \ell(x_j^*, u_j^*)$$
Thus:
$$J_N(x^+, \tilde{U}_N) = \sum_{j=1}^{N-1} \ell(x_j^*, u_j^*) + \ell(x_N^*, \kappa_f(x_N^*)) + V_f(f(x_N^*, \kappa_f(x_N^*)))$$

**Step 5: Algebraic Difference $J_N(x^+, \tilde{U}_N) - V_N^*(x)$:**
Subtract $V_N^*(x) = \ell(x, u_0^*) + \sum_{j=1}^{N-1} \ell(x_j^*, u_j^*) + V_f(x_N^*)$:
$$J_N(x^+, \tilde{U}_N) - V_N^*(x) = \sum_{j=1}^{N-1} \ell(x_j^*, u_j^*) + \ell(x_N^*, \kappa_f(x_N^*)) + V_f(f(x_N^*, \kappa_f(x_N^*))) - \left[ \ell(x, u_0^*) + \sum_{j=1}^{N-1} \ell(x_j^*, u_j^*) + V_f(x_N^*) \right]$$
The common summation $\sum_{j=1}^{N-1} \ell(x_j^*, u_j^*)$ cancels out completely:
$$J_N(x^+, \tilde{U}_N) - V_N^*(x) = -\ell(x, u_0^*) + \left[ V_f(f(x_N^*, \kappa_f(x_N^*))) - V_f(x_N^*) + \ell(x_N^*, \kappa_f(x_N^*)) \right]$$

**Step 6: Applying the Terminal Lyapunov Decrease Condition:**
By Assumption 5, for all $x_N^* \in \mathbb{X}_f$:
$$V_f(f(x_N^*, \kappa_f(x_N^*))) - V_f(x_N^*) \le -\ell(x_N^*, \kappa_f(x_N^*))$$
Rearranging:
$$V_f(f(x_N^*, \kappa_f(x_N^*))) - V_f(x_N^*) + \ell(x_N^*, \kappa_f(x_N^*)) \le 0$$
Therefore, the bracketed term in Step 5 is non-positive:
$$J_N(x^+, \tilde{U}_N) - V_N^*(x) \le -\ell(x, u_0^*)$$

**Step 7: Optimality Inequality:**
Because $V_N^*(x^+)$ is the minimum of $J_N(x^+, U)$ over all admissible sequences, and $\tilde{U}_N$ is a feasible sequence:
$$V_N^*(x^+) \le J_N(x^+, \tilde{U}_N)$$
Combining this with Step 6:
$$V_N^*(x^+) - V_N^*(x) \le J_N(x^+, \tilde{U}_N) - V_N^*(x) \le -\ell(x, u_0^*)$$
Since $x^+ = f(x, \kappa_{\text{mpc}}(x))$ and $u_0^* = \kappa_{\text{mpc}}(x)$:
$$V_N^*(f(x, \kappa_{\text{mpc}}(x))) - V_N^*(x) \le -\ell(x, \kappa_{\text{mpc}}(x))$$
This establishes the discrete Lyapunov decrease property.

**Step 8: Proving Asymptotic Stability:**
1. **Positive Definiteness:** By Assumption 3, $\ell(x, u) \ge \alpha_1(\|x\|)$. Since $V_N^*(x) \ge \ell(x, u_0^*) \ge \alpha_1(\|x\|)$, $V_N^*(x) > 0$ for all $x \neq 0$ and $V_N^*(0) = 0$.
2. **Decrescence:** By Assumption 6, $V_N^*(x) \le \alpha_2(\|x\|)$ in a neighborhood of the origin.
3. **Strict Negative Definiteness of the Lyapunov Difference:**
   $$\Delta V_N^*(x) \triangleq V_N^*(f(x, \kappa_{\text{mpc}}(x))) - V_N^*(x) \le -\ell(x, \kappa_{\text{mpc}}(x)) \le -\alpha_1(\|x\|) < 0 \quad \forall x \in \mathcal{X}_N \setminus \{0\}$$

By Lyapunov's Direct Method for discrete-time autonomous systems, $V_N^*(x)$ is a valid strict Lyapunov function.
Therefore, the origin $x = 0$ is asymptotically stable for the closed-loop MPC system, and every trajectory starting in $\mathcal{X}_N$ converges to the origin:
$$\lim_{t \to \infty} x_t = 0$$
with region of attraction $\mathcal{X}_N$. $\blacksquare$

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

### Illustration 2: 1D LQR Backward Riccati Recursion over $T=2$ Timesteps

**Problem:**
Consider a 1D discrete-time linear system over a finite horizon $T = 2$ ($t \in \{0, 1, 2\}$):
$$x_{t+1} = A x_t + B u_t$$
with parameters:
- Dynamics: $A = 1.1$, $B = 0.5$ (an open-loop unstable system since $|A| = 1.1 > 1.0$).
- State stage penalty: $Q = 1.0$.
- Control stage penalty: $R = 0.1$.
- Terminal state penalty: $Q_f = 2.0$.

1. Compute the backward Riccati sequence: terminal curvature $P_2$, feedback gain $K_1$ and curvature $P_1$ at $t = 1$, and feedback gain $K_0$ and curvature $P_0$ at $t = 0$ in both exact rational fraction and floating-point decimal form.
2. Given initial state $x_0 = 2.0$, execute the optimal forward rollout ($u_0^*, x_1, u_1^*, x_2$), calculate the stage and terminal costs, and verify that the cumulative trajectory cost matches the Bellman value $V_0(x_0) = \frac{1}{2} P_0 x_0^2$ down to machine precision.

**Solution:**

#### Step 1: Backward Riccati Recursion

- **Terminal Step $t = 2$:**
  $$P_2 = Q_f = 2.0 = \mathbf{\frac{2}{1}}$$

- **Step $t = 1$:**
  1. *Denominator of Feedback Gain:*
     $$D_1 = R + B^2 P_2 = 0.1 + (0.5)^2 \times 2.0 = \frac{1}{10} + \frac{1}{4} \times 2 = \frac{1}{10} + \frac{1}{2} = \frac{6}{10} = \mathbf{\frac{3}{5}} = 0.600000$$
  2. *Feedback Gain $K_1$:*
     $$K_1 = -\frac{B P_2 A}{D_1} = -\frac{0.5 \times 2.0 \times 1.1}{0.6} = -\frac{1.1}{0.6} = \mathbf{-\frac{11}{6}} \approx \mathbf{-1.833333}$$
  3. *Riccati Curvature $P_1$:*
     $$P_1 = Q + A^2 P_2 + A P_2 B K_1 = 1.0 + (1.1)^2 \times 2.0 + 1.1 \times 2.0 \times 0.5 \times \left(-\frac{11}{6}\right)$$
     $$= 1.0 + 1.21 \times 2.0 + 1.1 \times \left(-\frac{11}{6}\right) = 1.0 + 2.42 - \frac{12.1}{6} = 3.42 - \frac{121}{60}$$
     Expressing as rational fractions:
     $$3.42 = \frac{171}{50} \implies P_1 = \frac{171}{50} - \frac{121}{60} = \frac{1026 - 605}{300} = \mathbf{\frac{421}{300}} \approx \mathbf{1.403333}$$

- **Step $t = 0$:**
  1. *Denominator of Feedback Gain:*
     $$D_0 = R + B^2 P_1 = \frac{1}{10} + \left(\frac{1}{2}\right)^2 \times \frac{421}{300} = \frac{1}{10} + \frac{421}{1200} = \frac{120 + 421}{1200} = \mathbf{\frac{541}{1200}} \approx 0.450833$$
  2. *Numerator of Feedback Gain:*
     $$N_0 = B P_1 A = \frac{1}{2} \times \frac{421}{300} \times \frac{11}{10} = \frac{4631}{6000} \approx 0.771833$$
  3. *Feedback Gain $K_0$:*
     $$K_0 = -\frac{N_0}{D_0} = -\frac{4631 / 6000}{541 / 1200} = -\frac{4631}{6000} \times \frac{1200}{541} = -\frac{4631}{5 \times 541} = \mathbf{-\frac{4631}{2705}} \approx \mathbf{-1.712015}$$
  4. *Riccati Curvature $P_0$:*
     $$P_0 = Q + A^2 P_1 + A P_1 B K_0 = 1.0 + \left(\frac{11}{10}\right)^2 \times \frac{421}{300} + \frac{4631}{6000} \times \left(-\frac{4631}{2705}\right)$$
     $$= 1.0 + \frac{121}{100} \times \frac{421}{300} - \frac{21446161}{16230000} = 1.0 + \frac{50941}{30000} - \frac{21446161}{16230000}$$
     Finding common denominator $16230000 = 30000 \times 541$:
     $$\frac{80941}{30000} - \frac{21446161}{16230000} = \frac{80941 \times 541 - 21446161}{16230000} = \frac{43789081 - 21446161}{16230000} = \frac{22342920}{16230000} = \mathbf{\frac{186191}{135250}} \approx \mathbf{1.376643}$$

---

#### Step 2: Forward Rollout & Exact Energy Verification ($x_0 = 2.0$)

- **Timestep $t = 0$:**
  - State: $x_0 = 2.0 = \frac{2}{1}$
  - Action:
    $$u_0^* = K_0 x_0 = -\frac{4631}{2705} \times 2 = \mathbf{-\frac{9262}{2705}} \approx \mathbf{-3.424030}$$
  - Stage cost $\ell_0$:
    $$\ell_0 = \frac{1}{2} Q x_0^2 + \frac{1}{2} R (u_0^*)^2 = \frac{1}{2}(1)(4) + \frac{1}{2}\left(\frac{1}{10}\right)\left(-\frac{9262}{2705}\right)^2 = 2 + \frac{1}{20}\left(\frac{85784644}{7317025}\right) = \mathbf{\frac{94616411}{36585125}} \approx \mathbf{2.586199}$$
  - Next state $x_1$:
    $$x_1 = A x_0 + B u_0^* = \frac{11}{10}(2) + \frac{1}{2}\left(-\frac{9262}{2705}\right) = \frac{11}{5} - \frac{4631}{2705} = \frac{5951 - 4631}{2705} = \frac{1320}{2705} = \mathbf{\frac{264}{541}} \approx \mathbf{0.487985}$$

- **Timestep $t = 1$:**
  - State: $x_1 = \frac{264}{541} \approx 0.487985$
  - Action:
    $$u_1^* = K_1 x_1 = -\frac{11}{6} \times \frac{264}{541} = -\frac{11 \times 44}{541} = \mathbf{-\frac{484}{541}} \approx \mathbf{-0.894640}$$
  - Stage cost $\ell_1$:
    $$\ell_1 = \frac{1}{2}(1)\left(\frac{264}{541}\right)^2 + \frac{1}{2}\left(\frac{1}{10}\right)\left(-\frac{484}{541}\right)^2 = \frac{69696 + 23425.6}{2 \times 292681} = \mathbf{\frac{232804}{1463405}} \approx \mathbf{0.159084}$$
  - Next state $x_2$:
    $$x_2 = A x_1 + B u_1^* = \frac{11}{10}\left(\frac{264}{541}\right) + \frac{1}{2}\left(-\frac{484}{541}\right) = \frac{1452}{2705} - \frac{1210}{2705} = \mathbf{\frac{242}{2705}} \approx \mathbf{0.089464}$$

- **Terminal Step $t = 2$:**
  - Terminal cost $\ell_f$:
    $$\ell_f = \frac{1}{2} Q_f x_2^2 = \frac{1}{2}(2.0)\left(\frac{242}{2705}\right)^2 = \mathbf{\frac{58564}{7317025}} \approx \mathbf{0.008004}$$

- **Total Cumulative Rollout Cost vs. Riccati Value Function:**
  $$J^* = \ell_0 + \ell_1 + \ell_f = \frac{94616411}{36585125} + \frac{232804}{1463405} + \frac{58564}{7317025} = \mathbf{\frac{186191}{67625}} \approx \mathbf{2.753287}$$
  Riccati value function evaluated at initial state $x_0 = 2.0$:
  $$V_0(x_0) = \frac{1}{2} P_0 x_0^2 = \frac{1}{2}\left(\frac{186191}{135250}\right)(4) = \mathbf{\frac{186191}{67625}} \approx \mathbf{2.753287}$$

$$\mathbf{J^* \equiv V_0(x_0) = \frac{186191}{67625} \approx 2.753287} \quad \text{(Exact rational fraction match!)} \quad \blacksquare$$

---

### Illustration 3: iLQR 1-Step Backward Expansion with Nonlinear Pendulum Dynamics

**Problem:**
Consider the non-linear dynamics of a simple pendulum with state $x = [\theta, \omega]^\top \in \mathbb{R}^2$ (angle and angular velocity) and control torque $u \in \mathbb{R}$. With gravity $g = 10 \text{ m/s}^2$, length $l = 1.0 \text{ m}$, mass $m = 1.0 \text{ kg}$, and forward Euler discretization step $\Delta t = 0.1 \text{ s}$:
$$x_{t+1} = f(x_t, u_t) = \begin{bmatrix} \theta_t + \Delta t \, \omega_t \\ \omega_t - \Delta t \frac{g}{l} \sin\theta_t + \Delta t \frac{1}{m l^2} u_t \end{bmatrix} = \begin{bmatrix} \theta_t + 0.1 \omega_t \\ \omega_t - \sin\theta_t + 0.1 u_t \end{bmatrix}$$

Let the quadratic stage cost be:
$$\ell(x, u) = \frac{1}{2} x^\top Q x + \frac{1}{2} R u^2, \quad Q = \begin{bmatrix} 2.0 & 0 \\ 0 & 0.5 \end{bmatrix}, \quad R = 0.2$$
and the terminal cost at step $t+1$ be $\ell_f(x) = \frac{1}{2} x^\top Q_f x$ with $Q_f = \begin{bmatrix} 5.0 & 0 \\ 0 & 1.0 \end{bmatrix}$.

At nominal operating point:
$$\bar{x}_t = \begin{bmatrix} \bar{\theta} \\ \bar{\omega} \end{bmatrix} = \begin{bmatrix} \pi/6 \\ 0.2 \end{bmatrix} \approx \begin{bmatrix} 0.523599 \\ 0.200000 \end{bmatrix}, \quad \bar{u}_t = 0.0$$
Perform a complete 1-step iLQR backward expansion:
1. Compute next nominal state $\bar{x}_{t+1} = f(\bar{x}_t, \bar{u}_t)$.
2. Compute dynamics Jacobians $f_x, f_u$.
3. Compute future value function derivatives $V'_x, V'_{xx}$.
4. Form the quadratic action-value approximation matrices $Q_x, Q_u, Q_{xx}, Q_{uu}, Q_{ux}$.
5. Calculate the optimal feedforward adjustment $k_t$, feedback gain matrix $K_t$, and expected cost reduction $\Delta V_t$.
6. Update current value function derivatives $V_x(t), V_{xx}(t)$.

**Solution:**

#### Step 1: Forward Nominal State
$$\bar{\theta}_{t+1} = \bar{\theta} + 0.1 \bar{\omega} = \frac{\pi}{6} + 0.1(0.2) = 0.523599 + 0.02 = \mathbf{0.543599} \text{ rad}$$
$$\bar{\omega}_{t+1} = \bar{\omega} - \sin(\pi/6) + 0.1(0.0) = 0.2 - 0.5 = \mathbf{-0.300000} \text{ rad/s}$$
$$\bar{x}_{t+1} = \begin{bmatrix} 0.543599 \\ -0.300000 \end{bmatrix}$$

#### Step 2: Dynamics Jacobians
$$f_x = \begin{bmatrix} \frac{\partial f_1}{\partial \theta} & \frac{\partial f_1}{\partial \omega} \\ \frac{\partial f_2}{\partial \theta} & \frac{\partial f_2}{\partial \omega} \end{bmatrix} = \begin{bmatrix} 1 & 0.1 \\ -\cos(\pi/6) & 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.1 \\ -\frac{\sqrt{3}}{2} & 1.0 \end{bmatrix} \approx \begin{bmatrix} \mathbf{1.000000} & \mathbf{0.100000} \\ \mathbf{-0.866025} & \mathbf{1.000000} \end{bmatrix}$$
$$f_u = \begin{bmatrix} \frac{\partial f_1}{\partial u} \\ \frac{\partial f_2}{\partial u} \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix}$$

#### Step 3: Cost and Future Value Derivatives
- **Stage cost derivatives at $(\bar{x}_t, \bar{u}_t)$:**
  $$\ell_x = Q \bar{x}_t = \begin{bmatrix} 2.0 & 0 \\ 0 & 0.5 \end{bmatrix} \begin{bmatrix} 0.523599 \\ 0.200000 \end{bmatrix} = \begin{bmatrix} \mathbf{1.047198} \\ \mathbf{0.100000} \end{bmatrix}, \quad \ell_u = R \bar{u}_t = \mathbf{0.0}$$
  $$\ell_{xx} = Q = \begin{bmatrix} 2.0 & 0 \\ 0 & 0.5 \end{bmatrix}, \quad \ell_{uu} = R = 0.2, \quad \ell_{ux} = \begin{bmatrix} 0.0 & 0.0 \end{bmatrix}$$
- **Future value derivatives at $\bar{x}_{t+1}$:**
  $$V'_x = Q_f \bar{x}_{t+1} = \begin{bmatrix} 5.0 & 0 \\ 0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.543599 \\ -0.300000 \end{bmatrix} = \begin{bmatrix} \mathbf{2.717994} \\ \mathbf{-0.300000} \end{bmatrix}$$
  $$V'_{xx} = Q_f = \begin{bmatrix} 5.0 & 0 \\ 0 & 1.0 \end{bmatrix}$$

#### Step 4: Quadratic Expansion Coefficients
1. **Control gradient $Q_u$:**
   $$Q_u = \ell_u + f_u^\top V'_x = 0.0 + \begin{bmatrix} 0.0 & 0.1 \end{bmatrix} \begin{bmatrix} 2.717994 \\ -0.300000 \end{bmatrix} = 0.1 \times (-0.3) = \mathbf{-0.030000}$$
2. **Control curvature $Q_{uu}$:**
   $$Q_{uu} = \ell_{uu} + f_u^\top V'_{xx} f_u = 0.2 + \begin{bmatrix} 0.0 & 0.1 \end{bmatrix} \begin{bmatrix} 5.0 & 0 \\ 0 & 1.0 \end{bmatrix} \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix} = 0.2 + (0.1)^2 \times 1.0 = 0.2 + 0.01 = \mathbf{0.210000}$$
3. **Cross-coupling matrix $Q_{ux}$:**
   $$Q_{ux} = \ell_{ux} + f_u^\top V'_{xx} f_x = \begin{bmatrix} 0 & 0 \end{bmatrix} + \begin{bmatrix} 0.0 & 0.1 \end{bmatrix} \begin{bmatrix} 5.0 & 0 \\ 0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.1 \\ -0.866025 & 1.0 \end{bmatrix}$$
   $$= \begin{bmatrix} 0.0 & 0.1 \end{bmatrix} \begin{bmatrix} 1.0 & 0.1 \\ -0.866025 & 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.086603} & \mathbf{0.100000} \end{bmatrix}$$
4. **State gradient $Q_x$:**
   $$Q_x = \ell_x + f_x^\top V'_x = \begin{bmatrix} 1.047198 \\ 0.100000 \end{bmatrix} + \begin{bmatrix} 1.0 & -0.866025 \\ 0.1 & 1.0 \end{bmatrix} \begin{bmatrix} 2.717994 \\ -0.300000 \end{bmatrix}$$
   $$= \begin{bmatrix} 1.047198 \\ 0.100000 \end{bmatrix} + \begin{bmatrix} 2.717994 + 0.259808 \\ 0.271799 - 0.300000 \end{bmatrix} = \begin{bmatrix} \mathbf{4.024999} \\ \mathbf{0.071799} \end{bmatrix}$$
5. **State Hessian $Q_{xx}$:**
   $$Q_{xx} = \ell_{xx} + f_x^\top V'_{xx} f_x = \begin{bmatrix} 2.0 & 0 \\ 0 & 0.5 \end{bmatrix} + \begin{bmatrix} 1.0 & -0.866025 \\ 0.1 & 1.0 \end{bmatrix} \begin{bmatrix} 5.0 & 0.5 \\ -0.866025 & 1.0 \end{bmatrix}$$
   $$= \begin{bmatrix} 2.0 & 0 \\ 0 & 0.5 \end{bmatrix} + \begin{bmatrix} 5.0 + 0.75 & 0.5 - 0.866025 \\ 0.5 - 0.866025 & 0.05 + 1.0 \end{bmatrix} = \begin{bmatrix} \mathbf{7.750000} & \mathbf{-0.366025} \\ \mathbf{-0.366025} & \mathbf{1.550000} \end{bmatrix}$$

#### Step 5: Feedforward and Feedback Policy Gains
- **Feedforward torque adjustment $k_t$:**
  $$k_t = - Q_{uu}^{-1} Q_u = -\frac{-0.030000}{0.210000} = \mathbf{\frac{1}{7}} \approx \mathbf{+0.142857} \text{ N}\cdot\text{m}$$
  *(Physical meaning: Because the pendulum is dropping downward with negative angular velocity $\omega = -0.3$, the optimizer injects positive torque $k_t > 0$ to counteract gravity).*
- **Feedback gain matrix $K_t$:**
  $$K_t = - Q_{uu}^{-1} Q_{ux} = -\frac{1}{0.21} \begin{bmatrix} -0.086603 & 0.100000 \end{bmatrix} = \begin{bmatrix} \frac{0.086603}{0.21} & -\frac{0.100000}{0.21} \end{bmatrix} = \begin{bmatrix} \mathbf{+0.412393} & \mathbf{-0.476190} \end{bmatrix}$$
  *(Physical meaning: Positive angular error increases corrective torque, while positive velocity decreases torque, providing active derivative damping).*

#### Step 6: Expected Improvement and Value Updates
- **Expected cost reduction:**
  $$\Delta V_t = -\frac{1}{2} k_t^\top Q_{uu} k_t = -\frac{1}{2} \left(\frac{1}{7}\right) (0.21) \left(\frac{1}{7}\right) = -\frac{0.21}{98} = \mathbf{-\frac{3}{1400}} \approx \mathbf{-0.002143}$$
- **Current value gradient $V_x(t)$:**
  $$V_x(t) = Q_x + Q_{ux}^\top k_t = \begin{bmatrix} 4.024999 \\ 0.071799 \end{bmatrix} + \begin{bmatrix} -0.086603 \\ 0.100000 \end{bmatrix} \left(\frac{1}{7}\right) = \begin{bmatrix} 4.024999 - 0.012372 \\ 0.071799 + 0.014286 \end{bmatrix} = \begin{bmatrix} \mathbf{4.012627} \\ \mathbf{0.086085} \end{bmatrix}$$
- **Current value Hessian $V_{xx}(t)$:**
  $$V_{xx}(t) = Q_{xx} - K_t^\top Q_{uu} K_t = \begin{bmatrix} 7.750000 & -0.366025 \\ -0.366025 & 1.550000 \end{bmatrix} - 0.21 \begin{bmatrix} 0.412393 \\ -0.476190 \end{bmatrix} \begin{bmatrix} 0.412393 & -0.476190 \end{bmatrix}$$
  $$= \begin{bmatrix} 7.750000 - 0.035714 & -0.366025 + 0.041239 \\ -0.366025 + 0.041239 & 1.550000 - 0.047619 \end{bmatrix} = \begin{bmatrix} \mathbf{7.714286} & \mathbf{-0.324786} \\ \mathbf{-0.324786} & \mathbf{1.502381} \end{bmatrix} \quad \blacksquare$$

---

### Illustration 4: Cross-Entropy Method (CEM) Trajectory Sampling for MPC

**Problem:**
Consider a 1D continuous control system starting from $s_0 = 2.0$ with linear dynamics $s_{t+1} = s_t + u_t$ over a planning horizon $H = 2$ ($t = 0, 1$). The stage cost is $\ell(s, u) = s^2 + u^2$, and the terminal penalty is $\ell_f(s_2) = 2 s_2^2$.
The total trajectory objective for candidate control sequence $U = (u_0, u_1)$ is:
$$J(U) = (s_1^2 + u_0^2) + (2 s_2^2 + u_1^2), \quad \text{where } s_1 = s_0 + u_0, \quad s_2 = s_1 + u_1$$

Given $N = 10$ sampled candidate trajectories generated from an initial Gaussian prior $\mathcal{N}(\mu^{(0)}, \Sigma^{(0)})$ with $\mu^{(0)} = [0.0, 0.0]^\top$ and $\sigma_0 = 1.0$:
1. Compute the trajectory rollouts $(s_1, s_2)$ and evaluate the cumulative cost $J$ for each candidate.
2. Rank all $N = 10$ candidates and select the top $K = 3$ elites ($\mathcal{E}$).
3. Calculate the elite sample mean $\mu_{\text{elite}}$ and population standard deviation $\sigma_{\text{elite}}$.
4. Apply Polyak smoothing with momentum factor $\beta = 0.7$ to obtain the updated sampling distribution parameters $(\mu^{(1)}, \sigma^{(1)})$.

**Solution:**

#### Step 1: Candidate Rollout and Cost Evaluation Table

| Candidate $k$ | Sample Action $u_0$ | Sample Action $u_1$ | State $s_1 = 2.0 + u_0$ | State $s_2 = s_1 + u_1$ | Step Cost $\ell_0 = s_1^2 + u_0^2$ | Tail Cost $\ell_1 + \ell_f = 2 s_2^2 + u_1^2$ | Total Cost $J^{(k)}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $-1.2$ | $-0.6$ | $0.8$ | $0.2$ | $0.64 + 1.44 = 2.08$ | $2(0.04) + 0.36 = 0.44$ | **$2.520$** |
| **2** | $-0.5$ | $-0.2$ | $1.5$ | $1.3$ | $2.25 + 0.25 = 2.50$ | $2(1.69) + 0.04 = 3.42$ | **$5.920$** |
| **3** | $-1.5$ | $-0.4$ | $0.5$ | $0.1$ | $0.25 + 2.25 = 2.50$ | $2(0.01) + 0.16 = 0.18$ | **$2.680$** |
| **4** | $-0.8$ | $-1.0$ | $1.2$ | $0.2$ | $1.44 + 0.64 = 2.08$ | $2(0.04) + 1.00 = 1.08$ | **$3.160$** |
| **5** | $-1.0$ | $-0.8$ | $1.0$ | $0.2$ | $1.00 + 1.00 = 2.00$ | $2(0.04) + 0.64 = 0.72$ | **$2.720$** |
| **6** | $-0.2$ | $-0.5$ | $1.8$ | $1.3$ | $3.24 + 0.04 = 3.28$ | $2(1.69) + 0.25 = 3.63$ | **$6.910$** |
| **7** | $-1.4$ | $-0.5$ | $0.6$ | $0.1$ | $0.36 + 1.96 = 2.32$ | $2(0.01) + 0.25 = 0.27$ | **$2.590$** |
| **8** | $-0.7$ | $-0.3$ | $1.3$ | $1.0$ | $1.69 + 0.49 = 2.18$ | $2(1.00) + 0.09 = 2.09$ | **$4.270$** |
| **9** | $-1.3$ | $-0.5$ | $0.7$ | $0.2$ | $0.49 + 1.69 = 2.18$ | $2(0.04) + 0.25 = 0.33$ | **$2.510$** |
| **10** | $-0.4$ | $-0.8$ | $1.6$ | $0.8$ | $2.56 + 0.16 = 2.72$ | $2(0.64) + 0.64 = 1.92$ | **$4.640$** |

---

#### Step 2: Sorting and Elite Selection ($K = 3$)
Ordering candidates by ascending total cost:
1. **Rank 1:** Candidate 9 with $J = \mathbf{2.510}$, vector $U^{(9)} = [-1.3, -0.5]$
2. **Rank 2:** Candidate 1 with $J = \mathbf{2.520}$, vector $U^{(1)} = [-1.2, -0.6]$
3. **Rank 3:** Candidate 7 with $J = \mathbf{2.590}$, vector $U^{(7)} = [-1.4, -0.5]$
4. Rank 4: Candidate 3 ($J = 2.680$)
5. Rank 5: Candidate 5 ($J = 2.720$)
6. Rank 6: Candidate 4 ($J = 3.160$)
7. Rank 7: Candidate 8 ($J = 4.270$)
8. Rank 8: Candidate 10 ($J = 4.640$)
9. Rank 9: Candidate 2 ($J = 5.920$)
10. Rank 10: Candidate 6 ($J = 6.910$)

The elite set is $\mathcal{E} = \{ U^{(9)}, U^{(1)}, U^{(7)} \}$.

---

#### Step 3: Elite Parameter Fitting
- **Sample Mean:**
  $$\mu_{\text{elite}, 0} = \frac{-1.3 + (-1.2) + (-1.4)}{3} = \frac{-3.9}{3} = \mathbf{-1.300000}$$
  $$\mu_{\text{elite}, 1} = \frac{-0.5 + (-0.6) + (-0.5)}{3} = \frac{-1.6}{3} = \mathbf{-\frac{1.6}{3}} \approx \mathbf{-0.533333}$$
  $$\mu_{\text{elite}} = \begin{bmatrix} -1.300000 \\ -0.533333 \end{bmatrix}$$

- **Population Variance and Standard Deviation:**
  $$\sigma_{\text{elite}, 0}^2 = \frac{(-1.3 - (-1.3))^2 + (-1.2 - (-1.3))^2 + (-1.4 - (-1.3))^2}{3} = \frac{0 + (0.1)^2 + (-0.1)^2}{3} = \frac{0.02}{3} \approx 0.006667$$
  $$\sigma_{\text{elite}, 0} = \sqrt{\frac{0.02}{3}} \approx \mathbf{0.081650}$$
  $$\sigma_{\text{elite}, 1}^2 = \frac{(-0.5 - (-1.6/3))^2 + (-0.6 - (-1.6/3))^2 + (-0.5 - (-1.6/3))^2}{3} = \frac{(0.033333)^2 + (-0.066667)^2 + (0.033333)^2}{3} = \frac{0.006667}{3} \approx 0.002222$$
  $$\sigma_{\text{elite}, 1} = \sqrt{\frac{0.006667}{3}} \approx \mathbf{0.047140}$$
  $$\sigma_{\text{elite}} = \begin{bmatrix} 0.081650 \\ 0.047140 \end{bmatrix}$$

---

#### Step 4: Polyak Momentum Smoothing ($\beta = 0.7$)
With initial prior $\mu^{(0)} = [0.0, 0.0]^\top$ and $\sigma^{(0)} = [1.0, 1.0]^\top$:
- **Updated Mean:**
  $$\mu_0^{(1)} = 0.7 \times (-1.300000) + 0.3 \times 0.0 = \mathbf{-0.910000}$$
  $$\mu_1^{(1)} = 0.7 \times (-0.533333) + 0.3 \times 0.0 = \mathbf{-0.373333}$$
  $$\mu^{(1)} = \begin{bmatrix} \mathbf{-0.910000} \\ \mathbf{-0.373333} \end{bmatrix}$$
- **Updated Standard Deviation:**
  $$\sigma_0^{(1)} = 0.7 \times 0.081650 + 0.3 \times 1.0 = 0.057155 + 0.300000 = \mathbf{0.357155}$$
  $$\sigma_1^{(1)} = 0.7 \times 0.047140 + 0.3 \times 1.0 = 0.032998 + 0.300000 = \mathbf{0.332998}$$
  $$\sigma^{(1)} = \begin{bmatrix} \mathbf{0.357155} \\ \mathbf{0.332998} \end{bmatrix} \quad \blacksquare$$

---

### Illustration 5: Receding Horizon Execution Step: Open-Loop Planned Trajectory vs. Closed-Loop Executed Trajectory under Process Noise

**Problem:**
Consider the 1D dynamical plant from Section 5:
$$x_{t+1} = a x_t + b u_t + w_t$$
where $a = 1.0$, $b = 1.0$, stage penalties are $q = 2.0, r = 1.0$, terminal penalty is $q_f = 4.0$, and the initial state is $x_0 = 10.0$.
A receding horizon controller operates with horizon length $N = 2$. From Section 5.3, the stationary receding-horizon feedback gain is:
$$K_{\text{mpc}} = K_0 = -\frac{14}{19} \approx -0.736842, \quad K_1 = -0.800000$$

At execution time, the system is subjected to severe stochastic process disturbances:
$$w_0 = +2.0 \quad (\text{at step } t = 0), \quad w_1 = +2.0 \quad (\text{at step } t = 1)$$
Compare the step-by-step state trajectories, applied actions, and cumulative costs under:
1. **Mode A: Open-Loop Control:** Solves the plan at $t = 0$ assuming $w_t \equiv 0$, and blindly executes the scheduled open-loop actions $(u_0^{\text{plan}}, u_1^{\text{plan}})$.
2. **Mode B: Closed-Loop Receding Horizon MPC:** Senses the true state $x_t$ at each sampling time and dynamically re-computes the receding horizon feedback action $u_t = K_{\text{mpc}} x_t$.

**Solution:**

#### Step 1: Nominal Open-Loop Plan Computed at $t = 0$ (Assuming $w = 0$)
From $x_0 = 10.0$:
- Action $u_0^{\text{plan}} = K_0 x_0 = -\frac{14}{19}(10.0) = -\frac{140}{19} \approx \mathbf{-7.368421}$
- Predicted state $x_1^{\text{nom}} = a x_0 + b u_0^{\text{plan}} = 10.0 - 7.368421 = \frac{50}{19} \approx \mathbf{2.631579}$
- Action $u_1^{\text{plan}} = K_1 x_1^{\text{nom}} = -0.8000 \times \frac{50}{19} = -\frac{40}{19} \approx \mathbf{-2.105263}$
- Predicted terminal state $x_2^{\text{nom}} = a x_1^{\text{nom}} + b u_1^{\text{plan}} = \frac{50}{19} - \frac{40}{19} = \frac{10}{19} \approx \mathbf{0.526316}$

---

#### Step 2: Mode A — Open-Loop Blind Execution under Disturbance ($w_0 = +2.0, w_1 = +2.0$)
1. **Timestep $t = 0$:**
   - State: $x_0 = 10.0000$
   - Applied Action: $u_0 = u_0^{\text{plan}} = \mathbf{-7.368421}$
   - Stage cost: $\ell_0 = \frac{1}{2}(2.0)(10.0)^2 + \frac{1}{2}(1.0)(-7.368421)^2 = 100.0 + 27.146814 = \mathbf{127.146814}$
   - Next state under disturbance $w_0 = +2.0$:
     $$x_1^{\text{ol}} = a x_0 + b u_0 + w_0 = 10.0 - 7.368421 + 2.0 = \mathbf{4.631579}$$
2. **Timestep $t = 1$:**
   - True state: $x_1^{\text{ol}} = 4.631579$ (system has drifted significantly away from predicted $2.6316$).
   - Applied Action: Open-loop controller is blind to sensor feedback; executes pre-scheduled:
     $$u_1 = u_1^{\text{plan}} = \mathbf{-2.105263}$$
   - Stage cost: $\ell_1 = \frac{1}{2}(2.0)(4.631579)^2 + \frac{1}{2}(1.0)(-2.105263)^2 = 21.451524 + 2.216066 = \mathbf{23.667590}$
   - Terminal state under disturbance $w_1 = +2.0$:
     $$x_2^{\text{ol}} = a x_1^{\text{ol}} + b u_1^{\text{plan}} + w_1 = 4.631579 - 2.105263 + 2.0 = \mathbf{4.526316}$$
3. **Terminal Step $t = 2$:**
   - Terminal cost:
     $$\ell_f^{\text{ol}} = \frac{1}{2} q_f (x_2^{\text{ol}})^2 = \frac{1}{2}(4.0)(4.526316)^2 = 2.0 \times 20.487535 = \mathbf{40.975069}$$
4. **Total Realized Cost (Open-Loop):**
   $$J^{\text{ol}} = \ell_0 + \ell_1 + \ell_f^{\text{ol}} = 127.146814 + 23.667590 + 40.975069 = \mathbf{191.789474}$$

---

#### Step 3: Mode B — Closed-Loop Receding Horizon MPC Execution
1. **Timestep $t = 0$:**
   - State: $x_0 = 10.0000$
   - Applied Action: $u_0 = K_{\text{mpc}} x_0 = \mathbf{-7.368421}$
   - Stage cost: $\ell_0 = \mathbf{127.146814}$
   - Realized next state under disturbance $w_0 = +2.0$:
     $$x_1^{\text{mpc}} = 10.0 - 7.368421 + 2.0 = \mathbf{4.631579}$$
2. **Timestep $t = 1$:**
   - Sensed true state: $x_1^{\text{mpc}} = 4.631579$.
   - Receding horizon re-planning: The MPC controller observes the $+2.0$ disturbance error and computes fresh feedback control:
     $$u_1^{\text{mpc}} = K_{\text{mpc}} x_1^{\text{mpc}} = -\frac{14}{19} \times 4.631579 = \mathbf{-3.412742}$$
     *(Notice: The MPC controller increases corrective braking torque from $-2.1053$ to $-3.4127$, actively rejecting the disturbance!).*
   - Stage cost: $\ell_1 = \frac{1}{2}(2.0)(4.631579)^2 + \frac{1}{2}(1.0)(-3.412742)^2 = 21.451524 + 5.823405 = \mathbf{27.274929}$
   - Terminal state under disturbance $w_1 = +2.0$:
     $$x_2^{\text{mpc}} = a x_1^{\text{mpc}} + b u_1^{\text{mpc}} + w_1 = 4.631579 - 3.412742 + 2.0 = \mathbf{3.218837}$$
3. **Terminal Step $t = 2$:**
   - Terminal cost:
     $$\ell_f^{\text{mpc}} = \frac{1}{2} q_f (x_2^{\text{mpc}})^2 = \frac{1}{2}(4.0)(3.218837)^2 = 2.0 \times 10.360909 = \mathbf{20.721818}$$
4. **Total Realized Cost (Closed-Loop MPC):**
   $$J^{\text{mpc}} = \ell_0 + \ell_1 + \ell_f^{\text{mpc}} = 127.146814 + 27.274929 + 20.721818 = \mathbf{175.143561}$$

---

#### Step 4: Quantitative Side-by-Side Comparison

| Metric / Variable | Open-Loop Control (Mode A) | Closed-Loop MPC (Mode B) | Advantage of Receding Horizon MPC |
| :--- | :---: | :---: | :--- |
| **Action applied at $t=1$ ($u_1$)** | $-2.1053$ (blind) | **$-3.4127$** (adaptive feedback) | $+62.1\%$ larger corrective torque to reject disturbance |
| **Terminal State $x_2$** | $4.5263$ | **$3.2188$** | **$28.9\%$ reduction in final state error** |
| **Terminal Penalty Cost $\ell_f$** | $40.9751$ | **$20.7218$** | **$49.4\%$ reduction in terminal error penalty** |
| **Total Cumulative Cost $J$** | $191.7895$ | **$175.1436$** | **$16.65$ cost savings** despite identical physical disturbances |

$$\mathbf{\Delta J = J^{\text{ol}} - J^{\text{mpc}} = 191.7895 - 175.1436 = +16.6459} \quad \blacksquare$$

---

## 7. Deep Learning Connection & Modern Applications

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
