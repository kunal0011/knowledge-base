# Chapter 8.3: Gated Architectures (LSTM & GRU)

---

## 1. Intuition & 101 Motivation

In Chapter 8.2, we uncovered the fundamental mathematical dilemma of Vanilla Recurrent Neural Networks: repeatedly applying the non-linear recurrent transformation $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t)$ creates a long-term Jacobian chain product:
$$\prod_{k=t+1}^T \operatorname{diag}(1 - \mathbf{h}_k^2) \mathbf{W}_{hh}$$
whose norm contracts exponentially to zero whenever singular values are strictly bounded to prevent explosion.

In 1997, Sepp Hochreiter and Jürgen Schmidhuber solved this long-standing crisis with the **Long Short-Term Memory (LSTM)** network.

Their radical architectural innovation was:
1. **Separating Memory from Output:** The model decouples the internal linear memory accumulator—the **Cell State** $\mathbf{c}_t$—from the exposed, non-linearly bounded hidden state $\mathbf{h}_t$.
2. **The Constant Error Carousel (CEC):** Instead of overwriting memory through non-linear matrix multiplication, the cell state updates **additively**:
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
   When the forget gate $\mathbf{f}_t = \mathbf{1}$, the Jacobian $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{I}$, creating a pristine **gradient highway** through time.
3. **Multiplicative Gating Valves:** Three continuous, differentiable sigmoid gates $\in (0, 1)$ regulate information traffic:
   - **Forget Gate ($\mathbf{f}_t$):** What proportion of the old cell state to discard.
   - **Input Gate ($\mathbf{i}_t$):** What proportion of the candidate memory to write.
   - **Output Gate ($\mathbf{o}_t$):** What proportion of the internal memory to expose to the network.

In 2014, Kyunghyun Cho et al. introduced the **Gated Recurrent Unit (GRU)**, which compresses the 4-gate LSTM into an elegant 2-gate architecture without a separate cell state, reducing parameter count and compute by **$25\%$**.

---

## 2. Rigorous Mathematical Formulation

```
                     LONG SHORT-TERM MEMORY (LSTM) CELL
           c_{t-1} ──────[ x ]───────────────────[ + ]─────────────────► c_t
                          ▲                       ▲            │
                          │ f_t                   │ i_t * c~   │
                     ┌────┴────┐             ┌────┴────┐     [tanh]
                     │  sigma  │             │    *    │       │
                     └────▲────┘             └────▲────┘       ▼
                          │                  ┌────┴────┐     [ x ] ◄── o_t (sigma)
                          │             ┌───►│  sigma  │(i_t)  │
                          │             │    └─────────┘       │
                          │             │    ┌─────────┐       │
                          │             ├───►│  tanh   │(c~)   │
                          │             │    └─────────┘       │
                          │             │    ┌─────────┐       │
                          ├─────────────┴───►│  sigma  │(o_t)  ▼
                          │                  └─────────┘      h_t
                   [ h_{t-1}, x_t ]
```

---

### 2.1 The Long Short-Term Memory (LSTM) Formulation

Let $\mathbf{x}_t \in \mathbb{R}^{d_x}$ be the input vector at time step $t$.
Let $\mathbf{h}_{t-1} \in \mathbb{R}^{d_h}$ be the previous hidden state, and $\mathbf{c}_{t-1} \in \mathbb{R}^{d_h}$ be the previous cell state.

We concatenate input and recurrent activations into:
$$\mathbf{v}_t = \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} \in \mathbb{R}^{(d_h + d_x)}$$

#### The Six Governing Equations:
1. **Forget Gate:**
   $$\mathbf{f}_t = \sigma\left(\mathbf{W}_f \mathbf{v}_t + \mathbf{b}_f\right) \in (0, 1)^{d_h}$$
2. **Input / Write Gate:**
   $$\mathbf{i}_t = \sigma\left(\mathbf{W}_i \mathbf{v}_t + \mathbf{b}_i\right) \in (0, 1)^{d_h}$$
3. **Candidate Cell State:**
   $$\tilde{\mathbf{c}}_t = \tanh\left(\mathbf{W}_c \mathbf{v}_t + \mathbf{b}_c\right) \in (-1, 1)^{d_h}$$
4. **Cell State Update (The Constant Error Carousel):**
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \in \mathbb{R}^{d_h}$$
5. **Output / Read Gate:**
   $$\mathbf{o}_t = \sigma\left(\mathbf{W}_o \mathbf{v}_t + \mathbf{b}_o\right) \in (0, 1)^{d_h}$$
6. **Hidden State Emission:**
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \in (-1, 1)^{d_h}$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the element-wise logistic sigmoid function, and $\odot$ represents the Hadamard (element-wise) product.

---

### 2.2 The Constant Error Carousel (CEC) Mathematical Proof

Consider the temporal Jacobian of the cell state from step $t-1$ to $t$:
$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \operatorname{diag}(\mathbf{f}_t) + \underbrace{\frac{\partial \mathbf{f}_t}{\partial \mathbf{c}_{t-1}} \odot \mathbf{c}_{t-1} + \frac{\partial \mathbf{i}_t}{\partial \mathbf{c}_{t-1}} \odot \tilde{\mathbf{c}}_t + \frac{\partial \tilde{\mathbf{c}}_t}{\partial \mathbf{c}_{t-1}} \odot \mathbf{i}_t}_{\text{Indirect terms through } \mathbf{h}_{t-1}}$$

Because $\mathbf{f}_t, \mathbf{i}_t, \tilde{\mathbf{c}}_t$ depend on $\mathbf{h}_{t-1}$ (which depends on $\mathbf{c}_{t-1}$ via $\mathbf{o}_{t-1} \odot \tanh(\mathbf{c}_{t-1})$), the direct error propagation along the cell state backbone is governed by:
$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \operatorname{diag}(\mathbf{f}_t)$$

Over an extended horizon of $T - t$ time steps:
$$\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_t} = \prod_{k=t+1}^T \operatorname{diag}(\mathbf{f}_k) = \operatorname{diag}\left( \prod_{k=t+1}^T \mathbf{f}_k \right)$$

#### Key Theoretical Insights:
1. **Absence of Matrix Multiplications:** The Jacobian product does **not** involve repeated multiplication by weight matrices $\mathbf{W}$. It is a diagonal product of scalar gate activations!
2. **Gradient Preservation:** If the network learns to keep the forget gate open ($\mathbf{f}_k \approx \mathbf{1}$), the product $\prod_{k} \mathbf{f}_k \approx \mathbf{1}$. Gradients flow backwards across hundreds of steps with **zero exponential decay**.
3. **Selective Amnesia:** If an old memory is deemed obsolete (e.g., closing a parentheses or ending a sentence), setting $\mathbf{f}_k \to \mathbf{0}$ instantly zeroes both the forward memory and the backward gradient for that specific feature coordinate.

---

### 2.3 The Forget Gate Bias Initialization Trick (Jozefowicz et al., 2015)

In an exhaustive empirical search of over 10,000 RNN architectures, Rafal Jozefowicz, Wojciech Zaremba, and Ilya Sutskever demonstrated that the single most critical hyperparameter for LSTM stability is **initializing the forget gate bias to a positive constant**:
$$\mathbf{b}_f \sim \mathcal{U}(1.0, 2.0)$$
- If $\mathbf{b}_f = 0$, at initialization $\mathbf{f}_t \approx \sigma(0) = 0.5$. Over 20 steps, the gradient decays by $0.5^{20} \approx 10^{-6}$, re-introducing vanishing gradients on day one!
- If $\mathbf{b}_f = 1.0$ or $2.0$, at initialization $\mathbf{f}_t \approx \sigma(2.0) \approx 0.88$. The gates default to wide open, allowing gradients to propagate deeply from the very first training epoch.

---

### 2.4 The Gated Recurrent Unit (GRU) (Cho et al., 2014)

```
                     GATED RECURRENT UNIT (GRU) CELL
           h_{t-1} ──────[ x ]───────────────────[ + ]─────────────────► h_t
                          ▲                       ▲
                          │ (1 - z_t)             │ z_t * h~
                          │                  ┌────┴────┐
                          │             ┌───►│  sigma  │(z_t)
                          │             │    └─────────┘
                          │             │    ┌─────────┐
                          │             ├───►│  tanh   │(h~) ◄── r_t (sigma)
                          │             │    └─────────┘
                   [ h_{t-1}, x_t ] ────┴─────────────────────────────
```

The GRU eliminates the separate cell state $\mathbf{c}_t$, utilizing only the hidden state $\mathbf{h}_t$. It merges the input and forget gates into a single **Update Gate** $\mathbf{z}_t$, and controls historical context through a **Reset Gate** $\mathbf{r}_t$.

#### The Four Governing Equations:
1. **Reset Gate:**
   $$\mathbf{r}_t = \sigma\left(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r\right) \in (0, 1)^{d_h}$$
2. **Update Gate:**
   $$\mathbf{z}_t = \sigma\left(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z\right) \in (0, 1)^{d_h}$$
3. **Candidate Hidden State:**
   $$\tilde{\mathbf{h}}_t = \tanh\left(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \, \mathbf{x}_t] + \mathbf{b}_h\right) \in (-1, 1)^{d_h}$$
4. **Interpolation State Update:**
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \in (-1, 1)^{d_h}$$

#### Architectural Trade-offs:
- **Convex Combination:** The update equation is an exact linear interpolation between the previous state $\mathbf{h}_{t-1}$ and candidate state $\tilde{\mathbf{h}}_t$.
- When $\mathbf{z}_t = \mathbf{0}$, $\mathbf{h}_t = \mathbf{h}_{t-1}$ (pure identity bypass, zero gradient vanishing).
- When $\mathbf{z}_t = \mathbf{1}$, $\mathbf{h}_t = \tilde{\mathbf{h}}_t$ (complete overwrite).
- **Parameter Efficiency:** GRU requires $3$ gate matrices vs. $4$ for LSTM. For identical hidden dimensions, GRU has **$25\%$ fewer parameters** and trains significantly faster on small-to-medium datasets.

---

## 3. Geometric & Algebraic Interpretation

### The Gated Phase Space Topology
In a Vanilla RNN, the state transformation $\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h}_{t-1})$ warps, rotates, and squashes the entire phase space simultaneously.
In an LSTM:
- The cell state $\mathbf{c}_t$ evolves in an unbounded linear vector space $\mathbb{R}^{d_h}$.
- The gating signals $\mathbf{f}_t, \mathbf{i}_t, \mathbf{o}_t$ act as **coordinate-wise dynamic projection operators** restricted to the hypercube $[0, 1]^{d_h}$.
- Instead of twisting the entire coordinate frame, gating scales individual orthogonal coordinate axes independently. A neuron tracking a specific long-term attribute (e.g., whether a quotation mark is open) can freeze its axis by keeping $f_k = 1, i_k = 0$ while all other 511 dimensions actively process short-term grammatical tokens.

---

## 4. Real-World Analogy

### The Industrial Factory Assembly Line (LSTM)
- **Cell State ($\mathbf{c}_t$):** A continuous, heavy-duty industrial conveyor belt running through a massive factory. Items placed on the belt remain on the belt by inertia unless acted upon.
- **Forget Gate ($\mathbf{f}_t$):** A robotic scraper arm above the belt. When an obsolete part passes by, it pushes it into a recycling bin ($f \to 0$). If the part is still needed, the arm lifts up and lets it pass untouched ($f \to 1$).
- **Input Gate ($\mathbf{i}_t$) & Candidate ($\tilde{\mathbf{c}}_t$):** A stamping machine that inspects incoming raw supplies ($\mathbf{x}_t$), manufactures a new component ($\tilde{\mathbf{c}}_t$), and places it onto the belt if permitted ($i_t \to 1$).
- **Output Gate ($\mathbf{o}_t$):** A camera over the belt that photographs the current assembly, applies an optical filter ($\tanh$), and transmits the image to the factory dashboard ($\mathbf{h}_t$) without altering the physical belt itself.

---

## 5. Prof. Tom Yeh "AI by Hand" Visual Grids

Let us trace forward and backward propagation through a complete LSTM cell with concrete hand arithmetic.

### 5.1 "What Refers to What" Legend Protocol

| Symbol / Variable | Mathematical Representation | Shape / Dimensions | Description & Role in LSTM Cell |
| :--- | :--- | :--- | :--- |
| $x_t$ | Input Token | Scalar ($1.0$) | Input activation at time step $t$ |
| $h_{t-1}$ | Previous Hidden State | Scalar ($0.5$) | Hidden state from time step $t-1$ |
| $c_{t-1}$ | Previous Cell State | Scalar ($2.0$) | Memory accumulator from time step $t-1$ |
| $f_t$ | Forget Gate | Scalar $\in (0, 1)$ | $\sigma(W_f x_t + U_f h_{t-1} + b_f)$ |
| $i_t$ | Input Gate | Scalar $\in (0, 1)$ | $\sigma(W_i x_t + U_i h_{t-1} + b_i)$ |
| $\tilde{c}_t$ | Candidate Cell State | Scalar $\in (-1, 1)$ | $\tanh(W_c x_t + U_c h_{t-1} + b_c)$ |
| $c_t$ | Updated Cell State | Scalar | $c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t$ |
| $o_t$ | Output Gate | Scalar $\in (0, 1)$ | $\sigma(W_o x_t + U_o h_{t-1} + b_o)$ |
| $h_t$ | Updated Hidden State | Scalar $\in (-1, 1)$ | $h_t = o_t \cdot \tanh(c_t)$ |
| $\frac{\partial \mathcal{L}}{\partial h_t}$ | Upstream Hidden Grad | Scalar ($1.0$) | Sensitivity of objective loss w.r.t. current hidden state |
| $\frac{\partial \mathcal{L}}{\partial c_t}$ | Cell State Gradient | Scalar | Direct upstream gradient + gradient through $h_t$ |
| $\frac{\partial \mathcal{L}}{\partial c_{t-1}}$ | Prior Cell Gradient | Scalar | Error routed backwards along Constant Error Carousel |

---

### 5.2 Concrete Toy Numbers

#### State Inputs:
$$x_t = 1.0, \quad h_{t-1} = 0.5, \quad c_{t-1} = 2.0$$

#### Model Weights and Biases:
- **Forget Gate:** $W_f = 0.0, \quad U_f = 0.0, \quad b_f = 1.0 \implies z_f = 1.0$
- **Input Gate:** $W_i = 1.0, \quad U_i = -1.0, \quad b_i = 0.5 \implies z_i = 1.0(1.0) - 1.0(0.5) + 0.5 = 1.0$
- **Candidate State:** $W_c = 0.5, \quad U_c = 1.0, \quad b_c = -1.0 \implies z_c = 0.5(1.0) + 1.0(0.5) - 1.0 = 0.0$
- **Output Gate:** $W_o = 0.0, \quad U_o = 2.0, \quad b_o = -1.0 \implies z_o = 0.0(1.0) + 2.0(0.5) - 1.0 = 0.0$

---

### 5.3 Forward Pass Arithmetic

1. **Forget Gate Activation:**
   $$f_t = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} \approx \mathbf{0.731059}$$

2. **Input Gate Activation:**
   $$i_t = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} \approx \mathbf{0.731059}$$

3. **Candidate Memory:**
   $$\tilde{c}_t = \tanh(0.0) = \mathbf{0.0}$$

4. **Cell State Update ($c_t = f_t c_{t-1} + i_t \tilde{c}_t$):**
   $$c_t = (0.731059)(2.0) + (0.731059)(0.0) = 1.462118 + 0.0 = \mathbf{1.462118}$$

5. **Output Gate Activation:**
   $$o_t = \sigma(0.0) = \frac{1}{1 + e^{0}} = \frac{1}{2} = \mathbf{0.5}$$

6. **Hidden State Emission ($h_t = o_t \tanh(c_t)$):**
   $$\tanh(c_t) = \tanh(1.462118) \approx 0.898144$$
   $$h_t = (0.5)(0.898144) = \mathbf{0.449072}$$

---

### 5.4 Backward Pass Arithmetic (The Constant Error Carousel)

Suppose the upstream loss gradient at step $t$ is:
$$\frac{\partial \mathcal{L}}{\partial h_t} = 1.0, \quad \frac{\partial \mathcal{L}_{\text{direct}}}{\partial c_t} = 0.0$$

1. **Gradient w.r.t. Current Cell State ($\frac{\partial \mathcal{L}}{\partial c_t}$):**
   $$h_t = o_t \tanh(c_t)$$
   $$\frac{\partial \mathcal{L}}{\partial c_t} = \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial c_t} = \frac{\partial \mathcal{L}}{\partial h_t} \cdot o_t \cdot \left(1 - \tanh^2(c_t)\right)$$
   $$1 - \tanh^2(1.462118) = 1 - (0.898144)^2 = 1 - 0.806663 = 0.193337$$
   $$\frac{\partial \mathcal{L}}{\partial c_t} = (1.0)(0.5)(0.193337) = \mathbf{0.096669}$$

2. **Gradient w.r.t. Output Gate Pre-activation ($\delta^{z_o}$):**
   $$\frac{\partial \mathcal{L}}{\partial o_t} = \frac{\partial \mathcal{L}}{\partial h_t} \tanh(c_t) = (1.0)(0.898144) = 0.898144$$
   $$\delta^{z_o} = \frac{\partial \mathcal{L}}{\partial o_t} \cdot o_t(1 - o_t) = (0.898144)(0.5)(1 - 0.5) = (0.898144)(0.25) = \mathbf{0.224536}$$

3. **Gradient w.r.t. Previous Cell State ($\frac{\partial \mathcal{L}}{\partial c_{t-1}}$ - The CEC in Action):**
   $$c_t = f_t c_{t-1} + i_t \tilde{c}_t$$
   $$\frac{\partial \mathcal{L}}{\partial c_{t-1}} = \frac{\partial \mathcal{L}}{\partial c_t} \frac{\partial c_t}{\partial c_{t-1}} = \frac{\partial \mathcal{L}}{\partial c_t} \cdot f_t$$
   $$\frac{\partial \mathcal{L}}{\partial c_{t-1}} = (0.096669)(0.731059) = \mathbf{0.070671}$$

Look at this calculation: the gradient flows directly from $c_t$ to $c_{t-1}$ scaled cleanly by the forget gate $f_t = 0.731059$, without encountering any saturating matrix multiplications!

---

## 6. Solved Illustrations

### Illustration 1: Parameter Count Derivation: LSTM vs. GRU

**Problem:**
Derive the exact formula for the total number of trainable parameters in:
1. A standard LSTM cell with input dimension $d_x$ and hidden dimension $d_h$.
2. A standard GRU cell with input dimension $d_x$ and hidden dimension $d_h$.
Calculate the parameter counts for $d_x = 512, d_h = 512$.

**Solution:**
1. **LSTM Cell:**
   An LSTM contains 4 transformations ($f, i, c, o$), each taking concatenated input $[\mathbf{h}_{t-1}, \mathbf{x}_t] \in \mathbb{R}^{d_h + d_x}$ to an output $\in \mathbb{R}^{d_h}$:
   $$\text{Params}(\text{LSTM}) = 4 \times \left[ (d_h + d_x) \times d_h + d_h \right] = 4 d_h (d_h + d_x + 1)$$
   For $d_x = 512, d_h = 512$:
   $$\text{Params}(\text{LSTM}) = 4 \times 512 \times (512 + 512 + 1) = 2,048 \times 1,025 = \mathbf{2,099,200 \text{ parameters}}$$

2. **GRU Cell:**
   A GRU contains 3 transformations ($r, z, h$):
   $$\text{Params}(\text{GRU}) = 3 \times \left[ (d_h + d_x) \times d_h + d_h \right] = 3 d_h (d_h + d_x + 1)$$
   For $d_x = 512, d_h = 512$:
   $$\text{Params}(\text{GRU}) = 3 \times 512 \times (512 + 512 + 1) = 1,536 \times 1,025 = \mathbf{1,574,400 \text{ parameters}}$$

**Comparison:**
$$\frac{\text{Params}(\text{GRU})}{\text{Params}(\text{LSTM})} = \frac{3}{4} = \mathbf{75\%}$$
The GRU provides an exact **$25\%$ reduction in parameter memory and matrix multiplication FLOPs**.

---

### Illustration 2: Peephole Connections (Gers & Schmidhuber, 2000)

**Problem:**
What are **peephole connections** in an LSTM? Why were they introduced, and why did modern deep learning frameworks (PyTorch default `nn.LSTM`) largely abandon them?

**Solution:**
In standard LSTM, gates $f_t, i_t, o_t$ only observe the previous hidden state $\mathbf{h}_{t-1}$ and input $\mathbf{x}_t$; they cannot inspect the current cell state $\mathbf{c}_{t-1}$ or $\mathbf{c}_t$.
Felix Gers and Jürgen Schmidhuber introduced peephole connections by adding diagonal weights from the cell state directly into the gates:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{v}_t + \mathbf{p}_f \odot \mathbf{c}_{t-1} + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{v}_t + \mathbf{p}_i \odot \mathbf{c}_{t-1} + \mathbf{b}_i)$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{v}_t + \mathbf{p}_o \odot \mathbf{c}_t + \mathbf{b}_o)$$
- *Theoretical Motivation:* Allows gates to time precise intervals by directly monitoring the accumulation level of the linear integrator $\mathbf{c}_t$.
- *Why Abandoned:* Benchmarks across large-scale speech recognition and machine translation revealed negligible accuracy gains, while adding memory access synchronization overhead that prevents fusing all 4 gate matrix multiplications into a single cuBLAS GEMM call (`[W_f; W_i; W_c; W_o]`).

---

## 7. Deep Learning Connection & Application

### 1. Fused Gate Kernel Optimization in cuDNN
In production implementations (`torch.nn.LSTM`), PyTorch does not execute 4 separate matrix-vector products. Instead, it stacks all 4 weight matrices into a single monolithic weight tensor:
$$\mathbf{W}_{\text{all}} = \begin{bmatrix} \mathbf{W}_i \\ \mathbf{W}_f \\ \mathbf{W}_c \\ \mathbf{W}_o \end{bmatrix} \in \mathbb{R}^{4d_h \times (d_h + d_x)}$$
A single massive cuBLAS matrix multiplication computes all 4 pre-activations simultaneously:
$$\begin{bmatrix} \mathbf{a}_i \\ \mathbf{a}_f \\ \mathbf{a}_c \\ \mathbf{a}_o \end{bmatrix} = \mathbf{W}_{\text{all}} \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b}_{\text{all}}$$
Then, a fused CUDA element-wise kernel applies $\sigma$ and $\tanh$ activations in high-speed GPU registers without round-tripping to DRAM.

---

## 8. Code Implementation & Verification

The accompanying verification script implements:
1. `LSTMCellScratch`: Pure NumPy implementation of forward pass and analytical backward pass.
2. `GRUCellScratch`: Pure NumPy implementation of forward pass and analytical backward pass.
3. Exact numerical verification of the Part 5 Visual Grid hand arithmetic.
4. Finite-difference numerical gradient checks for LSTM and GRU parameters.
5. End-to-end parity validation against PyTorch's `torch.nn.LSTMCell` and `torch.nn.GRUCell` with discrepancies $< 10^{-12}$.

See implementation in:
[`08_recurrent_networks/code/03_gated_architectures_lstm_and_gru.py`](./code/03_gated_architectures_lstm_and_gru.py)
