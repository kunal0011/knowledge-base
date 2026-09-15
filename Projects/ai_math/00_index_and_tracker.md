# Deep Learning & Mathematical Foundations — University Course Index & Progress Tracker

Welcome to your university-level reference course on Mathematics for Deep Learning and Modern Deep Learning Architectures.

---

## Pedagogical Template (Applied to Every Chapter)
Every topic is authored in an uncompromised 8-part pedagogical framework:
1. **Part 1: Intuition & 101 Motivation** — Why does this exist? What problem does it solve?
2. **Part 2: Rigorous Mathematical Formulation** — Definitions, theorems, properties, and LaTeX equations.
3. **Part 3: Geometric & Physical Interpretation** — High-dimensional geometry, coordinate transformations, and mechanics.
4. **Part 4: Real-World Analogy** — An intuitive real-world mental model.
5. **Part 5: "AI by Hand" (Prof. Tom Yeh Style Visual Grids)** — Cell-by-cell matrix/tensor arithmetic walkthroughs with concrete numbers, visual grids, and zero black boxes.
6. **Part 6: Step-by-Step Solved Illustrations** — Manual arithmetic covering standard cases, boundary conditions, and pathological edge cases.
7. **Part 7: Deep Learning Connection & Application** — Architectural roles, loss function derivation, or optimizer mechanics.
8. **Part 8: Code Implementation & Verification** — Clean, vectorized NumPy / PyTorch code with unit tests.

---

## Course Progress & Roadmap

| Module | Title | Chapters Count | Status |
| :--- | :--- | :---: | :---: |
| **Module 01** | [Linear Algebra for Machine Learning](./01_linear_algebra) | 9 Chapters | 🟢 Completed |
| **Module 02** | [Multivariable Calculus & Automatic Differentiation](./02_multivariable_calculus) | 7 Chapters | 🟢 Completed |
| **Module 03** | [Probability Theory for Deep Learning](./03_probability_theory) | 7 Chapters | 🟢 Completed |
| **Module 04** | [Mathematical Statistics & Estimation](./04_mathematical_statistics) | 6 Chapters | 🟢 Completed |
| **Module 05** | [Optimization & Convex Analysis](./05_optimization) | 7 Chapters | 🟢 Completed |
| **Module 06** | [Deep Learning Foundations (NumPy & PyTorch)](./06_deep_learning_foundations) | 8 Chapters | 🟢 Completed |
| **Module 07** | [Convolutional Neural Networks & Vision](./07_convolutional_networks) | 4 Chapters | 🟢 Completed |
| **Module 08** | [Recurrent Networks & Sequence Modeling](./08_recurrent_networks) | 4 Chapters | 🟢 Completed |
| **Module 09** | [Transformers, Large Language Models & Alignment](./09_transformers_and_llms) | 8 Chapters | 🟢 Completed |
| **Module 10** | [Generative Modeling (VAEs, GANs, Diffusion)](./10_generative_models) | 4 Chapters | 🟢 Completed |
| **Module 11** | [Reinforcement Learning: Foundations to Frontier Reasoning](./11_reinforcement_learning) | 28 Chapters | 🟢 Completed |
| **Module 12** | [Modern LLM Architectures & Engineering from Scratch](./12_modern_llm_architectures) | 12 Chapters | 🟢 Completed |
| **Module 13** | [Frontier Reasoning & Inference-Time Compute from Scratch](./13_reasoning_and_test_time_compute) | 9 Chapters | 🟢 Completed |

---

### Detailed Chapter Breakdown

#### Module 1: Linear Algebra
- [x] **1.1 [Vector Spaces, Subspaces & Basis](./01_linear_algebra/01_vector_spaces_and_subspaces.md)**
- [x] **1.2 [Vector Norms, Metrics & Inner Products](./01_linear_algebra/02_vector_norms_and_metrics.md)**
- [x] **1.3 [Matrix Algebra, Column Spaces & Systems of Linear Equations](./01_linear_algebra/03_matrix_algebra_and_systems.md)**
- [x] **1.4 [Linear Transformations & Invertibility](./01_linear_algebra/04_linear_transformations_and_invertibility.md)**
- [x] **1.5 [Orthogonality, Projections & Least Squares](./01_linear_algebra/05_orthogonality_projections_and_least_squares.md)**
- [x] **1.6 [Eigendecomposition & Spectral Theory](./01_linear_algebra/06_eigendecomposition_and_spectral_theory.md)**
- [x] **1.7 [Singular Value Decomposition (SVD)](./01_linear_algebra/07_singular_value_decomposition.md)**
- [x] **1.8 [Quadratic Forms & Positive Semi-Definite (PSD) Matrices](./01_linear_algebra/08_quadratic_forms_and_psd.md)**
- [x] **1.9 [Tensors, Contractions & Einstein Summation Notation](./01_linear_algebra/09_tensors_and_einsum.md)**

#### Module 2: Multivariable Calculus & Automatic Differentiation
- [x] **2.1 [Differential Calculus Foundations & Taylor Expansions](./02_multivariable_calculus/01_differential_calculus_and_taylor.md)**
- [x] **2.2 [Directional Derivatives & Gradients](./02_multivariable_calculus/02_directional_derivatives_and_gradients.md)**
- [x] **2.3 [The Jacobian Matrix](./02_multivariable_calculus/03_the_jacobian_matrix.md)**
- [x] **2.4 [The Hessian Matrix & Curvature Analysis](./02_multivariable_calculus/04_the_hessian_matrix.md)**
- [x] **2.5 [Multivariate Chain Rule](./02_multivariable_calculus/05_multivariate_chain_rule.md)**
- [x] **2.6 [Matrix Calculus (Numerator & Denominator Layouts, Trace Tricks)](./02_multivariable_calculus/06_matrix_calculus.md)**
- [x] **2.7 [Computational Graphs & Automatic Differentiation (Reverse vs. Forward Mode)](./02_multivariable_calculus/07_computational_graphs_and_autodiff.md)**

#### Module 3: Probability Theory for Deep Learning
- [x] **3.1 [Probability Foundations, Axioms & Bayes' Theorem](./03_probability_theory/01_probability_foundations_and_bayes.md)**
- [x] **3.2 [Random Variables (Discrete vs. Continuous), PMF, PDF, CDF](./03_probability_theory/02_random_variables_pmf_pdf_cdf.md)**
- [x] **3.3 [Expectation, Variance, Covariance & Covariance Matrices](./03_probability_theory/03_expectation_variance_covariance.md)**
- [x] **3.4 [Parametric Distributions (Bernoulli, Categorical, Gaussian, Beta, Dirichlet)](./03_probability_theory/04_parametric_distributions.md)**
- [x] **3.5 [Joint, Marginal & Conditional Distributions](./03_probability_theory/05_joint_marginal_conditional.md)**
- [x] **3.6 [Asymptotic Theorems (LLN, CLT) & Inequalities](./03_probability_theory/06_asymptotic_theorems_and_inequalities.md)**
- [x] **3.7 [Information Theory (Entropy, Cross-Entropy, KL Divergence, Mutual Information)](./03_probability_theory/07_information_theory.md)**

#### Module 4: Mathematical Statistics & Estimation
- [x] **4.1 [Statistical Inference Foundations & Sampling](./04_mathematical_statistics/01_statistical_inference_and_sampling.md)**
- [x] **4.2 [Properties of Estimators (Bias, Variance, Consistency, Cramér-Rao)](./04_mathematical_statistics/02_properties_of_estimators.md)**
- [x] **4.3 [Maximum Likelihood Estimation (MLE) & Loss Function Derivations](./04_mathematical_statistics/03_maximum_likelihood_estimation.md)**
- [x] **4.4 [Maximum A Posteriori (MAP) & Regularization Links (L1/L2)](./04_mathematical_statistics/04_maximum_a_posteriori.md)**
- [x] **4.5 [The Bias-Variance Decomposition (Mathematical Derivation)](./04_mathematical_statistics/05_bias_variance_decomposition.md)**
- [x] **4.6 [Monte Carlo Methods & Importance Sampling](./04_mathematical_statistics/06_monte_carlo_and_importance_sampling.md)**

#### Module 5: Optimization & Convex Analysis
- [x] **5.1 [Convex Sets, Convex Functions & Jensen's Inequality](./05_optimization/01_convexity_and_jensens_inequality.md)**
- [x] **5.2 [Unconstrained Optimization, Stationary Points & Saddle Points](./05_optimization/02_unconstrained_optimization_and_saddles.md)**
- [x] **5.3 [Constrained Optimization, Lagrange Multipliers & KKT Conditions](./05_optimization/03_constrained_optimization_and_kkt.md)**
- [x] **5.4 [First-Order Algorithms (Batch GD, SGD, Mini-batch SGD)](./05_optimization/04_first_order_algorithms.md)**
- [x] **5.5 [Acceleration & Momentum (Polyak Momentum & Nesterov NAG)](./05_optimization/05_acceleration_and_momentum.md)**
- [x] **5.6 [Adaptive Learning Rates (AdaGrad, RMSprop, Adam, AdamW)](./05_optimization/06_adaptive_learning_rates.md)**
- [x] **5.7 [Second-Order Optimization & Natural Gradients (Newton, BFGS, Fisher Information)](./05_optimization/07_second_order_and_natural_gradients.md)**

#### Module 6: Deep Learning Foundations (NumPy & PyTorch)
- [x] **6.1 [The Perceptron, Linear Models & Perceptron Convergence Theorem](./06_deep_learning_foundations/01_perceptron_and_linear_models.md)**
- [x] **6.2 [Activation Functions & Gradient Flow (Sigmoid, Tanh, ReLU, GELU, Swish, Softmax)](./06_deep_learning_foundations/02_activation_functions_and_gradient_flow.md)**
- [x] **6.3 [Multilayer Perceptron (MLP) & Universal Approximation Theorem](./06_deep_learning_foundations/03_multilayer_perceptron_and_universal_approximation.md)**
- [x] **6.4 [Full Backpropagation Derivation & Scratch Autograd Engine](./06_deep_learning_foundations/04_backpropagation_and_scratch_autograd.md)**
- [x] **6.5 [Loss Functions Deep-Dive (MSE, BCE, Cross-Entropy, Focal, Triplet Loss)](./06_deep_learning_foundations/05_loss_functions_deep_dive.md)**
- [x] **6.6 [Weight Initialization Schemes (Xavier/Glorot, He/Kaiming, Orthogonal)](./06_deep_learning_foundations/06_weight_initialization_schemes.md)**
- [x] **6.7 [Regularization Techniques (Weight Decay, Dropout, Label Smoothing)](./06_deep_learning_foundations/07_regularization_techniques.md)**
- [x] **6.8 [Normalization Techniques (BatchNorm, LayerNorm, RMSNorm)](./06_deep_learning_foundations/08_normalization_techniques.md)**

#### Module 7: Convolutional Neural Networks (CNNs) & Computer Vision
- [x] **7.1 [The Convolution Operation (Strides, Padding, Dilation, Receptive Field)](./07_convolutional_networks/01_convolution_operation_and_receptive_field.md)**
- [x] **7.2 [Backpropagation Through Convolutions and Pooling Layers](./07_convolutional_networks/02_backpropagation_through_convolutions_and_pooling.md)**
- [x] **7.3 [Classic to Modern ConvNets (LeNet, AlexNet, VGG, ResNet, ConvNeXt)](./07_convolutional_networks/03_classic_to_modern_convnets.md)**
- [x] **7.4 [Vision Transformers (ViT) & Patch Embeddings](./07_convolutional_networks/04_vision_transformers_and_patch_embeddings.md)**

#### Module 8: Recurrent Neural Networks (RNNs) & Sequence Modeling
- [x] **8.1 [Sequence Processing, Autoregressive Models & Vanilla RNN (BPTT)](./08_recurrent_networks/01_sequence_processing_and_vanilla_rnn.md)**
- [x] **8.2 [The Exploding and Vanishing Gradient Problem in RNNs](./08_recurrent_networks/02_exploding_and_vanishing_gradients.md)**
- [x] **8.3 [Gated Architectures (LSTM & GRU)](./08_recurrent_networks/03_gated_architectures_lstm_and_gru.md)**
- [x] **8.4 [Sequence-to-Sequence Models & Classical Attention (Bahdanau, Luong)](./08_recurrent_networks/04_seq2seq_and_attention.md)**

#### Module 9: Transformers, Large Language Models & Alignment
- [x] **9.1 [Scaled Dot-Product & Multi-Head Attention (MHA)](./09_transformers_and_llms/01_scaled_dot_product_and_mha.md)**
- [x] **9.2 [Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi)](./09_transformers_and_llms/02_positional_encodings.md)**
- [x] **9.3 [Transformer Architectures (BERT, GPT, T5)](./09_transformers_and_llms/03_transformer_architectures.md)**
- [x] **9.4 [Modern Efficiency (MQA, GQA, FlashAttention, MoE, KV-Cache)](./09_transformers_and_llms/04_modern_efficiency.md)**
- [x] **9.5 [Pre-training Objectives & Compute-Optimal Scaling Laws (Chinchilla)](./09_transformers_and_llms/05_pretraining_and_scaling_laws.md)**
- [x] **9.6 [Alignment & Post-Training (SFT, RLHF with PPO, DPO)](./09_transformers_and_llms/06_alignment_sft_rlhf_dpo.md)**

#### Module 10: Generative Modeling
- [x] **10.1 [Taxonomy of Generative Models](./10_generative_models/01_taxonomy_of_generative_models.md)**
- [x] **10.2 [Variational Autoencoders (VAEs & ELBO Derivation)](./10_generative_models/02_variational_autoencoders_and_elbo.md)**
- [x] **10.3 [Generative Adversarial Networks (GANs & Wasserstein GAN)](./10_generative_models/03_generative_adversarial_networks_and_wgan.md)**
- [x] **10.4 [Diffusion Models (DDPM, Score-Based SDEs & CFG)](./10_generative_models/04_diffusion_models_ddpm_and_score_sdes.md)**

#### Module 11: Reinforcement Learning (Foundations to Frontier Reasoning)
- [x] **11.1 [Multi-Armed Bandits & Exploration-Exploitation Dilemma](./11_reinforcement_learning/01_multi_armed_bandits.md)**
- [x] **11.2 [Markov Decision Processes (MDPs) & Formal Definitions](./11_reinforcement_learning/02_markov_decision_processes.md)**
- [x] **11.3 [The Bellman Equations: State-Value and Action-Value Functions](./11_reinforcement_learning/03_bellman_equations.md)**
- [x] **11.4 [Dynamic Programming & The Banach Contraction Mapping Theorem](./11_reinforcement_learning/04_dynamic_programming_and_contraction.md)**
- [x] **11.5 [Monte Carlo Methods for Prediction and Control](./11_reinforcement_learning/05_monte_carlo_methods.md)**
- [x] **11.6 [Temporal-Difference Learning (TD(0))](./11_reinforcement_learning/06_temporal_difference_learning.md)**
- [x] **11.7 [On-Policy vs. Off-Policy Control: SARSA, Q-Learning & Expected SARSA](./11_reinforcement_learning/07_sarsa_and_q_learning.md)**
- [x] **11.8 [Multi-Step Bootstrapping & Eligibility Traces: TD(λ)](./11_reinforcement_learning/08_multi_step_and_eligibility_traces.md)**
- [x] **11.9 [Function Approximation & The Deadly Triad](./11_reinforcement_learning/09_function_approximation_and_deadly_triad.md)**
- [x] **11.10 [Deep Q-Networks (DQN)](./11_reinforcement_learning/10_deep_q_networks.md)**
- [x] **11.11 [Advanced DQN Extensions: The Rainbow Suite](./11_reinforcement_learning/11_rainbow_dqn_suite.md)**
- [x] **11.12 [Distributional Reinforcement Learning (C51 & QR-DQN)](./11_reinforcement_learning/12_distributional_reinforcement_learning.md)**
- [x] **11.13 [The Policy Gradient Theorem & REINFORCE](./11_reinforcement_learning/13_policy_gradient_and_reinforce.md)**
- [x] **11.14 [Advantage Actor-Critic (A2C & A3C)](./11_reinforcement_learning/14_advantage_actor_critic.md)**
- [x] **11.15 [Generalized Advantage Estimation (GAE)](./11_reinforcement_learning/15_generalized_advantage_estimation.md)**
- [x] **11.16 [Continuous Action Spaces: DDPG & TD3](./11_reinforcement_learning/16_continuous_action_spaces_ddpg_td3.md)**
- [x] **11.17 [Natural Policy Gradients & Information Geometry](./11_reinforcement_learning/17_natural_policy_gradients.md)**
- [x] **11.18 [Trust Region Policy Optimization (TRPO)](./11_reinforcement_learning/18_trust_region_policy_optimization.md)**
- [x] **11.19 [Proximal Policy Optimization (PPO)](./11_reinforcement_learning/19_proximal_policy_optimization.md)**
- [x] **11.20 [Soft Actor-Critic (SAC)](./11_reinforcement_learning/20_soft_actor_critic.md)**
- [x] **11.21 [Model-Based Foundations & Dyna-Q](./11_reinforcement_learning/21_model_based_dyna_q.md)**
- [x] **11.22 [Trajectory Optimization & Model Predictive Control (MPC)](./11_reinforcement_learning/22_trajectory_optimization_and_mpc.md)**
- [x] **11.23 [Monte Carlo Tree Search (MCTS) & AlphaZero](./11_reinforcement_learning/23_mcts_and_alphazero.md)**
- [x] **11.24 [World Models & Dreamer](./11_reinforcement_learning/24_world_models_and_dreamer.md)**
- [x] **11.25 [Offline (Batch) RL & Conservative Q-Learning (CQL)](./11_reinforcement_learning/25_offline_rl_cql.md)**
- [x] **11.26 [Decision Transformers & Trajectory Modeling](./11_reinforcement_learning/26_decision_transformer.md)**
- [x] **11.27 [Reinforcement Learning from Human Feedback (RLHF) with PPO](./11_reinforcement_learning/27_rlhf_ppo.md)**
- [x] **11.28 [Group Relative Policy Optimization (GRPO) & Frontier Reasoning (DeepSeek-R1)](./11_reinforcement_learning/28_grpo_deepseek_r1.md)**

#### Module 12: Modern LLM Architectures & Engineering from Scratch (Raschka Deep Dive)
- [x] **12.1 [Modern Tokenization from Scratch: BPE, Byte-Level BPE & Tiktoken](./12_modern_llm_architectures/01_tokenization_from_scratch.md)**
- [x] **12.2 [Modern Core Blocks: Pre-RMSNorm, SwiGLU & Residual Scaling](./12_modern_llm_architectures/02_modern_core_blocks.md)**
- [x] **12.3 [Advanced Positional Encodings: RoPE, YaRN & NTK-Aware Scaling](./12_modern_llm_architectures/03_rope_and_long_context.md)**
- [x] **12.4 [Attention Evolution: MHA, MQA, Grouped-Query Attention (GQA) & The KV Cache](./12_modern_llm_architectures/04_attention_and_kv_cache.md)**
- [x] **12.5 [Memory & Hardware Efficiency: FlashAttention-1/2/3 & Online Softmax](./12_modern_llm_architectures/05_flash_attention_online_softmax.md)**
- [x] **12.6 [Sparse Mixture of Experts (MoE): Top-K Routing, Auxiliary Losses & DeepSeek-V3 MoE / MLA](./12_modern_llm_architectures/06_sparse_mixture_of_experts.md)**
- [x] **12.7 [Pre-Training Engineering: Chinchilla Scaling Laws, WSD Learning Schedules & Distributed Training (FSDP/ZeRO)](./12_modern_llm_architectures/07_pretraining_and_scaling_laws.md)**
- [x] **12.8 [Parameter-Efficient Fine-Tuning: LoRA, QLoRA (NF4 Quantization) & DoRA](./12_modern_llm_architectures/08_peft_lora_and_qlora.md)**
- [x] **12.9 [Post-Training I: Supervised Instruction Fine-Tuning (SFT) & Multi-Turn Trajectory Packing](./12_modern_llm_architectures/09_instruction_tuning_sft.md)**
- [x] **12.10 [Post-Training II: Direct Preference Alignment (DPO, ORPO & SimPO)](./12_modern_llm_architectures/10_preference_alignment_dpo.md)**
- [x] **12.11 [Diffusion Transformers (DiT): Replacing UNet with Transformers for Generative Modeling](./12_modern_llm_architectures/11_diffusion_transformers_dit.md)**
- [x] **12.12 [Multimodal Vision-Language Architectures (Vision Encoders, Cross-Attention & Linear Projections)](./12_modern_llm_architectures/12_multimodal_vision_language.md)**

#### Module 13: Frontier Reasoning & Inference-Time Compute from Scratch
- [x] **13.1 [Foundations of LLM Reasoning: System 1 vs. System 2 & Chain-of-Thought Dynamics](./13_reasoning_and_test_time_compute/01_reasoning_foundations_cot.md)**
- [x] **13.2 [Rationale Bootstrapping: The Self-Taught Reasoner (STaR) & Quiet-STaR](./13_reasoning_and_test_time_compute/02_rationale_bootstrapping_star.md)**
- [x] **13.3 [Test-Time Compute & Inference Scaling Laws (Best-of-N, Beam Search & Budget Allocation)](./13_reasoning_and_test_time_compute/03_test_time_compute_scaling.md)**
- [x] **13.4 [Process Reward Models (PRMs) & Step-Level Value Estimation (PRM800K & Math-Shepherd)](./13_reasoning_and_test_time_compute/04_process_reward_models_prm.md)**
- [x] **13.5 [Tree Search & Monte Carlo Tree Search (MCTS) for Language Models (Tree of Thoughts & PUCT)](./13_reasoning_and_test_time_compute/05_mcts_tree_search.md)**
- [x] **13.6 [Pure Reinforcement Learning for Reasoning: DeepSeek-R1-Zero & The "Aha" Moment Emergence](./13_reasoning_and_test_time_compute/06_pure_rl_reasoning_r1_zero.md)**
- [x] **13.7 [Cold-Start Data, Multi-Stage Alignment & Reasoning Distillation](./13_reasoning_and_test_time_compute/07_distillation_and_cold_start.md)**
- [x] **13.8 [Self-Correction, Backtracking & External Verification Loops (Python Sandboxes & Lean 4)](./13_reasoning_and_test_time_compute/08_self_correction_and_backtracking.md)**
- [x] **13.9 [Formal Reasoning & Interactive Theorem Proving (Lean 4, AlphaProof & Autoformalization)](./13_reasoning_and_test_time_compute/09_formal_reasoning_lean4.md)**
