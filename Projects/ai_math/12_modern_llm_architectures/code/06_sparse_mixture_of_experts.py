"""
Sparse Mixture of Experts (MoE), Top-K Gating & Load Balancing
Implementation and Verification Suite
Module 12: Modern LLM Architectures & Engineering from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    x = torch.tensor([2.0, 1.0], dtype=torch.float32)
    W_g = torch.tensor([
        [1.0, -1.0,  2.0, 0.0],
        [0.0,  2.0, -1.0, 1.0]
    ], dtype=torch.float32)
    
    # Step 1: Compute router logits H = x W_g
    H = torch.matmul(x, W_g)
    expected_H = torch.tensor([2.0, 0.0, 3.0, 1.0], dtype=torch.float32)
    assert torch.allclose(H, expected_H, atol=1e-5), f"H mismatch: got {H}, expected {expected_H}"
    print(f"Router logits H: {H.tolist()} -> EXACT MATCH [2.0, 0.0, 3.0, 1.0]")
    
    # Step 2: Top-K (K=2)
    top_values, top_indices = torch.topk(H, k=2, dim=-1)
    # Indices should be 2 (Expert 3) and 0 (Expert 1)
    assert set(top_indices.tolist()) == {2, 0}, f"Top-2 indices mismatch: {top_indices}"
    print(f"Top-2 indices: {top_indices.tolist()} (Experts 3 and 1) -> EXACT MATCH")
    
    # Step 3: Softmax over top-2 logits
    softmax_weights = F.softmax(top_values, dim=-1)
    # Map back to expert indices
    # Notice top_values corresponds to top_indices
    weight_dict = {idx.item(): w.item() for idx, w in zip(top_indices, softmax_weights)}
    g1 = weight_dict[0]
    g3 = weight_dict[2]
    expected_g1 = 0.268941
    expected_g3 = 0.731059
    assert math.isclose(g1, expected_g1, abs_tol=1e-5), f"g1 mismatch: {g1}"
    assert math.isclose(g3, expected_g3, abs_tol=1e-5), f"g3 mismatch: {g3}"
    assert math.isclose(g1 + g3, 1.0, abs_tol=1e-6)
    print(f"Softmax weights: g1={g1:.6f}, g3={g3:.6f} -> EXACT MATCH")
    
    # Step 4: Expert outputs
    E1 = torch.tensor([1.0, 4.0], dtype=torch.float32)
    E2 = torch.tensor([0.0, 1.0], dtype=torch.float32)
    E3 = torch.tensor([3.0, -2.0], dtype=torch.float32)
    E4 = torch.tensor([2.0, 2.0], dtype=torch.float32)
    E_shared = torch.tensor([0.5, 1.0], dtype=torch.float32)
    
    y_routed = g1 * E1 + g3 * E3
    expected_y_routed = torch.tensor([2.462118, -0.386354], dtype=torch.float32)
    assert torch.allclose(y_routed, expected_y_routed, atol=1e-5), f"y_routed mismatch: {y_routed}"
    print(f"Routed mixture output: {y_routed.tolist()} -> EXACT MATCH")
    
    # Step 5: Add Shared Expert
    y_total = y_routed + E_shared
    expected_y_total = torch.tensor([2.962118, 0.613646], dtype=torch.float32)
    assert torch.allclose(y_total, expected_y_total, atol=1e-5), f"y_total mismatch: {y_total}"
    print(f"Total MoE output (routed + shared): {y_total.tolist()} -> EXACT MATCH\n")


def verify_load_balancing_loss():
    print("--- 2. Verifying Auxiliary Load Balancing Loss Calculation ---")
    # Batch T=2 tokens, E=2 experts
    probs = torch.tensor([
        [0.8, 0.2],
        [0.6, 0.4]
    ], dtype=torch.float32)
    alpha = 0.01
    
    # Top-1 assignment
    top1 = torch.argmax(probs, dim=-1) # [0, 0]
    E = 2
    T = 2
    
    # Frequency f_i
    f = torch.zeros(E)
    for idx in top1:
        f[idx] += 1.0
    f = f / T  # [1.0, 0.0]
    
    # Mean probability P_i
    P = probs.mean(dim=0)  # [0.70, 0.30]
    
    aux_loss = alpha * E * torch.sum(f * P)
    expected_loss = 0.0140
    assert math.isclose(aux_loss.item(), expected_loss, abs_tol=1e-5), f"Loss mismatch: {aux_loss}"
    print(f"Computed Aux Load Balancing Loss: {aux_loss.item():.4f} -> EXACT MATCH {expected_loss}\n")


class ExpertFFN(nn.Module):
    """Standard 2-layer MLP expert."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


class SparseMoELayer(nn.Module):
    """
    Production-grade Sparse Mixture of Experts Layer with Top-K Gating,
    Auxiliary Load Balancing Loss, and DeepSeek-style Shared Experts.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_routed_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 1,
        aux_loss_weight: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.aux_loss_weight = aux_loss_weight
        
        # Router gate
        self.gate = nn.Linear(d_model, num_routed_experts, bias=False)
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff) for _ in range(num_routed_experts)
        ])
        
        # Shared experts (DeepSeekMoE style)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertFFN(d_model, d_ff) for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            aux_loss: Scalar auxiliary load balancing loss
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (N, D), where N = B * S
        N = x_flat.shape[0]
        
        # 1. Router logits & probabilities
        router_logits = self.gate(x_flat)  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)
        
        # 2. Top-K selection
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (N, K)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (N, K) - normalized over top-k
        
        # 3. Compute auxiliary load balancing loss
        # f_i = fraction of tokens routed to expert i
        # P_i = average routing probability of expert i across batch
        tokens_per_expert = torch.zeros(self.num_routed_experts, device=x.device)
        for k in range(self.top_k):
            tokens_per_expert.scatter_add_(
                0, top_k_indices[:, k], torch.ones(N, device=x.device)
            )
        f = tokens_per_expert / (N * self.top_k)  # normalized fraction
        P = router_probs.mean(dim=0)             # (E,)
        aux_loss = self.aux_loss_weight * self.num_routed_experts * torch.sum(f * P)
        
        # 4. Dispatch and aggregate routed expert outputs
        y_routed = torch.zeros_like(x_flat)
        for i in range(N):
            token_out = torch.zeros(D, device=x.device)
            for k in range(self.top_k):
                expert_idx = top_k_indices[i, k].item()
                weight = top_k_weights[i, k]
                expert_out = self.routed_experts[expert_idx](x_flat[i:i+1])
                token_out += weight * expert_out.squeeze(0)
            y_routed[i] = token_out
            
        # 5. Shared expert forward pass
        y_shared = torch.zeros_like(x_flat)
        if self.shared_experts is not None:
            for shared_exp in self.shared_experts:
                y_shared += shared_exp(x_flat)
                
        y_total = (y_routed + y_shared).view(B, S, D)
        return y_total, aux_loss


def test_sparse_moe_layer():
    print("--- 3. Testing Full PyTorch Sparse MoE Layer ---")
    torch.manual_seed(42)
    B, S, D, D_ff = 2, 4, 16, 32
    E, K = 4, 2
    
    moe = SparseMoELayer(
        d_model=D,
        d_ff=D_ff,
        num_routed_experts=E,
        top_k=K,
        num_shared_experts=1,
        aux_loss_weight=0.01
    )
    
    x = torch.randn(B, S, D, requires_grad=True)
    out, aux_loss = moe(x)
    
    assert out.shape == (B, S, D), f"Output shape mismatch: {out.shape}"
    assert aux_loss.ndim == 0, "Aux loss must be scalar"
    assert aux_loss.item() > 0.0, "Aux loss should be strictly positive"
    
    # Backpropagation check
    total_loss = out.sum() + aux_loss
    total_loss.backward()
    
    assert x.grad is not None, "Gradients failed to flow back to input x"
    assert moe.gate.weight.grad is not None, "Router gate did not receive gradients"
    print(f"Forward output shape: {out.shape}")
    print(f"Aux loss value: {aux_loss.item():.6f}")
    print("Backward pass verified: gradients successfully flowed through all active paths.")
    print("All Sparse MoE tests passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_load_balancing_loss()
    test_sparse_moe_layer()
    print("🟢 Chapter 12.6 Verification 100% Complete.")
