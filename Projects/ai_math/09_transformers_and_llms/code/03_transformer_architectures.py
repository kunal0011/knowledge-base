"""
Chapter 9.3: Transformer Architectures (BERT, GPT, T5)
======================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification (Cross-Attention step)
2. Pre-LN vs Post-LN 30-Layer Gradient Highway Experiment
3. Full CrossAttentionBlock forward and backward pass
4. Bidirectional (BERT) vs Causal (GPT) representation leakage test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid Cross-Attention Verification
# ----------------------------------------------------------------------
def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid Cross-Attention Verification ---")
    K = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64) # (2, 2)
    V = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64) # (2, 2)
    Q = torch.tensor([[1.0, 1.0]], dtype=torch.float64)            # (1, 2)
    
    # 1. Scores S_cross = Q * K^T / sqrt(2)
    d_k = 2.0
    S_cross = torch.matmul(Q, K.t()) / np.sqrt(d_k)
    S_cross_exp = torch.tensor([[1.0 / np.sqrt(2), 2.0 / np.sqrt(2)]], dtype=torch.float64)
    print(f"Computed S_cross: {S_cross.numpy()} | Expected: {S_cross_exp.numpy()}")
    assert torch.allclose(S_cross, S_cross_exp), "S_cross mismatch!"
    
    # 2. Softmax A_cross
    A_cross = F.softmax(S_cross, dim=-1)
    e1 = np.exp(1.0 / np.sqrt(2))
    e2 = np.exp(2.0 / np.sqrt(2))
    A_cross_exp = torch.tensor([[e1 / (e1 + e2), e2 / (e1 + e2)]], dtype=torch.float64)
    print(f"Computed A_cross: {A_cross.numpy()} | Expected: {A_cross_exp.numpy()}")
    assert torch.allclose(A_cross, A_cross_exp), "A_cross mismatch!"
    
    # 3. Context C = A_cross * V
    C = torch.matmul(A_cross, V)
    C_exp = torch.tensor([[A_cross_exp[0, 0] * 1.0, A_cross_exp[0, 1] * 2.0]], dtype=torch.float64)
    print(f"Computed Context C: {C.numpy()} | Expected: {C_exp.numpy()}")
    assert torch.allclose(C, C_exp), "Context C mismatch!"
    print("✓ Part 5 Visual Grid Cross-Attention hand calculations verified!\n")

# ----------------------------------------------------------------------
# 2. Pre-LN vs Post-LN Blocks and The 30-Layer Gradient Highway
# ----------------------------------------------------------------------
class PreLNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, d_model, bias=False) # linear projection abstraction
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PostLNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.mlp(x))
        return x

def test_gradient_highway_pre_vs_post():
    print("--- 2. Pre-LN vs Post-LN 30-Layer Gradient Highway Experiment ---")
    num_layers = 30
    d_model = 32
    torch.manual_seed(42)
    
    pre_ln_net = nn.Sequential(*[PreLNBlock(d_model) for _ in range(num_layers)])
    post_ln_net = nn.Sequential(*[PostLNBlock(d_model) for _ in range(num_layers)])
    
    x_pre = torch.randn(4, 10, d_model, requires_grad=True)
    x_post = x_pre.clone().detach().requires_grad_(True)
    
    out_pre = pre_ln_net(x_pre)
    out_post = post_ln_net(x_post)
    
    out_pre.sum().backward()
    out_post.sum().backward()
    
    grad_norm_pre = x_pre.grad.norm().item()
    grad_norm_post = x_post.grad.norm().item()
    
    print(f"30-Layer Pre-LN  Input Gradient Norm: {grad_norm_pre:.6f}")
    print(f"30-Layer Post-LN Input Gradient Norm: {grad_norm_post:.6f}")
    
    # Pre-LN gradient highway maintains strong gradient flow
    assert grad_norm_pre > 0.5, f"Pre-LN gradient collapsed: {grad_norm_pre}"
    # Post-LN gradient is significantly smaller due to 1/sqrt(L) normalizations
    assert grad_norm_pre > grad_norm_post, "Pre-LN should preserve larger gradient flow than Post-LN!"
    print("✓ Pre-LN gradient highway theorem confirmed over 30 layers!\n")

# ----------------------------------------------------------------------
# 3. Full CrossAttentionBlock Implementation
# ----------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, dec_tokens, enc_memory, causal_mask=None):
        """
        dec_tokens: (B, T_dec, d_model)
        enc_memory: (B, T_enc, d_model)
        """
        # 1. Pre-LN Causal Self-Attention
        norm_dec = self.norm_self(dec_tokens)
        sa_out, _ = self.self_attn(norm_dec, norm_dec, norm_dec, 
                                   attn_mask=causal_mask, is_causal=causal_mask is not None)
        dec_tokens = dec_tokens + sa_out
        
        # 2. Pre-LN Cross-Attention (Q from dec, K/V from enc)
        norm_cross_q = self.norm_cross(dec_tokens)
        ca_out, _ = self.cross_attn(query=norm_cross_q, key=enc_memory, value=enc_memory)
        dec_tokens = dec_tokens + ca_out
        
        # 3. Pre-LN MLP
        dec_tokens = dec_tokens + self.mlp(self.norm_mlp(dec_tokens))
        return dec_tokens

def test_cross_attention_block():
    print("--- 3. Full CrossAttentionBlock (T5 Decoder Style) Verification ---")
    B, T_dec, T_enc, d_model = 2, 4, 7, 16
    block = CrossAttentionBlock(d_model=d_model, num_heads=2)
    
    dec = torch.randn(B, T_dec, d_model, requires_grad=True)
    enc = torch.randn(B, T_enc, d_model, requires_grad=True)
    
    out = block(dec, enc)
    assert out.shape == (B, T_dec, d_model), f"Output shape mismatch: {out.shape}"
    
    loss = out.sum()
    loss.backward()
    
    assert dec.grad is not None and enc.grad is not None
    assert not torch.isnan(dec.grad).any() and not torch.isnan(enc.grad).any()
    print("✓ CrossAttentionBlock forward and backward passes verified!\n")

# ----------------------------------------------------------------------
# 4. Bidirectional (BERT) vs Causal (GPT) Representation Leakage Test
# ----------------------------------------------------------------------
def test_bert_vs_gpt_leakage():
    print("--- 4. Bidirectional (BERT) vs Causal (GPT) Information Leakage ---")
    d_model = 16
    T = 4
    
    # Simple self-attention
    mha = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
    
    x = torch.randn(1, T, d_model)
    
    # 1. Bidirectional attention (BERT)
    out_bert, _ = mha(x, x, x)
    
    # Modify token 3 in input
    x_mod = x.clone()
    x_mod[0, 3] += 10.0 # modify the last token
    
    out_bert_mod, _ = mha(x_mod, x_mod, x_mod)
    # In BERT, changing token 3 affects token 0!
    diff_bert_tok0 = torch.norm(out_bert[0, 0] - out_bert_mod[0, 0]).item()
    print(f"BERT Token 0 change when Token 3 is modified: {diff_bert_tok0:.4f}")
    assert diff_bert_tok0 > 0.1, "BERT should allow future information to affect earlier tokens!"
    
    # 2. Causal attention (GPT)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(T)
    out_gpt, _ = mha(x, x, x, attn_mask=causal_mask, is_causal=True)
    out_gpt_mod, _ = mha(x_mod, x_mod, x_mod, attn_mask=causal_mask, is_causal=True)
    
    # In GPT, changing token 3 CANNOT affect token 0!
    diff_gpt_tok0 = torch.norm(out_gpt[0, 0] - out_gpt_mod[0, 0]).item()
    print(f"GPT  Token 0 change when Token 3 is modified: {diff_gpt_tok0:.2e}")
    assert diff_gpt_tok0 < 1e-6, "GPT causal attention leaked future information into past tokens!"
    print("✓ Causal isolation strictly confirmed: GPT token 0 is invariant to future token modifications!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 9.3 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_gradient_highway_pre_vs_post()
    test_cross_attention_block()
    test_bert_vs_gpt_leakage()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 9.3 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
