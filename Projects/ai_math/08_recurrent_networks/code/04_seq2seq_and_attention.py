"""
Chapter 8.4: Sequence-to-Sequence Models & Classical Attention (Bahdanau, Luong)
================================================================================
Rigorous verification test suite:
1. Part 5 Visual Grid Hand Arithmetic Verification (Dot-Product Attention)
2. Bahdanau (Additive) Attention module verification
3. Luong (Multiplicative: dot, general, concat) Attention module verification
4. End-to-end Seq2Seq with Attention (BiGRU Encoder + Attentional GRU Decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 1. Part 5 Visual Grid Hand Arithmetic Verification
# ----------------------------------------------------------------------
def test_part5_visual_grid():
    print("--- 1. Part 5 Visual Grid Attention Hand Arithmetic Verification ---")
    h1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    h2 = torch.tensor([0.0, 2.0], dtype=torch.float64)
    h3 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    H = torch.stack([h1, h2, h3]) # (3, 2)
    
    s = torch.tensor([1.0, 1.0], dtype=torch.float64)
    
    # 1. Scores e = s^T * H
    e = torch.matmul(H, s) # (3,)
    e_expected = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    print(f"Computed Scores e: {e.numpy()} | Expected: {e_expected.numpy()}")
    assert torch.allclose(e, e_expected), "Scores mismatch!"
    
    # 2. Softmax weights
    alpha = F.softmax(e, dim=0)
    exp_sum = np.exp(1.0) + np.exp(2.0) + np.exp(2.0)
    alpha_expected = torch.tensor([np.exp(1.0)/exp_sum, np.exp(2.0)/exp_sum, np.exp(2.0)/exp_sum], dtype=torch.float64)
    print(f"Computed Alpha: {alpha.numpy()} | Expected: {alpha_expected.numpy()}")
    assert torch.allclose(alpha, alpha_expected), "Alpha mismatch!"
    
    # 3. Context vector c = sum(alpha * H)
    c = torch.sum(alpha.unsqueeze(1) * H, dim=0)
    c1_exp = alpha_expected[0]*1.0 + alpha_expected[2]*1.0
    c2_exp = alpha_expected[1]*2.0 + alpha_expected[2]*1.0
    c_expected = torch.tensor([c1_exp, c2_exp], dtype=torch.float64)
    print(f"Computed Context c: {c.numpy()} | Expected: {c_expected.numpy()}")
    assert torch.allclose(c, c_expected), "Context vector mismatch!"
    print("✓ Part 5 Visual Grid Attention hand arithmetic strictly verified!\n")

# ----------------------------------------------------------------------
# 2. Bahdanau (Additive) Attention Implementation & Verification
# ----------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_a = nn.Linear(dec_dim, attn_dim, bias=False)
        self.U_a = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v_a = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys):
        """
        query: (B, dec_dim) - current/previous decoder state
        keys:  (B, T_x, enc_dim) - all encoder hidden states
        Returns:
        context: (B, enc_dim)
        weights: (B, T_x)
        """
        # query: (B, 1, attn_dim)
        q_proj = self.W_a(query).unsqueeze(1)
        # keys: (B, T_x, attn_dim)
        k_proj = self.U_a(keys)
        
        # energy: (B, T_x, 1) -> (B, T_x)
        energy = self.v_a(torch.tanh(q_proj + k_proj)).squeeze(2)
        weights = F.softmax(energy, dim=1) # (B, T_x)
        
        # context: (B, enc_dim) = sum_j (weights_j * keys_j)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

def test_bahdanau_attention():
    print("--- 2. Bahdanau (Additive) Attention Verification ---")
    B, T_x, enc_dim, dec_dim, attn_dim = 2, 5, 8, 8, 4
    attn = BahdanauAttention(enc_dim, dec_dim, attn_dim)
    
    query = torch.randn(B, dec_dim, requires_grad=True)
    keys = torch.randn(B, T_x, enc_dim, requires_grad=True)
    
    context, weights = attn(query, keys)
    
    assert context.shape == (B, enc_dim), f"Context shape mismatch: {context.shape}"
    assert weights.shape == (B, T_x), f"Weights shape mismatch: {weights.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(B)), "Weights must sum to 1.0!"
    
    # Backward test
    loss = context.sum()
    loss.backward()
    assert query.grad is not None and keys.grad is not None
    print("✓ Bahdanau Additive Attention forward and backward passes verified!\n")

# ----------------------------------------------------------------------
# 3. Luong (Multiplicative) Attention Implementation & Verification
# ----------------------------------------------------------------------
class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, method='dot'):
        super().__init__()
        self.method = method
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        
        if method == 'general':
            self.W_a = nn.Linear(enc_dim, dec_dim, bias=False)
        elif method == 'concat':
            self.W_a = nn.Linear(dec_dim + enc_dim, dec_dim, bias=False)
            self.v_a = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, query, keys):
        """
        query: (B, dec_dim)
        keys:  (B, T_x, enc_dim)
        """
        B, T_x, _ = keys.shape
        
        if self.method == 'dot':
            assert self.enc_dim == self.dec_dim, "Dot attention requires enc_dim == dec_dim!"
            # (B, T_x)
            energy = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        elif self.method == 'general':
            # keys: (B, T_x, dec_dim)
            k_proj = self.W_a(keys)
            energy = torch.bmm(k_proj, query.unsqueeze(2)).squeeze(2)
        elif self.method == 'concat':
            q_expand = query.unsqueeze(1).expand(-1, T_x, -1)
            concat = torch.cat([q_expand, keys], dim=2)
            energy = self.v_a(torch.tanh(self.W_a(concat))).squeeze(2)
            
        weights = F.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights

def test_luong_attention():
    print("--- 3. Luong Attention (Dot, General, Concat) Verification ---")
    B, T_x, dim = 2, 4, 6
    
    for method in ['dot', 'general', 'concat']:
        attn = LuongAttention(enc_dim=dim, dec_dim=dim, method=method)
        query = torch.randn(B, dim, requires_grad=True)
        keys = torch.randn(B, T_x, dim, requires_grad=True)
        
        context, weights = attn(query, keys)
        assert context.shape == (B, dim)
        assert torch.allclose(weights.sum(dim=1), torch.ones(B))
        context.sum().backward()
        assert query.grad is not None and keys.grad is not None
        print(f"✓ Luong [{method}] attention verified!")
    print()

# ----------------------------------------------------------------------
# 4. End-to-End Seq2Seq Model with Attention
# ----------------------------------------------------------------------
class AttentionalSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim, hid_dim):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, emb_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb_dim)
        
        # Bidirectional GRU Encoder
        self.encoder = nn.GRU(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        
        # Attention
        self.attn = BahdanauAttention(enc_dim=hid_dim * 2, dec_dim=hid_dim, attn_dim=hid_dim)
        
        # Decoder GRU: takes [embedding; context]
        self.decoder = nn.GRU(emb_dim + hid_dim * 2, hid_dim, batch_first=True)
        
        # Bridge layer from bidirectional encoder to unidirection decoder
        self.enc_to_dec = nn.Linear(hid_dim * 2, hid_dim)
        
        # Classification head
        self.fc_out = nn.Linear(hid_dim + hid_dim * 2, tgt_vocab)

    def forward(self, src, tgt):
        """
        src: (B, T_src)
        tgt: (B, T_tgt)
        """
        B, T_src = src.shape
        _, T_tgt = tgt.shape
        
        # 1. Encode
        src_emb = self.src_emb(src) # (B, T_src, emb_dim)
        enc_outputs, h_n = self.encoder(src_emb) # enc_outputs: (B, T_src, hid_dim * 2)
        
        # Combine forward & backward final states for decoder init
        h_combined = torch.cat([h_n[0], h_n[1]], dim=1) # (B, hid_dim * 2)
        dec_hidden = torch.tanh(self.enc_to_dec(h_combined)) # (B, hid_dim)
        
        # 2. Decode with Attention
        tgt_emb = self.tgt_emb(tgt) # (B, T_tgt, emb_dim)
        
        outputs = []
        all_weights = []
        
        for t in range(T_tgt):
            y_t = tgt_emb[:, t] # (B, emb_dim)
            
            # Compute attention context
            context, weights = self.attn(dec_hidden, enc_outputs)
            all_weights.append(weights)
            
            # Input to decoder GRU: [y_t; context]
            dec_in = torch.cat([y_t, context], dim=1).unsqueeze(1) # (B, 1, emb + 2*hid)
            out_dec, dec_hidden_next = self.decoder(dec_in, dec_hidden.unsqueeze(0))
            dec_hidden = dec_hidden_next.squeeze(0) # (B, hid_dim)
            
            # Predict logit from [dec_hidden; context]
            logit = self.fc_out(torch.cat([dec_hidden, context], dim=1)) # (B, tgt_vocab)
            outputs.append(logit)
            
        return torch.stack(outputs, dim=1), torch.stack(all_weights, dim=1)

def test_full_seq2seq():
    print("--- 4. Full Attentional Seq2Seq Pipeline Verification ---")
    B, T_src, T_tgt = 3, 7, 5
    src_vocab, tgt_vocab = 20, 25
    emb_dim, hid_dim = 16, 32
    
    model = AttentionalSeq2Seq(src_vocab, tgt_vocab, emb_dim, hid_dim)
    
    src = torch.randint(0, src_vocab, (B, T_src))
    tgt = torch.randint(0, tgt_vocab, (B, T_tgt))
    
    logits, attn_matrix = model(src, tgt)
    
    assert logits.shape == (B, T_tgt, tgt_vocab), f"Logits shape mismatch: {logits.shape}"
    assert attn_matrix.shape == (B, T_tgt, T_src), f"Attn matrix shape mismatch: {attn_matrix.shape}"
    
    loss = logits.sum()
    loss.backward()
    
    assert model.src_emb.weight.grad is not None
    assert model.tgt_emb.weight.grad is not None
    assert model.attn.W_a.weight.grad is not None
    print(f"Logits shape: {logits.shape} | Alignment matrix shape: {attn_matrix.shape}")
    print("✓ Full Attentional Seq2Seq forward and backward passes verified!\n")

def run_tests():
    print("=" * 65)
    print("RUNNING CHAPTER 8.4 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=" * 65)
    test_part5_visual_grid()
    test_bahdanau_attention()
    test_luong_attention()
    test_full_seq2seq()
    print("=" * 65)
    print("ALL TESTS IN CHAPTER 8.4 PASSED CLEANLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
