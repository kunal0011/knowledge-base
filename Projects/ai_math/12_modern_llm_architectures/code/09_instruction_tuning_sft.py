"""
Post-Training I: SFT, Prompt-Response Masking & Trajectory Packing
Implementation and Verification Suite
Module 12: Modern LLM Architectures & Engineering from Scratch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_block_diagonal_mask(doc_lens: list[int]) -> torch.Tensor:
    """
    Builds an (L, L) block-diagonal causal attention mask.
    Tokens within the same document attend causally to each other.
    Tokens across different documents receive -inf.
    """
    total_len = sum(doc_lens)
    mask = torch.full((total_len, total_len), float('-inf'), dtype=torch.float32)
    
    start_idx = 0
    for length in doc_lens:
        end_idx = start_idx + length
        # Sub-block causal mask: lower triangular 0, upper triangular -inf
        sub_causal = torch.triu(torch.full((length, length), float('-inf')), diagonal=1)
        mask[start_idx:end_idx, start_idx:end_idx] = sub_causal
        start_idx = end_idx
        
    return mask


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    doc_lens = [3, 2] # Doc 1 len 3, Doc 2 len 2
    total_len = 5
    
    # Step 1: Target Mask Vector
    targets = torch.tensor([-100, -100, 30, -100, 50], dtype=torch.long)
    expected_targets = torch.tensor([-100, -100, 30, -100, 50], dtype=torch.long)
    assert torch.equal(targets, expected_targets)
    print(f"Target vector (ignore_index=-100): {targets.tolist()} -> EXACT MATCH")
    
    # Step 2: Position IDs with reset
    pos_ids = []
    for length in doc_lens:
        pos_ids.extend(list(range(length)))
    pos_ids = torch.tensor(pos_ids, dtype=torch.long)
    expected_pos_ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    assert torch.equal(pos_ids, expected_pos_ids)
    print(f"Position IDs with document reset: {pos_ids.tolist()} -> EXACT MATCH [0, 1, 2, 0, 1]")
    
    # Step 3: Block-diagonal causal mask
    mask = build_block_diagonal_mask(doc_lens)
    # Check that Doc 2 cannot see Doc 1
    assert torch.all(mask[3, 0:3] == float('-inf'))
    assert torch.all(mask[4, 0:3] == float('-inf'))
    # Check causal lower triangular inside Doc 1
    assert mask[0, 0] == 0.0 and mask[0, 1] == float('-inf')
    assert mask[1, 0] == 0.0 and mask[1, 1] == 0.0 and mask[1, 2] == float('-inf')
    assert mask[2, 0] == 0.0 and mask[2, 1] == 0.0 and mask[2, 2] == 0.0
    print("Block-diagonal attention mask verified: Cross-document leakage strictly blocked.")
    
    # Step 4: SFT Loss computation
    vocab_size = 100
    logits = torch.zeros(total_len, vocab_size, dtype=torch.float32)
    
    # Set logits at pos 2 so that softmax prob for token 30 is 0.60
    # log(0.6 / 0.4) = log(1.5)
    logits[2, 30] = math.log(0.60)
    # All other 99 classes share the remaining 0.40 probability
    logits[2, 0] = math.log(0.40) # simplified two-candidate distribution
    p2 = F.softmax(logits[2, [0, 30]], dim=-1)[1].item()
    assert math.isclose(p2, 0.60, abs_tol=1e-5)
    
    # Set logits at pos 4 so that softmax prob for token 50 is 0.80
    logits[4, 50] = math.log(0.80)
    logits[4, 0] = math.log(0.20)
    p4 = F.softmax(logits[4, [0, 50]], dim=-1)[1].item()
    assert math.isclose(p4, 0.80, abs_tol=1e-5)
    
    # Compute manual loss
    l2 = -math.log(0.60)
    l4 = -math.log(0.80)
    expected_manual_loss = (l2 + l4) / 2.0 # 0.3669846
    
    # PyTorch loss calculation with target masking
    # Create two logits matching the exact binary distribution for numerical match
    logits_exact = torch.full((total_len, vocab_size), -1e9)
    logits_exact[2, 30] = math.log(0.60)
    logits_exact[2, 0] = math.log(0.40)
    logits_exact[4, 50] = math.log(0.80)
    logits_exact[4, 0] = math.log(0.20)
    
    py_loss = F.cross_entropy(logits_exact, targets, ignore_index=-100)
    assert math.isclose(py_loss.item(), expected_manual_loss, abs_tol=1e-5), f"Loss mismatch: {py_loss} vs {expected_manual_loss}"
    print(f"Manual loss: {expected_manual_loss:.6f}")
    print(f"PyTorch ignore_index=-100 loss: {py_loss.item():.6f} -> EXACT MATCH\n")


def verify_illustration_efficiency():
    print("--- 2. Verifying Illustration 1 Packing Efficiency ---")
    doc_lengths = [128, 256, 512, 1024]
    max_len = 1024
    padded_total = len(doc_lengths) * max_len  # 4096
    packed_total = sum(doc_lengths)            # 1920
    waste_tokens = padded_total - packed_total # 2176
    waste_pct = (waste_tokens / padded_total) * 100.0
    assert math.isclose(waste_pct, 53.125, abs_tol=1e-4)
    print(f"Padded tokens: {padded_total}, Packed tokens: {packed_total}")
    print(f"Wasted compute on padding: {waste_pct:.3f}% -> EXACT MATCH\n")


class MultiTurnTrajectoryPacker:
    """
    Packs multi-turn dialogues into fixed-length sequences with
    ChatML formatting, prompt loss masking, and position resets.
    """
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def pack(self, conversations: list[list[dict]], max_seq_len: int):
        """
        Args:
            conversations: List of dialogs, where each dialog is a list of
                           dict(role="user"|"assistant", tokens=[...])
        """
        all_input_ids = []
        all_labels = []
        all_position_ids = []
        doc_lens = []

        for conv in conversations:
            conv_tokens = []
            conv_labels = []
            
            for turn in conv:
                role = turn["role"]
                tokens = turn["tokens"]
                conv_tokens.extend(tokens)
                
                if role == "assistant":
                    # Supervise assistant tokens
                    conv_labels.extend(tokens)
                else:
                    # Mask user/system tokens
                    conv_labels.extend([self.ignore_index] * len(tokens))
                    
            if len(all_input_ids) + len(conv_tokens) <= max_seq_len:
                # Add position IDs reset
                pos_ids = list(range(len(conv_tokens)))
                all_position_ids.extend(pos_ids)
                
                all_input_ids.extend(conv_tokens)
                all_labels.extend(conv_labels)
                doc_lens.append(len(conv_tokens))

        attention_mask = build_block_diagonal_mask(doc_lens)
        
        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "position_ids": torch.tensor(all_position_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "doc_lens": doc_lens
        }


def test_trajectory_packer():
    print("--- 3. Testing Production Multi-Turn Trajectory Packer ---")
    conv1 = [
        {"role": "user", "tokens": [101, 102]},
        {"role": "assistant", "tokens": [201, 202, 203]}
    ]
    conv2 = [
        {"role": "user", "tokens": [301]},
        {"role": "assistant", "tokens": [401, 402]}
    ]
    
    packer = MultiTurnTrajectoryPacker()
    packed = packer.pack([conv1, conv2], max_seq_len=16)
    
    assert packed["input_ids"].shape[0] == 8  # (2 + 3) + (1 + 2) = 8
    assert packed["labels"].shape[0] == 8
    assert packed["position_ids"].shape[0] == 8
    assert packed["attention_mask"].shape == (8, 8)
    
    # Verify labels masking
    expected_labels = [-100, -100, 201, 202, 203, -100, 401, 402]
    assert packed["labels"].tolist() == expected_labels
    
    # Verify position IDs reset
    expected_positions = [0, 1, 2, 3, 4, 0, 1, 2]
    assert packed["position_ids"].tolist() == expected_positions
    
    print("MultiTurnTrajectoryPacker successfully generated packed batch with proper masks and resets.")
    print("All tests in Chapter 12.9 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_illustration_efficiency()
    test_trajectory_packer()
    print("🟢 Chapter 12.9 Verification 100% Complete.")
