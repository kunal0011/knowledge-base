"""
Pre-Training Engineering: Chinchilla Scaling, WSD Schedules & ZeRO/FSDP
Implementation and Verification Suite
Module 12: Modern LLM Architectures & Engineering from Scratch
"""

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def chinchilla_flops(N: float, D: float) -> float:
    """Computes total FLOPs C = 6 * N * D."""
    return 6.0 * N * D


def chinchilla_token_budget(C: float, N: float) -> float:
    """Computes required tokens D = C / (6 * N)."""
    return C / (6.0 * N)


def memory_accounting(N: float, P: int = 8):
    """
    Computes static memory footprint across DDP, ZeRO-1, ZeRO-2, and ZeRO-3.
    Args:
        N: Number of parameters (e.g. 5e9 for 5B).
        P: World size (number of GPUs).
    Returns:
        dict with memory components in Gigabytes (1 GB = 1e9 bytes).
    """
    weights_gb = (2.0 * N) / 1e9
    grads_gb = (2.0 * N) / 1e9
    adam_master_gb = (4.0 * N) / 1e9
    adam_m_gb = (4.0 * N) / 1e9
    adam_v_gb = (4.0 * N) / 1e9
    optimizer_gb = adam_master_gb + adam_m_gb + adam_v_gb
    total_static_gb = weights_gb + grads_gb + optimizer_gb

    ddp_gb = total_static_gb
    zero1_gb = weights_gb + grads_gb + (optimizer_gb / P)
    zero2_gb = weights_gb + ((grads_gb + optimizer_gb) / P)
    zero3_gb = total_static_gb / P

    return {
        "weights_gb": weights_gb,
        "grads_gb": grads_gb,
        "optimizer_gb": optimizer_gb,
        "total_static_gb": total_static_gb,
        "ddp_gb": ddp_gb,
        "zero1_gb": zero1_gb,
        "zero2_gb": zero2_gb,
        "zero3_gb": zero3_gb
    }


def compute_lr_cosine(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(progress * math.pi))


def compute_lr_wsd(step: int, total_steps: int, warmup_steps: int, decay_start: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * (step / warmup_steps)
    elif step < decay_start:
        return lr_max
    else:
        progress = (step - decay_start) / (total_steps - decay_start)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(progress * math.pi))


def verify_part5_hand_trace():
    print("--- 1. Verifying Part 5 'AI by Hand' Manual Trace ---")
    C = 6.0e20
    N = 5.0e9
    P = 8

    # Token budget
    D = chinchilla_token_budget(C, N)
    expected_D = 20.0e9
    assert math.isclose(D, expected_D, rel_tol=1e-6), f"D mismatch: {D} vs {expected_D}"
    ratio = D / N
    assert math.isclose(ratio, 4.0, rel_tol=1e-6)
    print(f"Token budget D: {D/1e9:.1f} Billion tokens (Ratio D/N = {ratio:.1f}) -> EXACT MATCH")

    # Memory accounting
    mem = memory_accounting(N, P)
    assert math.isclose(mem["weights_gb"], 10.0, abs_tol=1e-5)
    assert math.isclose(mem["grads_gb"], 10.0, abs_tol=1e-5)
    assert math.isclose(mem["optimizer_gb"], 60.0, abs_tol=1e-5)
    assert math.isclose(mem["total_static_gb"], 80.0, abs_tol=1e-5)
    assert math.isclose(mem["ddp_gb"], 80.0, abs_tol=1e-5)
    assert math.isclose(mem["zero1_gb"], 27.50, abs_tol=1e-5)
    assert math.isclose(mem["zero2_gb"], 18.75, abs_tol=1e-5)
    assert math.isclose(mem["zero3_gb"], 10.00, abs_tol=1e-5)
    print(f"Memory footprints on 8x GPU cluster:")
    print(f"  DDP: {mem['ddp_gb']:.2f} GB")
    print(f"  ZeRO-1: {mem['zero1_gb']:.2f} GB")
    print(f"  ZeRO-2: {mem['zero2_gb']:.2f} GB")
    print(f"  ZeRO-3/FSDP: {mem['zero3_gb']:.2f} GB -> EXACT MATCH")

    # Learning rate schedules
    total_steps = 1000
    warmup_steps = 100
    decay_start = 800
    lr_max = 1e-3
    lr_min = 1e-5

    # Step 50
    wsd_50 = compute_lr_wsd(50, total_steps, warmup_steps, decay_start, lr_max, lr_min)
    cos_50 = compute_lr_cosine(50, total_steps, warmup_steps, lr_max, lr_min)
    assert math.isclose(wsd_50, 0.000500, abs_tol=1e-6)
    assert math.isclose(cos_50, 0.000500, abs_tol=1e-6)

    # Step 500
    wsd_500 = compute_lr_wsd(500, total_steps, warmup_steps, decay_start, lr_max, lr_min)
    cos_500 = compute_lr_cosine(500, total_steps, warmup_steps, lr_max, lr_min)
    assert math.isclose(wsd_500, 0.001000, abs_tol=1e-6)
    assert math.isclose(cos_500, 0.000591, abs_tol=1e-5)

    # Step 900
    wsd_900 = compute_lr_wsd(900, total_steps, warmup_steps, decay_start, lr_max, lr_min)
    cos_900 = compute_lr_cosine(900, total_steps, warmup_steps, lr_max, lr_min)
    assert math.isclose(wsd_900, 0.000505, abs_tol=1e-6)
    assert math.isclose(cos_900, 0.0000398, abs_tol=1e-6)
    print(f"Step 50 LR:  WSD={wsd_50:.6f}, Cosine={cos_50:.6f} -> EXACT MATCH")
    print(f"Step 500 LR: WSD={wsd_500:.6f}, Cosine={cos_500:.6f} -> EXACT MATCH")
    print(f"Step 900 LR: WSD={wsd_900:.6f}, Cosine={cos_900:.6f} -> EXACT MATCH\n")


def verify_illustration_1():
    print("--- 2. Verifying Illustration 1 Chinchilla Calculation ---")
    C = 1.2e20
    N = 1.0e9
    D = chinchilla_token_budget(C, N)
    assert math.isclose(D, 2.0e10, rel_tol=1e-6)
    assert math.isclose(D / N, 20.0, rel_tol=1e-6)

    throughput = 64 * 150e12  # 64 GPUs * 150 TFLOP/s
    time_s = C / throughput
    time_hrs = time_s / 3600.0
    assert math.isclose(time_s, 12500.0, abs_tol=1e-2)
    assert math.isclose(time_hrs, 3.47222, abs_tol=1e-3)
    print(f"Chinchilla optimal token count: {D/1e9:.1f}B tokens (Ratio: {D/N:.1f})")
    print(f"Training duration: {time_s:.0f}s ({time_hrs:.2f} hours) -> EXACT MATCH\n")


def test_pytorch_wsd_scheduler():
    print("--- 3. Testing PyTorch WSD Learning Rate Scheduler ---")
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    total_steps = 100
    warmup_steps = 10
    decay_start = 80
    lr_max = 1.0e-3
    lr_min = 1.0e-5

    def lr_lambda(step):
        # returns multiplier factor for lr_max
        lr = compute_lr_wsd(step, total_steps, warmup_steps, decay_start, lr_max, lr_min)
        return lr / lr_max

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    # Verify warmup peak
    assert math.isclose(lrs[warmup_steps], lr_max, rel_tol=1e-5)
    # Verify stable plateau
    assert math.isclose(lrs[50], lr_max, rel_tol=1e-5)
    # Verify decayed tail
    assert lrs[-1] < lrs[decay_start]
    print("PyTorch WSD Scheduler successfully stepped through Warmup, Stable, and Decay phases.")
    print("All tests in Chapter 12.7 passed successfully!\n")


if __name__ == "__main__":
    verify_part5_hand_trace()
    verify_illustration_1()
    test_pytorch_wsd_scheduler()
    print("🟢 Chapter 12.7 Verification 100% Complete.")
