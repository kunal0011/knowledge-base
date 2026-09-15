"""
Variational Autoencoder (VAE) & ELBO Derivation Verification
===========================================================
Module 10: Generative Modeling - Chapter 10.2

This script verifies:
1. Part 5 Tom Yeh "AI by Hand" Complete Numerical Parity:
   - Reparameterization z = mu + sigma * eps
   - Decoder forward pass with Sigmoid
   - Binary Cross-Entropy (BCE) reconstruction loss
   - Analytical Gaussian KL divergence
   - Total VAE loss and autograd gradient equivalence (dLoss/dmu and dLoss/dgamma)
2. Analytical KL vs. High-Sample Monte Carlo KL Divergence
3. LERP vs SLERP Latent Interpolation Norm Invariance
4. End-to-End VAE Architecture on synthetic 2D manifold
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def test_part5_hand_arithmetic_parity():
    print("--- Test 1: Part 5 Tom Yeh Hand Arithmetic Parity ---")
    dtype = torch.float64
    
    # Inputs & Parameters matching Part 5 exactly
    x = torch.tensor([0.8, 0.2], dtype=dtype)
    mu = torch.tensor([0.5, -0.4], dtype=dtype, requires_grad=True)
    gamma = torch.tensor([0.0, -0.6931471805599453], dtype=dtype, requires_grad=True)  # log(0.5)
    eps = torch.tensor([0.2, -0.5], dtype=dtype)
    
    W_dec = torch.tensor([[1.0, -0.5], [0.5, 1.0]], dtype=dtype)
    b_dec = torch.tensor([0.0, 0.0], dtype=dtype)
    
    # 1. Reparameterization
    sigma = torch.exp(0.5 * gamma)
    z = mu + sigma * eps
    
    expected_z = [0.7000, -0.4 - 0.70710678 * 0.5]
    print(f"Computed z: [{z[0].item():.4f}, {z[1].item():.4f}], Hand z: [{expected_z[0]:.4f}, {expected_z[1]:.4f}]")
    assert torch.allclose(z, torch.tensor(expected_z, dtype=dtype), atol=1e-5)
    
    # 2. Decoder forward pass
    a = W_dec @ z + b_dec
    x_hat = torch.sigmoid(a)
    print(f"Pre-activation a: [{a[0].item():.4f}, {a[1].item():.4f}]")
    print(f"Reconstructed x_hat: [{x_hat[0].item():.4f}, {x_hat[1].item():.4f}]")
    assert abs(x_hat[0].item() - 0.7459) < 1e-3
    assert abs(x_hat[1].item() - 0.4004) < 1e-3
    
    # 3. BCE Reconstruction Loss (Sum over features)
    loss_bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
    print(f"BCE Loss: {loss_bce.item():.4f} (Hand: 1.1009)")
    assert abs(loss_bce.item() - 1.1009) < 1e-3
    
    # 4. Analytical Gaussian KL Divergence
    # D_KL = -0.5 * sum(1 + gamma - mu^2 - exp(gamma))
    loss_kl = -0.5 * torch.sum(1.0 + gamma - mu ** 2 - torch.exp(gamma))
    print(f"KL Loss: {loss_kl.item():.4f} (Hand: 0.3016)")
    assert abs(loss_kl.item() - 0.3016) < 1e-3
    
    # 5. Total Loss
    loss_total = loss_bce + loss_kl
    print(f"Total VAE Loss: {loss_total.item():.4f} (Hand: 1.4025)")
    assert abs(loss_total.item() - 1.4025) < 1e-3
    
    # 6. Backward Pass & Gradients
    loss_total.backward()
    
    dmu_autograd = mu.grad.clone()
    dgamma_autograd = gamma.grad.clone()
    
    print(f"Autograd dL/dmu:    [{dmu_autograd[0].item():.4f}, {dmu_autograd[1].item():.4f}]")
    print(f"Autograd dL/dgamma: [{dgamma_autograd[0].item():.5f}, {dgamma_autograd[1].item():.5f}]")
    
    # Hand calculations from notes:
    # dL/dmu = [0.5461, -0.1725]
    # dL/dgamma = [0.00461, -0.2902]
    assert abs(dmu_autograd[0].item() - 0.5461) < 1e-3
    assert abs(dmu_autograd[1].item() - (-0.1725)) < 1e-3
    assert abs(dgamma_autograd[0].item() - 0.00461) < 1e-3
    assert abs(dgamma_autograd[1].item() - (-0.2902)) < 1e-3
    
    print("✓ Part 5 Hand Arithmetic verified to exact autograd parity!\n")


def test_analytical_vs_monte_carlo_kl():
    print("--- Test 2: Analytical Gaussian KL vs Monte Carlo KL ---")
    torch.manual_seed(42)
    
    # Arbitrary Gaussian parameters
    mu = torch.tensor([1.5, -2.0, 0.75])
    log_var = torch.tensor([0.2, -0.5, 0.8])
    std = torch.exp(0.5 * log_var)
    
    # 1. Analytical KL
    kl_analytical = -0.5 * torch.sum(1.0 + log_var - mu ** 2 - torch.exp(log_var)).item()
    
    # 2. Monte Carlo KL
    N = 2_000_000
    eps = torch.randn(N, 3)
    z_samples = mu + std * eps
    
    # log q(z|x) = sum_j [-0.5*log(2*pi) - 0.5*log(sigma_j^2) - (z_j - mu_j)^2 / (2*sigma_j^2)]
    log_q = torch.sum(-0.5 * math.log(2 * math.pi) - 0.5 * log_var - 0.5 * ((z_samples - mu) / std) ** 2, dim=-1)
    # log p(z) = sum_j [-0.5*log(2*pi) - 0.5 * z_j^2]
    log_p = torch.sum(-0.5 * math.log(2 * math.pi) - 0.5 * (z_samples ** 2), dim=-1)
    
    kl_mc = torch.mean(log_q - log_p).item()
    
    print(f"Analytical KL:  {kl_analytical:.6f}")
    print(f"Monte Carlo KL: {kl_mc:.6f} (N={N:,})")
    abs_diff = abs(kl_analytical - kl_mc)
    print(f"Absolute Difference: {abs_diff:.6f}")
    assert abs_diff < 0.005, f"Monte Carlo KL differs significantly from Analytical KL: {abs_diff}"
    print("✓ Analytical vs Monte Carlo KL consistency confirmed!\n")


def test_lerp_vs_slerp():
    print("--- Test 3: LERP vs SLERP Norm Invariance in High Dimensions ---")
    torch.manual_seed(123)
    d = 128
    
    # Sample two random vectors on high-dim Gaussian sphere
    z_A = torch.randn(d)
    z_B = torch.randn(d)
    
    norm_A = torch.norm(z_A).item()
    norm_B = torch.norm(z_B).item()
    expected_radius = math.sqrt(d)
    print(f"Dimension d={d}, Expected Shell Radius sqrt(d) = {expected_radius:.3f}")
    print(f"Norm z_A: {norm_A:.3f}, Norm z_B: {norm_B:.3f}")
    
    # LERP at midpoint alpha = 0.5
    z_lerp = 0.5 * z_A + 0.5 * z_B
    norm_lerp = torch.norm(z_lerp).item()
    
    # SLERP formula:
    # Omega = arccos( (z_A . z_B) / (||z_A|| * ||z_B||) )
    cos_omega = torch.dot(z_A, z_B) / (norm_A * norm_B)
    omega = torch.acos(torch.clamp(cos_omega, -1.0, 1.0))
    alpha = 0.5
    z_slerp = (math.sin((1 - alpha) * omega) / math.sin(omega)) * z_A + (math.sin(alpha * omega) / math.sin(omega)) * z_B
    norm_slerp = torch.norm(z_slerp).item()
    
    print(f"Midpoint LERP Norm:  {norm_lerp:.3f} (Severe shrinkage into low-probability hole!)")
    print(f"Midpoint SLERP Norm: {norm_slerp:.3f} (Shell radius strictly preserved!)")
    
    assert norm_lerp < 0.8 * expected_radius, "LERP should exhibit significant norm shrinkage"
    assert abs(norm_slerp - norm_A) < 1.0, "SLERP should maintain near-identical norm to endpoints"
    print("✓ LERP vs SLERP geometric properties confirmed!\n")


class ToyVAE(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, latent_dim=2):
        super().__init__()
        # Encoder
        self.enc_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.dec_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def encode(self, x):
        h = self.enc_net(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
        
    def decode(self, z):
        return self.dec_net(z)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def test_vae_training_and_generation():
    print("--- Test 4: End-to-End VAE Training & Reconstruction ---")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate 2D Swiss-roll / S-curve toy data normalized in [-2, 2]
    N = 2000
    t = np.random.uniform(-1.5 * np.pi, 1.5 * np.pi, size=N)
    x_data = np.stack([np.sin(t), np.cos(t) * 0.5 + t * 0.2], axis=1).astype(np.float32)
    x_tensor = torch.from_numpy(x_data)
    
    vae = ToyVAE(in_dim=2, hidden_dim=64, latent_dim=2)
    optimizer = optim.Adam(vae.parameters(), lr=0.005)
    
    # Train for 200 epochs
    batch_size = 128
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    initial_loss = None
    final_loss = None
    
    for epoch in range(60):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(batch_x)
            
            recon_loss = F.mse_loss(recon_x, batch_x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1.0 + logvar - mu ** 2 - torch.exp(logvar))
            
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        epoch_loss /= N
        if epoch == 0:
            initial_loss = epoch_loss
        final_loss = epoch_loss
        
    print(f"Epoch 1 Loss: {initial_loss:.4f} -> Epoch 60 Loss: {final_loss:.4f}")
    assert final_loss < initial_loss * 0.5, "VAE failed to optimize reconstruction and ELBO"
    
    # Test generation from prior z ~ N(0, I)
    z_prior = torch.randn(10, 2)
    with torch.no_grad():
        samples = vae.decode(z_prior)
    print(f"Generated 10 samples from latent prior, sample shape: {samples.shape}")
    assert samples.shape == (10, 2)
    assert not torch.isnan(samples).any()
    print("✓ End-to-end VAE training and generation verified!\n")


if __name__ == "__main__":
    test_part5_hand_arithmetic_parity()
    test_analytical_vs_monte_carlo_kl()
    test_lerp_vs_slerp()
    test_vae_training_and_generation()
    print("=========================================================")
    print("ALL MODULE 10 CHAPTER 10.2 VERIFICATIONS PASSED (100%)")
    print("=========================================================")
