"""
Chapter 4.3: Maximum Likelihood Estimation (MLE) & Loss Function Derivations
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Grid & Closed-Form Linear Regression MLE.
2. Joint Gaussian MLE Bias Verification (E[sigma^2_MLE] = (N-1)/N * sigma^2).
3. Bernoulli MLE vs. PyTorch BCEWithLogitsLoss Optimization.
4. Linearly Separable Logistic Regression Weight Explosion & L2 Stabilization.
5. Heteroscedastic Aleatoric Uncertainty Loss in PyTorch.
"""

import numpy as np
import scipy.optimize as opt
import torch
import torch.nn as nn
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Visual Grid Verification
# =====================================================================
def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 4.3 Part 5:
    x = [1.0, 2.0, 3.0], y = [1.6, 4.2, 6.0], sigma^2 = 1.0.
    Model: y_hat = w * x
    """
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([1.6, 4.2, 6.0], dtype=np.float64)
    n = len(x)

    # 1. Sufficient Statistics
    sum_x2 = float(np.sum(x ** 2))
    sum_xy = float(np.sum(x * y))
    w_mle = sum_xy / sum_x2

    print("--- Part 5: Visual Grid Verification ---")
    print(f"Sum x^2:           {sum_x2:.4f} (Expected: 14.0000)")
    print(f"Sum x*y:           {sum_xy:.4f} (Expected: 28.0000)")
    print(f"Analytical w_MLE:  {w_mle:.4f} (Expected: 2.0000)")

    assert abs(sum_x2 - 14.0) < 1e-7
    assert abs(sum_xy - 28.0) < 1e-7
    assert abs(w_mle - 2.0) < 1e-7

    # 2. Evaluate candidate weights
    candidates = [1.0, 1.5, 2.0, 2.5]
    expected_sse = [14.2000, 3.7000, 0.2000, 3.7000]
    expected_ll = [-9.8568, -4.6068, -2.8568, -4.6068]

    c_norm = -0.5 * n * math.log(2.0 * math.pi)  # -2.756815599614018

    for cand, exp_s, exp_l in zip(candidates, expected_sse, expected_ll):
        preds = cand * x
        res = y - preds
        sse = float(np.sum(res ** 2))
        log_lik = c_norm - 0.5 * sse
        print(f"w = {cand:.1f} | SSE = {sse:.4f} (Exp: {exp_s:.4f}) | Log-Lik = {log_lik:.4f} (Exp: {exp_l:.4f})")
        assert abs(sse - exp_s) < 1e-4
        assert abs(log_lik - exp_l) < 1e-4

    # 3. Numerical optimization validation
    def neg_log_lik(w_val):
        return 0.5 * np.sum((y - w_val * x) ** 2) - c_norm

    res_opt = opt.minimize_scalar(neg_log_lik, bounds=(0.0, 4.0), method='bounded')
    print(f"Scipy Optimize w_MLE: {res_opt.x:.6f}")
    assert abs(res_opt.x - 2.0) < 1e-6
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Joint Gaussian MLE Bias Verification
# =====================================================================
def verify_gaussian_mle_bias(num_trials: int = 100_000):
    """
    Demonstrates that the MLE of Gaussian variance sigma^2 is biased:
    E[sigma^2_MLE] = ((N - 1) / N) * sigma^2.
    """
    np.random.seed(42)
    n = 10
    true_mu = 5.0
    true_sigma2 = 9.0
    true_sigma = math.sqrt(true_sigma2)

    samples = np.random.normal(loc=true_mu, scale=true_sigma, size=(num_trials, n))

    # MLE of mu is sample mean
    mu_mle = np.mean(samples, axis=1)
    # MLE of sigma^2 is (1/N) * sum(X_i - mu_mle)^2 (ddof=0)
    sigma2_mle = np.var(samples, axis=1, ddof=0)

    emp_mu = float(np.mean(mu_mle))
    emp_sigma2 = float(np.mean(sigma2_mle))
    theo_sigma2 = ((n - 1.0) / n) * true_sigma2  # (9 / 10) * 9.0 = 8.1000

    print("\n--- Joint Gaussian MLE Bias Simulation ---")
    print(f"True mu: {true_mu:.4f} | Empirical E[mu_MLE]: {emp_mu:.4f}")
    print(f"True sigma^2: {true_sigma2:.4f} | Theo E[sigma^2_MLE]: {theo_sigma2:.4f} | Empirical: {emp_sigma2:.4f}")

    assert abs(emp_mu - true_mu) < 0.02
    assert abs(emp_sigma2 - theo_sigma2) < 0.02
    print("✓ Gaussian MLE variance downward bias confirmed!")


# =====================================================================
# 3. Bernoulli MLE vs. PyTorch BCEWithLogitsLoss
# =====================================================================
def verify_bernoulli_mle_pytorch():
    """
    Verifies that minimizing PyTorch BCEWithLogitsLoss yields an output
    probability equal to the analytical sample proportion p_hat = sum(y) / N.
    """
    torch.manual_seed(42)
    # 70 ones and 30 zeros
    y_data = torch.tensor([1.0] * 70 + [0.0] * 30, dtype=torch.float32)
    n = len(y_data)
    analytical_p = float(torch.mean(y_data))  # 0.70

    # Single scalar logit parameter
    logit = torch.tensor([0.0], requires_grad=True)
    optimizer = torch.optim.Adam([logit], lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(logit.expand(n), y_data)
        loss.backward()
        optimizer.step()

    learned_p = float(torch.sigmoid(logit).item())
    print("\n--- Bernoulli MLE vs. PyTorch BCE Loss ---")
    print(f"Analytical Sample Proportion: {analytical_p:.4f}")
    print(f"PyTorch Learned Probability:  {learned_p:.4f}")

    assert abs(learned_p - analytical_p) < 0.001
    print("✓ PyTorch BCE optimization matched closed-form Bernoulli MLE!")


# =====================================================================
# 4. Linearly Separable Logistic Regression Weight Explosion
# =====================================================================
def verify_separable_weight_explosion():
    """
    Demonstrates that for linearly separable data, unregularized logistic
    regression weights diverge toward infinity, but L2 weight decay stabilizes them.
    """
    torch.manual_seed(42)
    # Separable 1D dataset: negative points at -2, -1; positive at +1, +2
    x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)

    # 1. Unregularized model
    w_unreg = torch.tensor([[0.5]], requires_grad=True)
    opt_unreg = torch.optim.SGD([w_unreg], lr=0.2)
    for _ in range(1500):
        opt_unreg.zero_grad()
        preds = torch.sigmoid(x @ w_unreg)
        loss = nn.functional.binary_cross_entropy(preds, y)
        loss.backward()
        opt_unreg.step()

    final_w_unreg = float(w_unreg.item())

    # 2. Regularized model (Weight Decay / L2 penalty = 0.1)
    w_reg = torch.tensor([[0.5]], requires_grad=True)
    opt_reg = torch.optim.SGD([w_reg], lr=0.2, weight_decay=0.1)
    for _ in range(1500):
        opt_reg.zero_grad()
        preds = torch.sigmoid(x @ w_reg)
        loss = nn.functional.binary_cross_entropy(preds, y)
        loss.backward()
        opt_reg.step()

    final_w_reg = float(w_reg.item())

    print("\n--- Linearly Separable Logistic Regression ---")
    print(f"Unregularized Weight after 1500 steps: {final_w_unreg:.4f} (Diverging toward infinity)")
    print(f"L2 Regularized Weight (decay=0.1):    {final_w_reg:.4f} (Stabilized at finite minimum)")

    assert final_w_unreg > 5.0, "Unregularized weight must grow large"
    assert final_w_reg < 3.0, "Regularized weight must be constrained"
    print("✓ Complete separation weight divergence & L2 regularization verified!")


# =====================================================================
# 5. Heteroscedastic Aleatoric Loss PyTorch Module
# =====================================================================
class HeteroscedasticRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Outputs [mu, log_var (s)]
        )

    def forward(self, x):
        out = self.fc(x)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        return mu, log_var


def gaussian_heteroscedastic_nll_loss(mu, log_var, target):
    """
    L = 0.5 * exp(-log_var) * (target - mu)^2 + 0.5 * log_var
    """
    precision = torch.exp(-log_var)
    loss = 0.5 * precision * ((target - mu) ** 2) + 0.5 * log_var
    return torch.mean(loss)


def verify_heteroscedastic_loss():
    """
    Trains a small model to predict input-dependent noise variance:
    y = 2*x + eps, where eps ~ N(0, (0.2 + 0.8*x)^2) for x in [0, 2].
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate synthetic data with variance growing with x
    n = 1000
    x_np = np.random.uniform(0.1, 2.0, size=(n, 1)).astype(np.float32)
    true_std = 0.2 + 0.6 * x_np
    noise = np.random.normal(0.0, true_std).astype(np.float32)
    y_np = 2.0 * x_np + noise

    x_tensor = torch.tensor(x_np)
    y_tensor = torch.tensor(y_np)

    model = HeteroscedasticRegressionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(600):
        optimizer.zero_grad()
        mu, log_var = model(x_tensor)
        loss = gaussian_heteroscedastic_nll_loss(mu, log_var, y_tensor)
        loss.backward()
        optimizer.step()

    # Test predictions at small x vs large x
    test_x = torch.tensor([[0.2], [1.8]])
    with torch.no_grad():
        mu_pred, log_var_pred = model(test_x)
        std_pred = torch.exp(0.5 * log_var_pred).numpy().flatten()

    print("\n--- Heteroscedastic Aleatoric Uncertainty Estimation ---")
    print(f"At x = 0.2: Predicted Std = {std_pred[0]:.4f} (True ~0.32)")
    print(f"At x = 1.8: Predicted Std = {std_pred[1]:.4f} (True ~1.28)")

    # Model must recognize that x=1.8 has higher noise than x=0.2
    assert std_pred[1] > 2.0 * std_pred[0]
    print("✓ Heteroscedastic NLL loss successfully captured input-dependent variance!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 4.3 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_gaussian_mle_bias()
    verify_bernoulli_mle_pytorch()
    verify_separable_weight_explosion()
    verify_heteroscedastic_loss()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 4.3 PASSED CLEANLY!")
    print("=================================================================")
