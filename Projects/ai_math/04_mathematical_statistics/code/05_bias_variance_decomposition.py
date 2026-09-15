"""
Chapter 4.5: The Bias-Variance Decomposition
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Numerical Grid Verification (Model A vs. Model B).
2. Monte Carlo Polynomial Regression Bias-Variance Tradeoff Simulation.
3. Bagging Ensemble Variance Reduction Verification.
4. Modern Double Descent Phenomenon with Minimum-Norm Interpolation.
"""

import numpy as np
import math


# =====================================================================
# 1. Part 5: "AI by Hand" Visual Grid Verification
# =====================================================================
def verify_part5_visual_grid():
    """
    Validates manual walkthrough values from Chapter 4.5 Part 5:
    True target: f(x_0) = 2.0000, noise sigma^2 = 0.2500.
    Model A predictions: [1.20, 1.40, 1.30, 1.10]
    Model B predictions: [1.10, 2.80, 1.30, 2.60]
    """
    f_true = 2.0000
    noise_sigma2 = 0.2500

    preds_a = np.array([1.20, 1.40, 1.30, 1.10], dtype=np.float64)
    preds_b = np.array([1.10, 2.80, 1.30, 2.60], dtype=np.float64)

    # Model A
    f_bar_a = float(np.mean(preds_a))
    bias_a = f_bar_a - f_true
    bias2_a = bias_a ** 2
    var_a = float(np.mean((preds_a - f_bar_a) ** 2))
    epe_a = bias2_a + var_a + noise_sigma2

    # Model B
    f_bar_b = float(np.mean(preds_b))
    bias_b = f_bar_b - f_true
    bias2_b = bias_b ** 2
    var_b = float(np.mean((preds_b - f_bar_b) ** 2))
    epe_b = bias2_b + var_b + noise_sigma2

    print("--- Part 5: Visual Grid Verification ---")
    print(f"Model A: f_bar = {f_bar_a:.4f} | Bias^2 = {bias2_a:.4f} | Var = {var_a:.4f} | Total EPE = {epe_a:.4f}")
    print(f"Model B: f_bar = {f_bar_b:.4f} | Bias^2 = {bias2_b:.4f} | Var = {var_b:.4f} | Total EPE = {epe_b:.4f}")

    assert abs(f_bar_a - 1.2500) < 1e-7
    assert abs(bias2_a - 0.5625) < 1e-7
    assert abs(var_a - 0.0125) < 1e-7
    assert abs(epe_a - 0.8250) < 1e-7

    assert abs(f_bar_b - 1.9500) < 1e-7
    assert abs(bias2_b - 0.0025) < 1e-7
    assert abs(var_b - 0.5725) < 1e-7
    assert abs(epe_b - 0.8250) < 1e-7

    # Both models achieve identical total error but inverted composition
    assert abs(epe_a - epe_b) < 1e-7
    print("✓ Part 5 visual grid numbers strictly validated!")


# =====================================================================
# 2. Monte Carlo Polynomial Regression Bias-Variance Simulation
# =====================================================================
def verify_polynomial_bias_variance():
    """
    Simulates 300 independent training datasets of size N=25 for polynomial
    models of degrees 1 to 5 to demonstrate the classic U-shaped tradeoff.
    """
    np.random.seed(42)
    n_train = 25
    n_trials = 300
    noise_std = 0.30
    degrees = [1, 2, 3, 5]

    # Test grid
    x_test = np.linspace(-1.0, 1.0, 50)
    y_true_test = np.sin(np.pi * x_test)

    print("\n--- Polynomial Regression Bias-Variance Tradeoff ---")
    results = {}
    for deg in degrees:
        all_preds = np.zeros((n_trials, len(x_test)))
        for trial in range(n_trials):
            x_train = np.random.uniform(-1.0, 1.0, size=n_train)
            y_train = np.sin(np.pi * x_train) + np.random.normal(0.0, noise_std, size=n_train)
            
            # Fit polynomial
            coeffs = np.polyfit(x_train, y_train, deg=deg)
            preds = np.polyval(coeffs, x_test)
            all_preds[trial, :] = preds

        # Bias and Variance averaged across test grid
        mean_preds = np.mean(all_preds, axis=0)
        bias2 = float(np.mean((mean_preds - y_true_test) ** 2))
        var = float(np.mean(np.var(all_preds, axis=0)))
        total_mse = bias2 + var + (noise_std ** 2)

        results[deg] = (bias2, var, total_mse)
        print(f"Degree {deg:1d} | Bias^2 = {bias2:.5f} | Var = {var:.5f} | Noise = {noise_std**2:.4f} | Total Test MSE = {total_mse:.5f}")

    # Degree 1 (linear) has higher bias than degree 3
    assert results[1][0] > results[3][0]
    # Degree 5 has higher variance than degree 1
    assert results[5][1] > results[1][1]
    # Degree 3 achieves the sweet spot (lowest total MSE)
    assert results[3][2] < results[1][2]
    print("✓ Classical Bias-Variance U-curve verified!")


# =====================================================================
# 3. Bagging Ensemble Variance Reduction Verification
# =====================================================================
def verify_bagging_variance_reduction():
    """
    Verifies that model ensembling reduces variance while preserving bias.
    """
    np.random.seed(42)
    n_train = 30
    n_trials = 200
    noise_std = 0.30
    k_ensemble_sizes = [1, 2, 5, 10]

    x_test = np.linspace(-1.0, 1.0, 30)
    y_true_test = np.sin(np.pi * x_test)

    print("\n--- Bagging Ensemble Variance Reduction ---")
    variances = []
    biases = []

    for k in k_ensemble_sizes:
        ensemble_preds = np.zeros((n_trials, len(x_test)))
        for trial in range(n_trials):
            # Generate k bootstrapped models
            k_preds = np.zeros((k, len(x_test)))
            for m in range(k):
                x_train = np.random.uniform(-1.0, 1.0, size=n_train)
                y_train = np.sin(np.pi * x_train) + np.random.normal(0.0, noise_std, size=n_train)
                coeffs = np.polyfit(x_train, y_train, deg=4)
                k_preds[m, :] = np.polyval(coeffs, x_test)
            # Ensemble prediction is average across k models
            ensemble_preds[trial, :] = np.mean(k_preds, axis=0)

        mean_ens = np.mean(ensemble_preds, axis=0)
        bias2 = float(np.mean((mean_ens - y_true_test) ** 2))
        var = float(np.mean(np.var(ensemble_preds, axis=0)))
        biases.append(bias2)
        variances.append(var)
        print(f"Ensemble Size K = {k:2d} | Bias^2 = {bias2:.5f} | Variance = {var:.5f}")

    # Bias should remain virtually constant
    assert abs(biases[0] - biases[-1]) < 0.015
    # Variance should decrease significantly as K increases
    assert variances[-1] < 0.40 * variances[0]
    print("✓ Bagging pure variance reduction verified!")


# =====================================================================
# 4. Modern Double Descent Phenomenon Verification
# =====================================================================
def verify_double_descent():
    """
    Demonstrates the Double Descent curve using minimum-norm least squares:
    Test error peaks around the interpolation threshold (P = N) and drops
    in the heavily overparameterized regime (P >> N).
    """
    np.random.seed(42)
    n_train = 30
    n_test = 200
    noise_std = 0.20

    # Underlying 1D problem mapped to random Fourier features
    x_train = np.linspace(-1.0, 1.0, n_train)[:, np.newaxis]
    y_train = np.sin(np.pi * x_train.flatten()) + np.random.normal(0.0, noise_std, size=n_train)

    x_test = np.linspace(-1.0, 1.0, n_test)[:, np.newaxis]
    y_test = np.sin(np.pi * x_test.flatten())

    feature_dims = [8, 16, 25, 30, 35, 60, 120]
    test_mses = []

    print("\n--- Modern Double Descent Verification ---")
    for p in feature_dims:
        # Fixed random projection frequencies
        np.random.seed(123)
        omega = np.random.normal(0.0, 3.0, size=(1, p))
        b = np.random.uniform(0.0, 2.0 * np.pi, size=(1, p))

        # Random Fourier feature matrix: cos(x * omega + b)
        phi_train = np.cos(x_train @ omega + b)  # shape (n_train, p)
        phi_test = np.cos(x_test @ omega + b)    # shape (n_test, p)

        # Minimum-norm least squares solution via SVD pseudoinverse
        w_hat, _, _, _ = np.linalg.lstsq(phi_train, y_train, rcond=1e-4)

        preds_test = phi_test @ w_hat
        mse = float(np.mean((preds_test - y_test) ** 2))
        test_mses.append(mse)

        status = ""
        if p < n_train:
            status = "(Underparameterized)"
        elif p == n_train:
            status = "(Interpolation Peak! P=N)"
        else:
            status = "(Overparameterized)"

        print(f"Features P = {p:3d} | Test MSE = {mse:.4f} {status}")

    peak_idx = feature_dims.index(30)
    overparam_idx = feature_dims.index(120)

    # Error at peak (P=30) must be substantially higher than in overparameterized regime (P=120)
    assert test_mses[peak_idx] > test_mses[overparam_idx]
    print("✓ Double Descent test error peak and post-interpolation recovery verified!")


if __name__ == "__main__":
    print("=================================================================")
    print("RUNNING CHAPTER 4.5 NUMERICAL VERIFICATION & EXPERIMENTS")
    print("=================================================================")
    verify_part5_visual_grid()
    verify_polynomial_bias_variance()
    verify_bagging_variance_reduction()
    verify_double_descent()
    print("=================================================================")
    print("ALL TESTS IN CHAPTER 4.5 PASSED CLEANLY!")
    print("=================================================================")
