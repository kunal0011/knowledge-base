"""
Chapter 3.5: Joint, Marginal & Conditional Distributions
Companion Verification Script & Numerical Experiments

Topics Covered:
1. Part 5 'AI by Hand' Triangular Joint Distribution Evaluation.
2. 2D Numerical Quadrature for Joint, Marginal, and Conditional Densities.
3. Law of Total Expectation (The Tower Property) Verification.
4. Law of Total Variance (Eve's Law) Decomposition Verification.
5. Monte Carlo Rejection Sampling from Joint Continuous Density.
"""

import numpy as np
import scipy.integrate as integrate


# =====================================================================
# 1. Part 5 Triangular Joint Distribution & Derived Formulas
# =====================================================================
def joint_pdf(x: float, y: float) -> float:
    """
    f_{X, Y}(x, y) = 8*x*y on 0 <= y <= x <= 1
    """
    if 0.0 <= x <= 1.0 and 0.0 <= y <= x:
        return float(8.0 * x * y)
    return 0.0


def marginal_x_pdf(x: float) -> float:
    """
    f_X(x) = 4*x^3 on [0, 1]
    """
    if 0.0 <= x <= 1.0:
        return float(4.0 * x**3)
    return 0.0


def marginal_y_pdf(y: float) -> float:
    """
    f_Y(y) = 4*y - 4*y^3 on [0, 1]
    """
    if 0.0 <= y <= 1.0:
        return float(4.0 * y - 4.0 * y**3)
    return 0.0


def conditional_y_given_x_pdf(y: float, x: float) -> float:
    """
    f_{Y|X}(y|x) = 2*y / x^2 on 0 <= y <= x
    """
    if 0.0 < x <= 1.0 and 0.0 <= y <= x:
        return float(2.0 * y / (x**2))
    return 0.0


# =====================================================================
# 2. Monte Carlo Rejection Sampling from Joint Distribution
# =====================================================================
def sample_joint(num_samples: int = 300_000):
    """
    Rejection sampling: max f(x, y) on domain is at (1, 1): f(1, 1) = 8.0
    """
    np.random.seed(42)
    samples_x = []
    samples_y = []
    batch_size = num_samples * 3

    while len(samples_x) < num_samples:
        u_x = np.random.uniform(0.0, 1.0, batch_size)
        u_y = np.random.uniform(0.0, 1.0, batch_size)
        # Filter domain: y <= x
        domain_mask = u_y <= u_x
        u_x = u_x[domain_mask]
        u_y = u_y[domain_mask]

        # Rejection threshold
        f_vals = 8.0 * u_x * u_y
        accept_mask = np.random.uniform(0.0, 8.0, len(u_x)) <= f_vals

        samples_x.extend(u_x[accept_mask])
        samples_y.extend(u_y[accept_mask])

    return np.array(samples_x[:num_samples]), np.array(samples_y[:num_samples])


# =====================================================================
# Main Verification Suite
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Chapter 3.5: Joint, Marginal & Conditional Distributions — Tests")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Part 5 "AI by Hand" Exact Numerical Verification
    # -------------------------------------------------------------
    print("\n[Test 1: Part 5 'AI by Hand' Triangular Joint Density Walkthrough]")
    x_test, y_test = 0.80, 0.40

    f_xy = joint_pdf(x_test, y_test)
    f_x = marginal_x_pdf(x_test)
    f_y = marginal_y_pdf(y_test)
    f_y_given_x = conditional_y_given_x_pdf(y_test, x_test)

    print(f"  Evaluation point:          (x = {x_test}, y = {y_test})")
    print(f"  f_XY(0.8, 0.4):            {f_xy:.6f} (Expected: 2.560000)")
    print(f"  f_X(0.8):                  {f_x:.6f} (Expected: 2.048000)")
    print(f"  f_Y(0.4):                  {f_y:.6f} (Expected: 1.344000)")
    print(f"  f_Y|X(0.4 | 0.8):          {f_y_given_x:.6f} (Expected: 1.250000)")
    print(f"  Ratio f_XY / f_X:          {f_xy / f_x:.6f} (Expected: 1.250000)")

    assert np.isclose(f_xy, 2.56)
    assert np.isclose(f_x, 2.048)
    assert np.isclose(f_y, 1.344)
    assert np.isclose(f_y_given_x, 1.25)
    assert np.isclose(f_xy / f_x, 1.25)
    print("  -> Part 5 exact numbers MATCHED perfectly!")

    # -------------------------------------------------------------
    # Test 2: 2D Numerical Integration Normalization Check
    # -------------------------------------------------------------
    print("\n[Test 2: 2D Quadrature Normalization Check]")
    # Integrate y from 0 to x, and x from 0 to 1
    total_vol, _ = integrate.dblquad(
        lambda y, x: 8.0 * x * y, 0.0, 1.0, lambda x: 0.0, lambda x: x
    )
    print(f"  Total Joint Volume:        {total_vol:.8f} (Expected: 1.00000000)")
    assert np.isclose(total_vol, 1.0, atol=1e-10)

    # Marginal x integration: int_0^1 4*x^3 dx = 1.0
    area_x, _ = integrate.quad(marginal_x_pdf, 0.0, 1.0)
    # Marginal y integration: int_0^1 (4y - 4y^3) dy = 1.0
    area_y, _ = integrate.quad(marginal_y_pdf, 0.0, 1.0)
    assert np.isclose(area_x, 1.0)
    assert np.isclose(area_y, 1.0)
    print("  -> Joint, Marginal X, and Marginal Y all integrate strictly to 1.0!")

    # -------------------------------------------------------------
    # Test 3: Law of Total Expectation (The Tower Property)
    # -------------------------------------------------------------
    print("\n[Test 3: The Tower Property E[Y] == E_X[ E[Y|X] ]]")
    # Direct E[Y] from marginal
    mean_y_direct, _ = integrate.quad(lambda y: y * marginal_y_pdf(y), 0.0, 1.0)
    expected_fraction_mean_y = 8.0 / 15.0

    # Iterated E_X[ E[Y|X] ] = E_X[ (2/3) X ]
    mean_y_iterated, _ = integrate.quad(
        lambda x: (2.0 / 3.0 * x) * marginal_x_pdf(x), 0.0, 1.0
    )

    print(f"  Direct E[Y]:               {mean_y_direct:.8f} (Expected: 8/15 = {expected_fraction_mean_y:.8f})")
    print(f"  Iterated E_X[ E[Y|X] ]:    {mean_y_iterated:.8f} (Expected: 8/15 = {expected_fraction_mean_y:.8f})")

    assert np.isclose(mean_y_direct, expected_fraction_mean_y)
    assert np.isclose(mean_y_iterated, expected_fraction_mean_y)
    print("  -> The Tower Property confirmed analytically!")

    # -------------------------------------------------------------
    # Test 4: Law of Total Variance (Eve's Law)
    # -------------------------------------------------------------
    print("\n[Test 4: Eve's Law Var(Y) == E[Var(Y|X)] + Var(E[Y|X])]")
    # Direct Var(Y) = 11 / 225
    expected_var_y = 11.0 / 225.0
    mean_y_sq, _ = integrate.quad(lambda y: (y**2) * marginal_y_pdf(y), 0.0, 1.0)
    var_y_direct = mean_y_sq - mean_y_direct**2

    # Term 1: E_X[ Var(Y|X) ] = E_X[ x^2 / 18 ] = 1 / 27
    term1, _ = integrate.quad(
        lambda x: ((x**2) / 18.0) * marginal_x_pdf(x), 0.0, 1.0
    )
    expected_term1 = 1.0 / 27.0

    # Term 2: Var_X( E[Y|X] ) = Var_X( 2/3 X ) = 8 / 675
    mean_gx, _ = integrate.quad(
        lambda x: (2.0 / 3.0 * x) * marginal_x_pdf(x), 0.0, 1.0
    )
    mean_gx_sq, _ = integrate.quad(
        lambda x: ((2.0 / 3.0 * x) ** 2) * marginal_x_pdf(x), 0.0, 1.0
    )
    term2 = mean_gx_sq - mean_gx**2
    expected_term2 = 8.0 / 675.0

    total_decomp = term1 + term2

    print(f"  Direct Var(Y):             {var_y_direct:.8f} (Expected: 11/225 = {expected_var_y:.8f})")
    print(f"  Term 1 E[Var(Y|X)]:        {term1:.8f} (Expected: 1/27 = {expected_term1:.8f})")
    print(f"  Term 2 Var(E[Y|X]):        {term2:.8f} (Expected: 8/675 = {expected_term2:.8f})")
    print(f"  Term 1 + Term 2:           {total_decomp:.8f}")

    assert np.isclose(var_y_direct, expected_var_y)
    assert np.isclose(term1, expected_term1)
    assert np.isclose(term2, expected_term2)
    assert np.isclose(total_decomp, var_y_direct)
    print("  -> Eve's Law decomposition confirmed mathematically!")

    # -------------------------------------------------------------
    # Test 5: Rejection Sampling Verification
    # -------------------------------------------------------------
    print("\n[Test 5: Monte Carlo Rejection Sampling Verification (300,000 Samples)]")
    samples_x, samples_y = sample_joint(num_samples=300_000)

    emp_mean_x = float(np.mean(samples_x))
    emp_mean_y = float(np.mean(samples_y))
    emp_var_y = float(np.var(samples_y))

    # Analytical E[X] = 4/5 = 0.80
    # Analytical E[Y] = 8/15 = 0.533333
    # Analytical Var(Y) = 11/225 = 0.048889
    print(f"  Empirical Mean X:          {emp_mean_x:.4f} (Expected: 0.8000)")
    print(f"  Empirical Mean Y:          {emp_mean_y:.4f} (Expected: 0.5333)")
    print(f"  Empirical Var Y:           {emp_var_y:.6f} (Expected: 0.048889)")

    assert np.isclose(emp_mean_x, 0.80, atol=0.01)
    assert np.isclose(emp_mean_y, 8.0 / 15.0, atol=0.01)
    assert np.isclose(emp_var_y, 11.0 / 225.0, atol=0.002)
    print("  -> Rejection sampler matches all analytical moments!")

    print("\n" + "=" * 65)
    print(">>> ALL CHAPTER 3.5 MATHEMATICAL & COMPUTATIONAL TESTS PASSED! <<<")
    print("=" * 65)
