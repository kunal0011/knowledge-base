---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 679: 24 Game"
tags:
  - leetcode
  - coding
  - backtracking
  - math
  - array
  - amazon
  - google
---

# LeetCode 679: 24 Game

**Target Companies:** Google (Signature Question), Meta, Amazon, Microsoft  
**Difficulty:** Hard  
**Topic:** Backtracking / State-Reduction Search / Floating-Point Arithmetic  

---

### Problem Statement

You are given an integer array `cards` of length 4. You have four cards, each containing a number from 1 to 9. You need to arrange these cards using the operators `['+', '-', '*', '/']` and parentheses `'('` and `')'` to get the value `24`.

You are restricted with the following rules:
- The division operator `'/'` represents real division, not integer division.
  - For example, $4 / (1 - 2/3) = 4 / (1/3) = 12$.
- Every operation done is between two numbers. In particular, we cannot use `'-'` as a unary operator.
  - For example, if `cards = [1, 1, 1, 1]`, the expression `"-1 - 1 - 1 - 1"` is not allowed.
- You cannot concatenate numbers together.
  - For example, if `cards = [1, 2, 1, 2]`, we cannot use the numbers to make `12 + 12`.

Return `true` if you can get such an expression that evaluates to `24`, or `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:** `cards: List[int]` of length 4
- **Output:** `bool` (`True` if 24 can be formed, `False` otherwise)
- **Constraints:**
  - `cards.length == 4`
  - $1 \le \text{cards}[i] \le 9$

---

### Key Idea & Intuition

- **State-Reduction Backtracking ($4 \to 3 \to 2 \to 1$):**
  - Any valid mathematical expression with binary operators and parentheses is equivalent to repeatedly picking **any two available numbers**, applying an operator, and replacing those two numbers with the result.
  - Thus, at each step of recursion:
    - We have a list of numbers (initially length 4).
    - Pick any two distinct numbers $a$ and $b$ (at indices $i < j$).
    - Try all possible results:
      1. $a + b$
      2. $a - b$
      3. $b - a$
      4. $a \times b$
      5. $a / b$ (if $|b| > 10^{-6}$)
      6. $b / a$ (if $|a| > 10^{-6}$)
    - Create a new list containing the unpicked numbers plus the result, and recurse.
  - When the list has only 1 number left, check if $|num - 24| < 10^{-5}$.
- **Floating-Point Precision:**
  - Standard integer division will fail on fractions. For example, cards `[3, 3, 8, 8]` can make 24 via:
    $$8 / (3 - 8/3) = 8 / (1/3) = 24$$
  - All numbers must be converted to `float` / `double`, and equality with 24 must use an epsilon threshold ($\epsilon = 10^{-5}$).
- **Total Search Space Size:**
  - Step $4 \to 3$: $\binom{4}{2} = 6$ pairs $\times 6$ operations $= 36$ branches.
  - Step $3 \to 2$: $\binom{3}{2} = 3$ pairs $\times 6$ operations $= 18$ branches.
  - Step $2 \to 1$: $\binom{2}{2} = 1$ pair $\times 6$ operations $= 6$ branches.
  - Total states $\le 36 \times 18 \times 6 = 3,888$ combinations! Exhaustive search is extremely fast ($\approx 1 \text{ ms}$).

---

### Solution Approach (Step-by-Step)

1. Convert `cards` to a list of floating-point numbers `nums = [float(x) for x in cards]`.
2. Define recursive helper `solve(nums)`:
   - If `len(nums) == 1`:
     - Return `abs(nums[0] - 24.0) < 1e-5`.
   - Iterate over all pairs $(i, j)$ with $0 \le i < j < \text{len}(nums)$:
     - Extract $a = \text{nums}[i]$ and $b = \text{nums}[j]$.
     - Create `remaining` containing all elements from `nums` except indices $i$ and $j$.
     - Candidate results:
       - $a + b$
       - $a - b$
       - $b - a$
       - $a \times b$
       - If $|b| > 10^{-6}$: $a / b$
       - If $|a| > 10^{-6}$: $b / a$
     - For each candidate:
       - If `solve(remaining + [candidate])` is `True`:
         - Return `True` (early exit).
   - Return `False`.
3. Return `solve(nums)`.

---

### Visual Algorithm Walkthrough

For `cards = [3, 3, 8, 8]`:

```
Level 1: nums = [3.0, 3.0, 8.0, 8.0]
Pick a = 8.0, b = 3.0.
Operation: a / b = 8.0 / 3.0 = 2.666666...
New list -> Level 2: [3.0, 8.0, 2.666666...]

Pick a = 3.0, b = 2.666666...
Operation: a - b = 3.0 - 2.666666... = 0.333333... (1/3)
New list -> Level 3: [8.0, 0.333333...]

Pick a = 8.0, b = 0.333333...
Operation: a / b = 8.0 / (1/3) = 24.0
New list -> Level 4: [24.0]

Base case: len == 1, abs(24.0 - 24.0) < 1e-5 -> TRUE!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `cards` | Winning Formula | Output |
| :--- | :--- | :--- | :--- |
| **Standard** | `[4, 1, 8, 7]` | $(8 - 4) \times (7 - 1) = 4 \times 6 = 24$ | `true` |
| **Fraction Division** | `[3, 3, 8, 8]` | $8 / (3 - 8/3) = 24$ | `true` |
| **Impossible** | `[1, 2, 1, 2]` | Max product is small, cannot reach 24 | `false` |
| **All Identical** | `[1, 1, 1, 1]` | Max is 4, min is 0 | `false` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        """
        Determines whether 24 can be reached using cards and operations +, -, *, /.
        Uses state-reduction DFS with floating-point tolerance.
        """
        EPSILON = 1e-5

        def solve(nums: List[float]) -> bool:
            if len(nums) == 1:
                return abs(nums[0] - 24.0) < EPSILON

            n = len(nums)
            for i in range(n):
                for j in range(i + 1, n):
                    # Remaining numbers not chosen
                    remaining = [nums[k] for k in range(n) if k != i and k != j]
                    a, b = nums[i], nums[j]

                    # All possible operations between a and b
                    candidates = [a + b, a - b, b - a, a * b]
                    if abs(b) > 1e-6:
                        candidates.append(a / b)
                    if abs(a) > 1e-6:
                        candidates.append(b / a)

                    for cand in candidates:
                        if solve(remaining + [cand]):
                            return True

            return False

        return solve([float(c) for c in cards])
```

#### C++17
```cpp
#include <vector>
#include <cmath>

class Solution {
public:
    bool judgePoint24(const std::vector<int>& cards) {
        std::vector<double> nums;
        for (int c : cards) {
            nums.push_back(static_cast<double>(c));
        }
        return solve(nums);
    }

private:
    const double EPS = 1e-5;

    bool solve(std::vector<double>& nums) {
        if (nums.size() == 1) {
            return std::abs(nums[0] - 24.0) < EPS;
        }

        int n = static_cast<int>(nums.size());
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                std::vector<double> next_nums;
                for (int k = 0; k < n; ++k) {
                    if (k != i && k != j) {
                        next_nums.push_back(nums[k]);
                    }
                }

                double a = nums[i];
                double b = nums[j];

                std::vector<double> candidates = {a + b, a - b, b - a, a * b};
                if (std::abs(b) > 1e-6) candidates.push_back(a / b);
                if (std::abs(a) > 1e-6) candidates.push_back(b / a);

                for (double cand : candidates) {
                    next_nums.push_back(cand);
                    if (solve(next_nums)) return true;
                    next_nums.pop_back(); // Backtrack
                }
            }
        }

        return false;
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    private static final double EPS = 1e-5;

    public boolean judgePoint24(int[] cards) {
        List<Double> nums = new ArrayList<>();
        for (int c : cards) {
            nums.add((double) c);
        }
        return solve(nums);
    }

    private boolean solve(List<Double> nums) {
        if (nums.size() == 1) {
            return Math.abs(nums.get(0) - 24.0) < EPS;
        }

        int n = nums.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                List<Double> nextNums = new ArrayList<>();
                for (int k = 0; k < n; k++) {
                    if (k != i && k != j) {
                        nextNums.add(nums.get(k));
                    }
                }

                double a = nums.get(i);
                double b = nums.get(j);

                List<Double> candidates = new ArrayList<>();
                candidates.add(a + b);
                candidates.add(a - b);
                candidates.add(b - a);
                candidates.add(a * b);
                if (Math.abs(b) > 1e-6) candidates.add(a / b);
                if (Math.abs(a) > 1e-6) candidates.add(b / a);

                for (double cand : candidates) {
                    nextNums.add(cand);
                    if (solve(nextNums)) return true;
                    nextNums.remove(nextNums.size() - 1); // Backtrack
                }
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(1)$ bounded constant time.
  - Number of pair selections across 3 levels: $\binom{4}{2} \times \binom{3}{2} \times \binom{2}{2} = 6 \times 3 \times 1 = 18$.
  - Number of operator assignments: at most 6 per step ($6^3 = 216$).
  - Total leaves evaluated $\le 18 \times 216 = 3,888 \ll 10^5$, executing in $< 3 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since recursion depth is fixed at 3 stack frames with arrays of length $\le 4$.

---

### Takeaway Pattern & Interview Traps

- **Non-Commutative Operators:** Addition and multiplication are commutative ($a + b = b + a$), but subtraction and division are not! You must explicitly check both $a - b$ and $b - a$, and both $a / b$ and $b / a$.
- **Floating-Point Comparison:** Never write `nums[0] == 24.0`. Divisions such as $8 / (1/3)$ introduce slight IEEE 754 precision loss. Always use `abs(nums[0] - 24.0) < 1e-5`.
- **Division by Zero:** Protect against division by zero by verifying `abs(divisor) > 1e-6` before dividing.