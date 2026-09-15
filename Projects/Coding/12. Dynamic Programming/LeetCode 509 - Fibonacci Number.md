---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 509: Fibonacci Number"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - math
  - memoization
  - amazon
  - apple
  - google
---

# LeetCode 509: Fibonacci Number

**Target Companies:** Amazon, Apple, Google, Microsoft, Meta  
**Difficulty:** Easy  
**Topic:** Dynamic Programming / Space Optimization / Recurrence Relations  

---

### Problem Statement

The **Fibonacci numbers**, commonly denoted `F(n)` form a sequence, called the **Fibonacci sequence**, such that each number is the sum of the two preceding ones, starting from `0` and `1`. That is:

- $F(0) = 0, \ F(1) = 1$
- $F(n) = F(n - 1) + F(n - 2), \quad \text{for } n > 1$

Given `n`, calculate `F(n)`.

---

### Input & Output Formats & Constraints

- **Input:** `n: int` — Non-negative integer index.
- **Output:** `int` — Value of $F(n)$.
- **Constraints:**
  - $0 \le n \le 30$

---

### Key Idea & Intuition

1. **The Canonical DP Subproblem:**
   - Naive recursion $F(n) = F(n-1) + F(n-2)$ branches into an exponential call tree of complexity $\mathcal{O}(2^n)$, re-evaluating overlapping subproblems repeatedly (e.g. $F(2)$ is computed independently many times).
   - By storing subproblem answers, we reduce runtime to linear $\mathcal{O}(n)$.

2. **State Reduction ($\mathcal{O}(1)$ Space):**
   - In bottom-up DP, to calculate $\text{dp}[i]$, we only ever need the two immediately preceding values: $\text{dp}[i - 1]$ and $\text{dp}[i - 2]$.
   - Maintaining a full table of size $n + 1$ is redundant. We can keep two rolling variables `prev2` and `prev1`, updating them in $\mathcal{O}(1)$ space.

3. **Logarithmic Time Alternative (Matrix Exponentiation):**
   - For massive $n$ ($n \le 10^9$):
     $$\begin{pmatrix} F(n+1) & F(n) \\ F(n) & F(n-1) \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n$$
   - Using binary exponentiation, this yields an $\mathcal{O}(\log n)$ solution.

---

### Solution Approach (Step-by-Step)

1. **Base Cases:**
   - If $n \le 1$, return $n$.
2. **Rolling State Updates:**
   - Initialize `prev2 = 0` ($F(0)$) and `prev1 = 1` ($F(1)$).
   - For $i$ from $2$ to $n$:
     - `curr = prev1 + prev2`
     - `prev2 = prev1`
     - `prev1 = curr`
3. **Return:**
   - Return `prev1`.

---

### Visual Algorithm Walkthrough

For $n = 6$:

```
Initial: prev2 = 0, prev1 = 1

i = 2:
  curr = prev1 + prev2 = 1 + 0 = 1
  prev2 = 1, prev1 = 1

i = 3:
  curr = prev1 + prev2 = 1 + 1 = 2
  prev2 = 1, prev1 = 2

i = 4:
  curr = prev1 + prev2 = 2 + 1 = 3
  prev2 = 2, prev1 = 3

i = 5:
  curr = prev1 + prev2 = 3 + 2 = 5
  prev2 = 3, prev1 = 5

i = 6:
  curr = prev1 + prev2 = 5 + 3 = 8
  prev2 = 5, prev1 = 8

Final Result: F(6) = 8
```

---

### Solved Examples with Multiple Inputs

| Case | `n` | Fibonacci Progression | Result | Explanation |
|---|---|---|---|---|
| **Base 0** | `0` | $F(0)$ | `0` | Standard definition |
| **Base 1** | `1` | $F(1)$ | `1` | Standard definition |
| **Small Value** | `2` | $F(0) + F(1) = 0 + 1$ | `1` | $F(2) = 1$ |
| **Standard 4** | `4` | $0, 1, 1, 2, 3$ | `3` | $F(4) = 3$ |
| **Upper Bound 30** | `30` | Linear progression | `832040` | Maximum test constraint |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — $\mathcal{O}(1)$ Space)
```python
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
            
        prev2, prev1 = 0, 1
        for _ in range(2, n + 1):
            curr = prev1 + prev2
            prev2 = prev1
            prev1 = curr
            
        return prev1
```

#### 2. C++ (C++17 / STL — $\mathcal{O}(1)$ Space)
```cpp
class Solution {
public:
    int fib(int n) {
        if (n <= 1) return n;

        int prev2 = 0;
        int prev1 = 1;

        for (int i = 2; i <= n; ++i) {
            int curr = prev1 + prev2;
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
};
```

#### 3. Java (Modern, Typed — $\mathcal{O}(1)$ Space)
```java
class Solution {
    public int fib(int n) {
        if (n <= 1) return n;

        int prev2 = 0;
        int prev1 = 1;

        for (int i = 2; i <= n; i++) {
            int curr = prev1 + prev2;
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$  
  The loop runs exactly $n - 1$ times for $n \ge 2$, with $\mathcal{O}(1)$ operations per step.
- **Space Complexity:** $\mathcal{O}(1)$  
  Only two integer variables are kept in memory.

---

### Takeaway Pattern & Interview Traps

1. **The Recursion Pitfall:**
   - Writing `return fib(n-1) + fib(n-2)` without memoization triggers catastrophic exponential complexity $\mathcal{O}(2^n)$.
2. **The Golden Ratio Formula (Binet's Formula):**
   - $F(n) = \frac{1}{\sqrt{5}} \left( \left(\frac{1 + \sqrt{5}}{2}\right)^n - \left(\frac{1 - \sqrt{5}}{2}\right)^n \right)$.
   - While mathematically elegant, using floating-point operations introduces IEEE 754 precision rounding errors for larger $n$. The integer rolling variable method is robust and accurate.