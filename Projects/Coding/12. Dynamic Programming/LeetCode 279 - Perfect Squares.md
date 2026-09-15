---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 279: Perfect Squares"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - math
  - bfs
  - amazon
  - google
  - microsoft
---

# LeetCode 279: Perfect Squares

**Target Companies:** Amazon, Google, Microsoft, Bloomberg, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Math / BFS  

---

### Problem Statement

Given an integer `n`, return *the least number of perfect square numbers that sum to `n`*.

A **perfect square** is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, `1`, `4`, `9`, and `16` are perfect squares while `3` and `11` are not.

---

### Input & Output Formats & Constraints

- **Input:** An integer `n` ($1 \le n \le 10^4$).
- **Output:** An integer denoting the minimum count of perfect squares summing to `n`.
- **Constraints:**
  - `1 <= n <= 10^4`

---

### Key Idea & Intuition

#### Unbounded Knapsack / Coin Change Equivalence
This problem is isomorphic to **Coin Change (LeetCode 322)**:
- Target amount: $n$.
- Available denominations: All perfect squares $j^2 \le n$ (i.e. $1, 4, 9, 16, \dots$).
- You can reuse each square infinitely many times.
- Goal: Minimize the total count of squares used.

#### Dynamic Programming Formulation
Let $dp[i]$ represent the minimum number of perfect squares needed to sum to $i$:
- **Base Case:** $dp[0] = 0$ (a sum of 0 requires 0 squares).
- **Recurrence Relation:** For every $i \in [1, n]$, we test all possible perfect squares $j^2 \le i$:
  $$dp[i] = 1 + \min_{1 \le j \le \lfloor \sqrt{i} \rfloor} dp[i - j^2]$$
- We initialize $dp[i] = \infty$ for all $i \ge 1$.

---

### Solution Approach (Step-by-Step)

1. **Initialize DP Table:**
   - Create array `dp` of size $n + 1$ with $dp[0] = 0$ and all other entries initialized to $\infty$.
2. **Iterative Computation:**
   - For $i$ from 1 to $n$:
     - $j = 1$
     - While $j \times j \le i$:
       - $dp[i] = \min(dp[i], \, 1 + dp[i - j \times j])$
       - $j += 1$
3. **Return Output:**
   - Return $dp[n]$.

---

### Visual Algorithm Walkthrough

#### Trace for $n = 12$
```
Target: n = 12
Available squares <= 12: [1, 4, 9]

i = 0: dp[0] = 0
i = 1: dp[1] = 1 + dp[0] = 1 (1^2)
i = 2: dp[2] = 1 + dp[1] = 2 (1+1)
i = 3: dp[3] = 1 + dp[2] = 3 (1+1+1)
i = 4: min(1 + dp[3], 1 + dp[0]) = min(4, 1) = 1 (2^2)
i = 5: min(1 + dp[4], 1 + dp[1]) = min(2, 2) = 2 (4+1)
i = 6: min(1 + dp[5], 1 + dp[2]) = min(3, 3) = 3 (4+1+1)
i = 7: min(1 + dp[6], 1 + dp[3]) = min(4, 4) = 4 (4+1+1+1)
i = 8: min(1 + dp[7], 1 + dp[4]) = min(5, 2) = 2 (4+4)
i = 9: min(..., 1 + dp[0]) = 1 (3^2)
i = 10: min(1+dp[9], 1+dp[6], 1+dp[1]) = min(2, 4, 2) = 2 (9+1)
i = 11: min(1+dp[10], 1+dp[7], 1+dp[2]) = min(3, 5, 3) = 3 (9+1+1)
i = 12:
  Try j=1: 1 + dp[11] = 1 + 3 = 4
  Try j=2: 1 + dp[8]  = 1 + 2 = 3  (4 + 4 + 4)
  Try j=3: 1 + dp[3]  = 1 + 3 = 4  (9 + 1 + 1 + 1)
  dp[12] = min(4, 3, 4) = 3

Result: dp[12] = 3 (12 = 4 + 4 + 4).
```

---

### Solved Examples with Multiple Inputs

| $n$ | Perfect Squares Used | Breakdown | Output |
|---|---|---|---|
| `12` | $4, 4, 4$ | $4 + 4 + 4 = 12$ | `3` |
| `13` | $4, 9$ | $4 + 9 = 13$ | `2` |
| `16` | $16$ | $16 = 4^2$ | `1` |
| `7` | $4, 1, 1, 1$ | $4 + 1 + 1 + 1 = 7$ | `4` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp: list[int] = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
                
        return dp[n]
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int numSquares(int n) {
        std::vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j * j <= i; ++j) {
                dp[i] = std::min(dp[i], dp[i - j * j] + 1);
            }
        }

        return dp[n];
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }

        return dp[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \sqrt{n})$. For each integer $i \in [1, n]$, the inner loop evaluates at most $\sqrt{i}$ candidate squares. The total work is $\sum_{i=1}^n \sqrt{i} \approx \int_0^n \sqrt{x} dx = \frac{2}{3} n^{1.5}$. For $n = 10^4$, this is at most $\frac{2}{3} \times 10^6 \approx 6.7 \times 10^5$ operations, completing in $< 15$ ms.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to maintain the 1D DP table.

---

### Takeaway Pattern & Interview Traps

1. **Legendre's Four-Square Theorem:**
   By Lagrange's theorem, every natural number is the sum of at most 4 perfect squares ($ans \le 4$). Furthermore, Legendre showed that $n$ is the sum of 4 squares if and only if $n = 4^a(8b + 7)$. Checking if $n$ is a square ($ans=1$), sum of 2 squares ($ans=2$), or fits $4^a(8b+7)$ ($ans=4$, else $ans=3$) gives an $\mathcal{O}(\sqrt{n})$ math solution!
2. **BFS Alternative:** Because every transition has equal unit weight ($+1$ square), this problem can also be viewed as finding the shortest path in an unweighted graph via Breadth-First Search (BFS), which often terminates even faster than DP.