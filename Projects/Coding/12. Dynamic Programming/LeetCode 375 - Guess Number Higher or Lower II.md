---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 375: Guess Number Higher or Lower II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - interval-dp
  - minimax
  - game-theory
  - google
  - amazon
---

# LeetCode 375: Guess Number Higher or Lower II

**Target Companies:** Google, Amazon, Microsoft, Uber  
**Difficulty:** Medium  
**Topic:** Interval Dynamic Programming / Minimax Game Theory  

---

### Problem Statement

We are playing the Guessing Game. The game will work as follows:

1. I pick a number between $1$ and $n$.
2. You guess a number.
3. If you guess the right number, **you win the game**.
4. If you guess the wrong number, then I will tell you whether the number I picked is higher or lower, and you will continue guessing.
5. Every time you guess a wrong number $x$, you pay $\$x$. If you run out of money, you lose the game.

Given a particular $n$, return the **minimum amount of money you need to guarantee a win regardless of what number I pick**.

---

### Input & Output Formats & Constraints

- **Input:** `n: int` — Upper bound of the number range $[1, n]$.
- **Output:** `int` — Minimum money required to guarantee a win in the worst case.
- **Constraints:**
  - $1 \le n \le 200$

---

### Key Idea & Intuition

1. **Why Standard Binary Search Fails:**
   - In standard binary search, each comparison has unit cost ($1$), so halving the search space minimizes the maximum number of steps.
   - Here, guessing $x$ costs $\$x$. Guessing smaller numbers is cheaper than guessing larger numbers.
   - Example: For range $[1, 2, 3]$:
     - If you guess $1$ (cost $1$): if wrong, number must be in $[2, 3]$. In $[2, 3]$, guessing $2$ costs $2$ (if higher, it is $3$). Total cost = $1 + 2 = 3$.
     - If you guess $2$ (cost $2$): if wrong, it's either $1$ (0 further cost) or $3$ (0 further cost). Total cost = $2 + 0 = 2$.
     - Guessing $2$ guarantees a win with $\$2$.
   - Because costs are non-uniform and state-dependent, we must use **Minimax Dynamic Programming**.

2. **Minimax Formulation (Interval DP):**
   - Let $\text{dp}[l][r]$ be the minimum money required to guarantee a win when the secret number is known to lie in the range $[l, r]$.
   - If $l \ge r$, only one candidate (or none) remains. The secret number is uniquely determined, requiring **zero** additional guesses:
     $$\text{dp}[l][r] = 0 \quad \text{for } l \ge r$$
   - If we pick guess $x \in [l, r]$:
     - We pay $x$ dollars.
     - The adversary forces us into the worse sub-problem:
       $$\max(\text{dp}[l][x - 1], \text{dp}[x + 1][r])$$
     - The worst-case cost for guessing $x$ is:
       $$\text{cost}(x) = x + \max(\text{dp}[l][x - 1], \text{dp}[x + 1][r])$$
   - We choose the guess $x$ that minimizes this worst-case cost:
     $$\text{dp}[l][r] = \min_{l \le x \le r} \left( x + \max(\text{dp}[l][x - 1], \text{dp}[x + 1][r]) \right)$$

---

### Solution Approach (Step-by-Step)

1. **Table Allocation:**
   - Create a 2D table `dp` of size $(n + 2) \times (n + 2)$ filled with $0$. (The $+2$ handles out-of-bounds boundary lookups $x - 1$ and $x + 1$ gracefully without index errors).
2. **Iterate by Interval Length:**
   - Iterate interval length $\text{length}$ from $2$ to $n$:
     - Iterate left boundary $l$ from $1$ to $n - \text{length} + 1$:
       - Right boundary $r = l + \text{length} - 1$.
       - Initialize $\text{dp}[l][r] = \infty$.
       - For every candidate guess $x \in [l, r]$:
         - $\text{cost} = x + \max(\text{dp}[l][x - 1], \text{dp}[x + 1][r])$.
         - $\text{dp}[l][r] = \min(\text{dp}[l][r], \text{cost})$.
3. **Return:**
   - Return $\text{dp}[1][n]$.

---

### Visual Algorithm Walkthrough

For $n = 4$:

```
Base Cases: All dp[i][i] = 0

Length = 2:
  dp[1][2]: Guess 1 -> 1 + dp[2][2] = 1; Guess 2 -> 2 + dp[1][1] = 2 => min = 1
  dp[2][3]: Guess 2 -> 2 + dp[3][3] = 2; Guess 3 -> 3 + dp[2][2] = 3 => min = 2
  dp[3][4]: Guess 3 -> 3 + dp[4][4] = 3; Guess 4 -> 4 + dp[3][3] = 4 => min = 3

Length = 3:
  dp[1][3]:
    Guess 1 -> 1 + dp[2][3] = 1 + 2 = 3
    Guess 2 -> 2 + max(dp[1][1], dp[3][3]) = 2 + 0 = 2
    Guess 3 -> 3 + dp[1][2] = 3 + 1 = 4
    min cost = 2 (guess 2)
  dp[2][4]:
    Guess 2 -> 2 + dp[3][4] = 2 + 3 = 5
    Guess 3 -> 3 + max(dp[2][2], dp[4][4]) = 3 + 0 = 3
    Guess 4 -> 4 + dp[2][3] = 4 + 2 = 6
    min cost = 3 (guess 3)

Length = 4 (dp[1][4]):
  Guess 1 -> 1 + dp[2][4] = 1 + 3 = 4
  Guess 2 -> 2 + max(dp[1][1], dp[3][4]) = 2 + 3 = 5
  Guess 3 -> 3 + max(dp[1][2], dp[4][4]) = 3 + max(1, 0) = 4
  Guess 4 -> 4 + dp[1][3] = 4 + 2 = 6
  min cost = min(4, 5, 4, 6) = 4

Result: dp[1][4] = 4
```

---

### Solved Examples with Multiple Inputs

| Case | `n` | Intermediate Values | Result | Explanation |
|---|---|---|---|---|
| **Trivial** | `1` | Range $[1, 1] \implies 0$ cost | `0` | Number is definitely 1 |
| **Two Numbers** | `2` | Range $[1, 2] \implies \text{guess } 1$ | `1` | Guess 1: if right 0 cost; if wrong, secret is 2 for cost 1 |
| **Three Numbers** | `3` | Guess 2 | `2` | Guess 2: wrong higher -> 3, wrong lower -> 1 |
| **Standard 4** | `4` | Guess 3 -> if lower, guess 1 (cost 3 + 1 = 4) | `4` | Minimax optimal play |
| **Small Range** | `10` | Full interval DP computation | `16` | Worst-case guaranteed threshold |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # dp[l][r] represents min cost for range [l, r]
        # (n + 2) x (n + 2) table handles boundary lookups seamlessly
        dp = [[0] * (n + 2) for _ in range(n + 2)]
        
        for length in range(2, n + 1):
            for l in range(1, n - length + 2):
                r = l + length - 1
                min_cost = float('inf')
                
                # Try all possible guesses x in [l, r]
                for x in range(l, r + 1):
                    cost = x + max(dp[l][x - 1], dp[x + 1][r])
                    if cost < min_cost:
                        min_cost = cost
                        
                dp[l][r] = min_cost
                
        return dp[1][n]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int getMoneyAmount(int n) {
        std::vector<std::vector<int>> dp(n + 2, std::vector<int>(n + 2, 0));

        for (int len = 2; len <= n; ++len) {
            for (int l = 1; l <= n - len + 1; ++l) {
                int r = l + len - 1;
                int min_cost = INT_MAX;

                for (int x = l; x <= r; ++x) {
                    int cost = x + std::max(dp[l][x - 1], dp[x + 1][r]);
                    min_cost = std::min(min_cost, cost);
                }

                dp[l][r] = min_cost;
            }
        }

        return dp[1][n];
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int getMoneyAmount(int n) {
        int[][] dp = new int[n + 2][n + 2];

        for (int len = 2; len <= n; len++) {
            for (int l = 1; l <= n - len + 1; l++) {
                int r = l + len - 1;
                int minCost = Integer.MAX_VALUE;

                for (int x = l; x <= r; x++) {
                    int cost = x + Math.max(dp[l][x - 1], dp[x + 1][r]);
                    minCost = Math.min(minCost, cost);
                }

                dp[l][r] = minCost;
            }
        }

        return dp[1][n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^3)$  
  There are $\mathcal{O}(n^2)$ sub-intervals $[l, r]$. For each interval, we iterate over all possible choices $x \in [l, r]$ ($\mathcal{O}(n)$ options). For $n \le 200$, operations $\approx \frac{200^3}{6} \approx 1.3 \times 10^6$, which runs in under $10$ ms.
- **Space Complexity:** $\mathcal{O}(n^2)$  
  A 2D array of size $(n + 2) \times (n + 2)$ stores subproblem solutions.

---

### Takeaway Pattern & Interview Traps

1. **Interval DP by Increasing Length:**
   - Always iterate intervals in increasing order of length (from $2$ to $n$) so that shorter ranges $[l, x - 1]$ and $[x + 1, r]$ are guaranteed to be evaluated before $[l, r]$.
2. **Sentinel Off-By-One Cushion:**
   - Using an $(n + 2) \times (n + 2)$ array avoids checking if $x - 1 < l$ or $x + 1 > r$. For $x = l$, $\text{dp}[l][l - 1] = 0$; for $x = r$, $\text{dp}[r + 1][r] = 0$.
3. **The "Minimize Max" Mindset:**
   - In game theory / competitive environments, inner $\max$ models the adversarial secret picker who always directs you toward the most expensive branch, while outer $\min$ models your optimal decision.