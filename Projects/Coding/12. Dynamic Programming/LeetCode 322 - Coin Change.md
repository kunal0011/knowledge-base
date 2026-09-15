---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 322: Coin Change"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - unbounded-knapsack
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 322: Coin Change

**Target Companies:** Amazon (Top DP problem), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Unbounded Knapsack / 1D Bottom-Up Dynamic Programming  

---

### Problem Statement

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return the **fewest number of coins** that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return `-1`.

You may assume that you have an **infinite number** of each kind of coin.

---

### Input & Output Formats & Constraints

- **Input:**
  - `coins: List[int]` — Array of distinct positive coin denominations.
  - `amount: int` — Target total amount of money.
- **Output:**
  - `int` — Minimum number of coins required to sum up to `amount`, or `-1` if impossible.
- **Constraints:**
  - $1 \le \text{coins.length} \le 12$
  - $1 \le \text{coins}[i] \le 2^{31} - 1$
  - $0 \le \text{amount} \le 10^4$

---

### Key Idea & Intuition

1. **Why Greedy Fails:**
   - A common instinct is to greedily pick the largest coin possible. However, consider `coins = [1, 3, 4, 5]` and `amount = 7`.
   - Greedy choice: $5 + 1 + 1 = 7$ (3 coins).
   - Optimal choice: $4 + 3 = 7$ (2 coins).
   - Because arbitrary coin denominations lack the "canonical coin system" property (like US coins), greedy choices lead to suboptimal solutions. We must explore combinations systematically using **Dynamic Programming**.

2. **Optimal Substructure (Unbounded Knapsack):**
   - Let $\text{dp}[a]$ be the minimum number of coins needed to make amount $a$.
   - To make amount $a$, if we use a coin $c \in \text{coins}$ (where $c \le a$), we need $1 + \text{dp}[a - c]$ coins.
   - We seek the minimum across all possible coin choices:
     $$\text{dp}[a] = \min_{c \in \text{coins}, c \le a} (1 + \text{dp}[a - c])$$

3. **Base Case & Sentinels:**
   - $\text{dp}[0] = 0$: 0 coins are needed to make an amount of 0.
   - Initialize $\text{dp}[a] = \infty$ (or $\text{amount} + 1$) for all $a \ge 1$. Since the smallest coin value is $\ge 1$, no valid combination can use more than $\text{amount}$ coins. Thus, $\text{amount} + 1$ serves as an unreachable sentinel $\infty$ that avoids 32-bit integer overflow when adding 1.

---

### Solution Approach (Step-by-Step)

1. **Allocate DP Array:**
   - Create an array `dp` of size `amount + 1` filled with `amount + 1`.
   - Set `dp[0] = 0`.
2. **Iterate Bottom-Up:**
   - For each sub-target $a$ from $1$ up to `amount`:
     - For each coin $c$ in `coins`:
       - If $a - c \ge 0$:
         - `dp[a] = min(dp[a], 1 + dp[a - c])`.
3. **Return Result:**
   - If `dp[amount] > amount`, it means the amount could not be formed by any coin combination; return `-1`.
   - Otherwise, return `dp[amount]`.

---

### Visual Algorithm Walkthrough

Suppose `coins = [1, 2, 5]` and `amount = 11`. Sentinel $\infty = 12$.

```
Index:    0   1   2   3   4   5   6   7   8   9  10  11
Init:    [0, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

a = 1:
  c = 1: dp[1] = min(12, 1 + dp[0]) = 1
  c = 2: 1 < 2, skip
  c = 5: 1 < 5, skip
  dp = [0, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]

a = 2:
  c = 1: dp[2] = min(12, 1 + dp[1]) = 2
  c = 2: dp[2] = min(2,  1 + dp[0]) = 1
  dp = [0, 1, 1, 12, 12, 12, 12, 12, 12, 12, 12, 12]

a = 3:
  c = 1: dp[3] = min(12, 1 + dp[2]) = 2
  c = 2: dp[3] = min(2,  1 + dp[1]) = 2
  dp = [0, 1, 1, 2, 12, 12, 12, 12, 12, 12, 12, 12]

...
a = 5:
  c = 1: dp[5] = min(..., 1 + dp[4]) = 3
  c = 2: dp[5] = min(3,  1 + dp[3]) = 3
  c = 5: dp[5] = min(3,  1 + dp[0]) = 1
  dp[5] = 1

...
a = 10:
  c = 5: dp[10] = 1 + dp[5] = 2

a = 11:
  c = 1: dp[11] = 1 + dp[10] = 3 (coins: 5 + 5 + 1)
  c = 2: dp[11] = 1 + dp[9]  = 1 + 3 = 4
  c = 5: dp[11] = 1 + dp[6]  = 1 + 2 = 3
  dp[11] = 3

Final Result: dp[11] = 3  (Coins: 5 + 5 + 1)
```

---

### Solved Examples with Multiple Inputs

| Case | `coins` | `amount` | Intermediate Transitions | Result | Explanation |
|---|---|---|---|---|---|
| **Standard** | `[1, 2, 5]` | `11` | `dp[10]=2` (5+5), `dp[11] = 1 + dp[10] = 3` | `3` | $5 + 5 + 1 = 11$ |
| **Impossible** | `[2]` | `3` | `dp[0]=0, dp[1]=4, dp[2]=1, dp[3]=4` | `-1` | Odd amount cannot be formed with even coin |
| **Zero Amount** | `[1]` | `0` | Base case `dp[0] = 0` | `0` | 0 coins needed for 0 amount |
| **Greedy Trap** | `[1, 3, 4, 5]` | `7` | `dp[7] = 1 + dp[7 - 4] = 1 + 1 = 2` (4+3) | `2` | Greedy gives $5+1+1$ (3), DP gives $4+3$ (2) |
| **Exact Large Coin** | `[1, 7, 10]` | `14` | `dp[14] = 1 + dp[7] = 2` ($7+7$) | `2` | Greedy takes $10+1+1+1+1$ (5 coins) |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Sentinel amount + 1 represents infinity without overflow risk
        inf = amount + 1
        dp = [inf] * (amount + 1)
        dp[0] = 0
        
        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - c])
                    
        return dp[amount] if dp[amount] != inf else -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int coinChange(std::vector<int>& coins, int amount) {
        // Sentinel amount + 1 avoids integer overflow when adding 1
        const int INF = amount + 1;
        std::vector<int> dp(amount + 1, INF);
        dp[0] = 0;

        for (int a = 1; a <= amount; ++a) {
            for (int c : coins) {
                if (a >= c) {
                    dp[a] = std::min(dp[a], 1 + dp[a - c]);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int coinChange(int[] coins, int amount) {
        int inf = amount + 1;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, inf);
        dp[0] = 0;

        for (int a = 1; a <= amount; a++) {
            for (int c : coins) {
                if (a >= c) {
                    dp[a] = Math.min(dp[a], 1 + dp[a - c]);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\text{amount} \times |\text{coins}|)$  
  The outer loop runs $\text{amount}$ iterations, and the inner loop checks each of the $|\text{coins}|$ denominations. With $\text{amount} \le 10^4$ and $|\text{coins}| \le 12$, maximum operations $\approx 1.2 \times 10^5$, executing in a few milliseconds.
- **Space Complexity:** $\mathcal{O}(\text{amount})$  
  A 1D table of size `amount + 1` is maintained.

---

### Takeaway Pattern & Interview Traps

1. **Unbounded Knapsack vs. 0/1 Knapsack:**
   - In 0/1 knapsack, each item can be picked at most once; iterating items outer and amount backwards prevents reuse.
   - In unbounded knapsack (Coin Change), each item can be reused infinitely; iterating amounts forwards allows repeated usage of the same coin denomination.
2. **Sentinel Initialization Trap (`INT_MAX`):**
   - If initializing with `INT_MAX` in C++ or Java, `1 + dp[a - c]` will cause 32-bit signed integer overflow into negative numbers (`INT_MIN`), resulting in wrong `min()` calculations. Always initialize with `amount + 1` or check for `dp[a - c] != INT_MAX`.
3. **Coin Ordering:**
   - Pre-sorting `coins` in ascending order allows an early `break` when $c > a$, speeding up runtime in practice.
