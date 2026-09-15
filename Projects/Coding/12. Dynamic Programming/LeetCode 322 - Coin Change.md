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
  - amazon
  - google
---

# LeetCode 322: Coin Change

**Target Companies:** Amazon (Top #1 DP), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Unbounded Knapsack / 1D Bottom-Up DP

---

### Problem Statement

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return the **fewest number of coins** that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return `-1`.

You may assume that you have an **infinite number** of each kind of coin.

---

### Input & Output Formats & Constraints

- **Input:** `coins: List[int]`, `amount: int`
- **Output:** `int` (minimum coins or -1)
- **Constraints:**
  - $1 \le \text{coins.length} \le 12$
  - $1 \le \text{coins}[i] \le 2^{31} - 1$
  - $0 \le \text{amount} \le 10^4$

---

### Key Idea & Intuition

- **Optimal Substructure:**
  - To make `amount` $A$, if we pick coin $c$, the remaining amount is $A - c$.
  - The minimum coins needed is:
    $$\text{DP}[A] = 1 + \min_{c \in \text{coins}} \text{DP}[A - c]$$
- **Base Case & Initialization:**
  - $\text{DP}[0] = 0$ (0 coins needed to make amount 0).
  - All other amounts initialized to $\infty$ (`amount + 1`).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    dp[a] = min(dp[a], 1 + dp[a - c])
                    
        return dp[amount] if dp[amount] != float('inf') else -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int coinChange(std::vector<int>& coins, int amount) {
        std::vector<int> dp(amount + 1, amount + 1);
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
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
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

- **Time Complexity:** $O(\text{amount} \times |\text{coins}|)$ — Outer loop runs `amount` times, inner loop checks each denomination.
- **Space Complexity:** $O(\text{amount})$ for the 1D DP table.
