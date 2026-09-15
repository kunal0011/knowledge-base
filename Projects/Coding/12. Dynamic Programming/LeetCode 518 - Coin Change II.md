---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 518: Coin Change II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - knapsack
  - unbounded-knapsack
  - combinations
  - amazon
  - google
---

# LeetCode 518: Coin Change II

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Unbounded Knapsack / Counting Combinations / 1D Dynamic Programming  

---

### Problem Statement

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return the **number of combinations** that make up that amount. If that amount of money cannot be made up by any combination of the coins, return `0`.

You may assume that you have an **infinite number** of each kind of coin.

The answer is guaranteed to fit into a signed **32-bit** integer.

---

### Input & Output Formats & Constraints

- **Input:**
  - `amount: int` — Target monetary sum.
  - `coins: List[int]` — Distinct positive coin denominations.
- **Output:**
  - `int` — Total count of unique combinations summing to `amount`.
- **Constraints:**
  - $1 \le \text{coins.length} \le 300$
  - $1 \le \text{coins}[i] \le 5000$
  - All the values of `coins` are **unique**.
  - $0 \le \text{amount} \le 5000$

---

### Key Idea & Intuition

1. **Combinations vs. Permutations (The Loop Order Law):**
   - In **LeetCode 377 (Combination Sum IV)**:
     - `amount` loop is outer, `coins` loop is inner.
     - Sequences like `[1, 2]` and `[2, 1]` are counted as separate distinct ways (permutations).
   - In **LeetCode 518 (Coin Change II)**:
     - `coins` loop is **outer**, `amount` loop is **inner**.
     - Each coin type is processed once in fixed succession. All combinations containing coin $c$ are built by extending earlier solutions that only used coins up to $c$.
     - Consequently, `[1, 2]` is generated, but `[2, 1]` cannot be generated afterwards $\implies$ strictly counts unordered combinations!

2. **Unbounded Knapsack Recurrence:**
   - Let $\text{dp}[a]$ be the number of ways to form amount $a$ using a subset of coin denominations.
   - Base Case: $\text{dp}[0] = 1$ (the empty set forms amount 0 in exactly 1 way).
   - For each coin $c \in \text{coins}$:
     - For amount $a$ from $c$ to $\text{amount}$:
       $$\text{dp}[a] += \text{dp}[a - c]$$
   - Forward iteration ($a = c \dots \text{amount}$) allows coin $c$ to be reused unlimited times.

---

### Solution Approach (Step-by-Step)

1. **Table Allocation:**
   - Create an array `dp` of size `amount + 1` initialized to `0`.
   - Set base case `dp[0] = 1`.
2. **Double Loop:**
   - Outer loop: for each coin $c$ in `coins`:
     - Inner loop: for $a$ from $c$ to `amount`:
       - `dp[a] += dp[a - c]`.
3. **Return:**
   - Return `dp[amount]`.

---

### Visual Algorithm Walkthrough

Suppose `coins = [1, 2, 5]` and `amount = 5`.

```
Initialize: dp = [1, 0, 0, 0, 0, 0]
Amount index:     0  1  2  3  4  5

Coin c = 1:
  Iterate a from 1 to 5:
  dp[1] += dp[0] = 1
  dp[2] += dp[1] = 1
  dp[3] += dp[2] = 1
  dp[4] += dp[3] = 1
  dp[5] += dp[4] = 1
  dp = [1, 1, 1, 1, 1, 1]

Coin c = 2:
  Iterate a from 2 to 5:
  a = 2: dp[2] += dp[0] = 1 + 1 = 2  ([1,1], [2])
  a = 3: dp[3] += dp[1] = 1 + 1 = 2  ([1,1,1], [1,2])
  a = 4: dp[4] += dp[2] = 1 + 2 = 3  ([1,1,1,1], [1,1,2], [2,2])
  a = 5: dp[5] += dp[3] = 1 + 2 = 3  ([1,1,1,1,1], [1,1,1,2], [1,2,2])
  dp = [1, 1, 2, 2, 3, 3]

Coin c = 5:
  Iterate a from 5 to 5:
  a = 5: dp[5] += dp[0] = 3 + 1 = 4  (+ [5])
  dp = [1, 1, 2, 2, 3, 4]

Final Result: dp[5] = 4
Combinations:
1) 5 = 1 + 1 + 1 + 1 + 1
2) 5 = 1 + 1 + 1 + 2
3) 5 = 1 + 2 + 2
4) 5 = 5
```

---

### Solved Examples with Multiple Inputs

| Case | `amount` | `coins` | Stepwise Progression | Result |
|---|---|---|---|---|
| **Standard** | `5` | `[1, 2, 5]` | After 1: `[1,1,1,1,1,1]`, after 2: `[1,1,2,2,3,3]`, after 5: `[1,1,2,2,3,4]` | `4` |
| **No Combination** | `3` | `[2]` | `dp[3] = 0` | `0` |
| **Zero Target** | `0` | `[7]` | Base case `dp[0] = 1` | `1` |
| **Single Exact Coin** | `10` | `[10]` | Only `[10]` forms 10 | `1` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        # Outer loop over coins prevents duplicate permutations
        for coin in coins:
            for a in range(coin, amount + 1):
                dp[a] += dp[a - coin]
                
        return dp[amount]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int change(int amount, std::vector<int>& coins) {
        // Use unsigned int to prevent potential overflow in intermediate states
        std::vector<unsigned int> dp(amount + 1, 0);
        dp[0] = 1;

        for (int coin : coins) {
            for (int a = coin; a <= amount; ++a) {
                dp[a] += dp[a - coin];
            }
        }

        return static_cast<int>(dp[amount]);
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;

        for (int coin : coins) {
            for (int a = coin; a <= amount; a++) {
                dp[a] += dp[a - coin];
            }
        }

        return dp[amount];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\text{amount} \times |\text{coins}|)$  
  Outer loop runs $|\text{coins}|$ times, inner loop runs $\text{amount} - c + 1 \le \text{amount}$ times. With $|\text{coins}| \le 300$ and $\text{amount} \le 5000$, maximum operations $\approx 300 \times 5000 = 1.5 \times 10^6$, finishing in $< 5$ ms.
- **Space Complexity:** $\mathcal{O}(\text{amount})$  
  Only a 1D table of size `amount + 1` is maintained in memory.

---

### Takeaway Pattern & Interview Traps

1. **The Ultimate Knapsack Distinction:**
   - **0/1 Knapsack:** Items outer loop, capacity **backwards** ($W \dots w$).
   - **Unbounded Combinations (LC 518):** Items outer loop, capacity **forwards** ($w \dots W$).
   - **Unbounded Permutations (LC 377):** Capacity **outer** loop ($1 \dots W$), items inner loop.
2. **Edge Case: Amount = 0:**
   - When $\text{amount} = 0$, the return value must be `1` (there is exactly 1 way to make 0: pick no coins).