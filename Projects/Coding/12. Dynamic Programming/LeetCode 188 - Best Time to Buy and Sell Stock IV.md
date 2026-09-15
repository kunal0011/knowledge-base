---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 188: Best Time to Buy and Sell Stock IV"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 188: Best Time to Buy and Sell Stock IV

**Target Companies:** Amazon, Google, Meta, Bloomberg, Microsoft  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Array  

---

### Problem Statement

You are given an integer array `prices` where `prices[i]` is the price of a given stock on the $i^{\text{th}}$ day, and an integer `k`.

Find the maximum profit you can achieve. You may complete at most `k` transactions: i.e. you may buy at most `k` times and sell at most `k` times.

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

---

### Input & Output Formats & Constraints

- **Input:**
  - An integer `k` ($1 \le k \le 100$).
  - An integer array `prices` ($1 \le |prices| \le 1000$).
- **Output:** An integer representing the maximum profit achievable with at most `k` transactions.
- **Constraints:**
  - `1 <= k <= 100`
  - `1 <= prices.length <= 1000`
  - `0 <= prices[i] <= 1000`

---

### Key Idea & Intuition

#### 1. Optimization for $k \ge n / 2$
A complete transaction requires at least 2 days (buy on one day, sell on another).
Therefore, in an array of length $n$, the maximum number of non-overlapping transactions possible is $\lfloor n / 2 \rfloor$.
If $k \ge n / 2$:
- The transaction limit is non-binding; this reduces directly to **LeetCode 122 (Best Time to Buy and Sell Stock II)** with unlimited transactions!
- We simply harvest every positive price difference in $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ space:
  $$\text{Profit} = \sum_{i=1}^{n-1} \max(0, prices[i] - prices[i - 1])$$

#### 2. Generalizing the State Machine to $k$ Transactions
When $k < n / 2$, we generalize the 4-state machine from LeetCode 123 into two arrays of length $k + 1$:
- `buy[t]`: Maximum cash balance after executing the $t^{\text{th}}$ **buy** ($1 \le t \le k$).
- `sell[t]`: Maximum cash balance after executing the $t^{\text{th}}$ **sell** ($1 \le t \le k$).

**Transitions for each price $p$:**
For transaction index $t$ from $1$ to $k$:
1. Buying the $t^{\text{th}}$ stock consumes $p$ dollars from our earnings after the $(t - 1)^{\text{th}}$ sell:
   $$buy[t] = \max(buy[t], \, sell[t - 1] - p)$$
2. Selling the $t^{\text{th}}$ stock yields $p$ dollars added to our position after the $t^{\text{th}}$ buy:
   $$sell[t] = \max(sell[t], \, buy[t] + p)$$

#### Initial Base Conditions
- For all $t \in [1, k]$: $buy[t] = -prices[0]$ and $sell[t] = 0$.
- $sell[0] = 0$ (0 transactions yield 0 profit).

This yields a space complexity of strictly $\mathcal{O}(k)$ and time complexity of $\mathcal{O}(n \cdot k)$.

---

### Solution Approach (Step-by-Step)

1. **Boundary & Unlimited Shortcuts:**
   - If $n \le 1$ or $k == 0$, return 0.
   - If $k \ge n // 2$, compute sum of positive differences and return.
2. **Initialize State Arrays:**
   - Create `buy = [-prices[0]] * (k + 1)`
   - Create `sell = [0] * (k + 1)`
3. **Iterate Across Prices and Transactions:**
   - For price $p$ in `prices[1:]`:
     - For $t$ from 1 to $k$:
       - $buy[t] = \max(buy[t], sell[t - 1] - p)$
       - $sell[t] = \max(sell[t], buy[t] + p)$
4. **Return:**
   - Return `sell[k]`.

---

### Visual Algorithm Walkthrough

#### Trace for `k = 2`, `prices = [3, 2, 6, 5, 0, 3]`
```
k = 2 < 6 / 2 = 3 (Bounded transactions DP)

Initial State (Day 0, price = 3):
  buy = [0, -3, -3]
  sell = [0,  0,  0]

Day 1 (price = 2):
  t = 1: buy[1] = max(-3, 0 - 2) = -2, sell[1] = max(0, -2 + 2) = 0
  t = 2: buy[2] = max(-3, 0 - 2) = -2, sell[2] = max(0, -2 + 2) = 0

Day 2 (price = 6):
  t = 1: buy[1] = max(-2, -6) = -2, sell[1] = max(0, -2 + 6) = 4
  t = 2: buy[2] = max(-2, 4 - 6) = -2, sell[2] = max(0, -2 + 6) = 4

Day 3 (price = 5):
  t = 1: buy[1] = -2, sell[1] = 4
  t = 2: buy[2] = max(-2, 4 - 5) = -1, sell[2] = max(4, -1 + 5) = 4

Day 4 (price = 0):
  t = 1: buy[1] = max(-2, 0) = 0, sell[1] = 4
  t = 2: buy[2] = max(-1, 4 - 0) = 4, sell[2] = 4

Day 5 (price = 3):
  t = 1: buy[1] = 0, sell[1] = 4
  t = 2: buy[2] = 4, sell[2] = max(4, 4 + 3) = 7

Final Answer: sell[2] = 7.
(Trade 1: Buy at 2, Sell at 6 -> Profit 4; Trade 2: Buy at 0, Sell at 3 -> Profit 3. Total = 7).
```

---

### Solved Examples with Multiple Inputs

| `k` | `prices` | $k \ge n/2$? | Transactions Chosen | Output |
|---|---|---|---|---|
| `2` | `[2, 4, 1]` | Yes ($2 \ge 1.5$) | Buy at 2, sell at 4 | `2` |
| `2` | `[3, 2, 6, 5, 0, 3]` | No ($2 < 3$) | $(6-2) + (3-0) = 4 + 3$ | `7` |
| `1` | `[1, 2, 4, 2, 5, 7, 2, 4, 9, 0]` | No | Buy at 1, sell at 9 | `8` |
| `0` | `[1, 2, 3]` | No | 0 transactions allowed | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxProfit(self, k: int, prices: list[int]) -> int:
        n = len(prices)
        if n <= 1 or k == 0:
            return 0
            
        # Optimization: unlimited transactions if k >= n // 2
        if k >= n // 2:
            return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, n))
            
        # buy[t] = max profit with t buys; sell[t] = max profit with t sells
        buy = [-prices[0]] * (k + 1)
        sell = [0] * (k + 1)
        
        for p in prices[1:]:
            for t in range(1, k + 1):
                buy[t] = max(buy[t], sell[t - 1] - p)
                sell[t] = max(sell[t], buy[t] + p)
                
        return sell[k]
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <numeric>

class Solution {
public:
    int maxProfit(int k, const std::vector<int>& prices) {
        int n = static_cast<int>(prices.size());
        if (n <= 1 || k == 0) return 0;

        // Unlimited transactions optimization
        if (k >= n / 2) {
            int max_profit = 0;
            for (int i = 1; i < n; ++i) {
                if (prices[i] > prices[i - 1]) {
                    max_profit += prices[i] - prices[i - 1];
                }
            }
            return max_profit;
        }

        // Bounded transactions DP in O(k) space
        std::vector<int> buy(k + 1, -prices[0]);
        std::vector<int> sell(k + 1, 0);

        for (int i = 1; i < n; ++i) {
            int p = prices[i];
            for (int t = 1; t <= k; ++t) {
                buy[t] = std::max(buy[t], sell[t - 1] - p);
                sell[t] = std::max(sell[t], buy[t] + p);
            }
        }

        return sell[k];
    }
};
```

#### Java 17
```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        if (n <= 1 || k == 0) {
            return 0;
        }

        // Fast path for unlimited transactions
        if (k >= n / 2) {
            int maxProfit = 0;
            for (int i = 1; i < n; i++) {
                if (prices[i] > prices[i - 1]) {
                    maxProfit += prices[i] - prices[i - 1];
                }
            }
            return maxProfit;
        }

        // Space-optimized O(k) DP
        int[] buy = new int[k + 1];
        int[] sell = new int[k + 1];

        for (int t = 0; t <= k; t++) {
            buy[t] = -prices[0];
            sell[t] = 0;
        }

        for (int i = 1; i < n; i++) {
            int p = prices[i];
            for (int t = 1; t <= k; t++) {
                buy[t] = Math.max(buy[t], sell[t - 1] - p);
                sell[t] = Math.max(sell[t], buy[t] + p);
            }
        }

        return sell[k];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \cdot k)$, where $n = |prices|$. The outer loop runs $n$ times, and the inner loop runs $k$ times with $\mathcal{O}(1)$ updates. When $k \ge n / 2$, runtime is strictly $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(k)$ auxiliary space for the two arrays `buy` and `sell` of size $k + 1$.

---

### Takeaway Pattern & Interview Traps

1. **The $k \ge n / 2$ Memory & Time Saver:** In competitive programming or old LeetCode test cases, $k$ could be as large as $10^9$. Without the $k \ge n / 2$ early exit, allocating an array of size $k$ leads to Memory Limit Exceeded (MLE).
2. **Unified State-Machine Architecture:** The progression across the stock series is unified:
   - LC 121: 1 transaction ($k = 1$)
   - LC 122: $\infty$ transactions
   - LC 123: 2 transactions ($k = 2$)
   - LC 188: $k$ transactions
   - LC 309: $\infty$ transactions with 1-day cooldown
   - LC 714: $\infty$ transactions with fee
3. **Loop Ordering:** Updating `buy[t]` followed immediately by `sell[t]` on the same day is safe because same-day transactions net to $0$ profit.