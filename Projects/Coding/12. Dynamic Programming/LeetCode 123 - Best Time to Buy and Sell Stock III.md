---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 123: Best Time to Buy and Sell Stock III"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - amazon
  - google
  - meta
  - microsoft
  - citadel
---

# LeetCode 123: Best Time to Buy and Sell Stock III

**Target Companies:** Amazon, Google, Meta, Microsoft, Citadel, Bloomberg  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Array  

---

### Problem Statement

You are given an array `prices` where `prices[i]` is the price of a given stock on the $i^{\text{th}}$ day.

Find the maximum profit you can achieve. You may complete **at most two transactions**.

**Note:** You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

---

### Input & Output Formats & Constraints

- **Input:** An integer array `prices` ($1 \le |prices| \le 10^5$).
- **Output:** An integer denoting the maximum profit achievable with at most two transactions.
- **Constraints:**
  - `1 <= prices.length <= 10^5`
  - `0 <= prices[i] <= 10^5`

---

### Key Idea & Intuition

#### The Four-State Machine Formulation
At any point in time during the trading sequence, an investor can be in one of four distinct non-trivial financial positions:
1. `buy1`: The maximum cash balance after **buying the first stock**.
2. `sell1`: The maximum cash balance after **selling the first stock**.
3. `buy2`: The maximum cash balance after **buying the second stock**.
4. `sell2`: The maximum cash balance after **selling the second stock**.

#### Transition Equations on Day $i$
For each price $p$ on day $i$:
1. **First Buy (`buy1`):**
   - Either keep prior position, or buy today for $-p$:
     $$buy1 = \max(buy1, -p)$$
2. **First Sell (`sell1`):**
   - Either keep prior position, or sell the first stock today at price $p$:
     $$sell1 = \max(sell1, buy1 + p)$$
3. **Second Buy (`buy2`):**
   - Either keep prior position, or buy the second stock today using profit from the first trade ($sell1 - p$):
     $$buy2 = \max(buy2, sell1 - p)$$
4. **Second Sell (`sell2`):**
   - Either keep prior position, or sell the second stock today at price $p$:
     $$sell2 = \max(sell2, buy2 + p)$$

#### Initial Base Conditions
On day 0:
- $buy1 = -prices[0]$
- $sell1 = 0$
- $buy2 = -prices[0]$ (buying and immediately selling and buying again on day 0)
- $sell2 = 0$

Because each variable depends only on variables from the same or earlier transactions on that day, updating them sequentially in the order `buy1` $\to$ `sell1` $\to$ `buy2` $\to$ `sell2` works seamlessly in $\mathcal{O}(1)$ space.

---

### Solution Approach (Step-by-Step)

1. **State Initialization:**
   - Set `buy1 = -prices[0]`, `sell1 = 0`.
   - Set `buy2 = -prices[0]`, `sell2 = 0`.
2. **Iterate Across Prices:**
   - For each price $p$ in `prices`:
     - `buy1 = max(buy1, -p)`
     - `sell1 = max(sell1, buy1 + p)`
     - `buy2 = max(buy2, sell1 - p)`
     - `sell2 = max(sell2, buy2 + p)`
3. **Return Output:**
   - Return `sell2` (if only 1 transaction is optimal, `sell2` will equal `sell1`).

---

### Visual Algorithm Walkthrough

#### Trace for `prices = [3, 3, 5, 0, 0, 3, 1, 4]`
```
Day 0 (price = 3):
  buy1 = -3, sell1 = 0, buy2 = -3, sell2 = 0

Day 1 (price = 3):
  buy1 = max(-3, -3) = -3
  sell1 = max(0, -3 + 3) = 0
  buy2 = max(-3, 0 - 3) = -3
  sell2 = max(0, -3 + 3) = 0

Day 2 (price = 5):
  buy1 = max(-3, -5) = -3
  sell1 = max(0, -3 + 5) = 2   (Sold 1st stock for +2 profit)
  buy2 = max(-3, 2 - 5) = -3
  sell2 = max(0, -3 + 5) = 2

Day 3 (price = 0):
  buy1 = max(-3, -0) = 0      (Better buy price available: 0)
  sell1 = max(2, 0 + 0) = 2
  buy2 = max(-3, 2 - 0) = 2   (Bought 2nd stock at 0 with 2 in bank)
  sell2 = max(2, 2 + 0) = 2

Day 4 (price = 0):
  States remain: buy1 = 0, sell1 = 2, buy2 = 2, sell2 = 2

Day 5 (price = 3):
  buy1 = max(0, -3) = 0
  sell1 = max(2, 0 + 3) = 3
  buy2 = max(2, 3 - 3) = 2
  sell2 = max(2, 2 + 3) = 5   (Sold 2nd stock for +3, total = 5)

Day 6 (price = 1):
  buy1 = max(0, -1) = 0
  sell1 = max(3, 0 + 1) = 3
  buy2 = max(2, 3 - 1) = 2
  sell2 = max(5, 2 + 1) = 5

Day 7 (price = 4):
  buy1 = max(0, -4) = 0
  sell1 = max(3, 0 + 4) = 4
  buy2 = max(2, 4 - 4) = 2
  sell2 = max(5, 2 + 4) = 6   (Sold 2nd stock at 4, total = 6)

Final Profit: sell2 = 6.
Transaction 1: Buy at 3 (Day 0), Sell at 5 (Day 2) -> Profit 2.
Transaction 2: Buy at 0 (Day 3), Sell at 4 (Day 7) -> Profit 4.
Total Profit = 2 + 4 = 6.
```

---

### Solved Examples with Multiple Inputs

| `prices` | `buy1, sell1` | `buy2, sell2` | Explanation | Output |
|---|---|---|---|---|
| `[3, 3, 5, 0, 0, 3, 1, 4]` | $(0, 4)$ | $(2, 6)$ | Two transactions: $(5-3) + (4-0) = 6$ | `6` |
| `[1, 2, 3, 4, 5]` | $(-1, 4)$ | $(-1, 4)$ | Monotonically increasing: 1 transaction suffices | `4` |
| `[7, 6, 4, 3, 1]` | $(-1, 0)$ | $(-1, 0)$ | Strictly decreasing: no transactions taken | `0` |
| `[1]` | $(-1, 0)$ | $(-1, 0)$ | Single price point | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        if not prices:
            return 0
            
        buy1: float = -prices[0]
        sell1: int = 0
        buy2: float = -prices[0]
        sell2: int = 0
        
        for p in prices:
            buy1 = max(buy1, -p)
            sell1 = max(sell1, buy1 + p)
            buy2 = max(buy2, sell1 - p)
            sell2 = max(sell2, buy2 + p)
            
        return sell2
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxProfit(const std::vector<int>& prices) {
        if (prices.empty()) return 0;
        
        int buy1 = -prices[0];
        int sell1 = 0;
        int buy2 = -prices[0];
        int sell2 = 0;
        
        for (int p : prices) {
            buy1 = std::max(buy1, -p);
            sell1 = std::max(sell1, buy1 + p);
            buy2 = std::max(buy2, sell1 - p);
            sell2 = std::max(sell2, buy2 + p);
        }
        
        return sell2;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        
        int buy1 = -prices[0];
        int sell1 = 0;
        int buy2 = -prices[0];
        int sell2 = 0;
        
        for (int p : prices) {
            buy1 = Math.max(buy1, -p);
            sell1 = Math.max(sell1, buy1 + p);
            buy2 = Math.max(buy2, sell1 - p);
            sell2 = Math.max(sell2, buy2 + p);
        }
        
        return sell2;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |prices|$. We iterate through the array once, performing four $\mathcal{O}(1)$ updates per element.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Exactly four scalar integer variables are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Simultaneous Same-Day Buy/Sell:** If a single transaction is optimal (e.g. `[1, 2, 3, 4, 5]`), the second transaction can be executed on the same day ($buy2 = sell1 - p$, $sell2 = buy2 + p = sell1$), effectively neutralizing the second transaction and returning the 1-transaction maximum.
2. **Generalization to $K$ Transactions (LeetCode 188):** Instead of four separate variables `buy1, sell1, buy2, sell2`, maintain two arrays `buy[k]` and `sell[k]`. The transition logic remains identically $\mathcal{O}(1)$ per state!
3. **Alternative Bidirectional DP:** An alternative $\mathcal{O}(N)$ time and $\mathcal{O}(N)$ space method precomputes prefix maximum profits $left[i]$ and suffix maximum profits $right[i]$, taking $\max_{i} (left[i] + right[i])$. Mentioning the 4-state constant-space approach demonstrates superior optimization capability.