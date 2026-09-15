---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 122: Best Time to Buy and Sell Stock II"
tags:
  - leetcode
  - coding
  - greedy
  - dynamic-programming
  - array
  - amazon
  - google
---

# LeetCode 122: Best Time to Buy and Sell Stock II

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Dynamic Programming / Array  

---

### Problem Statement

You are given an integer array `prices` where `prices[i]` is the price of a given stock on the $i$-th day.

On each day, you may decide to buy and/or sell the stock. You can only hold **at most one** share of the stock at any time. However, you can buy it and then immediately sell it on the **same day**.

Find and return the **maximum profit** you can achieve.

---

### Input & Output Formats & Constraints

- **Input:**
  - `prices`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{prices.length} \le 3 \times 10^4$).
- **Output:**
  - `int` — the total maximum profit achievable with unlimited transactions.
- **Constraints:**
  - $1 \le \text{prices.length} \le 3 \times 10^4$
  - $0 \le \text{prices}[i] \le 10^4$

---

### Key Idea & Intuition

Unlike LeetCode 121 (where you are restricted to at most one transaction), here you are allowed **unlimited transactions**.

#### The Telescoping Sum Invariant:
Consider buying a stock at day $a$ and selling it at a peak day $b$ ($a < b$) where prices are monotonically increasing: $P_a \le P_{a+1} \le \dots \le P_b$.
The total profit is:
$$P_b - P_a = (P_b - P_{b-1}) + (P_{b-1} - P_{b-2}) + \dots + (P_{a+1} - P_a)$$

Notice that the net profit of a multi-day holding period is **identically equal** to the sum of daily profits across each single day in that range!
- If the price goes up tomorrow ($P_{i} > P_{i-1}$), we capture the delta $P_{i} - P_{i-1}$.
- If the price goes down tomorrow ($P_{i} \le P_{i-1}$), we do nothing (we would not hold stock overnight during a downturn).

Therefore, the globally optimal strategy is equivalent to the pure greedy local rule:
$$\text{Max Profit} = \sum_{i=1}^{n-1} \max(0, \text{prices}[i] - \text{prices}[i - 1])$$

There is never any benefit to skipping a positive daily price increase.

---

### Solution Approach (Step-by-Step)

1. Initialize `max_profit = 0`.
2. Loop `i` from $1$ to $n - 1$:
   - If `prices[i] > prices[i - 1]`:
     - `max_profit += prices[i] - prices[i - 1]`
3. Return `max_profit`.

---

### Visual Algorithm Walkthrough

For `prices = [7, 1, 5, 3, 6, 4]`:

```
Day 0: price = 7
Day 1: price = 1 -> 1 < 7: price dropped, no trade.
Day 2: price = 5 -> 5 > 1: gain = 5 - 1 = 4. profit += 4 (total: 4)
Day 3: price = 3 -> 3 < 5: price dropped, no trade.
Day 4: price = 6 -> 6 > 3: gain = 6 - 3 = 3. profit += 3 (total: 7)
Day 5: price = 4 -> 4 < 6: price dropped, no trade.

Total Profit = 4 + 3 = 7.
```

Visual Price Curve:
```
Price
  7 |  *
  6 |              *
  5 |        *
  4 |                    *
  3 |           *
  2 |
  1 |     *
    +----------------------- Day
       0  1  2  3  4  5

Slopes collected:
  Day 1 -> Day 2: +4
  Day 3 -> Day 4: +3
  Sum = 7
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `prices = [7, 1, 5, 3, 6, 4]`
- **Output:** `7` (Buy day 1, sell day 2; buy day 3, sell day 4)

#### Example 2 (Monotonically Increasing):
- **Input:** `prices = [1, 2, 3, 4, 5]`
- **Tracing:** Gains: $(2-1) + (3-2) + (4-3) + (5-4) = 4$.
- **Output:** `4`

#### Example 3 (Monotonically Decreasing):
- **Input:** `prices = [7, 6, 4, 3, 1]`
- **Tracing:** Prices strictly decrease every day. No transactions made.
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        
        for i in range(1, len(prices)):
            # Capture every positive upward slope
            if prices[i] > prices[i - 1]:
                max_profit += prices[i] - prices[i - 1]
                
        return max_profit
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxProfit(const std::vector<int>& prices) {
        int max_profit = 0;
        int n = static_cast<int>(prices.size());
        
        for (int i = 1; i < n; ++i) {
            if (prices[i] > prices[i - 1]) {
                max_profit += (prices[i] - prices[i - 1]);
            }
        }
        
        return max_profit;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                maxProfit += prices[i] - prices[i - 1];
            }
        }
        
        return maxProfit;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - A single linear scan through the array of length $n$, performing one subtraction and conditional addition per day.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Uses only a single integer accumulator `max_profit`.

---

### Takeaway Pattern & Interview Traps

- **Why Same-Day Buy/Sell Doesn't Violate "Hold at Most One":** Selling at day $i$ and buying again at day $i$ is logically identical to holding the stock from day $i-1$ through day $i+1$. The telescoping mathematical identity guarantees the total profit is unchanged.
- **Contrast with Transaction Fees (LeetCode 714) or Cooldown (LeetCode 309):** The greedy slope strategy works **only** when transactions have zero friction (no fees, no cooldowns). When fees or cooldowns exist, you must switch to State Machine Dynamic Programming.