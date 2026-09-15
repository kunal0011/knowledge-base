---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 122: Best Time to Buy and Sell Stock II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - greedy
  - array
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 122: Best Time to Buy and Sell Stock II

**Target Companies:** Amazon, Google, Meta, Bloomberg, Microsoft  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Greedy / Array  

---

### Problem Statement

You are given an integer array `prices` where `prices[i]` is the price of a given stock on the $i^{\text{th}}$ day.

On each day, you may decide to buy and/or sell the stock. You can only hold **at most one share** of the stock at any time. However, you can buy it then immediately sell it on the same day.

Find and return *the **maximum profit** you can achieve*.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `prices` ($1 \le |prices| \le 3 \times 10^4$).
- **Output:** An integer denoting the maximum profit achievable with unlimited transactions.
- **Constraints:**
  - `1 <= prices.length <= 3 * 10^4`
  - `0 <= prices[i] <= 10^4`

---

### Key Idea & Intuition

#### 1. The Greedy Angle: Accumulating Every Upward Slope
Because we can trade an unlimited number of times with no transaction fees, any multi-day price increase from day $A$ to day $B$ ($prices[B] - prices[A]$ where $prices[A] < prices[A+1] < \dots < prices[B]$) can be decomposed into a sum of single-day gains:
$$prices[B] - prices[A] = \sum_{k=A}^{B-1} (prices[k+1] - prices[k])$$
Therefore, we can simply harvest the profit on every single day where the price rises:
$$\text{Total Profit} = \sum_{i=1}^{n-1} \max(0, prices[i] - prices[i - 1])$$

#### 2. The Dynamic Programming State-Machine Angle
To build intuition for constrained stock problems (LC 123, 188, 309, 714), we model the daily decision space using two state variables:
- `not_hold`: Maximum profit on day $i$ without owning a share.
- `hold`: Maximum profit on day $i$ while holding a share.

**Transitions on day $i$:**
1. `not_hold` = $\max(\text{not\_hold}, \text{hold} + prices[i])$
   *(Either stay without stock, or sell stock held previously at today's price)*
2. `hold` = $\max(\text{hold}, \text{not\_hold} - prices[i])$
   *(Either continue holding stock, or buy a new share using our accumulated `not_hold` cash)*

Notice the critical distinction from LeetCode 121: because unlimited transactions are allowed, buying stock today builds upon previously accumulated profits (`not_hold - prices[i]`) rather than resetting to `0 - prices[i]`.

---

### Solution Approach (Step-by-Step)

1. **Initialize State Machine:**
   - `not_hold = 0` (initial state with 0 cash, no stock)
   - `hold = -prices[0]` (buying on day 0)
2. **Daily Transitions:**
   - For price $p$ in `prices[1:]`:
     - `prev_not_hold = not_hold`
     - `not_hold = max(not_hold, hold + p)`
     - `hold = max(hold, prev_not_hold - p)`
3. **Return Output:**
   - Return `not_hold` (at market close, an unheld position is strictly $\ge$ a held position).

---

### Visual Algorithm Walkthrough

#### Trace for `prices = [7, 1, 5, 3, 6, 4]`
```
Day 0: price = 7
  not_hold = 0
  hold = -7

Day 1: price = 1
  not_hold = max(0, -7 + 1) = 0
  hold = max(-7, 0 - 1) = -1

Day 2: price = 5
  not_hold = max(0, -1 + 5) = 4
  hold = max(-1, 0 - 5) = -1

Day 3: price = 3
  not_hold = max(4, -1 + 3) = 4
  hold = max(-1, 4 - 3) = 1

Day 4: price = 6
  not_hold = max(4, 1 + 6) = 7
  hold = max(1, 4 - 6) = 1

Day 5: price = 4
  not_hold = max(7, 1 + 4) = 7
  hold = max(1, 7 - 4) = 3

Final Result: not_hold = 7.
(Buy at 1, sell at 5 -> +4; Buy at 3, sell at 6 -> +3. Total = 7).
```

---

### Solved Examples with Multiple Inputs

| Prices Input | Daily Price Slopes $(prices[i] - prices[i-1])$ | Positive Slopes Harvested | Total Profit |
|---|---|---|---|
| `[7, 1, 5, 3, 6, 4]` | $-6, +4, -2, +3, -2$ | $+4 + 3 = 7$ | `7` |
| `[1, 2, 3, 4, 5]` | $+1, +1, +1, +1$ | $1 + 1 + 1 + 1 = 4$ | `4` |
| `[7, 6, 4, 3, 1]` | $-1, -2, -1, -2$ | None | `0` |
| `[2, 2, 2, 2]` | $0, 0, 0$ | None | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        not_hold: int = 0
        hold: int = -prices[0]
        
        for p in prices[1:]:
            prev_not_hold: int = not_hold
            not_hold = max(not_hold, hold + p)
            hold = max(hold, prev_not_hold - p)
            
        return not_hold
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxProfit(const std::vector<int>& prices) {
        int not_hold = 0;
        int hold = -prices[0];
        int n = static_cast<int>(prices.size());
        
        for (int i = 1; i < n; ++i) {
            int prev_not_hold = not_hold;
            not_hold = std::max(not_hold, hold + prices[i]);
            hold = std::max(hold, prev_not_hold - prices[i]);
        }
        
        return not_hold;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxProfit(int[] prices) {
        int notHold = 0;
        int hold = -prices[0];
        
        for (int i = 1; i < prices.length; i++) {
            int prevNotHold = notHold;
            notHold = Math.max(notHold, hold + prices[i]);
            hold = Math.max(hold, prevNotHold - prices[i]);
        }
        
        return notHold;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |prices|$. We perform a single linear sweep across the price array.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only two state scalar variables (`not_hold`, `hold`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Greedy vs. DP Proof of Equivalence:** While the greedy "sum of positive differences" $\sum \max(0, prices[i] - prices[i-1])$ is shorter to write, explaining the DP state machine shows deep mastery and effortlessly scales to problems with cooldowns (LC 309) or transaction fees (LC 714).
2. **Same-Day Transactions:** The problem explicitly allows buying and selling on the same day. In the DP formulation, doing so incurs $-p + p = 0$, causing no negative side effects.