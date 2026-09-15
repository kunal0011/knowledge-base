---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 1011: Capacity To Ship Packages Within D Days"
tags:
  - leetcode
  - coding
  - binary-search
  - binary-search-on-answer
  - greedy
  - amazon
  - google
---

# LeetCode 1011: Capacity To Ship Packages Within D Days

**Target Companies:** Amazon (Top Signature Supply Chain Question), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Binary Search on Answer / Greedy Conveyor Belt Capacity Partitioning  

---

### Problem Statement

A conveyor belt has packages that must be shipped from one port to another within `days` days.

The $i$-th package on the conveyor belt has a weight of `weights[i]`. Each day, we load the ship with packages on the conveyor belt (in the order given by `weights`). We may not load more weight than the maximum weight capacity of the ship.

Return the **least weight capacity** of the ship that will result in all the packages on the conveyor belt being shipped within `days` days.

---

### Input & Output Formats & Constraints

- **Input:** `weights: List[int]`, `days: int`
- **Output:** `int` (minimum possible ship capacity)
- **Constraints:**
  - $1 \le \text{days} \le \text{weights.length} \le 5 \times 10^4$
  - $1 \le \text{weights}[i] \le 500$

---

### Key Idea & Intuition

The packages must be loaded in the **exact conveyor order** given.
Notice that the ship's capacity $C$ has a strict **monotonic feasibility property**:
- If capacity $C$ can ship all packages in $\le \text{days}$, then any capacity $C' > C$ can also ship them in $\le \text{days}$.
- If capacity $C$ cannot ship all packages in $\le \text{days}$, then no smaller capacity $C'' < C$ can either.

This monotonic relationship enables **Binary Search on the Answer**:
- **Minimum feasible capacity (`left`):** $\max(\text{weights})$ — The ship must at least be able to carry the single heaviest individual package, otherwise that package could never be loaded.
- **Maximum feasible capacity (`right`):** $\sum \text{weights}$ — At this capacity, all packages are shipped together on Day 1.

#### Greedy Verification Function (`can_ship(cap)`):
Traverse the packages in order. Accumulate package weights on the current day. If adding the next package causes the current day's load to exceed `cap`, increment `days_needed` by 1 and place the package on the next day.
If `days_needed <= days`, then `cap` is feasible $\implies$ seek a smaller capacity (`right = mid - 1`).

---

### Solution Approach (Step-by-Step)

1. Determine search boundaries:
   - `left = max(weights)`
   - `right = sum(weights)`
   - `ans = right`
2. Define helper `can_ship(cap: int) -> bool`:
   - `days_needed = 1`, `curr_load = 0`.
   - For `w` in `weights`:
     - If `curr_load + w > cap`:
       - `days_needed += 1`
       - `curr_load = w`
     - Else:
       - `curr_load += w`
   - Return `days_needed <= days`.
3. While `left <= right`:
   - `mid = left + (right - left) // 2`.
   - If `can_ship(mid)`:
     - `ans = mid`
     - `right = mid - 1` (try smaller capacity)
   - Else:
     - `left = mid + 1` (capacity too small, increase)
4. Return `ans`.

---

### Visual Algorithm Walkthrough

```
weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], days = 5

Search Space:
  left = max(weights) = 10
  right = sum(weights) = 55

Iteration 1:
  mid = (10 + 55) // 2 = 32
  can_ship(32):
    Day 1: 1+2+3+4+5+6+7 = 28
    Day 2: 8+9+10 = 27
    Days needed = 2 <= 5 (FEASIBLE!)
    ans = 32, right = mid - 1 = 31

Iteration 2:
  mid = (10 + 31) // 2 = 20
  can_ship(20):
    Day 1: 1+2+3+4+5 = 15
    Day 2: 6+7 = 13
    Day 3: 8+9 = 17
    Day 4: 10
    Days needed = 4 <= 5 (FEASIBLE!)
    ans = 20, right = mid - 1 = 19

Iteration 3:
  mid = (10 + 19) // 2 = 14
  can_ship(14):
    Day 1: 1+2+3+4 = 10
    Day 2: 5+6 = 11
    Day 3: 7
    Day 4: 8
    Day 5: 9
    Day 6: 10
    Days needed = 6 > 5 (INFEASIBLE!)
    left = mid + 1 = 15

Iteration 4:
  mid = 15
  can_ship(15):
    Day 1: 1+2+3+4+5 = 15
    Day 2: 6+7 = 13
    Day 3: 8
    Day 4: 9
    Day 5: 10
    Days needed = 5 <= 5 (FEASIBLE!)
    ans = 15, right = 14

Loop terminates (left > right).
Minimum capacity = 15.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Multi-Day Loading
- **Input:** `weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`, `days = 5`
- **Output:** `15`

#### Example 2: Tight Delivery Schedule ($days = 3$)
- **Input:** `weights = [3, 2, 2, 4, 1, 4]`, `days = 3`
- **Days Breakdown at Capacity 6:**
  - Day 1: `[3, 2]` (weight 5)
  - Day 2: `[2, 4]` (weight 6)
  - Day 3: `[1, 4]` (weight 5)
- **Output:** `6`

#### Example 3: Minimal Possible Days ($days = 1$)
- **Input:** `weights = [1, 2, 3, 1, 1]`, `days = 1`
- **Output:** `8` (sum of all weights)

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def can_ship(cap: int) -> bool:
            days_needed = 1
            curr_load = 0
            for w in weights:
                if curr_load + w > cap:
                    days_needed += 1
                    curr_load = w
                else:
                    curr_load += w
            return days_needed <= days
            
        left, right = max(weights), sum(weights)
        ans = right
        
        while left <= right:
            mid = left + (right - left) // 2
            if can_ship(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
                
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int shipWithinDays(std::vector<int>& weights, int days) {
        int left = *std::max_element(weights.begin(), weights.end());
        int right = std::accumulate(weights.begin(), weights.end(), 0);
        int ans = right;

        auto canShip = [&](int cap) -> bool {
            int daysNeeded = 1;
            int currLoad = 0;
            for (int w : weights) {
                if (currLoad + w > cap) {
                    daysNeeded++;
                    currLoad = w;
                } else {
                    currLoad += w;
                }
            }
            return daysNeeded <= days;
        };

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (canShip(mid)) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return ans;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int shipWithinDays(int[] weights, int days) {
        int left = 0;
        int right = 0;

        for (int w : weights) {
            left = Math.max(left, w);
            right += w;
        }

        int ans = right;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (canShip(weights, days, mid)) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return ans;
    }

    private boolean canShip(int[] weights, int days, int cap) {
        int daysNeeded = 1;
        int currLoad = 0;

        for (int w : weights) {
            if (currLoad + w > cap) {
                daysNeeded++;
                currLoad = w;
            } else {
                currLoad += w;
            }
        }

        return daysNeeded <= days;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \cdot \log(\sum \text{weights} - \max(\text{weights})))$ — Binary search executes $O(\log(\sum W))$ times. Each check traverses all $N$ packages in $O(N)$.
- **Space Complexity:** $O(1)$ auxiliary space — Only primitive scalar counters.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Binary Search on Continuous Conveyor Belt Capacity (Greedy Feasibility Check).
- **Trap:** Forgetting that packages **cannot be reordered**: this is strictly sequential partitioning, which is why greedy packing from left to right is optimal and runs in $O(N)$.