---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 1011: Capacity To Ship Packages Within D Days"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 1011: Capacity To Ship Packages Within D Days

**Target Companies:** Amazon (Signature Logistics Problem), Google

---

### Problem Statement

A conveyor belt carries packages with weights. Ship all packages within `days` in the given order. Return least ship capacity.

---

### Key Observation

* Minimum ship capacity must be at least `max(weights)` (to ship the heaviest item).
* Maximum capacity needed is `sum(weights)` (ship all items in 1 day).
* Capacity feasibility is strictly monotonic $
  ightarrow$ Binary search capacity in range `[max(weights), sum(weights)]`.

---

### Core Technique: Binary Search on Capacity Feasibility

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def can_ship(capacity: int) -> bool:
            days_needed = 1
            curr_load = 0
            for w in weights:
                if curr_load + w > capacity:
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

---

### Worked-Out Example

```python
weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Range: [10, 55]
mid = 32 -> 3 days (valid) -> ans=32, right=31
mid = 15 -> 5 days (valid) -> ans=15, right=14
Result = 15
```

---

### Complexity Analysis

* **Time Complexity:** `O(n * log(sum - max))`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Standard Amazon inventory/logistics pattern: minimize capacity by testing monotonic multi-day packing.