---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 875: Koko Eating Bananas"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 875: Koko Eating Bananas

---

### Problem Statement

Koko wants to eat all bananas within `h` hours. Return the minimum integer eating speed `k` per hour.

---

### Key Observation

* The speed `k` lies in monotonic range `[1, max(piles)]`.
* If speed `k` is sufficient to eat in `h` hours, any speed `> k` is also valid.
* This monotonicity enables Binary Search on the Answer.

---

### Core Technique: Binary Search on Monotonic Feasibility Function

---

### Python 3 Solution (with typing)

```python
import math
from typing import List

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        ans = right
        
        while left <= right:
            mid = left + (right - left) // 2
            total_hours = sum(math.ceil(p / mid) for p in piles)
            
            if total_hours <= h:
                ans = mid
                right = mid - 1  # try finding a smaller valid speed
            else:
                left = mid + 1   # speed too slow
                
        return ans
```

---

### Worked-Out Example

```python
piles = [3, 6, 7, 11], h = 8
left = 1, right = 11
mid = 6: hours = ceil(3/6)+ceil(6/6)+ceil(7/6)+ceil(11/6) = 1+1+2+2 = 6 <= 8 (valid) -> ans=6, right=5
mid = 3: hours = 1+2+3+4 = 10 > 8 (invalid) -> left=4
mid = 4: hours = 1+2+2+3 = 8 <= 8 (valid) -> ans=4, right=3
Result = 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(n * log(max(piles)))`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Whenever asked to minimize/maximize an answer with a monotonic verification condition, use Binary Search on Answer.