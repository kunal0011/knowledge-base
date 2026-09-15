---
date: "2026-08-29"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1499: Max Value of Equation"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 1499: Max Value of Equation

**Target Companies:** Google (Top Asked Hard)

---

### Problem Statement

Given array `points` sorted by x, where `points[i] = [x_i, y_i]`, find max value of equation `y_i + y_j + |x_i - x_j|` where `|x_i - x_j| <= k` and `i < j`.

---

### Key Observation

* Since `x` is sorted and `i < j`, `|x_i - x_j| = x_j - x_i`.
* Rewrite equation: `y_i + y_j + x_j - x_i = (y_j + x_j) + (y_i - x_i)`.
* For each `j`, we need to maximize `(y_i - x_i)` subject to `x_j - x_i <= k`.
* Use a **Monotonic Decreasing Deque** storing pairs `(y_i - x_i, x_i)`.

---

### Core Technique: Algebraic Formula Factorization + Monotonic Max Deque

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        q = deque()  # stores (y - x, x), decreasing in (y - x)
        max_val = float('-inf')
        
        for x, y in points:
            # 1. Remove expired points (x - x_prev > k)
            while q and x - q[0][1] > k:
                q.popleft()
                
            # 2. Best candidate is at front of deque
            if q:
                max_val = max(max_val, (y + x) + q[0][0])
                
            # 3. Maintain monotonic decreasing order of (y - x)
            curr_diff = y - x
            while q and q[-1][0] <= curr_diff:
                q.pop()
            q.append((curr_diff, x))
            
        return max_val
```

---

### Worked-Out Example

```python
points = [[1,3],[2,0],[5,10],[6,-10]], k = 1
point [1,3]: diff = 3-1 = 2 -> q=[(2, 1)]
point [2,0]: x_diff = 2-1 <= 1 (valid) -> val = (0+2) + 2 = 4 -> max_val = 4
             curr_diff = 0-2 = -2 -> q=[(2, 1), (-2, 2)]
point [5,10]: 5-1 > 1 (expire 1), 5-2 > 1 (expire 2) -> q=[] -> q=[(5, 5)]
point [6,-10]: 6-5 <= 1 -> val = (-10+6) + 5 = 1
Max value = 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Factorize 2-variable coordinate formulas into independent `(y + x) + (y\_prev - x\_prev)` to query max in O(1) via Monotonic Deque.