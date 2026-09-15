---
date: "2026-08-29"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 239: Sliding Window Maximum"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 239: Sliding Window Maximum

**Target Companies:** Amazon (Top #1), Google, Meta

---

### Problem Statement

You are given an array of integers `nums` and a sliding window of size `k`. Return the max sliding window values in `O(n)` time.

---

### Key Observation

* Use a **Monotonic Decreasing Deque** storing indices of elements.
* When adding `nums[i]`, pop elements from back while `nums[deque[-1]] < nums[i]` (smaller elements can never be the maximum again).
* Evict front indices if they fall out of window boundary (`deque[0] < i - k + 1`).
* `deque[0]` is always the maximum for current window.

---

### Core Technique: Monotonic Decreasing Deque (O(n) Sliding Window Max)

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()  # stores indices, values are strictly decreasing
        res = []
        
        for i, num in enumerate(nums):
            # 1. Maintain decreasing order (pop smaller elements from back)
            while q and nums[q[-1]] < num:
                q.pop()
            q.append(i)
            
            # 2. Evict out-of-window index from front
            if q[0] < i - k + 1:
                q.popleft()
                
            # 3. Append max once window size reaches k
            if i >= k - 1:
                res.append(nums[q[0]])
                
        return res
```

---

### Worked-Out Example

```python
nums = [1,3,-1,-3,5,3,6,7], k = 3
i=0 (1): q=[0 (1)]
i=1 (3): pop 0, q=[1 (3)]
i=2 (-1): q=[1 (3), 2 (-1)] -> max = 3
i=3 (-3): q=[1 (3), 2 (-1), 3 (-3)] -> max = 3
i=4 (5): pop 3, 2, 1, q=[4 (5)] -> max = 5
Result = [3, 3, 5, 5, 6, 7]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n) (each index is pushed and popped at most once)`
* **Space Complexity:** `O(k) for deque`

---

### Takeaway Pattern

Monotonic Decreasing Deque guarantees O(1) query of sliding window maximums.