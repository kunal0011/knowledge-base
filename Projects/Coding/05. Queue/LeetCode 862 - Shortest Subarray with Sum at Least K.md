---
date: "2026-08-29"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 862: Shortest Subarray with Sum at Least K"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 862: Shortest Subarray with Sum at Least K

**Target Companies:** Google (Signature Hard), Amazon

---

### Problem Statement

Given an integer array `nums` (which can contain negative numbers) and an integer `k`, return length of shortest non-empty subarray with sum at least `k` (-1 if none).

---

### Key Observation

* With negative numbers, standard two-pointer sliding window fails because prefix sums are non-monotonic.
* Prefix sum condition: `P[j] - P[i] >= k` $
  ightarrow$ `P[i] <= P[j] - k` for shortest `j - i`.
* Maintain a **Monotonic Increasing Deque of Prefix Sum indices**:
* 1) While `P[j] - P[deque[0]] >= k`: update `min_len = min(min_len, j - deque.popleft())` (no future `j' > j` can beat this length with that `i`).
* 2) While `P[j] <= P[deque[-1]]`: `deque.pop()` (a larger prefix sum at a smaller index is strictly worse).

---

### Core Technique: Monotonic Increasing Deque on Prefix Sums

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Prefix sum array of size n + 1
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
            
        q = deque()  # stores indices of prefix, prefix values are increasing
        min_len = float('inf')
        
        for j in range(n + 1):
            # Check if valid window found with leftmost candidate
            while q and prefix[j] - prefix[q[0]] >= k:
                min_len = min(min_len, j - q.popleft())
                
            # Maintain monotonic increasing order
            while q and prefix[j] <= prefix[q[-1]]:
                q.pop()
                
            q.append(j)
            
        return min_len if min_len != float('inf') else -1
```

---

### Worked-Out Example

```python
nums = [2, -1, 2], k = 3
prefix = [0, 2, 1, 3]
j=0 (0): q=[0]
j=1 (2): prefix[1]-prefix[0]=2 < 3, q=[0, 1]
j=2 (1): prefix[2] (1) <= prefix[1] (2) -> pop 1! q=[0, 2]
j=3 (3): prefix[3]-prefix[0]=3 >= 3 -> min_len = min(inf, 3-0) = 3, pop 0
         prefix[3]-prefix[2]=2 < 3 -> q=[2, 3]
Result = 3
```

---

### Complexity Analysis

* **Time Complexity:** `O(n) (each index pushed and popped at most once)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

When negative values break sliding window monotonicity, maintain a Monotonic Deque over Prefix Sums.