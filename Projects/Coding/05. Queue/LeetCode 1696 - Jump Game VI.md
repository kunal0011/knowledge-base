---
date: "2026-08-29"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1696: Jump Game VI"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 1696: Jump Game VI

**Target Companies:** Amazon, Google, Microsoft

---

### Problem Statement

You are given a 0-indexed integer array `nums` and an integer `k`. You can jump at most `k` steps forward. Return the maximum score you can get to reach the last index.

---

### Key Observation

* DP recurrence: `dp[i] = nums[i] + max(dp[i - k ... i - 1])`.
* A naive linear search for the max in the last `k` states gives `O(n * k)` (TLE).
* Use a **Monotonic Decreasing Deque** storing indices of `dp` values to maintain the sliding window maximum of the past `k` states in `O(1)` amortized time.

---

### Core Technique: Dynamic Programming with Monotonic Deque Window Optimization

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        
        q = deque([0])  # stores indices, dp values are strictly decreasing
        
        for i in range(1, n):
            # 1. Evict elements outside k-step jump window
            if q[0] < i - k:
                q.popleft()
                
            # 2. State transition: dp[i] = nums[i] + max(dp[j])
            dp[i] = nums[i] + dp[q[0]]
            
            # 3. Maintain monotonic decreasing order in deque
            while q and dp[q[-1]] <= dp[i]:
                q.pop()
            q.append(i)
            
        return dp[-1]
```

---

### Worked-Out Example

```python
nums = [1, -1, -2, 4, -7, 3], k = 2
dp[0] = 1, q = [0]
i=1 (num=-1): dp[1] = -1 + dp[0] = 0, q = [0 (dp=1), 1 (dp=0)]
i=2 (num=-2): dp[2] = -2 + dp[0] = -1, q = [1 (dp=0), 2 (dp=-1)] (0 expired)
i=3 (num=4): dp[3] = 4 + dp[1] = 4, pop 2 and 1 -> q = [3 (dp=4)]
...
Final score = 7
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Optimize 1D DP window transitions from O(n\*k) to optimal O(n) using a Monotonic Deque.