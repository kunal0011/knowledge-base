---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 300: Longest Increasing Subsequence"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 300: Longest Increasing Subsequence

**Target Companies:** Google, Amazon, Meta

---

### Problem Statement

Given an integer array `nums`, return the length of the longest strictly increasing subsequence in `O(n log n)` time.

---

### Key Observation

* Standard DP is `O(n^2)`. To achieve `O(n log n)`, use **Patience Sorting**.
* Maintain a list `tails` where `tails[i]` stores the smallest tail of all increasing subsequences of length `i + 1`.
* For each `x` in `nums`, use Binary Search (`bisect_left`) to find the first element in `tails >= x` and replace it. If no such element exists, append `x`.

---

### Core Technique: Patience Sorting with Binary Search (bisect\_left)

---

### Python 3 Solution (with typing)

```python
import bisect
from typing import List

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        tails = []
        for x in nums:
            idx = bisect.bisect_left(tails, x)
            if idx == len(tails):
                tails.append(x)
            else:
                tails[idx] = x
        return len(tails)
```

---

### Worked-Out Example

```python
nums = [10, 9, 2, 5, 3, 7, 101, 18]
10 -> tails = [10]
9  -> tails = [9]
2  -> tails = [2]
5  -> tails = [2, 5]
3  -> tails = [2, 3]
7  -> tails = [2, 3, 7]
101-> tails = [2, 3, 7, 101]
18 -> tails = [2, 3, 7, 18]
Result length = len(tails) = 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(n log n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Tails array maintains minimum possible ending elements for each subsequence length, naturally ordered for binary search.