---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 41: First Missing Positive"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 41: First Missing Positive

---

### Problem Statement

Given an unsorted integer array `nums`, return the smallest missing positive integer in `O(n)` time and `O(1)` auxiliary space.

---

### Key Observation

* The answer must be in the range `[1, n + 1]`.
* Use the array itself as a Hash Table: place every positive number `x` (where `1 <= x <= n`) at its correct index `x - 1` via in-place swapping (Cyclic Sort).
* The first index `i` where `nums[i] != i + 1` gives the missing positive `i + 1`.

---

### Core Technique: In-Place Cyclic Indexing

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                correct_idx = nums[i] - 1
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
                
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1
```

---

### Worked-Out Example

```python
nums = [3, 4, -1, 1] (n = 4)
i = 0: nums[0] = 3 -> swap with index 2 -> [-1, 4, 3, 1]
i = 1: nums[1] = 4 -> swap with index 3 -> [-1, 1, 3, 4] -> swap nums[1] (1) with index 0 -> [1, -1, 3, 4]
Scan:
nums[0] == 1 (OK)
nums[1] == -1 != 2 -> missing positive is 2!
```

---

### Complexity Analysis

* **Time Complexity:** `O(n) (each number is placed in its correct position at most once)`
* **Space Complexity:** `O(1) auxiliary in-place`

---

### Takeaway Pattern

When O(n) time and O(1) space are strictly required on range 1..n, use array indices as in-place hash buckets.