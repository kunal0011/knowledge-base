---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 704: Binary Search"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 704: Binary Search

---

### Problem Statement

Given an array of integers `nums` sorted in ascending order, and an integer `target`, write a function to search `target` in `nums` in `O(log n)` time.

---

### Key Observation

* Since array is sorted, comparing target with midpoint halves the search space at each step.
* Use `mid = left + (right - left) // 2` to prevent integer overflow.

---

### Core Technique: Standard Closed-Interval Binary Search

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
```

---

### Worked-Out Example

```python
nums = [-1, 0, 3, 5, 9, 12], target = 9
left = 0, right = 5 -> mid = 2 (nums[2] = 3 < 9) -> left = 3
left = 3, right = 5 -> mid = 4 (nums[4] = 9 == target) -> return 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(log n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

The foundational template for all logarithmic search algorithms.