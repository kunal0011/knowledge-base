---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 15: 3Sum"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 15: 3Sum

---

### Problem Statement

Given an integer array nums, return all triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0` without duplicate triplets.

---

### Key Observation

* Sort the array in `O(n log n)`.
* Fix the first element `nums[i]`, and reduce the remaining problem to Two Pointers (Left and Right) looking for `-nums[i]`.
* Skip duplicate elements for both the outer loop and the two pointers to prevent identical triplets.

---

### Core Technique: Sorting + Two Pointers with Duplicate Skipping

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        n = len(nums)
        
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue  # skip duplicate fixed element
            if nums[i] > 0:
                break     # sum cannot be 0 if smallest number > 0
                
            left, right = i + 1, n - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1  # skip duplicate left
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1 # skip duplicate right
                    left += 1
                    right -= 1
        return res
```

---

### Worked-Out Example

```python
nums = [-1, 0, 1, 2, -1, -4] -> sorted: [-4, -1, -1, 0, 1, 2]
i = 1, nums[1] = -1:
  left = 2 (nums[2] = -1), right = 5 (nums[5] = 2)
  total = -1 + (-1) + 2 = 0 -> triplet: [-1, -1, 2]
  left = 3 (0), right = 4 (1)
  total = -1 + 0 + 1 = 0 -> triplet: [-1, 0, 1]
Result: [[-1, -1, 2], [-1, 0, 1]]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n^2)`
* **Space Complexity:** `O(1) auxiliary (or O(n) sorting space)`

---

### Takeaway Pattern

Reduce 3Sum to 2Sum on sorted array using a fixed anchor and two converging pointers.