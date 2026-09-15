---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 1: Two Sum"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 1: Two Sum

---

### Problem Statement

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

You may assume that each input would have **exactly one solution**, and you may not use the same element twice.

---

### Key Observation

* For each element `x`, we need to check if `target - x` already exists in the elements we have seen so far.
* Using a Hash Map (dictionary), we can look up complements in **O(1)** time instead of O(n) scan.

---

### Core Technique: One-Pass Hash Map

Iterate through the array once. For each number, calculate complement `target - num`. If found in the map, return indices. Otherwise, store current number with its index.

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in seen:
                return [seen[complement], i]
            seen[num] = i
        return []
```

---

### Worked-Out Example

#### Example 1

```python
nums = [2, 7, 11, 15], target = 9

i = 0, num = 2: complement = 7 (not in seen) -> seen = {2: 0}
i = 1, num = 7: complement = 2 (in seen! index 0) -> return [0, 1]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)` (single pass with O(1) hash map operations)
* **Space Complexity:** `O(n)` (hash map storing up to n elements)

---

### Takeaway Pattern

Whenever you need to find pairs with a given sum or difference, think **Hash Map Complement Lookup**.