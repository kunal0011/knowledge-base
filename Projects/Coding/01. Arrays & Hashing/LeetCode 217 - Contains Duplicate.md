---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 217: Contains Duplicate"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 217: Contains Duplicate

---

### Problem Statement

Given an integer array `nums`, return `true` if any value appears at least twice in the array, and return `false` if every element is distinct.

---

### Key Observation

* A Hash Set allows checking element presence in `O(1)` average time.
* If `len(nums) != len(set(nums))`, duplicate exists.

---

### Core Technique: Hash Set Membership

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
```

---

### Worked-Out Example

```python
nums = [1, 2, 3, 1]
seen = {}
num = 1 -> add to seen -> {1}
num = 2 -> add to seen -> {1, 2}
num = 3 -> add to seen -> {1, 2, 3}
num = 1 -> 1 is in seen! -> return True
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Use a Hash Set for instant duplicate detection.