---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 268: Missing Number"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 268: Missing Number

---

### Problem Statement

Given an array `nums` containing `n` distinct numbers in the range `[0, n]`, return the only number in the range that is missing from the array.

---

### Key Observation

* XOR all indices `0..n` with all values in `nums`.
* Every number present in the array will cancel out with its corresponding index, leaving the missing number.

---

### Core Technique: XOR Index Value Cancellation

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        res = len(nums)
        for i, num in enumerate(nums):
            res ^= (i ^ num)
        return res
```

---

### Worked-Out Example

```text
nums = [3, 0, 1] (n = 3)
res = 3 ^ (0 ^ 3) ^ (1 ^ 0) ^ (2 ^ 1)
res = (3 ^ 3) ^ (0 ^ 0) ^ (1 ^ 1) ^ 2 = 0 ^ 0 ^ 0 ^ 2 = 2
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

XOR index range with array elements to find missing values without arithmetic overflow risks.