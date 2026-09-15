---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 238: Product of Array Except Self"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 238: Product of Array Except Self

---

### Problem Statement

Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all elements of `nums` except `nums[i]`. Division is not permitted.

---

### Key Observation

* The product for index `i` is `(prefix_product_before_i) * (suffix_product_after_i)`.
* We can compute prefixes on a left pass, then accumulate suffixes on a right pass in `O(1)` auxiliary space.

---

### Core Technique: Prefix & Suffix Accumulation

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        
        # Left pass: prefix products
        prefix = 1
        for i in range(n):
            res[i] = prefix
            prefix *= nums[i]
            
        # Right pass: suffix products
        suffix = 1
        for i in range(n - 1, -1, -1):
            res[i] *= suffix
            suffix *= nums[i]
            
        return res
```

---

### Worked-Out Example

```text
nums = [1, 2, 3, 4]
After prefix pass: res = [1, 1, 2, 6]
Suffix pass:
  i = 3: res[3] = 6 * 1 = 6, suffix = 4
  i = 2: res[2] = 2 * 4 = 8, suffix = 12
  i = 1: res[1] = 1 * 12 = 12, suffix = 24
  i = 0: res[0] = 1 * 24 = 24
Result = [24, 12, 8, 6]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1) auxiliary (output array excluded)`

---

### Takeaway Pattern

When excluded element products are needed without division, split the problem into prefix and suffix passes.