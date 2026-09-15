---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 136: Single Number"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 136: Single Number

---

### Problem Statement

Given a non-empty array of integers `nums` where every element appears twice except for one, find that single one in linear time and `O(1)` space.

---

### Key Observation

* XOR properties: `x ^ x = 0` and `x ^ 0 = x`.
* XOR is commutative and associative: XORing all elements cancels out pairs and leaves the unique number.

---

### Core Technique: XOR Cancellation

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
```

---

### Worked-Out Example

```text
nums = [4, 1, 2, 1, 2]
res = 0 ^ 4 ^ 1 ^ 2 ^ 1 ^ 2
res = 4 ^ (1 ^ 1) ^ (2 ^ 2) = 4 ^ 0 ^ 0 = 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

XOR identical elements to reduce them to 0 in O(1) space.