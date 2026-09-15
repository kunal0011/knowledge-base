---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 201: Bitwise AND of Numbers Range"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 201: Bitwise AND of Numbers Range

---

### Problem Statement

Given two integers `left` and `right` that represent the range `[left, right]`, return the bitwise AND of all numbers in this range, inclusive.

---

### Key Observation

* As numbers increment, lower bits alternate between 0 and 1, so the bitwise AND of any changing bit position becomes `0`.
* The result is simply the **common binary prefix** of `left` and `right`, padded with zeros.
* Right shift both numbers until `left == right`, then shift back.

---

### Core Technique: Common Binary Prefix Alignment

---

### Python 3 Solution (with typing)

```python
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        shifts = 0
        while left < right:
            left >>= 1
            right >>= 1
            shifts += 1
        return left << shifts
```

---

### Worked-Out Example

```python
left = 5 (0101), right = 7 (0111)
shift 1: left = 2 (0010), right = 3 (0011)
shift 2: left = 1 (0001), right = 1 (0001) -> equal!
result = 1 << 2 = 4 (0100)
```

---

### Complexity Analysis

* **Time Complexity:** `O(1) (at most 32 shifts)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Range bitwise AND always equals the common binary prefix of the endpoints.