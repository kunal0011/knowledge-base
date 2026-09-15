---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 371: Sum of Two Integers"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 371: Sum of Two Integers

---

### Problem Statement

Given two integers `a` and `b`, return the sum of the two integers without using the operators `+` and `-`.

---

### Key Observation

* Bitwise XOR (`a ^ b`) performs addition without carry.
* Bitwise AND shifted left (`(a & b) << 1`) computes the carry.
* Repeat until carry is 0. In Python, apply a 32-bit mask (`0xFFFFFFFF`) to handle negative two's complement.

---

### Core Technique: Half Adder Circuit via XOR & AND

---

### Python 3 Solution (with typing)

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        while (b & mask) > 0:
            carry = (a & b) << 1
            a = a ^ b
            b = carry
        return (a & mask) if b > 0 else a
```

---

### Worked-Out Example

```python
a = 1 (0001), b = 2 (0010)
carry = (0001 & 0010) << 1 = 0
a = 0001 ^ 0010 = 0011 (3)
carry == 0 -> return 3
```

---

### Complexity Analysis

* **Time Complexity:** `O(1) (at most 32 iterations for 32-bit integers)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Addition is composed of sum `a ^ b` and carry `(a & b) << 1`.