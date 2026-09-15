---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 191: Number of 1 Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 191: Number of 1 Bits

---

### Problem Statement

Write a function that takes the binary representation of a positive integer and returns the number of set bits ('1's) it has (Hamming weight).

---

### Key Observation

* The operation `n & (n - 1)` flips the lowest set bit of `n` to `0`.
* Looping until `n == 0` executes in iterations equal to the number of set bits.

---

### Core Technique: Brian Kernighan's Algorithm

---

### Python 3 Solution (with typing)

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n > 0:
            n &= (n - 1)
            count += 1
        return count
```

---

### Worked-Out Example

```python
n = 11 (binary 1011)
iter 1: 1011 & 1010 = 1010 (count = 1)
iter 2: 1010 & 1001 = 1000 (count = 2)
iter 3: 1000 & 0111 = 0000 (count = 3)
n == 0 -> return 3
```

---

### Complexity Analysis

* **Time Complexity:** `O(k) where k is number of set bits (at most 32)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

`n & (n - 1)` clears the lowest set bit in O(1) bitwise operations.