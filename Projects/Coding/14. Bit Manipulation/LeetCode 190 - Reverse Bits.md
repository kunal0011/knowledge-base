---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 190: Reverse Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 190: Reverse Bits

---

### Problem Statement

Reverse bits of a given 32 bits unsigned integer.

---

### Key Observation

* Extract the lowest bit of `n` using `n & 1`.
* Shift the result left by 1 and append the extracted bit.
* Shift `n` right by 1. Repeat exactly 32 times.

---

### Core Technique: Bit Shift & Accumulation

---

### Python 3 Solution (with typing)

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res
```

---

### Worked-Out Example

```
n = 00000010100101000001111010011100
After 32 shifts, res = 00111001011110000010100101000000 (964176192)
```

---

### Complexity Analysis

* **Time Complexity:** `O(1) (fixed 32 iterations)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Extract bits with `& 1` and assemble the reversed integer with `(res << 1) | bit`.