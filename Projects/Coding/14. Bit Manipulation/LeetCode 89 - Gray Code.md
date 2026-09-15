---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 89: Gray Code"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 89: Gray Code

---

### Problem Statement

An n-bit Gray code sequence is a sequence of `2^n` integers where every adjacent pair differs by exactly one bit. Given `n`, return any valid n-bit Gray code sequence.

---

### Key Observation

* The standard binary to Gray code formula is: `gray(i) = i ^ (i >> 1)`.
* Iterating `i` from `0` to `2^n - 1` and applying the formula directly produces a valid sequence.

---

### Core Technique: Direct Mathematical Bitwise Gray Transformation

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def grayCode(self, n: int) -> List[int]:
        return [i ^ (i >> 1) for i in range(1 << n)]
```

---

### Worked-Out Example

```
n = 2 (range 0..3)
i = 0 (00): 0 ^ 0 = 0 (00)
i = 1 (01): 1 ^ 0 = 1 (01)
i = 2 (10): 2 ^ 1 = 3 (11)
i = 3 (11): 3 ^ 1 = 2 (10)
Result = [0, 1, 3, 2]
```

---

### Complexity Analysis

* **Time Complexity:** `O(2^n)`
* **Space Complexity:** `O(1) auxiliary (excluding return array)`

---

### Takeaway Pattern

Binary to Gray code mapping is directly computed via `i ^ (i >> 1)`.