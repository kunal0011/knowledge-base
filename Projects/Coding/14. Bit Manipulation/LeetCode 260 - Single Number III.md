---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 260: Single Number III"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 260: Single Number III

---

### Problem Statement

Given an integer array `nums`, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements in linear time and `O(1)` space.

---

### Key Observation

* XORing all elements yields `x = a ^ b` where `a` and `b` are the unique numbers.
* Find any set bit in `x` using `diff = x & (-x)` (lowest set bit).
* This bit is 1 in either `a` or `b`, but not both!
* Partition array into two groups based on this bit and XOR separately.

---

### Core Technique: Lowest Set Bit Partitioning (Two-Group XOR)

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xor_all = 0
        for num in nums:
            xor_all ^= num
            
        # Isolate lowest set bit
        diff_bit = xor_all & (-xor_all)
        
        a = 0
        b = 0
        for num in nums:
            if num & diff_bit:
                a ^= num
            else:
                b ^= num
                
        return [a, b]
```

---

### Worked-Out Example

```python
nums = [1, 2, 1, 3, 2, 5]
xor_all = 3 ^ 5 = (011 ^ 101) = 110 (binary 6)
diff_bit = 110 & (-110) = 010 (binary 2)
Group 1 (bit 1 set): [2, 2, 3] -> XOR = 3
Group 2 (bit 1 zero): [1, 1, 5] -> XOR = 5
Result = [3, 5]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Use `x & (-x)` to isolate differing bits and partition twin-element XOR spaces.