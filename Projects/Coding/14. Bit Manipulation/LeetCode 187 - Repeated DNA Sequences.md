---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 187: Repeated DNA Sequences"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 187: Repeated DNA Sequences

**Target Companies:** Google, Amazon, LinkedIn

---

### Problem Statement

Given a DNA string `s` (composed of 'A', 'C', 'G', 'T'), return all 10-letter-long sequences that occur more than once.

---

### Key Observation

* A 10-character string takes 10 bytes as string.
* There are only 4 DNA characters ('A': 00, 'C': 01, 'G': 10, 'T': 11) $
  ightarrow$ Each character needs only 2 bits!
* A 10-letter window fits in **20 bits** (single 32-bit integer).
* Maintain a rolling bitmask: shift left by 2, bitwise-OR new 2-bit char, and mask with `0xFFFFF` (20 set bits).

---

### Core Technique: 2-Bit Rolling Bitmask Hash

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        if len(s) < 10:
            return []
            
        to_2bit = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        mask_20bit = 0xFFFFF  # (1 << 20) - 1
        
        seen = set()
        res = set()
        
        # Build first 10-char bitmask
        curr_mask = 0
        for i in range(10):
            curr_mask = (curr_mask << 2) | to_2bit[s[i]]
        seen.add(curr_mask)
        
        # Rolling bitmask window
        for i in range(10, len(s)):
            curr_mask = ((curr_mask << 2) & mask_20bit) | to_2bit[s[i]]
            if curr_mask in seen:
                res.add(s[i - 9:i + 1])
            else:
                seen.add(curr_mask)
                
        return list(res)
```

---

### Worked-Out Example

```python
s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
10-char window encoded into integer:
"AAAAACCCCC" -> 00 00 00 00 00 01 01 01 01 01_2
When seen again, add string to result set.
Output: ["AAAAACCCCC", "CCCCCAAAAA"]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n) with 4x memory savings using integers instead of strings`

---

### Takeaway Pattern

Encode 4-symbol alphabets with 2 bits per char to perform O(1) bitwise rolling window hashes.