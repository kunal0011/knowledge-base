---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 393: UTF-8 Validation"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 393: UTF-8 Validation

**Target Companies:** Google (Top Asked Classic), Amazon

---

### Problem Statement

Given an integer array `data` representing bytes, return `true` if it represents a valid UTF-8 encoding.

---

### Key Observation

* 1-byte char: `0xxxxxxx` (starts with 0).
* 2-byte char: `110xxxxx 10xxxxxx` (starts with 110).
* 3-byte char: `1110xxxx 10xxxxxx 10xxxxxx` (starts with 1110).
* 4-byte char: `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` (starts with 11110).
* Continuation bytes must start with `10xxxxxx` (`byte >> 6 == 0b10`).

---

### Core Technique: Bitmask Byte Header Inspection

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        remaining_bytes = 0
        
        for num in data:
            byte = num & 0xFF  # only consider lower 8 bits
            
            if remaining_bytes == 0:
                if (byte >> 7) == 0b0:
                    remaining_bytes = 0
                elif (byte >> 5) == 0b110:
                    remaining_bytes = 1
                elif (byte >> 4) == 0b1110:
                    remaining_bytes = 2
                elif (byte >> 3) == 0b11110:
                    remaining_bytes = 3
                else:
                    return False
            else:
                if (byte >> 6) != 0b10:
                    return False
                remaining_bytes -= 1
                
        return remaining_bytes == 0
```

---

### Worked-Out Example

```python
data = [197, 130, 1] (binary: [11000101, 10000010, 00000001])
byte 1 (197): starts with 110 -> 2-byte char, expect 1 continuation byte
byte 2 (130): starts with 10 -> valid continuation byte! remaining = 0
byte 3 (1): starts with 0 -> valid 1-byte char!
All bytes consumed -> return True
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Use bit-shifts `byte >> k` to directly inspect byte header prefixes against binary masks.