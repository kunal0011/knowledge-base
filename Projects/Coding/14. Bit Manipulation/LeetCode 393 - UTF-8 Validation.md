---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 393: UTF-8 Validation"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - utf8
  - state-machine
  - google
  - amazon
---

# LeetCode 393: UTF-8 Validation

**Target Companies:** Google (Top Signature Systems Question), Amazon, Apple, Meta  
**Difficulty:** Medium  
**Topic:** Byte Header Bitwise Masking & UTF-8 Encoding State Validation  

---

### Problem Statement

Given an integer array `data` representing the data, return whether it is a valid **UTF-8** encoding (i.e. it translates to a sequence of valid UTF-8 encoded characters).

A character in UTF-8 can be from **1 to 4 bytes** long, subjected to the following rules:
1. For a **1-byte** character, the first bit is a `0`, followed by its Unicode code point.
2. For an **$n$-bytes** character, the first $n$ bits are all one's, the $n+1$ bit is `0`, followed by $n-1$ bytes with the most significant $2$ bits being `10`.

This is how the UTF-8 encoding would work:
```
Number of Bytes | UTF-8 Octet Sequence (binary)
   1            | 0xxxxxxx
   2            | 110xxxxx 10xxxxxx
   3            | 1110xxxx 10xxxxxx 10xxxxxx
   4            | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```
*Note: The input is an array of integers. Only the **least significant 8 bits** of each integer is used to store the data.*

---

### Input & Output Formats & Constraints

- **Input:** `data: List[int]`
- **Output:** `bool` (`True` if valid UTF-8, `False` otherwise)
- **Constraints:**
  - $1 \le \text{data.length} \le 2 \times 10^4$
  - $0 \le \text{data}[i] \le 255$

---

### Key Idea & Intuition

We can model this process as a **finite state machine** tracking `remaining_bytes`:
- `remaining_bytes == 0`: We are expecting a **leading byte** that initiates a new character:
  - `byte >> 7 == 0b0`: 1-byte character $\to$ `remaining_bytes = 0`.
  - `byte >> 5 == 0b110`: 2-byte character $\to$ expects 1 continuation byte $\to$ `remaining_bytes = 1`.
  - `byte >> 4 == 0b1110`: 3-byte character $\to$ expects 2 continuation bytes $\to$ `remaining_bytes = 2`.
  - `byte >> 3 == 0b11110`: 4-byte character $\to$ expects 3 continuation bytes $\to$ `remaining_bytes = 3`.
  - Any other leading prefix (e.g., `10xxxxxx` or `11111xxx`) is **invalid** $\to$ return `False`.
- `remaining_bytes > 0`: We are inside a multi-byte character and expect a **continuation byte**:
  - The top 2 bits must be `10`: `(byte >> 6) == 0b10`.
  - If valid, decrement `remaining_bytes -= 1`.
  - If invalid, return `False`.
- **Termination Invariant:** After all bytes are processed, `remaining_bytes == 0` must hold (no incomplete character left trailing).

---

### Solution Approach (Step-by-Step)

1. Initialize `remaining_bytes = 0`.
2. For each `num` in `data`:
   - Mask to lower 8 bits: `byte = num & 0xFF`.
   - If `remaining_bytes == 0`:
     - If `(byte >> 7) == 0`: continue (1-byte char).
     - Else if `(byte >> 5) == 0b110`: `remaining_bytes = 1`.
     - Else if `(byte >> 4) == 0b1110`: `remaining_bytes = 2`.
     - Else if `(byte >> 3) == 0b11110`: `remaining_bytes = 3`.
     - Else: return `False` (invalid leading byte).
   - Else:
     - If `(byte >> 6) != 0b10`: return `False` (invalid continuation byte).
     - `remaining_bytes -= 1`.
3. Return `remaining_bytes == 0`.

---

### Visual Algorithm Walkthrough

```
Input: data = [197, 130, 1]

Byte 1: 197 -> Binary: 11000101
  remaining_bytes = 0
  197 >> 5 = 11000101 >> 5 = 00000110_2 == 0b110!
  Valid 2-byte header!
  Expect 1 continuation byte: remaining_bytes = 1.

Byte 2: 130 -> Binary: 10000010
  remaining_bytes = 1 (expecting continuation byte)
  130 >> 6 = 10000010 >> 6 = 00000010_2 == 0b10!
  Valid continuation byte!
  remaining_bytes = 1 - 1 = 0.

Byte 3: 1 -> Binary: 00000001
  remaining_bytes = 0
  1 >> 7 = 0b0!
  Valid 1-byte character!
  remaining_bytes = 0.

Loop finished. remaining_bytes == 0? YES!
Return True.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Valid 2-Byte Character + 1-Byte Character
- **Input:** `data = [197, 130, 1]`
- **Binary:** `[11000101, 10000010, 00000001]`
- **Output:** `true`

#### Example 2: Invalid Continuation Byte
- **Input:** `data = [235, 140, 4]`
- **Binary:** `[11101011, 10001100, 00000100]`
- **Trace:**
  - `235` starts with `1110` (3-byte char $\to$ needs two `10` continuation bytes).
  - First continuation byte `140` starts with `10` (valid).
  - Second byte `4` is `00000100` (starts with `00`, NOT `10`!).
  - Invalid!
- **Output:** `false`

#### Example 3: Incomplete Multi-Byte Character at End of Data
- **Input:** `data = [235, 140]`
- **Trace:** 3-byte character lacks its final byte; loop terminates with `remaining_bytes = 1 != 0`.
- **Output:** `false`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        remaining_bytes = 0
        
        for num in data:
            byte = num & 0xFF  # Only consider least significant 8 bits
            
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

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    bool validUtf8(std::vector<int>& data) {
        int remainingBytes = 0;

        for (int num : data) {
            int byte = num & 0xFF;

            if (remainingBytes == 0) {
                if ((byte >> 7) == 0) {
                    remainingBytes = 0;
                } else if ((byte >> 5) == 0b110) {
                    remainingBytes = 1;
                } else if ((byte >> 4) == 0b1110) {
                    remainingBytes = 2;
                } else if ((byte >> 3) == 0b11110) {
                    remainingBytes = 3;
                } else {
                    return false;
                }
            } else {
                if ((byte >> 6) != 0b10) {
                    return false;
                }
                remainingBytes--;
            }
        }

        return remainingBytes == 0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean validUtf8(int[] data) {
        int remainingBytes = 0;

        for (int num : data) {
            int b = num & 0xFF;

            if (remainingBytes == 0) {
                if ((b >> 7) == 0) {
                    remainingBytes = 0;
                } else if ((b >> 5) == 0b110) {
                    remainingBytes = 1;
                } else if ((b >> 4) == 0b1110) {
                    remainingBytes = 2;
                } else if ((b >> 3) == 0b11110) {
                    remainingBytes = 3;
                } else {
                    return false;
                }
            } else {
                if ((b >> 6) != 0b10) {
                    return false;
                }
                remainingBytes--;
            }
        }

        return remainingBytes == 0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Single pass over `data` with constant-time bitwise shifts and comparisons per byte.
- **Space Complexity:** $O(1)$ auxiliary space — Only a single integer state variable `remaining_bytes`.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Network Protocol / Binary Format Header Parsing with State Machine.
- **Trap:** Forgetting to check `remaining_bytes == 0` after the loop: if the input ends abruptly before a multi-byte sequence completes, returning `True` would be incorrect.
- **Trap:** 5-byte sequences: The UTF-8 specification restricts code points to at most 4 bytes. A header starting with `111110xx` is explicitly invalid.