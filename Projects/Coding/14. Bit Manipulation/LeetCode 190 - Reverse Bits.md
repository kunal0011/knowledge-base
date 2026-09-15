---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 190: Reverse Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - bit-shifts
  - amazon
  - google
---

# LeetCode 190: Reverse Bits

**Target Companies:** Google, Amazon, Apple, Microsoft  
**Difficulty:** Easy  
**Topic:** 32-Bit Reversal via Serial Shifting & Divide-and-Conquer Byte Swapping  

---

### Problem Statement

Reverse bits of a given 32 bits unsigned integer.

**Follow up:** If this function is called many times, how would you optimize it?

---

### Input & Output Formats & Constraints

- **Input:** `n: int` (32-bit unsigned integer)
- **Output:** `int` (32-bit unsigned integer with reversed bits)
- **Constraints:**
  - The input must be a binary string of length $32$ or an unsigned 32-bit integer.

---

### Key Idea & Intuition

#### Method 1: Serial 32-Bit Squeeze ($O(1)$ Time)
1. Initialize `res = 0`.
2. Iterate exactly 32 times:
   - Shift `res` left by 1: `res <<= 1`.
   - Extract the lowest bit of `n`: `n & 1`.
   - Insert it into the vacated position of `res`: `res |= (n & 1)`.
   - Shift `n` right by 1: `n >>= 1` (or `n >>>= 1` in Java to prevent sign-bit extension).
3. After 32 iterations, the least significant bit of `n` has been shifted to the most significant bit of `res`.

#### Method 2: Divide and Conquer / Byte Cache (Optimal for Many Calls)
If called billions of times, serial loop iteration is slow. We can reverse bits using hierarchical block swapping (similar to Merge Sort):
1. Swap adjacent 16-bit halves: `n = (n >> 16) | (n << 16)`.
2. Swap adjacent 8-bit bytes: `n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8)`.
3. Swap adjacent 4-bit nibbles: `n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4)`.
4. Swap adjacent 2-bit pairs: `n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2)`.
5. Swap adjacent 1-bit pairs: `n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1)`.
This executes in just **5 parallel bitwise operations** without any branching or loops!

---

### Solution Approach (Step-by-Step)

1. Initialize `res = 0`.
2. Loop 32 times:
   - `res = (res << 1) | (n & 1)`
   - `n >>= 1` (in Java, use `n >>>= 1`)
3. Return `res`.

---

### Visual Algorithm Walkthrough

```
Input n (8-bit simplified demonstration):
  n = 00010110_2, res = 00000000_2

Iter 1: n & 1 = 0 -> res = 00000000, n = 00001011
Iter 2: n & 1 = 1 -> res = 00000001, n = 00000101
Iter 3: n & 1 = 1 -> res = 00000011, n = 00000010
Iter 4: n & 1 = 0 -> res = 00000110, n = 00000001
Iter 5: n & 1 = 1 -> res = 00001101, n = 00000000
Iter 6: n & 1 = 0 -> res = 00011010, n = 00000000
Iter 7: n & 1 = 0 -> res = 00110100, n = 00000000
Iter 8: n & 1 = 0 -> res = 01101000, n = 00000000

Result: 01101000_2 (exact mirror reversal of 00010110_2).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Binary Reversal
- **Input:** `00000010100101000001111010011100` (43261596)
- **Output:** `00111001011110000010100101000000` (964176192)

#### Example 2: Terminal Bit Set
- **Input:** `11111111111111111111111111111101`
- **Output:** `10111111111111111111111111111111`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <cstdint>

class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t res = 0;
        for (int i = 0; i < 32; ++i) {
            res = (res << 1) | (n & 1);
            n >>= 1;
        }
        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
public class Solution {
    // you need treat n as an unsigned value
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res = (res << 1) | (n & 1);
            n >>>= 1; // Logical unsigned right shift
        }
        return res;
    }
}
```

#### 4. Divide-and-Conquer 5-Step Optimization (C++)
```cpp
class SolutionOptimized {
public:
    uint32_t reverseBits(uint32_t n) {
        n = (n >> 16) | (n << 16);
        n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8);
        n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4);
        n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2);
        n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1);
        return n;
    }
};
```

---

### Complexity Analysis

- **Time Complexity:** $O(1)$ — Strictly 32 iterations (or 5 operations in the divide-and-conquer version).
- **Space Complexity:** $O(1)$ auxiliary space.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Serial Bit Extraction with Accumulator Left Shifting.
- **Trap:** Java Signed Arithmetic Shift (`>>` vs `>>>`): In Java, `>>` preserves the sign bit (filling with 1s if negative). You must use the unsigned logical right shift `>>>= 1` to fill vacated bits with zeroes.
