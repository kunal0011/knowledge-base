---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 201: Bitwise AND of Numbers Range"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - common-prefix
  - brian-kernighan
  - amazon
  - google
---

# LeetCode 201: Bitwise AND of Numbers Range

**Target Companies:** Google, Amazon, Microsoft, ByteDance  
**Difficulty:** Medium  
**Topic:** Common Binary Prefix Alignment & Rightmost Set Bit Stripping  

---

### Problem Statement

Given two integers `left` and `right` that represent the range $[left, right]$, return the bitwise AND of all numbers in this range, inclusive.

---

### Input & Output Formats & Constraints

- **Input:** `left: int`, `right: int`
- **Output:** `int` (bitwise AND of all integers in $[left, right]$)
- **Constraints:**
  - $0 \le \text{left} \le \text{right} \le 2^{31} - 1$

---

### Key Idea & Intuition

A naive iteration across all numbers from `left` to `right` takes $O(\text{right} - \text{left})$ time, which triggers a TLE when range is $2^{31} - 1 \approx 2 \times 10^9$.

Notice what happens during bitwise AND across a continuous range:
- For any bit position that changes between `0` and `1` even once within $[left, right]$, the cumulative AND at that position **must evaluate to `0`** (since $x \ \& \ 0 = 0$).
- Which bits remain `1`? Only the bits that **never change** across the entire range!
- The only bits that never change from `left` to `right` are their **longest common binary prefix**.
- All bits after the common prefix toggle between 0 and 1 at least once as the numbers increment, and are therefore wiped to `0`.

#### Two Clean Implementation Strategies:
1. **Bit Shift Method:**
   Shift both `left` and `right` to the right until they become identical (`left == right`), tracking the number of `shifts`. Then shift back: `left << shifts`.
2. **Brian Kernighan's Clear Method:**
   Since any 1-bit in `right` that is greater than `left` will inevitably be turned to `0` by some intermediate number, we repeatedly clear the lowest set bit of `right`:
   $$\text{while } right > left: \quad right \ \&= (right - 1)$$
   When $right \le left$, `right` has been reduced to the exact common prefix!

---

### Solution Approach (Step-by-Step)

#### Approach 1 (Common Prefix Shifting):
1. Initialize `shifts = 0`.
2. While `left < right`:
   - `left >>= 1`
   - `right >>= 1`
   - `shifts += 1`
3. Return `left << shifts`.

#### Approach 2 (Brian Kernighan Stripping):
1. While `right > left`:
   - `right &= (right - 1)`
2. Return `right`.

---

### Visual Algorithm Walkthrough

```
Range: left = 9 (1001_2), right = 12 (1100_2)

Binary Representations:
  9:  1 0 0 1
 10:  1 0 1 0
 11:  1 0 1 1
 12:  1 1 0 0
-----------------
AND:  1 0 0 0  (8)

Notice:
- Bit 3 (MSB): stays '1' for all numbers -> PRESERVED!
- Bits 2, 1, 0: change between 0 and 1 -> ZEROED OUT!
Common Prefix: '1 0 0 0' -> 8.

Brian Kernighan trace:
  right = 12 (1100) > left = 9 (1001)
  right = 12 & 11 = 1100 & 1011 = 1000 (8)
  right = 8 <= left = 9 -> STOP!
  Return right = 8.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Consecutive Span
- **Input:** `left = 5`, `right = 7`
- **Binary:** $5 = 101_2$, $6 = 110_2$, $7 = 111_2$.
- **Common Prefix:** `100_2` ($4$).
- **Output:** `4`

#### Example 2: Identical Left and Right
- **Input:** `left = 0`, `right = 0`
- **Output:** `0`

#### Example 3: Full 32-Bit Range
- **Input:** `left = 1`, `right = 2147483647`
- **Trace:** Spans across powers of 2 $\implies$ common prefix is `0`.
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        # Method 1: Common prefix right-shift
        shifts = 0
        while left < right:
            left >>= 1
            right >>= 1
            shifts += 1
        return left << shifts
```

#### 2. C++ (C++17 / STL — Brian Kernighan Optimal)
```cpp
class Solution {
public:
    int rangeBitwiseAnd(int left, int right) {
        // Method 2: Strip lowest 1-bits of right until right <= left
        while (right > left) {
            right &= (right - 1);
        }
        return right;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int rangeBitwiseAnd(int left, int right) {
        // Strip rightmost bits of right
        while (right > left) {
            right &= (right - 1);
        }
        return right;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(1)$ — In Method 1, at most 32 shifts occur. In Method 2, at most the number of set bits in `right` ($\le 32$) are cleared. Strictly $O(1)$.
- **Space Complexity:** $O(1)$ auxiliary space.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Common Binary Prefix Identification across Continuous Integer Intervals.
- **Trap:** Writing a `for` loop from `left` to `right`: will instantly time out for large ranges ($right - left \approx 2 \cdot 10^9$). Always operate on the bit representation of the boundaries.