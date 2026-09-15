---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 137: Single Number II"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 137: Single Number II

**Target Companies:** Google (Signature Bit Question), Amazon, Apple  
**Difficulty:** Medium  
**Topic:** Digital Logic Design / Modulo 3 Bit Counting

---

### Problem Statement

Given an integer array `nums` where every element appears **three times** except for one, which appears **exactly once**. Find the single element and return it.

You must implement a solution with a **linear runtime complexity** and use only **constant extra space**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (the unique number)
- **Constraints:**
  - $1 \le \text{nums.length} \le 3 \times 10^4$
  - $-2^{31} \le \text{nums}[i] \le 2^{31} - 1$
  - Each element in `nums` appears exactly three times except for one element which appears once.

---

### Key Idea & Intuition

- **General Bit Counting Approach (Modulo 3):**
  - If we sum the $i$-th bit of all numbers in the array, the sum must be of the form $3k$ or $3k + 1$.
  - $\sum \text{bit}_i \pmod 3$ gives the $i$-th bit of the unique number!
- **Digital Logic Finite State Machine (FSM):**
  - Instead of looping through all 32 bits, we can track counts of bits modulo 3 across the entire 32-bit integers simultaneously using two bitmask variables `ones` and `twos`:
    - `ones`: Contains bits that have appeared $1$ time (or $4, 7, \dots$).
    - `twos`: Contains bits that have appeared $2$ times (or $5, 8, \dots$).
  - When a bit appears a 3rd time, both `ones` and `twos` reset that bit to `0`!
- **Bitwise State Transitions:**
  - `ones = (ones ^ num) & ~twos`
  - `twos = (twos ^ num) & ~ones`
  - After iterating through the entire array, `ones` holds the bits of the number that appeared exactly once!

---

### Solution Approach (Step-by-Step)

1. Initialize `ones = 0`, `twos = 0`.
2. For each `num` in `nums`:
   - `ones = (ones ^ num) & ~twos`
   - `twos = (twos ^ num) & ~ones`
3. Return `ones`.

---

### Visual Algorithm Walkthrough

```
Truth Table for a single bit transition on arrival of input 'num':

  Current State     Input    Next State
(twos, ones)        num     (twos, ones)
----------------------------------------
   0, 0 (count 0)    1         0, 1 (count 1)
   0, 1 (count 1)    1         1, 0 (count 2)
   1, 0 (count 2)    1         0, 0 (count 3 -> resets to 0!)
   Any state         0         No change

Formula:
ones = (ones ^ num) & ~twos
twos = (twos ^ num) & ~ones
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Positive Numbers
- **Input:** `nums = [2, 2, 3, 2]`
- **Trace:**
  - `num = 2`: `ones = 2, twos = 0`
  - `num = 2`: `ones = 0, twos = 2`
  - `num = 3`:
    - `ones = (0 ^ 3) & ~2 = 3 & ~2 = 1`
    - `twos = (2 ^ 3) & ~1 = 1 & ~1 = 0`
  - `num = 2`:
    - `ones = (1 ^ 2) & ~0 = 3 & ~0 = 3` -> wait, bit 1 resets to 0, bit 0 is 1 -> `ones = 3`
  - Final `ones = 3`.
- **Output:** `3`

#### Example 2: Negative Numbers
- **Input:** `nums = [0, 1, 0, 1, 0, 1, 99]`
- **Output:** `99`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ones, twos = 0, 0
        for num in nums:
            ones = (ones ^ num) & ~twos
            twos = (twos ^ num) & ~ones
        return ones
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int singleNumber(std::vector<int>& nums) {
        int ones = 0, twos = 0;
        for (int num : nums) {
            ones = (ones ^ num) & ~twos;
            twos = (twos ^ num) & ~ones;
        }
        return ones;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        for (int num : nums) {
            ones = (ones ^ num) & ~twos;
            twos = (twos ^ num) & ~ones;
        }
        return ones;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Exactly one single pass over the array with simple bitwise operations.
- **Space Complexity:** $O(1)$ — Only two scalar variables `ones` and `twos`.
