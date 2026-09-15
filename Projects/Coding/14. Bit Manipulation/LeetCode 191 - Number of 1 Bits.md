---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 191: Number of 1 Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - brian-kernighan
  - amazon
  - google
---

# LeetCode 191: Number of 1 Bits (Hamming Weight)

**Target Companies:** Amazon, Google, Microsoft, Apple, Meta  
**Difficulty:** Easy  
**Topic:** Brian Kernighan's Algorithm (`n & (n - 1)`)  

---

### Problem Statement

Given a positive integer `n`, write a function that returns the number of set bits (also known as the **Hamming weight**) in its binary representation.

---

### Input & Output Formats & Constraints

- **Input:** `n: int` (positive integer $\le 2^{31} - 1$)
- **Output:** `int` (count of bits equal to `1`)
- **Constraints:**
  - $1 \le n \le 2^{31} - 1$

---

### Key Idea & Intuition

A naive approach tests all 32 bits one-by-one by shifting right (`n >>= 1`). This always takes exactly 32 iterations, even if $n = 1$.

#### Brian Kernighan's Algorithm:
Notice what happens when subtracting $1$ from a binary number:
$$n = \dots 1 0 0 0_2 \implies n - 1 = \dots 0 1 1 1_2$$
- The rightmost set bit (lowest 1-bit) turns into `0`.
- All trailing zeroes to its right turn into `1`s.
- All bits to its left remain completely unchanged.

When we compute the bitwise AND:
$$n \ \& \ (n - 1)$$
The rightmost `1` bit of $n$ is cancelled out to `0`, while leaving all other bits intact!

Each operation $n = n \ \& \ (n - 1)$ strips off exactly one set bit. The loop runs **only as many times as there are 1-bits** ($K$ iterations where $K \le 32$), running in optimal $O(K)$ time.

---

### Solution Approach (Step-by-Step)

1. Initialize `count = 0`.
2. While `n != 0`:
   - `n &= (n - 1)` (strips the lowest set bit).
   - `count += 1`.
3. Return `count`.

---

### Visual Algorithm Walkthrough

```
n = 11 (Binary: 1011_2)

Iteration 1:
  n       = 1011_2  (11)
  n - 1   = 1010_2  (10)
  n & n-1 = 1010_2  (Lowest 1-bit at index 0 removed!)
  count = 1

Iteration 2:
  n       = 1010_2  (10)
  n - 1   = 1001_2  (9)
  n & n-1 = 1000_2  (Lowest 1-bit at index 1 removed!)
  count = 2

Iteration 3:
  n       = 1000_2  (8)
  n - 1   = 0111_2  (7)
  n & n-1 = 0000_2  (Lowest 1-bit at index 3 removed!)
  count = 3

n is now 0. Loop terminates.
Hamming weight = 3.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Positive Integer
- **Input:** `n = 11` (binary `1011`)
- **Output:** `3`

#### Example 2: Power of Two (Single Set Bit)
- **Input:** `n = 128` (binary `10000000`)
- **Trace:** $128 \ \& \ 127 = 0 \implies$ loop runs exactly 1 time!
- **Output:** `1`

#### Example 3: Large Integer with 30 Set Bits
- **Input:** `n = 2147483645`
- **Output:** `30`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n &= (n - 1)
            count += 1
        return count
```

#### 2. C++ (C++17 / STL)
```cpp
class Solution {
public:
    int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n &= (n - 1);
            count++;
        }
        return count;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n &= (n - 1);
            count++;
        }
        return count;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(K)$ where $K$ is the number of 1-bits in the binary representation of $n$ ($0 \le K \le 32$). Best case is $O(1)$ for powers of two.
- **Space Complexity:** $O(1)$ auxiliary space — Only a single counter variable.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Brian Kernighan's Bit Eraser `n & (n - 1)`.
- **Extension / Follow-Up:** How to test if a number is a power of two? If $n > 0$ and `(n & (n - 1)) == 0`, $n$ has exactly one set bit and is thus a pure power of 2!
