---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 371: Sum of Two Integers"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - digital-logic
  - amazon
  - google
---

# LeetCode 371: Sum of Two Integers

**Target Companies:** Google, Amazon, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Digital Half/Full Adder Circuit Emulation via XOR and AND Shifts  

---

### Problem Statement

Given two integers `a` and `b`, return the sum of the two integers without using the operators `+` and `-`.

---

### Input & Output Formats & Constraints

- **Input:** `a: int`, `b: int`
- **Output:** `int` (sum $a + b$)
- **Constraints:**
  - $-1000 \le a, b \le 1000$

---

### Key Idea & Intuition

In digital hardware architecture, an **adder** adds two binary numbers using logic gates:
1. **Sum without Carry (XOR Gate):**
   The sum of two bits without carry is identical to XOR:
   $$0 \oplus 0 = 0, \quad 1 \oplus 0 = 1, \quad 0 \oplus 1 = 1, \quad 1 \oplus 1 = 0$$
   Therefore, `a ^ b` computes the sum of the bits where no carry is generated.
2. **Carry Generation (AND Gate + Left Shift):**
   A carry is generated at a bit position if and only if both bits are `1`:
   $$1 \ \& \ 1 = 1 \quad (\text{all other combinations produce } 0)$$
   Because a carry propagates to the next higher significance column, we shift it left by 1:
   $$\text{carry} = (a \ \& \ b) \ll 1$$
3. **Iterative Reduction:**
   We update `a = a ^ b` (the partial sum) and `b = carry`. We repeat this loop until the carry `b` becomes $0$.

#### The Python 3 Two's Complement Caveat:
In C++ and Java, 32-bit integers naturally truncate on overflow. In Python, integers have **arbitrary precision** (infinite width) without native 32-bit overflow.
- A negative carry can shift left indefinitely, causing an infinite loop.
- To simulate 32-bit signed integers in Python:
  - Apply `mask = 0xFFFFFFFF` at each step: `a = (a ^ b) & mask`.
  - At the end, if the 31st sign bit is set (`a > 0x7FFFFFFF`), convert it back to Python's signed representation using:
    $$\sim(a \oplus \text{mask})$$

---

### Solution Approach (Step-by-Step)

1. Mask to 32 bits: `mask = 0xFFFFFFFF`.
2. While `b & mask != 0`:
   - `carry = (a & b) << 1`
   - `a = a ^ b`
   - `b = carry`
3. If `b > 0` (in Python due to masking): return `(a & mask) if (a & mask) <= 0x7FFFFFFF else ~(a ^ mask)`.
4. In C++/Java: simply loop `while (b != 0)` with `carry = (unsigned)(a & b) << 1`, `a = a ^ b`, `b = carry`.

---

### Visual Algorithm Walkthrough

```
Compute 5 + 7 without +:
  a = 5 (0101_2)
  b = 7 (0111_2)

Iteration 1:
  carry = (a & b) << 1 = (0101 & 0111) << 1 = (0101) << 1 = 1010_2 (10)
  a     = a ^ b        = 0101 ^ 0111        = 0010_2 (2)
  b     = carry = 1010_2 (10)

Iteration 2:
  carry = (a & b) << 1 = (0010 & 1010) << 1 = (0010) << 1 = 0100_2 (4)
  a     = a ^ b        = 0010 ^ 1010        = 1000_2 (8)
  b     = carry = 0100_2 (4)

Iteration 3:
  carry = (a & b) << 1 = (1000 & 0100) << 1 = 0000 << 1 = 0
  a     = a ^ b        = 1000 ^ 0100        = 1100_2 (12)
  b     = carry = 0

Carry b == 0! Loop terminates.
Return a = 12 (since 5 + 7 = 12).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Positive Addition
- **Input:** `a = 1`, `b = 2`
- **Output:** `3`

#### Example 2: Positive and Negative (Subtraction via Two's Complement)
- **Input:** `a = 2`, `b = 3`
- **Output:** `5`

#### Example 3: Negative Sum
- **Input:** `a = -2`, `b = 3`
- **Trace:** $3 + (-2) = 1$.
- **Output:** `1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        max_int = 0x7FFFFFFF
        
        while (b & mask) != 0:
            carry = (a & b) << 1
            a = (a ^ b) & mask
            b = carry & mask
            
        # Convert 32-bit unsigned representation back to Python signed integer
        return a if a <= max_int else ~(a ^ mask)
```

#### 2. C++ (C++17 / STL)
```cpp
class Solution {
public:
    int getSum(int a, int b) {
        while (b != 0) {
            // Cast to unsigned int to avoid signed shift undefined behavior
            int carry = static_cast<int>(static_cast<unsigned int>(a & b) << 1);
            a = a ^ b;
            b = carry;
        }
        return a;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int getSum(int a, int b) {
        while (b != 0) {
            int carry = (a & b) << 1;
            a = a ^ b;
            b = carry;
        }
        return a;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(1)$ — At each iteration, the lowest carry bit shifts left by at least 1 position. For 32-bit integers, the carry will vanish in at most 32 iterations.
- **Space Complexity:** $O(1)$ auxiliary space.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Hardware Adder Logic: `sum = a ^ b`, `carry = (a & b) << 1`.
- **Trap:** Python infinite loop with negative numbers: In Python, negative numbers have infinite leading `1` bits (`...11111110`). Without masking to 32 bits (`& 0xFFFFFFFF`), `carry << 1` grows infinitely large and never reaches 0.
- **Trap:** C++ Signed Shift Overflow: In C++, shifting a negative signed integer left is undefined behavior. Always cast `a & b` to `unsigned int` before shifting.