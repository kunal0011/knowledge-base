---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 89: Gray Code"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - math
  - backtracking
  - amazon
  - google
---

# LeetCode 89: Gray Code

**Target Companies:** Amazon, Google, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Direct Bitwise Formula $i \oplus (i \gg 1)$ & Mirror-Reflection Construction  

---

### Problem Statement

An **$n$-bit gray code sequence** is a sequence of $2^n$ integers where:
1. Every integer is in the inclusive range $[0, 2^n - 1]$,
2. The first integer is $0$,
3. An integer appears **no more than once** in the sequence,
4. The binary representation of every pair of **adjacent** integers differs by **exactly one bit**, and
5. The binary representation of the **first and last** integers differs by **exactly one bit**.

Given an integer $n$, return **any valid $n$-bit gray code sequence**.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `List[int]` (array of size $2^n$)
- **Constraints:**
  - $1 \le n \le 16$

---

### Key Idea & Intuition

#### Method 1: The Direct Bitwise Formula ($i \oplus (i \gg 1)$)
A mathematically proven closed-form expression translates any binary index $i$ into its corresponding Gray code representation:
$$G(i) = i \oplus \lfloor \frac{i}{2} \rfloor = i \oplus (i \gg 1)$$
Why does this satisfy the single-bit difference property?
- Incrementing $i$ by $1$ in standard binary causes a cascade of trailing bits $011\dots1 \to 100\dots0$.
- In $i \oplus (i \gg 1)$, adjacent numbers $i$ and $i + 1$ differ in only the highest bit changed by the addition of 1, while all subsequent bit flips cancel each other out in the XOR.
- The entire sequence of length $2^n$ is generated in a single loop from $0$ to $2^n - 1$.

#### Method 2: The Mirror Reflection Property (Inductive Construction)
- For $n = 0$: `[0]`
- For $n = 1$: `[0, 1]`
- To construct $n = k + 1$ from $n = k$:
  - Take the existing sequence for $k$.
  - Reflect (reverse) the sequence.
  - Prefix the reflected sequence with a set $k$-th bit (i.e. add $1 \ll k$).
  - Concatenate: `next_sequence = current + [x | (1 << k) for x in reversed(current)]`.
  - The boundary between the original and reflected parts differs by only the $k$-th bit, and the cyclic wraparound between index $0$ and the last index also differs by only the $k$-th bit!

---

### Solution Approach (Step-by-Step)

#### Approach 1 (Direct Formula):
1. Preallocate result array of size $2^n = 1 \ll n$.
2. For $i$ from $0$ to $(1 \ll n) - 1$:
   - Compute `gray = i ^ (i >> 1)`.
   - Store in `res[i]`.
3. Return `res`.

---

### Visual Algorithm Walkthrough

```
Inductive Mirror Reflection Construction:

n = 1:
  [0, 1]

n = 2:
  Original:  [0, 1]
  Reversed:  [1, 0]
  Add 1<<1 (2) to reversed: [1|2, 0|2] = [3, 2]
  Merged:    [0, 1, 3, 2]
  Binary:    00 -> 01 -> 11 -> 10 (each differs by exactly 1 bit!)
             Notice 10 and 00 also differ by exactly 1 bit (cyclic).

n = 3:
  Original:  [0, 1, 3, 2]
  Reversed:  [2, 3, 1, 0]
  Add 1<<2 (4) to reversed: [6, 7, 5, 4]
  Merged:    [0, 1, 3, 2, 6, 7, 5, 4]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: $n = 2$
- **Input:** `n = 2`
- **Output:** `[0, 1, 3, 2]`
- **Binary Trace:**
  - $0 \to 00_2$
  - $1 \to 01_2$ (differs at bit 0)
  - $3 \to 11_2$ (differs at bit 1)
  - $2 \to 10_2$ (differs at bit 0)
  - Wraparound: $2 (10_2) \to 0 (00_2)$ (differs at bit 1)

#### Example 2: $n = 1$
- **Input:** `n = 1`
- **Output:** `[0, 1]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def grayCode(self, n: int) -> List[int]:
        # Direct closed-form bitwise formula
        total = 1 << n
        return [i ^ (i >> 1) for i in range(total)]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> grayCode(int n) {
        int total = 1 << n;
        std::vector<int> res(total);
        for (int i = 0; i < total; ++i) {
            res[i] = i ^ (i >> 1);
        }
        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> grayCode(int n) {
        int total = 1 << n;
        List<Integer> res = new ArrayList<>(total);
        for (int i = 0; i < total; i++) {
            res.add(i ^ (i >> 1));
        }
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(2^n)$ — We generate exactly $2^n$ numbers, spending $O(1)$ constant bitwise operations per number. For $n \le 16$, $2^{16} = 65,536$ operations ($< 1$ millisecond).
- **Space Complexity:** $O(1)$ auxiliary space beyond the required $O(2^n)$ output array.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Binary to Gray Code Conversion $G(i) = i \oplus (i \gg 1)$ and Mirror Reflection.
- **Trap:** Recursive backtracking: Trying to find Gray codes via DFS backtracking can result in $O(2^n!)$ or exponential timeouts. The mathematical closed-form formula runs deterministically in strictly optimal linear time with respect to the output size.