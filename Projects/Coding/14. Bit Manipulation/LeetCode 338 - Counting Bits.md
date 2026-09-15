---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 338: Counting Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 338: Counting Bits

**Target Companies:** Amazon, Google, Microsoft, Apple, Meta  
**Difficulty:** Easy  
**Topic:** Dynamic Programming via Bitwise Transition Recurrences  

---

### Problem Statement

Given an integer $n$, return an array `ans` of length $n + 1$ such that for each $i$ ($0 \le i \le n$), `ans[i]` is the **number of $1$'s** in the binary representation of $i$.

Follow up:
- Can you do it in linear time $O(n)$ in a single pass?
- Can you do it without using any built-in function (like `__builtin_popcount` in C++)?

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `List[int]` (array of size $n + 1$)
- **Constraints:**
  - $0 \le n \le 10^5$

---

### Key Idea & Intuition

Computing the Hamming weight independently for each number from $0$ to $n$ costs $O(n \log n)$ time.
To achieve linear $O(n)$ time in a single pass, we use **Dynamic Programming**:

#### DP Relation 1: Right Shift + LSB (Even/Odd Transition)
Notice that shifting a number right by 1 (`i >> 1` or `i // 2`) removes its least significant bit:
- If $i$ is even ($i \ \& \ 1 == 0$), its binary representation ends in `0`. Dividing by 2 simply removes this trailing zero, leaving the count of 1-bits unchanged:
  $$\text{ans}[i] = \text{ans}[i \gg 1]$$
- If $i$ is odd ($i \ \& \ 1 == 1$), its binary representation ends in `1`. Dividing by 2 removes this set bit:
  $$\text{ans}[i] = \text{ans}[i \gg 1] + 1$$
Unified transition:
$$\text{ans}[i] = \text{ans}[i \gg 1] + (i \ \& \ 1)$$

#### DP Relation 2: Brian Kernighan Transition (Clearing Lowest Set Bit)
Since $i \ \& \ (i - 1)$ strips the lowest set bit from $i$:
$$\text{ans}[i] = \text{ans}[i \ \& \ (i - 1)] + 1$$
Because $i \ \& \ (i - 1) < i$, its Hamming weight was already computed in a previous step!

---

### Solution Approach (Step-by-Step)

1. Preallocate array `ans = [0] * (n + 1)`.
2. For $i$ from $1$ to $n$:
   - `ans[i] = ans[i >> 1] + (i & 1)`
3. Return `ans`.

---

### Visual Algorithm Walkthrough

```
Computing ans for n = 5:

ans[0] = 0 (Base case: binary 000_2 has 0 set bits)

i = 1: i >> 1 = 0, i & 1 = 1
       ans[1] = ans[0] + 1 = 0 + 1 = 1   (001_2)

i = 2: i >> 1 = 1, i & 1 = 0
       ans[2] = ans[1] + 0 = 1 + 0 = 1   (010_2)

i = 3: i >> 1 = 1, i & 1 = 1
       ans[3] = ans[1] + 1 = 1 + 1 = 2   (011_2)

i = 4: i >> 1 = 2, i & 1 = 0
       ans[4] = ans[2] + 0 = 1 + 0 = 1   (100_2)

i = 5: i >> 1 = 2, i & 1 = 1
       ans[5] = ans[2] + 1 = 1 + 1 = 2   (101_2)

Final Result: [0, 1, 1, 2, 1, 2]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: $n = 2$
- **Input:** `n = 2`
- **Output:** `[0, 1, 1]`

#### Example 2: $n = 5$
- **Input:** `n = 5`
- **Output:** `[0, 1, 1, 2, 1, 2]`

#### Example 3: $n = 0$
- **Input:** `n = 0`
- **Output:** `[0]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> countBits(int n) {
        std::vector<int> ans(n + 1, 0);
        for (int i = 1; i <= n; ++i) {
            ans[i] = ans[i >> 1] + (i & 1);
        }
        return ans;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int[] countBits(int n) {
        int[] ans = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            ans[i] = ans[i >> 1] + (i & 1);
        }
        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Exactly $N$ loop iterations with $O(1)$ constant time bitwise operations per number.
- **Space Complexity:** $O(1)$ auxiliary space beyond the required $O(N)$ return array.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Bitwise Dynamic Programming Transition (`dp[i] = dp[i >> 1] + (i & 1)`).
- **Alternative:** `ans[i] = ans[i & (i - 1)] + 1`: Mentioning both the right-shift and the lowest-set-bit-clearing recurrence demonstrates deep mastery of bitwise state design.