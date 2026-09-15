---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 338: Counting Bits"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - bit-manipulation
  - amazon
  - google
  - meta
  - apple
---

# LeetCode 338: Counting Bits

**Target Companies:** Amazon, Google, Meta, Apple, Microsoft  
**Difficulty:** Easy  
**Topic:** Dynamic Programming / Bit Manipulation / LSB Recurrence  

---

### Problem Statement

Given an integer `n`, return an array `ans` of length `n + 1` such that for each `i` ($0 \le i \le n$), `ans[i]` is the **number of `1`s** in the binary representation of `i`.

---

### Input & Output Formats & Constraints

- **Input:** `n: int` — Non-negative integer up to which bit counts are calculated.
- **Output:** `List[int]` — Array of length `n + 1` where index `i` contains the popcount of `i`.
- **Constraints:**
  - $0 \le n \le 10^5$
  - Follow up: Can you do it in linear time $\mathcal{O}(n)$ and in a single pass without using built-in popcount functions?

---

### Key Idea & Intuition

1. **Brute Force vs. Dynamic Programming:**
   - Counting bits of each number $i$ individually using `i.bit_count()` or `__builtin_popcount(i)` requires $\mathcal{O}(\log i)$ operations per number, yielding $\mathcal{O}(n \log n)$ total time.
   - However, numbers share binary prefixes. By relating the popcount of $i$ to an already-computed smaller integer, we can calculate each entry in $\mathcal{O}(1)$ time.

2. **Transition 1: Right-Shift (LSB Elimination) — Preferred:**
   - Right-shifting $i$ by 1 bit (`i >> 1` or `i // 2`) strips the least significant bit (LSB).
   - The number of set bits in $i$ is simply the number of set bits in `i >> 1` plus $1$ if the dropped bit was `1`, or $0$ if it was `0`:
     $$\text{dp}[i] = \text{dp}[i \gg 1] + (i \ \& \ 1)$$
   - Since $i \gg 1 < i$ for all $i \ge 1$, $\text{dp}[i \gg 1]$ is already computed when iterating from $1$ to $n$.

3. **Transition 2: Brian Kernighan's Bit Reset (Alternative):**
   - The expression $i \ \& \ (i - 1)$ clears the lowest set bit of $i$.
   - Therefore:
     $$\text{dp}[i] = \text{dp}[i \ \& \ (i - 1)] + 1$$
   - This relation is equally valid and optimal.

---

### Solution Approach (Step-by-Step)

1. **Initialize Table:**
   - Allocate an integer array `ans` of size `n + 1` with default zeroes (`ans[0] = 0`).
2. **Iterate from 1 to $n$:**
   - For each integer $i \in [1, n]$:
     - Set `ans[i] = ans[i >> 1] + (i & 1)`.
3. **Return:**
   - Return `ans`.

---

### Visual Algorithm Walkthrough

For $n = 7$:

```
Index i | Binary | i >> 1 | dp[i >> 1] | i & 1 | Calculation         | dp[i]
--------|--------|--------|------------|-------|---------------------|-------
   0    |  0000  |    -   |      -     |   -   | Base case           |   0
   1    |  0001  |    0   |   dp[0]=0  |   1   | 0 + 1               |   1
   2    |  0010  |    1   |   dp[1]=1  |   0   | 1 + 0               |   1
   3    |  0011  |    1   |   dp[1]=1  |   1   | 1 + 1               |   2
   4    |  0100  |    2   |   dp[2]=1  |   0   | 1 + 0               |   1
   5    |  0101  |    2   |   dp[2]=1  |   1   | 1 + 1               |   2
   6    |  0110  |    3   |   dp[3]=2  |   0   | 2 + 0               |   2
   7    |  0111  |    3   |   dp[3]=2  |   1   | 2 + 1               |   3

Final Array: [0, 1, 1, 2, 1, 2, 2, 3]
```

---

### Solved Examples with Multiple Inputs

| Case | `n` | Intermediate Sequence (`dp[0...n]`) | Output | Explanation |
|---|---|---|---|---|
| **Zero** | `0` | `[0]` | `[0]` | $0$ has 0 set bits |
| **Small Power of 2** | `2` | `dp[1]=1, dp[2]=1` | `[0, 1, 1]` | `0` (00), `1` (01), `2` (10) |
| **Odd Bound** | `5` | `dp[0..5] = [0, 1, 1, 2, 1, 2]` | `[0, 1, 1, 2, 1, 2]` | Matches binary forms `0, 1, 10, 11, 100, 101` |
| **Power of 2 Edge** | `8` | `dp[8] = dp[4] + 0 = 1 + 0 = 1` | `[0, 1, 1, 2, 1, 2, 2, 3, 1]` | `8` is $1000_2$, exactly 1 set bit |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            # dp[i] = dp[i >> 1] + (i & 1)
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
            // Note parentheses: + has higher precedence than & in C++!
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
            // Note parentheses: + has higher precedence than & in Java!
            ans[i] = ans[i >> 1] + (i & 1);
        }
        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$  
  The loop executes exactly $n$ iterations. In each iteration, one bitwise shift `>>`, one bitwise AND `&`, one addition, and one array lookup are performed, all in $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space  
  Excluding the returned output array of size $n + 1$, no additional heap memory or data structures are allocated.

---

### Takeaway Pattern & Interview Traps

1. **Operator Precedence Trap (`+` vs `&`):**
   - In C++ and Java, arithmetic addition `+` has **higher** precedence than bitwise AND `&`.
   - Writing `ans[i >> 1] + i & 1` evaluates as `(ans[i >> 1] + i) & 1`, which is completely incorrect! Always write `ans[i >> 1] + (i & 1)`.
2. **Follow-Up Invariant:**
   - Interviewers explicitly forbid `Integer.bitCount()` or `__builtin_popcount()`. The single-pass dynamic programming transition satisfies the follow-up constraints strictly with $\mathcal{O}(n)$ time and no built-in library calls.