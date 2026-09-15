---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 633: Sum of Square Numbers"
tags:
  - leetcode
  - coding
  - two-pointers
  - math
  - binary-search
  - google
  - amazon
---

# LeetCode 633: Sum of Square Numbers

**Target Companies:** Google, Amazon, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Two Pointers / Mathematical Monotonic Convergence  

---

### Problem Statement

Given a non-negative integer $c$, determine whether there exist two integers $a$ and $b$ such that:
$$a^2 + b^2 = c$$

Return `true` if such integers exist, or `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:** `c: int`
- **Output:** `bool`
- **Constraints:**
  - $0 \le c \le 2^{31} - 1$

---

### Key Idea & Intuition

Since $a^2 \ge 0$ and $b^2 \ge 0$, neither $a$ nor $b$ can exceed $\lfloor\sqrt{c}\rfloor$. 

Because the squared function $f(x) = x^2$ is strictly monotonically increasing for $x \ge 0$, the search space is bounded to the domain $[0, \lfloor\sqrt{c}\rfloor]$:
- Initialize two pointers at the boundaries: `left = 0` and `right = math.isqrt(c)`.
- At each step, compute $S = \text{left}^2 + \text{right}^2$:
  - If $S == c$: Valid pair found! Return `true`.
  - If $S < c$: Sum is too small; increase `left++`.
  - If $S > c$: Sum is too large; decrease `right--`.
- Repeat while `left <= right`. (Note that $a$ and $b$ can be identical, so `left == right` is valid, e.g., $c = 2 = 1^2 + 1^2$).

---

### Solution Approach (Step-by-Step)

1. Compute upper bound: `right = int(math.isqrt(c))`, `left = 0`.
2. While `left <= right`:
   - Compute `total = left * left + right * right`.
   - If `total == c`: return `True`.
   - Else if `total < c`: `left += 1`.
   - Else: `right -= 1`.
3. If no pair satisfies the condition after loop completes, return `False`.

---

### Visual Algorithm Walkthrough

```
c = 5
right = isqrt(5) = 2
left  = 0

Pointers:
  left = 0, right = 2
  0^2 + 2^2 = 0 + 4 = 4 < 5 -> sum too small, left++

  left = 1, right = 2
  1^2 + 2^2 = 1 + 4 = 5 == c -> MATCH FOUND!
  Pair (1, 2) satisfies 1^2 + 2^2 = 5.

Result: True
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Sum of Squares
- **Input:** `c = 5`
- **Pairs:** $1^2 + 2^2 = 1 + 4 = 5$.
- **Output:** `true`

#### Example 2: Prime Not Expressible as Sum of Squares
- **Input:** `c = 3`
- **Trace:**
  - `left = 0, right = 1`: $0^2 + 1^2 = 1 < 3 \implies left = 1$.
  - `left = 1, right = 1`: $1^2 + 1^2 = 2 < 3 \implies left = 2$.
  - `left > right` ($2 > 1$) $\implies$ loop terminates.
- **Output:** `false`

#### Example 3: Single Perfect Square
- **Input:** `c = 4`
- **Pairs:** $0^2 + 2^2 = 0 + 4 = 4$.
- **Output:** `true`

#### Example 4: Large Integer Boundary
- **Input:** `c = 2147483646`
- **Trace:** Correctly avoids 32-bit integer overflow using 64-bit precision in C++/Java.
- **Output:** `false`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
import math

class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        left: int = 0
        right: int = math.isqrt(c)
        
        while left <= right:
            total: int = left * left + right * right
            if total == c:
                return True
            elif total < c:
                left += 1
            else:
                right -= 1
                
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <cmath>

class Solution {
public:
    bool judgeSquareSum(int c) {
        long long left = 0;
        long long right = static_cast<long long>(std::sqrt(c));
        
        while (left <= right) {
            long long total = left * left + right * right;
            if (total == c) {
                return true;
            } else if (total < c) {
                left++;
            } else {
                right--;
            }
        }
        
        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean judgeSquareSum(int c) {
        long left = 0;
        long right = (long) Math.sqrt(c);

        while (left <= right) {
            long total = left * left + right * right;
            if (total == c) {
                return true;
            } else if (total < c) {
                left++;
            } else {
                right--;
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\sqrt{c})$ — The pointers converge inwards within the domain $[0, \sqrt{c}]$, performing at most $\sqrt{c}$ checks.
- **Space Complexity:** $O(1)$ — Only constant primitive scalar variables are allocated.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Two-Pointer Inward Squeeze on Monotonically Increasing Function Domains.
- **Trap:** 32-bit Integer Overflow in C++ and Java. Since $c \le 2^{31} - 1$, $right \approx 46340$. If `left` and `right` were `int`, `left * left + right * right` could exceed $2^{31} - 1$ and wrap around to negative numbers. Always declare `left`, `right`, and `total` as 64-bit `long long` (or `long`).