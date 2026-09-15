---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 396: Rotate Function"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - math
  - prefix-sum
  - amazon
  - google
---

# LeetCode 396: Rotate Function

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Mathematical Derivation / Sliding Rotation  

---

### Problem Statement

You are given an integer array `nums` of length `n`.

Assume `arr_k` to be an array obtained by rotating `nums` by `k` positions clock-wise. We define the **rotation function** `F` on `nums` as follows:

$$F(k) = 0 \times \text{arr}_k[0] + 1 \times \text{arr}_k[1] + \dots + (n - 1) \times \text{arr}_k[n - 1]$$

Return the **maximum value of $F(0), F(1), \dots, F(n - 1)$**.

The test cases are generated so that the answer fits in a **32-bit** signed integer.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of integers of length $n$.
- **Output:** `int` — Maximum value of the rotation function across all rotations.
- **Constraints:**
  - $n == \text{nums.length}$
  - $1 \le n \le 10^5$
  - $-100 \le \text{nums}[i] \le 100$

---

### Key Idea & Intuition

1. **Brute Force vs. Mathematical Recurrence:**
   - Computing each $F(k)$ independently by simulating the rotation takes $\mathcal{O}(n)$ time per rotation, resulting in an $\mathcal{O}(n^2)$ overall time complexity, which will result in Time Limit Exceeded (TLE) for $n = 10^5$.
   - Notice the algebraic relationship between consecutive rotations $F(k - 1)$ and $F(k)$:
     Let $n = 4$ and array be $[A, B, C, D]$:
     $$F(0) = 0A + 1B + 2C + 3D$$
     Rotate clockwise by 1: $[D, A, B, C]$:
     $$F(1) = 0D + 1A + 2B + 3C$$
   - Subtracting $F(0)$ from $F(1)$:
     $$\begin{aligned} F(1) - F(0) &= (1A - 0A) + (2B - 1B) + (3C - 2C) + (0D - 3D) \\ &= A + B + C - 3D \\ &= (A + B + C + D) - 4D \\ &= \text{totalSum} - n \times \text{nums}[n - 1] \end{aligned}$$
   - In general, when rotating clockwise by 1 from rotation $k-1$ to $k$:
     - The element at the previous end ($\text{nums}[n - k]$) jumps from coefficient $n - 1$ down to $0$.
     - Every other element's coefficient increases by exactly $1$.
     - Hence, the recurrence relation is:
       $$F(k) = F(k - 1) + \text{totalSum} - n \times \text{nums}[n - k]$$

2. **Rolling Variable Dynamic Programming:**
   - Each state $F(k)$ depends strictly on the immediately preceding state $F(k - 1)$.
   - We only need an $\mathcal{O}(1)$ space rolling variable `curr_F` rather than a full array of size $n$.

---

### Solution Approach (Step-by-Step)

1. **Calculate Baseline:**
   - Compute `total_sum = sum(nums)`.
   - Compute initial rotation function $F(0)$:
     `curr = sum(i * nums[i] for i in range(n))`.
   - Initialize `max_val = curr`.
2. **Iterate from $k = 1$ to $n - 1$:**
   - The element shifted from the end to the front is $\text{nums}[n - k]$.
   - Update `curr = curr + total_sum - n * nums[n - k]`.
   - Update `max_val = max(max_val, curr)`.
3. **Return:**
   - Return `max_val`.

---

### Visual Algorithm Walkthrough

For `nums = [4, 3, 2, 6]`, $n = 4$:
- `total_sum` $= 4 + 3 + 2 + 6 = 15$.
- $F(0) = 0(4) + 1(3) + 2(2) + 3(6) = 0 + 3 + 4 + 18 = 25$.

```
k = 1:
  Element shifted from tail: nums[4 - 1] = nums[3] = 6
  F(1) = F(0) + total_sum - 4 * 6
       = 25 + 15 - 24 = 16
  max_val = max(25, 16) = 25

k = 2:
  Element shifted from tail: nums[4 - 2] = nums[2] = 2
  F(2) = F(1) + total_sum - 4 * 2
       = 16 + 15 - 8 = 23
  max_val = max(25, 23) = 25

k = 3:
  Element shifted from tail: nums[4 - 3] = nums[1] = 3
  F(3) = F(2) + total_sum - 4 * 3
       = 23 + 15 - 12 = 26
  max_val = max(25, 26) = 26

Final Result: 26 (achieved at rotation k = 3)
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | `total_sum` | $F(0)$ | Sequence of $F(k)$ | Result |
|---|---|---|---|---|---|
| **Standard** | `[4, 3, 2, 6]` | `15` | `25` | `F(0)=25, F(1)=16, F(2)=23, F(3)=26` | `26` |
| **Single Element** | `[100]` | `100` | `0` | `F(0)=0` | `0` |
| **All Negative** | `[-1, -2, -3]` | `-6` | `0(-1)+1(-2)+2(-3)=-8` | `F(0)=-8, F(1)=-9, F(2)=-7` | `-7` |
| **All Equal** | `[5, 5, 5, 5]` | `20` | `0(5)+1(5)+2(5)+3(5)=30` | All $F(k) = 30$ | `30` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # Calculate F(0)
        curr = sum(i * nums[i] for i in range(n))
        max_f = curr
        
        # Calculate F(1) ... F(n-1) in O(1) per step
        for k in range(1, n):
            curr = curr + total_sum - n * nums[n - k]
            if curr > max_f:
                max_f = curr
                
        return max_f
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int maxRotateFunction(std::vector<int>& nums) {
        int n = nums.size();
        long long total_sum = 0;
        long long curr = 0;

        for (int i = 0; i < n; ++i) {
            total_sum += nums[i];
            curr += static_cast<long long>(i) * nums[i];
        }

        long long max_f = curr;

        for (int k = 1; k < n; ++k) {
            curr = curr + total_sum - static_cast<long long>(n) * nums[n - k];
            max_f = std::max(max_f, curr);
        }

        return static_cast<int>(max_f);
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int maxRotateFunction(int[] nums) {
        int n = nums.length;
        long totalSum = 0;
        long curr = 0;

        for (int i = 0; i < n; i++) {
            totalSum += nums[i];
            curr += (long) i * nums[i];
        }

        long maxF = curr;

        for (int k = 1; k < n; k++) {
            curr = curr + totalSum - (long) n * nums[n - k];
            maxF = Math.max(maxF, curr);
        }

        return (int) maxF;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$  
  Initial summation of elements and baseline $F(0)$ requires a single pass $\mathcal{O}(n)$. The subsequent loop executes $n - 1$ times with $\mathcal{O}(1)$ arithmetic operations per step. Total time is linear $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(1)$  
  Only scalar accumulators (`total_sum`, `curr`, `max_f`) are used.

---

### Takeaway Pattern & Interview Traps

1. **Differential Transition Recognition:**
   - Whenever an objective function updates uniformly across an array rotation or sliding window, compute the algebraic difference $\Delta = F(k) - F(k-1)$. This transforms quadratic $\mathcal{O}(n^2)$ recalculation into an optimal linear $\mathcal{O}(n)$ recurrence.
2. **64-bit Integer Overflow Safeguard:**
   - In languages with fixed-width integers like C++ and Java, $n \times \text{nums}[i]$ and accumulated products can briefly exceed standard 32-bit limits if values were larger (though constraints state final answer fits in 32-bit). Using `long long` / `long` for intermediate arithmetic prevents unexpected signed integer overflow.