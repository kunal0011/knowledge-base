---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 713: Subarray Product Less Than K"
tags:
  - leetcode
  - coding
  - sliding-window
  - two-pointers
  - array
  - amazon
  - google
---

# LeetCode 713: Subarray Product Less Than K

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg, Apple  
**Difficulty:** Medium  
**Topic:** Sliding Window / Two Pointers / Array  

---

### Problem Statement

Given an array of integers `nums` and an integer `k`, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than `k`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — array of positive integers ($1 \le \text{nums.length} \le 3 \times 10^4$).
  - `k`: `int` ($0 \le k \le 10^6$).
- **Output:**
  - `int` — total number of contiguous subarrays with product $< k$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 3 \times 10^4$
  - $1 \le \text{nums}[i] \le 1000$
  - $0 \le k \le 10^6$

---

### Key Idea & Intuition

Because all numbers in `nums` are positive integers ($\text{nums}[i] \ge 1$):
1. Multiplying by $\text{nums}[\text{right}]$ is strictly non-decreasing.
2. If a subarray $[l, r]$ has product $< k$, then **every sub-segment ending at $r$** (namely $[l, r], [l+1, r], \dots, [r, r]$) also has product $< k$, because dividing out elements $\ge 1$ cannot increase the product.
3. Therefore, for a fixed right boundary $r$, once $l$ is the smallest valid left index such that $\prod_{i=l}^r \text{nums}[i] < k$, the number of valid subarrays ending at $r$ is exactly:
   $$\text{count}(r) = r - l + 1$$

#### Edge Case $k \le 1$:
Since $\text{nums}[i] \ge 1$, the smallest possible product of any non-empty subarray is $1$. If $k \le 1$, no subarray can have a product strictly less than $k$. We can return $0$ immediately, avoiding division by zero or infinite shrink loops.

---

### Solution Approach (Step-by-Step)

1. If $k \le 1$, return $0$.
2. Initialize `product = 1`, `left = 0`, and `count = 0`.
3. Loop `right` from $0$ to $n - 1$:
   - Multiply `product *= nums[right]`.
   - While `product >= k`:
     - Divide `product /= nums[left]`.
     - Increment `left += 1`.
   - Add the number of valid subarrays ending at `right`:
     `count += right - left + 1`.
4. Return `count`.

---

### Visual Algorithm Walkthrough

For `nums = [10, 5, 2, 6]`, `k = 100`:

```
Indices:   0     1    2    3
Values:   [10,   5,   2,   6]

right = 0 (val = 10):
  product = 1 * 10 = 10 < 100
  Window [0..0]: [10]
  New subarrays ending at 0: 0 - 0 + 1 = 1 ([10])
  total count = 1

right = 1 (val = 5):
  product = 10 * 5 = 50 < 100
  Window [0..1]: [10, 5]
  New subarrays ending at 1: 1 - 0 + 1 = 2 ([10, 5], [5])
  total count = 1 + 2 = 3

right = 2 (val = 2):
  product = 50 * 2 = 100 >= 100 (INVALID!)
  Shrink left:
    product /= nums[0] (10) -> product = 10
    left moves to 1
  product = 10 < 100 (VALID!)
  Window [1..2]: [5, 2]
  New subarrays ending at 2: 2 - 1 + 1 = 2 ([5, 2], [2])
  total count = 3 + 2 = 5

right = 3 (val = 6):
  product = 10 * 6 = 60 < 100
  Window [1..3]: [5, 2, 6]
  New subarrays ending at 3: 3 - 1 + 1 = 3 ([5, 2, 6], [2, 6], [6])
  total count = 5 + 3 = 8

Total Result = 8.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [10, 5, 2, 6]`, `k = 100`
- **Output:** `8` (Subarrays: `[10]`, `[5]`, `[2]`, `[6]`, `[10, 5]`, `[5, 2]`, `[2, 6]`, `[5, 2, 6]`)

#### Example 2:
- **Input:** `nums = [1, 2, 3]`, `k = 0`
- **Output:** `0` (Since $k = 0 \le 1$)

#### Example 3:
- **Input:** `nums = [1, 1, 1]`, `k = 2`
- **Tracing:** All products are 1, which is $< 2$. Total subarrays = $\frac{3 \times 4}{2} = 6$.
- **Output:** `6`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0
            
        product = 1
        result = 0
        left = 0
        
        for right, val in enumerate(nums):
            product *= val
            while product >= k:
                product //= nums[left]
                left += 1
            result += right - left + 1
            
        return result
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int numSubarrayProductLessThanK(const std::vector<int>& nums, int k) {
        if (k <= 1) return 0;
        
        long long product = 1;
        int result = 0;
        int left = 0;
        int n = static_cast<int>(nums.size());
        
        for (int right = 0; right < n; ++right) {
            product *= nums[right];
            while (product >= k) {
                product /= nums[left];
                ++left;
            }
            result += (right - left + 1);
        }
        
        return result;
    }
};
```

#### Java 17
```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        
        long product = 1;
        int result = 0;
        int left = 0;
        
        for (int right = 0; right < nums.length; right++) {
            product *= nums[right];
            while (product >= k) {
                product /= nums[left];
                left++;
            }
            result += (right - left + 1);
        }
        
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Pointer `right` traverses each index from $0$ to $n - 1$.
  - Pointer `left` monotonically advances and never moves backwards; it advances at most $n$ times in total.
  - Multiplication and division are integer arithmetic taking $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Uses only scalar variables (`product`, `result`, `left`, `right`).

---

### Takeaway Pattern & Interview Traps

- **Counting Subarrays Ending at $R$:** In two-pointer sliding window problems counting valid subarrays, always compute `count += right - left + 1`. This counts the new subarrays introduced by element `right` and completely eliminates double-counting.
- **The $k \le 1$ Corner Case:** Without `if k <= 1: return 0`, a while loop `while product >= k:` with $k = 1$ or $k = 0$ will advance `left` past `right`, causing index out of bounds or division by zero!