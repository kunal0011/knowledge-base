---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 413: Arithmetic Slices"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - amazon
  - google
  - meta
---

# LeetCode 413: Arithmetic Slices

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Contiguous Subarrays / Difference Recurrence  

---

### Problem Statement

An integer array is called arithmetic if it consists of **at least three elements** and if the difference between any two consecutive elements is the same.

- For example, `[1,3,5,7,9]`, `[7,7,7,7]`, and `[3,-1,-5,-9]` are arithmetic sequences.

Given an integer array `nums`, return the **number of arithmetic subarrays** of `nums`.

A **subarray** is a contiguous subsequence of the array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of integers.
- **Output:** `int` — Total count of arithmetic subarrays of length $\ge 3$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 5000$
  - $-1000 \le \text{nums}[i] \le 1000$

---

### Key Idea & Intuition

1. **Subarray Extension Invariant:**
   - An arithmetic slice must have length $\ge 3$ with constant common difference $d = \text{nums}[i] - \text{nums}[i-1]$.
   - Let $\text{dp}[i]$ denote the number of valid arithmetic slices that **end at index $i$**.
   - If $\text{nums}[i] - \text{nums}[i-1] == \text{nums}[i-1] - \text{nums}[i-2]$:
     - Every arithmetic slice ending at index $i-1$ can be extended by appending $\text{nums}[i]$ (giving $\text{dp}[i-1]$ slices).
     - Furthermore, the 3-element window $(\text{nums}[i-2], \text{nums}[i-1], \text{nums}[i])$ forms a brand-new slice of length $3$ ($+1$ slice).
     - Therefore:
       $$\text{dp}[i] = \text{dp}[i-1] + 1$$
   - If the difference does not match:
     $$\text{dp}[i] = 0$$

2. **Total Count & Space Optimization:**
   - Total valid slices is simply the sum of all $\text{dp}[i]$ for $i \ge 2$:
     $$\text{Total} = \sum_{i=2}^{n-1} \text{dp}[i]$$
   - Notice that $\text{dp}[i]$ depends solely on $\text{dp}[i-1]$. We can reduce space from $\mathcal{O}(n)$ to $\mathcal{O}(1)$ using a single variable `curr`.

---

### Solution Approach (Step-by-Step)

1. **Base Check:**
   - If `len(nums) < 3`, return `0`.
2. **Initialize Counters:**
   - `curr = 0` (number of arithmetic slices ending at current index).
   - `total = 0` (cumulative total across all indices).
3. **Linear Sweep:**
   - Iterate $i$ from $2$ to $n - 1$:
     - If `nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]`:
       - `curr += 1`
       - `total += curr`
     - Else:
       - `curr = 0`
4. **Return:**
   - Return `total`.

---

### Visual Algorithm Walkthrough

Given `nums = [1, 2, 3, 4, 5]`:

```
i = 0, 1: Too short to form a slice of length >= 3.

i = 2 (nums[2] = 3):
  nums[2] - nums[1] = 1, nums[1] - nums[0] = 1 (equal!)
  curr = curr + 1 = 0 + 1 = 1   -> Slice: [1, 2, 3]
  total = 0 + 1 = 1

i = 3 (nums[3] = 4):
  nums[3] - nums[2] = 1, nums[2] - nums[1] = 1 (equal!)
  curr = curr + 1 = 1 + 1 = 2   -> Slices: [2, 3, 4] and [1, 2, 3, 4]
  total = 1 + 2 = 3

i = 4 (nums[4] = 5):
  nums[4] - nums[3] = 1, nums[3] - nums[2] = 1 (equal!)
  curr = curr + 1 = 2 + 1 = 3   -> Slices: [3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5]
  total = 3 + 3 = 6

Final Output: 6
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | Stepwise `curr` values | Result | Explanation |
|---|---|---|---|---|
| **Standard** | `[1, 2, 3, 4]` | $i=2: 1, i=3: 2$ | `3` | `[1,2,3], [2,3,4], [1,2,3,4]` |
| **Length < 3** | `[1, 2]` | Length $< 3$, immediate exit | `0` | Impossible to form slice of length 3 |
| **Alternating / Broken** | `[1, 2, 3, 5, 7, 9]` | $i=2: 1, i=3: 0, i=4: 0, i=5: 1$ | `2` | `[1,2,3]` and `[5,7,9]` |
| **All Constant** | `[7, 7, 7, 7]` | $i=2: 1, i=3: 2$ | `3` | Difference is 0: `[7,7,7], [7,7,7], [7,7,7,7]` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        curr = 0
        total = 0
        
        for i in range(2, n):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                curr += 1
                total += curr
            else:
                curr = 0
                
        return total
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int numberOfArithmeticSlices(std::vector<int>& nums) {
        int n = nums.size();
        if (n < 3) return 0;

        int curr = 0;
        int total = 0;

        for (int i = 2; i < n; ++i) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                curr += 1;
                total += curr;
            } else {
                curr = 0;
            }
        }

        return total;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        if (n < 3) return 0;

        int curr = 0;
        int total = 0;

        for (int i = 2; i < n; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                curr += 1;
                total += curr;
            } else {
                curr = 0;
            }
        }

        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$  
  A single linear pass from index $2$ to $N - 1$. Each iteration performs $\mathcal{O}(1)$ arithmetic subtractions and comparisons.
- **Space Complexity:** $\mathcal{O}(1)$  
  Only two scalar accumulator variables (`curr`, `total`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Subarray (Contiguous) vs. Subsequence (Non-Contiguous):**
   - LC 413 asks for arithmetic **subarrays** (contiguous), solvable in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space.
   - Contrast this with LC 446 (Arithmetic Slices II - Subsequence), which asks for arithmetic **subsequences** (non-contiguous), requiring a 2D hash map DP of complexity $\mathcal{O}(N^2)$.
2. **Formulaic Aggregation:**
   - An contiguous segment of length $L$ with identical differences contains $\frac{(L - 1)(L - 2)}{2}$ slices. The incremental accumulator `curr += 1; total += curr` automatically calculates this triangular sum on the fly.