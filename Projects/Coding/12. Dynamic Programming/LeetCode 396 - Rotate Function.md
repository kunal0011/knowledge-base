---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 396: Rotate Function"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 396: Rotate Function

**LeetCode 396 – Rotate Function**, with **formal state definition, transition derivation, DP table construction, and a worked example**.

---

## LeetCode 396 — Rotate Function

### Problem Statement

Given an integer array `nums` of length `n`, define the **rotation function**:

[  
F(k) = \sum\_{i=0}^{n-1} i \times B\_k[i]  
]

Where `B_k` is the array obtained by rotating `nums` **clockwise by `k` positions**.

Return the **maximum value of `F(k)`** for `k ∈ [0, n-1]`.

---

## Key Observation (Core DP Insight)

A **brute force** computation of each `F(k)` costs `O(n)`, leading to `O(n²)`.

However, observe a **recurrence relation** between consecutive rotations:

[  
F(k) = F(k-1) + \text{sum(nums)} - n \times \text{nums}[n-k]  
]

This allows us to compute all rotations in **O(n)**.

---

## DP Formulation

### 1. DP State Definition

Let:

[  
dp[k] = \text{value of the rotation function } F(k)  
]

Where:

* `k` = number of clockwise rotations
* `k ∈ [0, n-1]`

---

### 2. Base Case

For `k = 0` (no rotation):

[  
dp[0] = \sum\_{i=0}^{n-1} i \times nums[i]  
]

---

### 3. State Transition

When rotating from `k-1` → `k`:

* All elements shift one position to the right
* The last element moves to index `0`

Mathematically:

[  
dp[k] = dp[k-1] + totalSum - n \times nums[n-k]  
]

Where:

* `totalSum = sum(nums)`
* `nums[n-k]` is the element that moves from index `0` to index `n-1`

---

### 4. DP Table Construction

We compute `dp[0]`, then iteratively compute `dp[1] ... dp[n-1]`.

---

## Worked Example

### Input

```text
nums = [4, 3, 2, 6]
n = 4
```

### Step 1: Compute `totalSum`

[  
totalSum = 4 + 3 + 2 + 6 = 15  
]

---

### Step 2: Compute Base Case `dp[0]`

[  
dp[0] = 0×4 + 1×3 + 2×2 + 3×6 = 25  
]

---

### Step 3: Build DP Table

| k | Formula | dp[k] |
| --- | --- | --- |
| 0 | Base case | **25** |
| 1 | `25 + 15 − 4×6` | **16** |
| 2 | `16 + 15 − 4×2` | **23** |
| 3 | `23 + 15 − 4×3` | **26** |

---

### Step 4: Answer

[  
\max(dp) = 26  
]

---

## Python 3 DP Solution (With Typing)

```python
from typing import List

class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = sum(nums)

        # dp[0]
        curr = sum(i * nums[i] for i in range(n))
        ans = curr

        # dp[1] to dp[n-1]
        for k in range(1, n):
            curr = curr + total_sum - n * nums[n - k]
            ans = max(ans, curr)

        return ans
```

---

## Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n)** |
| Space | **O(1)** (DP optimized to rolling variable) |

---

## Why This Is DP (Conceptually)

* **Optimal substructure**: `F(k)` is derived from `F(k-1)`
* **Overlapping computation avoided**: Each rotation reused previous result
* **DP table**: `dp[k]` stores rotation function value

---

## Summary

* State: `dp[k] = F(k)`
* Transition: `dp[k] = dp[k-1] + totalSum − n × nums[n-k]`
* Base case: `dp[0]`
* Result: `max(dp)`

If you want, I can also:

* Derive the formula **step-by-step mathematically**
* Show **rotation diagrams**
* Convert this into **prefix-sum intuition**
* Compare with brute-force visually

Just tell me.