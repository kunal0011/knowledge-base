---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 368: Largest Divisible Subset"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 368: Largest Divisible Subset

**LeetCode 368 – Largest Divisible Subset**, focused on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 368 – Largest Divisible Subset

### Problem Statement

Given a set of **distinct positive integers `nums`**, return the **largest subset** such that for every pair  
`(Si, Sj)` in the subset:

```
Si % Sj == 0  OR  Sj % Si == 0
```

If multiple answers exist, return **any**.

---

## Key Observation

1. **Divisibility is transitive when numbers are sorted**

   * If `a | b` and `b | c`, then `a | c`
2. Sorting enables us to treat this as a **Longest Increasing Subsequence (LIS)-style DP**
3. We only need to check **previous smaller numbers**

---

## Step 1: Sort the Input

```text
nums = [1, 2, 3, 4, 6, 8, 24]
```

Sorting ensures:

* If `nums[j]` divides `nums[i]` and `j < i`, the chain remains valid.

---

## DP State Definition

### DP Array

```
dp[i] = length of the largest divisible subset
        ending at index i (nums[i] must be included)
```

### Parent Array (for reconstruction)

```
parent[i] = index of previous element in the subset chain
```

---

## DP Initialization

```
dp[i] = 1          (each number alone is a valid subset)
parent[i] = -1     (no previous element initially)
```

---

## DP Transition

For every `i` from `0 → n-1`  
Check all `j` from `0 → i-1`

```
if nums[i] % nums[j] == 0:
    dp[i] = max(dp[i], dp[j] + 1)
```

If we update `dp[i]`, record the parent:

```
parent[i] = j
```

---

## Example Walkthrough

### Input

```text
nums = [1, 2, 3, 4, 6, 8, 24]
```

---

### DP Table Construction

| i | nums[i] | dp[i] | parent[i] | Reason |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1 | -1 | Base case |
| 1 | 2 | 2 | 0 | 2 % 1 = 0 |
| 2 | 3 | 2 | 0 | 3 % 1 = 0 |
| 3 | 4 | 3 | 1 | 4 % 2 = 0 |
| 4 | 6 | 3 | 1 | 6 % 2 = 0 |
| 5 | 8 | 4 | 3 | 8 % 4 = 0 |
| 6 | 24 | 5 | 5 | 24 % 8 = 0 |

---

### Final DP Arrays

```
dp     = [1, 2, 2, 3, 3, 4, 5]
parent = [-1,0,0,1,1,3,5]
```

---

## Step 2: Reconstruct the Subset

1. Find index of **maximum dp value**

   ```
   max_len = 5 at index = 6 (value = 24)
   ```
2. Follow `parent[]` backwards

```
24 → 8 → 4 → 2 → 1
```

### Result

```
[1, 2, 4, 8, 24]
```

---

## Python 3 Solution (Typed)

```python
from typing import List

class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        if not nums:
            return []

        nums.sort()
        n = len(nums)

        dp = [1] * n
        parent = [-1] * n

        max_len = 1
        max_idx = 0

        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        parent[i] = j

            if dp[i] > max_len:
                max_len = dp[i]
                max_idx = i

        # reconstruct subset
        result = []
        while max_idx != -1:
            result.append(nums[max_idx])
            max_idx = parent[max_idx]

        return result[::-1]
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n²)** |
| Space | **O(n)** |

---

## Conceptual Mapping

This problem is essentially:

* **LIS-style DP**
* Condition replaced from `<` to **divisibility**
* Requires **path reconstruction**

---

If you want next:

* Backtracking tree interpretation (conceptual)
* Optimization discussion
* Similar problems mapping (368 ↔ 300 ↔ 354)
* Why greedy fails here

Tell me how deep you want to go.