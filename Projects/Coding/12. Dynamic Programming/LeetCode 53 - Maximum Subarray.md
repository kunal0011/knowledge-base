---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 53: Maximum Subarray"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 53: Maximum Subarray

**LeetCode 53 – Maximum Subarray**, with **explicit state definition, transition, DP table construction, and a worked example**. This is Kadane’s Algorithm expressed rigorously as Dynamic Programming.

---

## Problem Statement (LeetCode 53)

Given an integer array `nums`, find the **contiguous subarray** (containing at least one number) which has the **largest sum**, and return its sum.

**Example**

```text
Input:  nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6
```

---

## 1. DP State Definition

### State

Let

```
dp[i] = maximum subarray sum that ends exactly at index i
```

Key point:

* The subarray **must include nums[i]**
* We only care about subarrays **ending at i**, not anywhere before

This restriction is what makes the transition clean.

---

## 2. State Transition

At index `i`, there are **only two choices**:

1. **Extend** the previous subarray ending at `i-1`
2. **Start a new subarray** at index `i`

### Transition Formula

```
dp[i] = max(
    nums[i],          # start new subarray
    dp[i-1] + nums[i] # extend previous subarray
)
```

Why this works:

* If `dp[i-1]` is positive → extending helps
* If `dp[i-1]` is negative → better to drop it and restart

---

## 3. Base Case

```
dp[0] = nums[0]
```

Because the only subarray ending at index `0` is `[nums[0]]`.

---

## 4. Final Answer

The maximum subarray **can end anywhere**, so:

```
answer = max(dp[0..n-1])
```

---

## 5. DP Table Construction (Step-by-Step Example)

### Input

```text
nums = [-2,1,-3,4,-1,2,1,-5,4]
```

### DP Table

| i | nums[i] | dp[i] = max(nums[i], dp[i-1] + nums[i]) |
| --- | --- | --- |
| 0 | -2 | -2 |
| 1 | 1 | max(1, -2+1) = **1** |
| 2 | -3 | max(-3, 1-3) = **-2** |
| 3 | 4 | max(4, -2+4) = **4** |
| 4 | -1 | max(-1, 4-1) = **3** |
| 5 | 2 | max(2, 3+2) = **5** |
| 6 | 1 | max(1, 5+1) = **6** |
| 7 | -5 | max(-5, 6-5) = **1** |
| 8 | 4 | max(4, 1+4) = **5** |

### DP Array

```
dp = [-2, 1, -2, 4, 3, 5, 6, 1, 5]
```

### Result

```
max(dp) = 6
```

---

## 6. Subarray Identification (Intuition)

The max value `6` occurs at index `6`.

Tracing backward:

```
[4, -1, 2, 1] → sum = 6
```

---

## 7. Python 3 Implementation (DP Style)

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n

        dp[0] = nums[0]
        ans = dp[0]

        for i in range(1, n):
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
            ans = max(ans, dp[i])

        return ans
```

---

## 8. Space Optimization (Still DP)

Observation:

* `dp[i]` only depends on `dp[i-1]`

### Optimized Version

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curr_sum = nums[0]
        max_sum = nums[0]

        for i in range(1, len(nums)):
            curr_sum = max(nums[i], curr_sum + nums[i])
            max_sum = max(max_sum, curr_sum)

        return max_sum
```

This is **Kadane’s Algorithm**, derived directly from DP.

---

## 9. DP Pattern Recognition

| Aspect | Value |
| --- | --- |
| DP Type | 1D DP |
| Pattern | “Best subarray ending at i” |
| Decision | Extend vs Restart |
| Time | O(n) |
| Space | O(1) optimized |

---

If you want, I can next:

* Show **why greedy works because of DP**
* Extend this to **max circular subarray**
* Show **how to print the subarray itself**
* Compare with **prefix-sum approach**

Tell me how deep you want to go.