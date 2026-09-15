---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 198: House Robber"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 198: House Robber

## LeetCode 198 — House Robber

### Problem Statement

You are given an integer array `nums` where `nums[i]` represents the amount of money in the `i`-th house along a street.  
You **cannot rob two adjacent houses**.

Return the **maximum amount of money** you can rob without triggering the alarm.

---

## Dynamic Programming Approach

This is a **classic linear DP optimization problem** where, at each index, you decide:

* **Rob the current house**, or
* **Skip the current house**

---

## 1. State Definition

Let:

```
dp[i] = maximum money that can be robbed from houses [0 .. i]
```

This definition ensures:

* Every subproblem is independent
* Each state builds on previously solved states

---

## 2. State Transition

At house `i`, you have **two choices**:

### Option 1: Skip house `i`

You take whatever you earned until house `i - 1`.

```
dp[i - 1]
```

### Option 2: Rob house `i`

If you rob this house, you **must skip house `i - 1`**, so you add `nums[i]` to the best you could do until `i - 2`.

```
dp[i - 2] + nums[i]
```

### Transition Formula

```
dp[i] = max(
    dp[i - 1],           # skip current house
    dp[i - 2] + nums[i]  # rob current house
)
```

---

## 3. Base Cases

```
dp[0] = nums[0]
dp[1] = max(nums[0], nums[1])
```

Edge cases:

* If `nums` is empty → return `0`
* If only one house → return `nums[0]`

---

## 4. DP Table Construction (Step-by-Step Example)

### Example Input

```text
nums = [2, 7, 9, 3, 1]
```

### DP Table Creation

| i | nums[i] | dp[i-2] | dp[i-1] | dp[i] = max(dp[i-1], dp[i-2]+nums[i]) |
| --- | --- | --- | --- | --- |
| 0 | 2 | — | — | 2 |
| 1 | 7 | — | 2 | max(2, 7) = 7 |
| 2 | 9 | 2 | 7 | max(7, 2+9=11) = 11 |
| 3 | 3 | 7 | 11 | max(11, 7+3=10) = 11 |
| 4 | 1 | 11 | 11 | max(11, 11+1=12) = 12 |

### Final Answer

```
12
```

Rob houses with money: `2 + 9 + 1`

---

## 5. Python 3 DP Solution (with Typing)

```python
from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        
        dp = [0] * n
        
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            dp[i] = max(
                dp[i - 1],          # skip current house
                dp[i - 2] + nums[i] # rob current house
            )
        
        return dp[-1]
```

---

## 6. Time and Space Complexity

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)` (DP table)

> This problem can be further optimized to **O(1) space** using two variables (`prev1`, `prev2`) if needed.

---

## 7. Key Takeaway

* This problem is a **decision DP** problem
* Every index asks: **take or skip**
* The adjacency constraint naturally leads to `i-2` dependency
* This DP pattern reappears in many problems (e.g., circular houses, trees)

If you want, I can also show:

* Space-optimized version
* Backtracking vs DP comparison
* House Robber II (circular) or House Robber III (tree) next