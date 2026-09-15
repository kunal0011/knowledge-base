---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 45: Jump Game II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 45: Jump Game II

**LeetCode 45 – Jump Game II**, with **clear state definition, transition, DP table construction, and a worked example**.  
This is intentionally **not the greedy solution**, but the conceptual DP that leads to it.

---

## Problem Statement (LeetCode 45)

You are given an integer array `nums`, where `nums[i]` represents the **maximum jump length from index `i`**.

You start at index `0`.  
Return the **minimum number of jumps** required to reach the **last index**.

It is guaranteed that you can always reach the last index.

---

## DP Thinking

### 1. State Definition

Let:

```
dp[i] = minimum number of jumps needed to reach index i
```

* `dp[0] = 0` (you start at index 0 with no jumps)
* We want `dp[n-1]`

---

### 2. Transition (Core DP Relation)

From index `j`, you can jump to any index `i` such that:

```
j < i ≤ j + nums[j]
```

So for every reachable `i`:

```
dp[i] = min(dp[i], dp[j] + 1)
```

This means:

> If you can reach `i` from `j`, then reaching `i` costs one more jump than reaching `j`.

---

### 3. Initialization

```
n = len(nums)
dp = [∞] * n
dp[0] = 0
```

---

### 4. DP Table Construction (Bottom-Up)

We process indices **from left to right**, because jumps always go forward.

For each index `j`:

* Try all jumps `1` to `nums[j]`
* Update future positions

---

## Python 3 DP Solution (with typing)

```python
from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [float('inf')] * n
        dp[0] = 0

        for j in range(n):
            for step in range(1, nums[j] + 1):
                i = j + step
                if i < n:
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n - 1]
```

---

## Example Walkthrough

### Input

```text
nums = [2, 3, 1, 1, 4]
```

Indices:

```
0   1   2   3   4
```

---

### Step-by-Step DP Table Update

#### Initial DP

```
dp = [0, ∞, ∞, ∞, ∞]
```

---

### j = 0

`nums[0] = 2` → can jump to `1` and `2`

```
dp[1] = min(∞, 0 + 1) = 1
dp[2] = min(∞, 0 + 1) = 1
```

```
dp = [0, 1, 1, ∞, ∞]
```

---

### j = 1

`nums[1] = 3` → can jump to `2, 3, 4`

```
dp[2] = min(1, 1 + 1) = 1
dp[3] = min(∞, 1 + 1) = 2
dp[4] = min(∞, 1 + 1) = 2
```

```
dp = [0, 1, 1, 2, 2]
```

---

### j = 2

`nums[2] = 1` → can jump to `3`

```
dp[3] = min(2, 1 + 1) = 2
```

(No change)

---

### j = 3

`nums[3] = 1` → can jump to `4`

```
dp[4] = min(2, 2 + 1) = 2
```

---

### Final DP Table

```
Index:  0  1  2  3  4
dp:     0  1  1  2  2
```

### Answer

```
dp[4] = 2
```

Minimum jumps = **2**

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n²)** (nested loops) |
| Space | **O(n)** |

---

## Key Insight (Why DP Works)

* Each index represents a **state**
* Each jump is a **state transition**
* We minimize transitions to reach the final state

This DP formulation is the **conceptual foundation** from which the famous **O(n) greedy solution** is derived.

---

If you want next:

* DP → Greedy optimization mapping
* Visual jump-level (BFS-style) explanation
* Comparison: Jump Game I vs II
* Backtracking tree (why it TLEs)

Tell me how deep you want to go.