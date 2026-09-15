---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 410: Split Array Largest Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 410: Split Array Largest Sum

**LeetCode 410 – Split Array Largest Sum**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 410 — Split Array Largest Sum

### Problem Statement

You are given an integer array `nums` and an integer `k`.

Split `nums` into **k non-empty contiguous subarrays** such that the **largest subarray sum** is minimized.

Return this minimum possible largest subarray sum.

---

## Key Observation

* The split must be **contiguous**
* We are minimizing a **maximum** → classic **minimax DP**
* Greedy alone does not work because early local decisions affect later partitions

This naturally leads to **Dynamic Programming with partitioning**.

---

## DP State Definition

Let:

```
dp[i][j] = minimum possible largest subarray sum
           when splitting first i elements into j subarrays
```

### Meaning

* We consider `nums[0 ... i-1]`
* We split them into exactly `j` contiguous parts
* Among all valid splits, `dp[i][j]` stores the **minimum possible maximum subarray sum**

---

## Prefix Sum (Preprocessing)

To compute subarray sums efficiently:

```
prefix[0] = 0
prefix[i] = sum(nums[0 ... i-1])
```

Subarray sum from index `x` to `i-1`:

```
sum(x, i) = prefix[i] - prefix[x]
```

---

## DP Transition (Core Logic)

To compute `dp[i][j]`:

We try placing the **last cut** at position `x`, where:

```
j-1 ≤ x < i
```

* First `x` elements → `j-1` subarrays
* Last subarray → `nums[x ... i-1]`

### Transition Formula

```
dp[i][j] = min over x (
               max(
                   dp[x][j-1],
                   prefix[i] - prefix[x]
               )
           )
```

### Explanation

* `dp[x][j-1]` → largest sum in previous partitions
* `prefix[i] - prefix[x]` → sum of last subarray
* The **maximum** of these two defines the cost of this split
* We **minimize** over all possible `x`

This is classic **partition DP with minimax optimization**.

---

## Base Cases

### 1. One subarray

```
dp[i][1] = prefix[i]
```

If only one partition, entire array is one subarray.

### 2. Zero elements

```
dp[0][0] = 0
```

### 3. Invalid states

Initialize others as infinity.

---

## DP Table Dimensions

```
dp size = (n+1) × (k+1)
```

* `i`: 0 → n
* `j`: 0 → k

---

## Example Walkthrough

### Input

```text
nums = [7, 2, 5, 10, 8]
k = 2
```

### Prefix Sum

```
index:   0  1  2  3   4   5
prefix: [0, 7, 9, 14, 24, 32]
```

---

### Step 1: Base Case (j = 1)

| i | dp[i][1] |
| --- | --- |
| 1 | 7 |
| 2 | 9 |
| 3 | 14 |
| 4 | 24 |
| 5 | 32 |

---

### Step 2: Compute dp[i][2]

#### dp[2][2]

Possible splits:

* x = 1

```
max(dp[1][1], prefix[2] - prefix[1])
= max(7, 2) = 7
```

```
dp[2][2] = 7
```

---

#### dp[3][2]

x = 1 → max(7, 7) = 7  
x = 2 → max(9, 5) = 9

```
dp[3][2] = 7
```

---

#### dp[4][2]

x = 1 → max(7, 17) = 17  
x = 2 → max(9, 15) = 15  
x = 3 → max(14, 10) = 14

```
dp[4][2] = 14
```

---

#### dp[5][2]

x = 1 → max(7, 25) = 25  
x = 2 → max(9, 23) = 23  
x = 3 → max(14, 18) = 18  
x = 4 → max(24, 8) = 24

```
dp[5][2] = 18
```

---

## Final DP Table (Relevant)

| i \ j | 1 | 2 |
| --- | --- | --- |
| 1 | 7 | — |
| 2 | 9 | 7 |
| 3 | 14 | 7 |
| 4 | 24 | 14 |
| 5 | 32 | 18 |

---

## Final Answer

```
dp[5][2] = 18
```

Optimal split:

```
[7, 2, 5] | [10, 8]
```

Largest sum = `max(14, 18) = 18`

---

## Python 3 DP Solution (Typed)

```python
from typing import List
import math

class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # Prefix sum
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        # DP table
        dp = [[math.inf] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # Base case: one subarray
        for i in range(1, n + 1):
            dp[i][1] = prefix[i]

        # Fill DP
        for j in range(2, k + 1):
            for i in range(j, n + 1):
                for x in range(j - 1, i):
                    dp[i][j] = min(
                        dp[i][j],
                        max(dp[x][j - 1], prefix[i] - prefix[x])
                    )

        return dp[n][k]
```

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | **O(n² · k)** |
| Space | **O(n · k)** |

---

## Why This DP Is Correct

* Explores **all valid partition points**
* Enforces **exactly k subarrays**
* Minimizes the **worst subarray sum**
* Uses prefix sums for efficient range queries

---

If you want, next we can:

* Derive the **binary search + greedy** solution
* Optimize DP with **monotonicity**
* Visualize the **partition tree**
* Compare DP vs Binary Search trade-offs

Tell me how you want to proceed.