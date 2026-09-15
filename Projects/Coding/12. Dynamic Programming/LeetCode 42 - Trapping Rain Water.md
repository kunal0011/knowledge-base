---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 42: Trapping Rain Water"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 42: Trapping Rain Water

**LeetCode 42 – Trapping Rain Water**, focused on **state definition, transitions, DP table construction, and a worked example**.

---

## LeetCode 42 – Trapping Rain Water

### Problem Statement

You are given an array `height` where `height[i]` represents the elevation at index `i`.  
Compute how much water can be trapped after raining.

---

## Core Insight (Why DP Works)

Water trapped at index `i` depends on:

* The **maximum height to its left**
* The **maximum height to its right**

If both sides are tall, water can be trapped above `height[i]`.

---

## DP State Definition

We define **two DP arrays**:

1. **Left Max Array**

```
left_max[i] = maximum height from index 0 to i
```

2. **Right Max Array**

```
right_max[i] = maximum height from index i to n-1
```

---

## DP Transition

### Left Max Transition

```
left_max[0] = height[0]
left_max[i] = max(left_max[i-1], height[i])
```

### Right Max Transition

```
right_max[n-1] = height[n-1]
right_max[i] = max(right_max[i+1], height[i])
```

---

## Water Trapped at Index `i`

Once both DP arrays are built:

```
water[i] = min(left_max[i], right_max[i]) - height[i]
```

If the value is negative, treat it as `0`.

---

## Final Answer

```
total_water = sum(water[i]) for all i
```

---

## Example Walkthrough

### Input

```
height = [0,1,0,2,1,0,1,3,2,1,2,1]
```

---

### Step 1: Build `left_max`

| i | height | left\_max |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2 | 0 | 1 |
| 3 | 2 | 2 |
| 4 | 1 | 2 |
| 5 | 0 | 2 |
| 6 | 1 | 2 |
| 7 | 3 | 3 |
| 8 | 2 | 3 |
| 9 | 1 | 3 |
| 10 | 2 | 3 |
| 11 | 1 | 3 |

---

### Step 2: Build `right_max`

| i | height | right\_max |
| --- | --- | --- |
| 11 | 1 | 1 |
| 10 | 2 | 2 |
| 9 | 1 | 2 |
| 8 | 2 | 2 |
| 7 | 3 | 3 |
| 6 | 1 | 3 |
| 5 | 0 | 3 |
| 4 | 1 | 3 |
| 3 | 2 | 3 |
| 2 | 0 | 3 |
| 1 | 1 | 3 |
| 0 | 0 | 3 |

---

### Step 3: Compute Water at Each Index

| i | height | min(left, right) | water |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0 |
| 2 | 0 | 1 | 1 |
| 3 | 2 | 2 | 0 |
| 4 | 1 | 2 | 1 |
| 5 | 0 | 2 | 2 |
| 6 | 1 | 2 | 1 |
| 7 | 3 | 3 | 0 |
| 8 | 2 | 2 | 0 |
| 9 | 1 | 2 | 1 |
| 10 | 2 | 2 | 0 |
| 11 | 1 | 1 | 0 |

---

### Total Water Trapped

```
1 + 1 + 2 + 1 + 1 = 6
```

---

## Python 3 DP Implementation (with Typing)

```python
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        n = len(height)
        left_max = [0] * n
        right_max = [0] * n

        # Build left_max DP
        left_max[0] = height[0]
        for i in range(1, n):
            left_max[i] = max(left_max[i - 1], height[i])

        # Build right_max DP
        right_max[n - 1] = height[n - 1]
        for i in range(n - 2, -1, -1):
            right_max[i] = max(right_max[i + 1], height[i])

        # Calculate trapped water
        water = 0
        for i in range(n):
            water += min(left_max[i], right_max[i]) - height[i]

        return water
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | `O(n)` |
| Space | `O(n)` (DP arrays) |

---

## Why This DP Is Important

* Introduces **prefix & suffix DP**
* Foundation for **two-pointer optimization**
* Clear separation of **state**, **transition**, and **result aggregation**

---

If you want, next I can:

* Convert this DP into **two-pointer O(1) space**
* Draw the **conceptual DP dependency graph**
* Compare DP vs Stack vs Two-Pointer approaches formally