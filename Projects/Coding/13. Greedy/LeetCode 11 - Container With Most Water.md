---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 11: Container With Most Water"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 11: Container With Most Water

**LeetCode 11 – Container With Most Water**, structured for clarity and rigor.

---

## Problem Statement

You are given an integer array `height` of length `n`.  
Each element represents the height of a vertical line drawn at index `i`.

Choose **two different lines** such that together with the x-axis they form a container that can store the **maximum amount of water**.

Return the **maximum water** the container can store.

### Constraints

* `2 ≤ n ≤ 10^5`
* `0 ≤ height[i] ≤ 10^4`

---

## Key Observation

For any two indices `i < j`:

```
Water Area = min(height[i], height[j]) × (j − i)
```

To maximize the area:

* We want a **large width** `(j − i)`
* We want a **large minimum height**

Brute force checks all pairs → **O(n²)** (too slow).

---

## Greedy Insight (Why Two Pointers Work)

Start with the **widest container**:

* Left pointer at index `0`
* Right pointer at index `n − 1`

Now consider:

* The width is maximum initially
* The area is limited by the **shorter line**

### Critical Greedy Rule

> Always move the pointer pointing to the **shorter height**.

### Why this is correct

Assume:

```
height[left] < height[right]
```

* The area is limited by `height[left]`
* Moving `right` inward only **reduces width**, while height constraint remains
* Moving `left` may find a **taller line**, potentially increasing area

Thus:

* Move the **shorter side**
* Never miss the optimal solution

This is a classic **proof by elimination** greedy strategy.

---

## Algorithm (Two Pointer Greedy)

1. Initialize:

   * `left = 0`
   * `right = n − 1`
   * `max_area = 0`
2. While `left < right`:

   * Compute current area
   * Update `max_area`
   * Move the pointer with smaller height
3. Return `max_area`

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        left: int = 0
        right: int = len(height) - 1
        max_area: int = 0

        while left < right:
            width: int = right - left
            curr_height: int = min(height[left], height[right])
            area: int = width * curr_height
            max_area = max(max_area, area)

            # Move the pointer with smaller height
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area
```

---

## Worked Example

### Input

```
height = [1,8,6,2,5,4,8,3,7]
```

### Step-by-Step

| Left | Right | Heights | Width | Area | Action |
| --- | --- | --- | --- | --- | --- |
| 0 | 8 | (1,7) | 8 | 8 | Move left |
| 1 | 8 | (8,7) | 7 | 49 | Move right |
| 1 | 7 | (8,3) | 6 | 18 | Move right |
| 1 | 6 | (8,8) | 5 | 40 | Move either |
| 1 | 5 | (8,4) | 4 | 16 | Move right |
| 1 | 4 | (8,5) | 3 | 15 | Move right |
| 1 | 3 | (8,2) | 2 | 4 | Move right |
| 1 | 2 | (8,6) | 1 | 6 | Stop |

### Maximum Area

```
49
```

Formed by:

* Index 1 → height = 8
* Index 8 → height = 7
* Width = 7

---

## Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

## Interview Takeaways

* This problem is a **greedy + two pointer** classic
* Key skill: **prove why moving the shorter pointer is safe**
* Frequently asked in FAANG interviews
* Cannot be solved optimally with sliding window or DP

---

If you want, I can also provide:

* Formal proof sketch for the greedy move
* Common wrong intuitions
* Comparison with brute force
* Visualization diagram for pointer movement