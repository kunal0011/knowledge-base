---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 42: Trapping Rain Water"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 42: Trapping Rain Water

Below is a structured, interview-ready explanation of **LeetCode 42 – Trapping Rain Water**, aligned with your usual preference for *key observations, core insight, and a fully worked example*.

---

## LeetCode 42 — Trapping Rain Water

### Problem Statement

You are given an array `height` where `height[i]` represents the height of a vertical bar at index `i`.  
Each bar has a width of `1`.

Return the **total amount of rainwater** that can be trapped after raining.

**Constraints**

* `1 ≤ height.length ≤ 2 * 10^4`
* `0 ≤ height[i] ≤ 10^5`

---

## Key Observation (Conceptual)

Water trapped at any index `i` depends on:

```
min(max height to the left, max height to the right) - height[i]
```

However:

* Precomputing left/right max arrays works but uses extra space.
* A **monotonic decreasing stack** allows us to compute trapped water **on the fly** by identifying **bounded valleys**.

---

## Stack-Based Key Insight (Why Stack Works)

### Core Idea

* Maintain a **monotonic decreasing stack** of indices.
* The stack represents bars that are waiting to find a **right boundary** taller than themselves.
* When we encounter a bar taller than the top of the stack:

  * We have found a **right boundary**
  * The popped element becomes the **bottom of a container**
  * The new stack top is the **left boundary**

### When Water is Trapped

Water is trapped when:

```
left_boundary > bottom AND right_boundary > bottom
```

Height of trapped water:

```
min(left_height, right_height) - bottom_height
```

Width:

```
current_index - left_index - 1
```

---

## Algorithm (Stack Approach)

1. Initialize:

   * Empty stack (stores indices)
   * `water = 0`
2. Iterate through `height` with index `i`
3. While stack is not empty and `height[i] > height[stack[-1]]`:

   * Pop bottom index
   * If stack becomes empty → no left boundary → break
   * Compute:

     * `distance = i - stack[-1] - 1`
     * `bounded_height = min(height[i], height[stack[-1]]) - height[bottom]`
   * Add `distance * bounded_height` to `water`
4. Push `i` to stack
5. Return `water`

---

## Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        stack: List[int] = []
        water: int = 0

        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                bottom = stack.pop()

                # No left boundary
                if not stack:
                    break

                left = stack[-1]
                distance = i - left - 1
                bounded_height = min(height[i], height[left]) - height[bottom]

                water += distance * bounded_height

            stack.append(i)

        return water
```

---

## Worked Example (Step-by-Step)

### Input

```
height = [0,1,0,2,1,0,1,3,2,1,2,1]
```

---

### Key Trapping Moments

#### Step 1: Index 3 (height = 2)

Stack before: `[1, 2]`

* Bottom = index `2` (height = 0)
* Left boundary = index `1` (height = 1)
* Right boundary = index `3` (height = 2)

```
distance = 3 - 1 - 1 = 1
bounded_height = min(1, 2) - 0 = 1
water += 1 * 1 = 1
```

---

#### Step 2: Index 7 (height = 3)

Multiple pops happen:

**First pop**

* Bottom = index `6` (height = 1)
* Left = index `4` (height = 1)

```
distance = 7 - 4 - 1 = 2
bounded_height = min(1, 3) - 1 = 0
water += 0
```

**Second pop**

* Bottom = index `4` (height = 1)
* Left = index `3` (height = 2)

```
distance = 7 - 3 - 1 = 3
bounded_height = min(2, 3) - 1 = 1
water += 3
```

---

#### Step 3: Index 10 (height = 2)

* Bottom = index `9` (height = 1)
* Left = index `8` (height = 2)

```
distance = 10 - 8 - 1 = 1
bounded_height = min(2, 2) - 1 = 1
water += 1
```

---

### Final Result

```
Total trapped water = 6
```

---

## Complexity Analysis

* **Time Complexity:** `O(n)` (each index pushed and popped once)
* **Space Complexity:** `O(n)` (stack)

---

## Why This Approach is Interview-Favorite

* Demonstrates understanding of **monotonic stacks**
* Avoids extra arrays
* Clean geometric interpretation (valleys and boundaries)
* Scales optimally for large inputs

If you want, I can also:

* Contrast this with **two-pointer solution**
* Draw a **stack evolution diagram**
* Convert this into a **pattern explanation (Monotonic Stack – Water Trapping)**