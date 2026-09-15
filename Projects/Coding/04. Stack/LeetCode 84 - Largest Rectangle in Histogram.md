---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 84: Largest Rectangle in Histogram"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 84: Largest Rectangle in Histogram

Below is a **complete, structured explanation of LeetCode 84 – Largest Rectangle in Histogram**, aligned with interview expectations and competitive programming rigor.

---

## LeetCode 84: Largest Rectangle in Histogram

### Problem Statement

You are given an array `heights` of length `n`, where each element represents the height of a bar in a histogram.  
Each bar has a width of **1**.

Return the **area of the largest rectangle** that can be formed within the histogram.

---

### Key Observation

A rectangle in the histogram is defined by:

* **Height** = height of the *shortest bar* in the rectangle
* **Width** = number of contiguous bars where all heights ≥ this height

Brute force checking all rectangles is **O(n²)** and will time out.

The critical insight:

> For each bar, treat it as the **minimum height** of a rectangle and expand **left and right** until a smaller bar is found.

To do this efficiently, we need to know:

* The **first smaller element on the left**
* The **first smaller element on the right**

This is where a **monotonic stack** becomes essential.

---

### Stack Key Insight (Core Idea)

We maintain a **monotonic increasing stack** of indices.

Why increasing?

* It allows us to detect when a bar can no longer extend to the right.

When processing a bar `h[i]`:

* If `h[i] < h[stack.top]`, then:

  * The bar at `stack.top` has found its **right boundary**
  * Pop it and compute the rectangle area using:

    * Height = popped bar height
    * Width = `current_index - previous_smaller_index - 1`

Each bar is:

* **Pushed once**
* **Popped once**

⇒ Overall **O(n)** time.

---

### Algorithm Steps

1. Append a `0` to `heights` (forces stack cleanup).
2. Initialize an empty stack (stores indices).
3. Iterate through all indices:

   * While stack is not empty and current height is smaller:

     * Pop the top index
     * Compute rectangle area
   * Push current index into the stack.
4. Track the maximum area.

---

### Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Add sentinel to flush stack at the end
        heights.append(0)
        stack: List[int] = []
        max_area = 0

        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                left = stack[-1] if stack else -1
                width = i - left - 1
                max_area = max(max_area, h * width)
            stack.append(i)

        return max_area
```

---

### Worked Example

#### Input

```
heights = [2, 1, 5, 6, 2, 3]
```

We append `0`:

```
[2, 1, 5, 6, 2, 3, 0]
```

---

#### Step-by-Step Stack Processing

| Index | Height | Action | Stack | Area Computed |
| --- | --- | --- | --- | --- |
| 0 | 2 | push | [0] | — |
| 1 | 1 | pop 2 | [] | 2 × 1 = 2 |
|  |  | push | [1] |  |
| 2 | 5 | push | [1,2] | — |
| 3 | 6 | push | [1,2,3] | — |
| 4 | 2 | pop 6 | [1,2] | 6 × 1 = 6 |
|  |  | pop 5 | [1] | 5 × 2 = **10** |
|  |  | push | [1,4] |  |
| 5 | 3 | push | [1,4,5] | — |
| 6 | 0 | pop 3 | [1,4] | 3 × 1 = 3 |
|  |  | pop 2 | [1] | 2 × 4 = 8 |
|  |  | pop 1 | [] | 1 × 6 = 6 |

---

### Final Answer

```
Maximum Area = 10
```

(Rectangle using heights `[5,6]` with height `5` and width `2`)

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)` (stack)

---

### Interview Summary (One-Liner)

> Use a monotonic increasing stack to compute, for each bar, the maximum rectangle where it is the minimum height by finding its nearest smaller bars on both sides in linear time.

If you want, I can also provide:

* Stack visualization diagrams
* Left/Right Smaller Element derivation
* Common mistakes & edge cases
* Comparison with brute force and divide-and-conquer approaches