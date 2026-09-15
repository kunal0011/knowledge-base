---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 85: Maximal Rectangle"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 85: Maximal Rectangle

Below is a complete, structured explanation of **LeetCode 85 – Maximal Rectangle**, aligned with how stack-based problems are typically reasoned about in interviews and competitive programming.

---

## 1. Problem Statement

You are given a 2D binary matrix `matrix` filled with `'0'` and `'1'`.

Return the **area of the largest rectangle containing only `'1'`s**.

### Example

```text
Input:
matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]

Output: 6
```

The maximal rectangle consists of six `1`s.

---

## 2. Key Observation

A 2D rectangle problem can be **reduced to a series of 1D histogram problems**.

* Treat **each row as the base of a histogram**
* Each column stores the **height of consecutive `'1'`s ending at the current row**
* For every row, compute the **largest rectangle in the histogram**

This transforms the problem into repeatedly solving:

> **Largest Rectangle in Histogram (LeetCode 84)**

---

## 3. Stack Key Insight (Core Idea)

For a histogram:

* The **maximum rectangle** is always determined when a bar is the **shortest bar** in that rectangle.
* A **monotonic increasing stack** helps find:

  * The **previous smaller element**
  * The **next smaller element**

When a bar is popped:

```
height = popped_bar
width = current_index - index_of_previous_smaller - 1
area = height * width
```

This guarantees:

* Every bar is processed exactly once
* Time complexity stays linear

---

## 4. Algorithm Breakdown

### Step 1: Build Heights Row-by-Row

For each cell:

```
if matrix[r][c] == '1':
    heights[c] += 1
else:
    heights[c] = 0
```

### Step 2: Compute Largest Rectangle in Histogram

Use a **monotonic increasing stack** with a sentinel `0` at the end.

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        cols = len(matrix[0])
        heights = [0] * cols
        max_area = 0

        for row in matrix:
            # Build histogram heights
            for c in range(cols):
                if row[c] == '1':
                    heights[c] += 1
                else:
                    heights[c] = 0

            # Compute largest rectangle in histogram
            max_area = max(max_area, self.largestRectangleArea(heights))

        return max_area

    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        max_area = 0

        for i in range(len(heights) + 1):
            curr_height = heights[i] if i < len(heights) else 0

            while stack and curr_height < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)

            stack.append(i)

        return max_area
```

---

## 6. Worked-Out Example (Step-by-Step)

### Input Matrix

```
[
  1 0 1 0 0
  1 0 1 1 1
  1 1 1 1 1
  1 0 0 1 0
]
```

---

### Row 1 → Heights

```
[1, 0, 1, 0, 0]
Max histogram area = 1
```

---

### Row 2 → Heights

```
[2, 0, 2, 1, 1]
Max histogram area = 3
```

---

### Row 3 → Heights

```
[3, 1, 3, 2, 2]
```

Histogram visualization:

```
3 | █   █
2 | █   █ █ █
1 | █ █ █ █ █
    0 1 2 3 4
```

* Rectangle using height = 2, width = 3 → **area = 6**
* This is the global maximum

---

### Row 4 → Heights

```
[4, 0, 0, 3, 0]
Max histogram area = 4
```

---

## 7. Final Answer

```
Maximum Rectangle Area = 6
```

---

## 8. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(rows × cols)** |
| Space Complexity | **O(cols)** (stack + heights) |

---

## 9. Interview Takeaway

* **2D → 1D reduction** is the key insight
* Monotonic stack solves the inner problem optimally
* This problem is a **direct extension of LeetCode 84**

If you want, I can also provide:

* Stack execution trace for one histogram
* Comparison with brute-force DP
* Visual stack push/pop walkthrough