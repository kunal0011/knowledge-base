---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 452: Minimum Number of Arrows to Burst Balloons"
tags:
  - leetcode
  - coding
  - greedy
  - interval
  - sorting
  - amazon
  - google
---

# LeetCode 452: Minimum Number of Arrows to Burst Balloons

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Interval Scheduling / Sorting  

---

### Problem Statement

There are some spherical balloons taped onto a flat wall that represents the XY-plane. The balloons are represented as a 2D integer array `points` where `points[i] = [x_start, x_end]` denotes a balloon that extends horizontally from `x_start` to `x_end`. You do not know the exact y-coordinates of the balloons.

Arrows can be shot up **directly vertically** (in the positive y-direction) from different points along the x-axis. A balloon with `[x_start, x_end]` is burst by an arrow shot at $x$ if `x_start <= x <= x_end`. There is no limit to the number of arrows that can be shot. A shot arrow keeps traveling up infinitely, bursting any balloons in its path.

Given the array `points`, return the **minimum number of arrows** that must be shot to burst all balloons.

---

### Input & Output Formats & Constraints

- **Input:**
  - `points`: `List[List[int]]` / `vector<vector<int>>` / `int[][]` ($1 \le \text{points.length} \le 10^5$).
- **Output:**
  - `int` — the minimum count of arrows required.
- **Constraints:**
  - $1 \le \text{points.length} \le 10^5$
  - `points[i].length == 2`
  - $-2^{31} \le x_{\text{start}} < x_{\text{end}} \le 2^{31} - 1$

---

### Key Idea & Intuition

To burst all balloons with the minimum number of arrows, each arrow should burst as many overlapping balloons as possible.

#### The Earliest End Position Greedy Invariant:
Consider the balloon that ends earliest on the x-axis: $[x_1, \text{end}_1]$.
- Any arrow that bursts this balloon **must** be shot at some position $x \le \text{end}_1$.
- To maximize the chance of also bursting other balloons that start later, where along $[x_1, \text{end}_1]$ should we shoot the arrow?
- **Greedily shoot at the furthest possible point to the right: $\text{arrow} = \text{end}_1$.**
- Any other balloon starting at or before $\text{end}_1$ ($\text{start}_k \le \text{end}_1$) will automatically be burst by this same arrow!
- If a balloon starts strictly after $\text{end}_1$ ($\text{start}_k > \text{end}_1$), it cannot be reached by this arrow, forcing us to launch a new arrow at its own ending coordinate.

This is the exact continuous dual of **Non-overlapping Intervals (LeetCode 435)**.

---

### Solution Approach (Step-by-Step)

1. If `points` is empty, return $0$.
2. Sort `points` in ascending order of their end coordinate `points[i][1]`.
3. Place the first arrow at the end of the first balloon:
   - `arrows = 1`
   - `arrow_pos = points[0][1]`
4. Loop through the remaining balloons from index $1$ to $N - 1$:
   - If `points[i][0] > arrow_pos`:
     - The current arrow cannot hit this balloon.
     - We must fire a new arrow: `arrows += 1`.
     - Update arrow position: `arrow_pos = points[i][1]`.
5. Return `arrows`.

---

### Visual Algorithm Walkthrough

For `points = [[10,16], [2,8], [1,6], [7,12]]`:

```
1. Sorted by end coordinate:
   [1, 6], [2, 8], [7, 12], [10, 16]

Balloons:
[1, 6]   : |--Balloon 1--|
[2, 8]   :   |--Balloon 2----|
[7, 12]  :                 |--Balloon 3---|
[10, 16] :                          |--Balloon 4---|

Trace:
- Balloon 1 [1, 6]:
  Shoot Arrow 1 at x = 6.
  arrows = 1, arrow_pos = 6.

- Balloon 2 [2, 8]:
  start = 2 <= 6 (Hit by Arrow 1! Burst!)

- Balloon 3 [7, 12]:
  start = 7 > 6 (Missed by Arrow 1!)
  Shoot Arrow 2 at x = 12.
  arrows = 2, arrow_pos = 12.

- Balloon 4 [10, 16]:
  start = 10 <= 12 (Hit by Arrow 2! Burst!)

All balloons burst! Total arrows = 2.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `points = [[10,16],[2,8],[1,6],[7,12]]`
- **Output:** `2`

#### Example 2:
- **Input:** `points = [[1,2],[3,4],[5,6],[7,8]]`
- **Tracing:** No balloons overlap. Each requires its own arrow.
- **Output:** `4`

#### Example 3 (Touching Balloons):
- **Input:** `points = [[1,2],[2,3],[3,4],[4,5]]`
- **Tracing:**
  - Balloon [1, 2] and [2, 3] both burst at $x = 2$.
  - Balloon [3, 4] and [4, 5] both burst at $x = 4$.
- **Output:** `2`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
            
        # Sort by ending coordinate
        points.sort(key=lambda x: x[1])
        
        arrows = 1
        arrow_pos = points[0][1]
        
        for i in range(1, len(points)):
            # If the balloon starts after the current arrow, shoot a new arrow
            if points[i][0] > arrow_pos:
                arrows += 1
                arrow_pos = points[i][1]
                
        return arrows
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int findMinArrowShots(std::vector<std::vector<int>>& points) {
        if (points.empty()) return 0;
        
        // Sort by ending coordinate
        std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
            return a[1] < b[1];
        });
        
        int arrows = 1;
        int arrow_pos = points[0][1];
        int n = static_cast<int>(points.size());
        
        for (int i = 1; i < n; ++i) {
            if (points[i][0] > arrow_pos) {
                arrows++;
                arrow_pos = points[i][1];
            }
        }
        
        return arrows;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int findMinArrowShots(int[][] points) {
        if (points == null || points.length == 0) {
            return 0;
        }
        
        // Use Integer.compare to avoid integer subtraction overflow
        Arrays.sort(points, (a, b) -> Integer.compare(a[1], b[1]));
        
        int arrows = 1;
        int arrowPos = points[0][1];
        
        for (int i = 1; i < points.length; i++) {
            if (points[i][0] > arrowPos) {
                arrows++;
                arrowPos = points[i][1];
            }
        }
        
        return arrows;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$
  - Sorting $n$ intervals takes $\mathcal{O}(n \log n)$ time.
  - A single linear scan evaluates each balloon in $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Sorting is performed in-place with scalar registers.

---

### Takeaway Pattern & Interview Traps

- **Integer Subtraction Overflow in Java/C++:**
  The coordinates range from $-2^{31}$ to $2^{31} - 1$.
  Writing `(a, b) -> a[1] - b[1]` in Java will produce **signed 32-bit integer underflow/overflow** when subtracting negative coordinates from positive ones (e.g. $10^9 - (-10^9)$)!
  Always use `Integer.compare(a[1], b[1])` in Java and `<` in C++.
- **Touching Endpoints:** In this problem, touching endpoints `[1, 2]` and `[2, 3]` **are** burst by a single arrow at $x = 2$, so we use strict inequality `points[i][0] > arrow_pos` rather than `>=`.