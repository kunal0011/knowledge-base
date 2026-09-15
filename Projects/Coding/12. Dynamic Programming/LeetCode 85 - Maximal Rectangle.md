---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 85: Maximal Rectangle"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - monotonic-stack
  - matrix
  - amazon
  - google
  - meta
---

# LeetCode 85: Maximal Rectangle

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, ByteDance  
**Difficulty:** Hard  
**Topic:** 1D Dynamic Programming / Monotonic Stack / Histogram Reduction  

---

### Problem Statement

Given a `rows x cols` binary `matrix` filled with `'0'`s and `'1'`s, find the largest rectangle containing only `'1'`s and return its area.

---

### Input & Output Formats & Constraints

- **Input:** `matrix: List[List[str]]` — 2D grid containing `'0'` and `'1'`.
- **Output:** `int` — Area of the largest all-1 sub-rectangle.
- **Constraints:**
  - $rows == \text{matrix.length}$
  - $cols == \text{matrix}[i].\text{length}$
  - $1 \le rows, cols \le 200$
  - `matrix[i][j]` is `'0'` or `'1'`.

---

### Key Idea & Intuition

1. **Reduction to Largest Rectangle in Histogram (LeetCode 84):**
   - Consider each row of the matrix as the horizontal baseline of a histogram.
   - For any column $c$, the height of the bar at row $r$ is the number of consecutive `'1'`s extending vertically upwards ending at row $r$.
   - **DP Transition for Histogram Heights:**
     $$\text{height}[c] = \begin{cases} \text{height}[c] + 1 & \text{if } \text{matrix}[r][c] == '1' \\ 0 & \text{if } \text{matrix}[r][c] == '0' \end{cases}$$
   - For each row, after updating the 1D DP `heights` array in $\mathcal{O}(cols)$ time, we solve the **Largest Rectangle in Histogram** problem in $\mathcal{O}(cols)$ using a **Monotonic Increasing Stack**.

2. **Monotonic Stack Mechanics:**
   - Maintain a stack of column indices in strictly increasing order of their heights.
   - When encountering a column with height smaller than the top of the stack, pop the stack:
     - The popped height $h = \text{heights}[\text{top}]$.
     - The right boundary is the current index $i$.
     - The left boundary is the new stack top (or $-1$ if stack becomes empty).
     - $\text{width} = i - \text{stack.top}() - 1$.
     - $\text{area} = h \times \text{width}$.
   - Appending a sentinel $0$ at index $cols$ flushes all remaining elements from the stack cleanly.

---

### Solution Approach (Step-by-Step)

1. **Check Base Case:**
   - If `not matrix or not matrix[0]`, return `0`.
2. **Initialize Heights Array:**
   - `heights = [0] * cols`
   - `max_area = 0`
3. **Row-by-Row Processing:**
   - For each row $r$ from $0$ to $rows - 1$:
     - For each column $c$ from $0$ to $cols - 1$:
       - `heights[c] = heights[c] + 1 if matrix[r][c] == '1' else 0`.
     - **Compute Histogram Max Area:**
       - Initialize `stack = []`.
       - For $i$ from $0$ to $cols$:
         - `cur_h = heights[i] if i < cols else 0`.
         - While `stack` and `cur_h < heights[stack[-1]]`:
           - `h = heights[stack.pop()]`.
           - `w = i if not stack else i - stack[-1] - 1`.
           - `max_area = max(max_area, h * w)`.
         - `stack.append(i)`.
4. **Return:**
   - Return `max_area`.

---

### Visual Algorithm Walkthrough

Given `matrix`:
```
[ "1", "0", "1", "0", "0" ]
[ "1", "0", "1", "1", "1" ]
[ "1", "1", "1", "1", "1" ]
[ "1", "0", "0", "1", "0" ]
```

**Row 0:**
- Heights: `[1, 0, 1, 0, 0]`
- Max area = $1$.

**Row 1:**
- Heights: `[2, 0, 2, 1, 1]`
- Histogram evaluation: Columns $2 \dots 4$ have height $\ge 1 \implies 1 \times 3 = 3$.
- Max area = $3$.

**Row 2:**
- Heights: `[3, 1, 3, 2, 2]`
- Histogram representation:
```
Col:  0  1  2  3  4
h=3:  █     █
h=2:  █     █  █  █
h=1:  █  █  █  █  █
```
- Rectangle spanning columns $2 \dots 4$ with height $2$:
  $\text{width} = 4 - 2 + 1 = 3$, $\text{height} = 2 \implies \text{area} = 2 \times 3 = 6$.
- Max area = $6$.

**Row 3:**
- Heights: `[4, 0, 0, 3, 0]`
- Max area remains $6$.

Final Answer: $6$.

---

### Solved Examples with Multiple Inputs

| Case | `matrix` | Peak Histogram Row | Max Dimensions | Result |
|---|---|---|---|---|
| **Standard** | `[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]` | Row 2: `[3, 1, 3, 2, 2]` | $2 \times 3$ | `6` |
| **All Ones** | `[["1","1"],["1","1"]]` | Row 1: `[2, 2]` | $2 \times 2$ | `4` |
| **Single Zero** | `[["0"]]` | Row 0: `[0]` | $0$ | `0` |
| **Single One** | `[["1"]]` | Row 0: `[1]` | $1 \times 1$ | `1` |
| **Checkerboard** | `[["1","0"],["0","1"]]` | Every row | $1 \times 1$ | `1` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — DP + Monotonic Stack)
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
            # Update heights DP array
            for c in range(cols):
                heights[c] = heights[c] + 1 if row[c] == '1' else 0
                
            # Largest Rectangle in Histogram on current heights
            stack = []
            for i in range(cols + 1):
                cur_h = heights[i] if i < cols else 0
                while stack and cur_h < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                stack.append(i)
                
        return max_area
```

#### 2. C++ (C++17 / STL — DP + Monotonic Stack)
```cpp
#include <vector>
#include <stack>
#include <algorithm>

class Solution {
public:
    int maximalRectangle(std::vector<std::vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;

        int cols = matrix[0].size();
        std::vector<int> heights(cols, 0);
        int max_area = 0;

        for (const auto& row : matrix) {
            for (int c = 0; c < cols; ++c) {
                heights[c] = (row[c] == '1') ? heights[c] + 1 : 0;
            }

            // Monotonic stack for histogram
            std::stack<int> st;
            for (int i = 0; i <= cols; ++i) {
                int cur_h = (i < cols) ? heights[i] : 0;
                while (!st.empty() && cur_h < heights[st.top()]) {
                    int h = heights[st.top()];
                    st.pop();
                    int w = st.empty() ? i : (i - st.top() - 1);
                    max_area = std::max(max_area, h * w);
                }
                st.push(i);
            }
        }

        return max_area;
    }
};
```

#### 3. Java (Modern, Typed — DP + Monotonic Stack)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int cols = matrix[0].length;
        int[] heights = new int[cols];
        int maxArea = 0;

        for (char[] row : matrix) {
            for (int c = 0; c < cols; c++) {
                heights[c] = (row[c] == '1') ? heights[c] + 1 : 0;
            }

            // Monotonic stack for histogram
            Deque<Integer> stack = new ArrayDeque<>();
            for (int i = 0; i <= cols; i++) {
                int curH = (i < cols) ? heights[i] : 0;
                while (!stack.isEmpty() && curH < heights[stack.peek()]) {
                    int h = heights[stack.pop()];
                    int w = stack.isEmpty() ? i : (i - stack.peek() - 1);
                    maxArea = Math.max(maxArea, h * w);
                }
                stack.push(i);
            }
        }

        return maxArea;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(rows \times cols)$  
  Updating the heights array takes $\mathcal{O}(cols)$ per row. In the histogram sub-routine, each column index is pushed and popped from the monotonic stack at most once ($\mathcal{O}(cols)$). Across all $rows$, total time is strictly $\mathcal{O}(rows \times cols)$. For $rows, cols \le 200$, operations $\le 4 \times 10^4$, executing in $< 5$ ms.
- **Space Complexity:** $\mathcal{O}(cols)$  
  The `heights` DP array and the monotonic stack store at most $cols + 1$ integer elements.

---

### Takeaway Pattern & Interview Traps

1. **The 2D to 1D Histogram Paradigm:**
   - Whenever an interview question involves finding maximal orthogonal structures (rectangles, squares) in binary grids, project the 2D matrix row-by-row into 1D continuous heights:
     - LeetCode 84: Largest Rectangle in Histogram (Core primitive)
     - LeetCode 85: Maximal Rectangle (2D extension using 1D DP heights)
     - LeetCode 221: Maximal Square (2D square variant)
2. **Sentinel $0$ Flush:**
   - The loop runs up to index $cols$ with `cur_h = 0`. This dummy element flushes all remaining non-zero heights from the stack without requiring extra post-processing code.