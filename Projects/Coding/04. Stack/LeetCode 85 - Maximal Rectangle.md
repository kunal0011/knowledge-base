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
  - monotonic-stack
  - dynamic-programming
  - matrix
  - amazon
  - google
---

# LeetCode 85: Maximal Rectangle

**Target Companies:** Amazon, Google, Microsoft, Meta, Bloomberg, Apple  
**Difficulty:** Hard  
**Topic:** Monotonic Stack / 2D to 1D Reduction / Histogram Application

---

### Problem Statement

Given a `rows x cols` binary `matrix` filled with `'0'`'s and `'1'`'s, find the largest rectangle containing only `'1'`'s and return *its area*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `matrix`: `List[List[str]]`, where $R = \text{len}(matrix)$ and $C = \text{len}(matrix[0])$.
  - $matrix[i][j]$ is either `'0'` or `'1'`.
- **Output:**
  - `int`: Maximum area of a rectangle consisting entirely of `'1'`s.
- **Constraints:**
  - $rows == matrix.length$
  - $cols == matrix[i].length$
  - $1 \le rows, cols \le 200$
  - $matrix[i][j]$ is `'0'` or `'1'`.

---

### Key Idea & Intuition

Finding the maximal rectangle in a 2D matrix can be elegantly reduced to solving **LeetCode 84 (Largest Rectangle in Histogram)** row by row.

#### 2D Matrix to 1D Histogram Transformation:
Treat each row of the matrix as the horizontal base line of a histogram:
- Maintain an array `heights` of length $C$ (number of columns).
- For each row $r$ from $0$ to $R - 1$:
  - For each column $c$:
    - If $matrix[r][c] == \text{'1'}$: the vertical column of consecutive 1's increases by 1:
      $$heights[c] = heights[c] + 1$$
    - If $matrix[r][c] == \text{'0'}$: the vertical chain of 1's is broken by ground level:
      $$heights[c] = 0$$
  - The array `heights` now forms a valid 1D histogram resting on row $r$!
  - We run the $\mathcal{O}(C)$ Monotonic Increasing Stack algorithm from LeetCode 84 to compute the largest rectangle standing on this row base.
  - Update the global maximum area.

---

### Solution Approach (Step-by-Step)

1. **Edge Case:**
   - If `matrix` is empty or `matrix[0]` is empty, return `0`.
2. **Initialize:**
   - `cols = len(matrix[0])`
   - `heights = [0] * cols`
   - `max_area = 0`
3. **Iterate Through Each Row:**
   - For each cell $(r, c)$:
     - If $matrix[r][c] == \text{'1'}$: `heights[c] += 1`
     - Else: `heights[c] = 0`
   - Compute `largestRectangleArea(heights)` using monotonic stack:
     - Initialize `stack = []`.
     - For $i \in [0, cols]$:
       - `curr_h = heights[i] if i < cols else 0`
       - While `stack` and `curr_h < heights[stack[-1]]`:
         - `h = heights[stack.pop()]`
         - `w = i if not stack else (i - stack[-1] - 1)`
         - `max_area = max(max_area, h * w)`
       - `stack.append(i)`
4. Return `max_area`.

---

### Visual Algorithm Walkthrough

Given the input matrix:
```
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
```

#### Row by Row Histogram Construction:
```
Row 0: ["1","0","1","0","0"]
heights: [1, 0, 1, 0, 0]
Largest Histogram Area on Row 0 = 1

-------------------------------------------------------------------------
Row 1: ["1","0","1","1","1"]
heights: [2, 0, 2, 1, 1]
Largest Histogram Area on Row 1 = 3 (height 1, width 3 across cols 2, 3, 4)

-------------------------------------------------------------------------
Row 2: ["1","1","1","1","1"]
heights: [3, 1, 3, 2, 2]

Histogram visualization:
3 | #   #
2 | #   # # #
1 | # # # # #
----+--------
col 0 1 2 3 4

Largest Histogram Area on Row 2:
- Evaluating height = 2 across columns 2, 3, 4 -> width = 3 -> Area = 2 * 3 = 6!
max_area updated to 6.

-------------------------------------------------------------------------
Row 3: ["1","0","0","1","0"]
heights: [4, 0, 0, 3, 0]  (notice cols 1, 2, 4 reset to 0!)
Largest Histogram Area on Row 3 = 4 (height 4, width 1 on col 0)

-------------------------------------------------------------------------
Final Maximum Area = 6
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Matrix

- **Input:**
  ```python
  matrix = [
      ["1","0","1","0","0"],
      ["1","0","1","1","1"],
      ["1","1","1","1","1"],
      ["1","0","0","1","0"]
  ]
  ```
- **Output:** `6`

#### Example 2: Single Cell Matrices

- **Input:** `matrix = [["0"]]` $\implies$ `Output: 0`
- **Input:** `matrix = [["1"]]` $\implies$ `Output: 1`

#### Example 3: Full Block of Ones

- **Input:**
  ```python
  matrix = [
      ["1","1"],
      ["1","1"]
  ]
  ```
- **Execution:**
  - Row 0 heights: `[1, 1]` $\implies$ Area $1 \times 2 = 2$.
  - Row 1 heights: `[2, 2]` $\implies$ Area $2 \times 2 = 4$.
- **Output:** `4`

---

### Multi-Language Implementations

#### Python 3

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
            # 1. Update running heights array
            for c in range(cols):
                if row[c] == '1':
                    heights[c] += 1
                else:
                    heights[c] = 0

            # 2. Monotonic stack for largest rectangle in histogram
            stack = []
            for i in range(cols + 1):
                curr_h = heights[i] if i < cols else 0
                while stack and curr_h < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                stack.append(i)

        return max_area
```

#### C++17

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maximalRectangle(const std::vector<std::vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;

        int rows = static_cast<int>(matrix.size());
        int cols = static_cast<int>(matrix[0].size());
        std::vector<int> heights(cols, 0);
        int max_area = 0;

        for (int r = 0; r < rows; ++r) {
            // Update histogram heights for current row
            for (int c = 0; c < cols; ++c) {
                if (matrix[r][c] == '1') {
                    heights[c] += 1;
                } else {
                    heights[c] = 0;
                }
            }

            // Largest rectangle in histogram using monotonic stack
            std::vector<int> stack;
            for (int i = 0; i <= cols; ++i) {
                int curr_h = (i < cols) ? heights[i] : 0;

                while (!stack.empty() && curr_h < heights[stack.back()]) {
                    int h = heights[stack.back()];
                    stack.pop_back();
                    int w = stack.empty() ? i : (i - stack.back() - 1);
                    max_area = std::max(max_area, h * w);
                }

                stack.push_back(i);
            }
        }

        return max_area;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int rows = matrix.length;
        int cols = matrix[0].length;
        int[] heights = new int[cols];
        int maxArea = 0;

        for (int r = 0; r < rows; r++) {
            // Update heights
            for (int c = 0; c < cols; c++) {
                if (matrix[r][c] == '1') {
                    heights[c] += 1;
                } else {
                    heights[c] = 0;
                }
            }

            // Histogram calculation using ArrayDeque
            Deque<Integer> stack = new ArrayDeque<>();
            for (int i = 0; i <= cols; i++) {
                int currH = (i < cols) ? heights[i] : 0;

                while (!stack.isEmpty() && currH < heights[stack.peek()]) {
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

- **Time Complexity:** $\mathcal{O}(R \times C)$
  - There are $R$ rows.
  - In each row, updating the `heights` array takes $\mathcal{O}(C)$ time.
  - The monotonic stack histogram solver processes each of the $C + 1$ column indices in $\mathcal{O}(C)$ time.
  - Total Time: $\mathcal{O}(R \times C)$, where $R$ is rows and $C$ is columns.
- **Space Complexity:** $\mathcal{O}(C)$
  - The `heights` array and stack require auxiliary memory proportional only to the number of columns $C$.

---

### Takeaway Pattern & Interview Traps

1. **The Consecutive 1's Invariant:**
   - Always remember to reset `heights[c] = 0` when `matrix[r][c] == '0'`. A common bug is writing `heights[c]--` or forgetting to clear it, which would falsely carry forward pillars across gaps.
2. **Sentinel Flush:**
   - Iterating $i$ up to `cols` inclusive with `curr_h = 0` flushes all remaining taller columns in the monotonic stack automatically without redundant loops.