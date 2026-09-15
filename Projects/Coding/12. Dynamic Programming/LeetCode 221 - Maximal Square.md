---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 221: Maximal Square"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - matrix
  - amazon
  - google
  - meta
  - apple
---

# LeetCode 221: Maximal Square

**Target Companies:** Amazon, Google, Meta, Apple, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Array / Matrix  

---

### Problem Statement

Given an `m x n` binary matrix filled with `'0'`s and `'1'`s, find the largest square containing only `'1'`s and return *its area*.

---

### Input & Output Formats & Constraints

- **Input:** A 2D list of characters `matrix` of dimensions $m \times n$.
- **Output:** An integer representing the area (side length squared) of the largest all-`1` square.
- **Constraints:**
  - `m == matrix.length`
  - `n == matrix[i].length`
  - `1 <= m, n <= 300`
  - `matrix[i][j]` is `'0'` or `'1'`.

---

### Key Idea & Intuition

#### The Bottom-Right Corner Invariant
Let $dp[i][j]$ represent the **side length of the largest all-`1` square whose bottom-right corner is at cell $(i, j)$**.

For a square of side length $k$ to end at $(i, j)$:
1. The cell $(i, j)$ itself must be `'1'`.
2. The square must extend $k - 1$ units up, $k - 1$ units left, and $k - 1$ units diagonally up-left.
3. This means that its three immediate adjacent neighbors:
   - Top: $(i - 1, j)$
   - Left: $(i, j - 1)$
   - Top-Left Diagonal: $(i - 1, j - 1)$
   must all serve as the bottom-right corners of valid squares of size at least $k - 1$.

#### Recurrence Relation
Because any gap of `'0'` in the top, left, or diagonal neighbors bottlenecks the square from expanding, the maximum side length ending at $(i, j)$ is bounded by the **minimum** of its three neighbors:
$$dp[i][j] = \begin{cases} 1 + \min\Big(dp[i-1][j], \, dp[i][j-1], \, dp[i-1][j-1]\Big) & \text{if } matrix[i][j] == \text{'1'} \\ 0 & \text{if } matrix[i][j] == \text{'0'} \end{cases}$$

#### Space Optimization to $\mathcal{O}(n)$
Calculating row $i$ only requires the previous row $i - 1$ and the current row's previous column $j - 1$. We can maintain a single 1D array of size $n + 1$ with a variable `prev_diag` storing $dp[i-1][j-1]$, achieving $\mathcal{O}(n)$ space.

---

### Solution Approach (Step-by-Step)

1. **Initialize 1D DP Array:**
   - Create array `dp` of size $n + 1$ initialized to 0.
   - Maintain `max_side = 0`.
2. **Iterative Matrix Scan:**
   - For $i$ from 0 to $m - 1$:
     - `prev_diag = 0`
     - For $j$ from 0 to $n - 1$:
       - `temp = dp[j + 1]` (saves value for next column's diagonal)
       - If $matrix[i][j] == \text{'1'}$:
         - $dp[j + 1] = 1 + \min(dp[j], dp[j + 1], prev\_diag)$
         - $max\_side = \max(max\_side, dp[j + 1])$
       - Else:
         - $dp[j + 1] = 0$
       - `prev_diag = temp`
3. **Return Area:**
   - Return $max\_side \times max\_side$.

---

### Visual Algorithm Walkthrough

#### Trace for Matrix:
```
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]

DP Table Evolution (showing side lengths):
Row 0: [ 1,  0,  1,  0,  0 ]
Row 1: [ 1,  0,  1,  1,  1 ]
Row 2: [ 1,  1,  1,  2,  2 ]
Row 3: [ 1,  0,  0,  1,  0 ]

Key Transition at (2, 3):
- matrix[2][3] == '1'
- Top neighbor (1, 3): dp[1][3] = 1
- Left neighbor (2, 2): dp[2][2] = 1
- Diagonal neighbor (1, 2): dp[1][2] = 1
- dp[2][3] = 1 + min(1, 1, 1) = 2
Meaning a 2x2 square ends at (2, 3):
  [1, 1]
  [1, 1]

Max Side Length = 2
Max Area = 2 * 2 = 4.
```

---

### Solved Examples with Multiple Inputs

| `matrix` | Maximum Side Found | Formed Square | Output Area |
|---|---|---|---|
| `[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]` | 2 | $2 \times 2$ square at rows 1-2, cols 2-3 | `4` |
| `[["0","1"],["1","0"]]` | 1 | Isolated single cells | `1` |
| `[["0"]]` | 0 | No 1s | `0` |
| `[["1","1"],["1","1"]]` | 2 | Entire grid | `4` |

---

### Multi-Language Implementations

#### Python 3 (Space-Optimized $\mathcal{O}(n)$)
```python
class Solution:
    def maximalSquare(self, matrix: list[list[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
            
        m, n = len(matrix), len(matrix[0])
        dp: list[int] = [0] * (n + 1)
        max_side: int = 0
        
        for i in range(m):
            prev_diag = 0
            for j in range(n):
                temp = dp[j + 1]
                if matrix[i][j] == '1':
                    dp[j + 1] = 1 + min(dp[j], dp[j + 1], prev_diag)
                    max_side = max(max_side, dp[j + 1])
                else:
                    dp[j + 1] = 0
                prev_diag = temp
                
        return max_side * max_side
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maximalSquare(const std::vector<std::vector<char>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;

        int m = static_cast<int>(matrix.size());
        int n = static_cast<int>(matrix[0].size());
        std::vector<int> dp(n + 1, 0);
        int max_side = 0;

        for (int i = 0; i < m; ++i) {
            int prev_diag = 0;
            for (int j = 0; j < n; ++j) {
                int temp = dp[j + 1];
                if (matrix[i][j] == '1') {
                    dp[j + 1] = 1 + std::min({dp[j], dp[j + 1], prev_diag});
                    max_side = std::max(max_side, dp[j + 1]);
                } else {
                    dp[j + 1] = 0;
                }
                prev_diag = temp;
            }
        }

        return max_side * max_side;
    }
};
```

#### Java 17
```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int m = matrix.length;
        int n = matrix[0].length;
        int[] dp = new int[n + 1];
        int maxSide = 0;

        for (int i = 0; i < m; i++) {
            int prevDiag = 0;
            for (int j = 0; j < n; j++) {
                int temp = dp[j + 1];
                if (matrix[i][j] == '1') {
                    dp[j + 1] = 1 + Math.min(Math.min(dp[j], dp[j + 1]), prevDiag);
                    maxSide = Math.max(maxSide, dp[j + 1]);
                } else {
                    dp[j + 1] = 0;
                }
                prevDiag = temp;
            }
        }

        return maxSide * maxSide;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$, where $m$ is the number of rows and $n$ is the number of columns. We inspect each cell exactly once and perform $\mathcal{O}(1)$ min operations.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space using the 1D rolling array.

---

### Takeaway Pattern & Interview Traps

1. **Why 3 Neighbors:** A square of size $k$ requires full coverage of the $(k-1) \times (k-1)$ overlapping sub-squares. Checking top, left, and diagonal guarantees that the three overlapping sub-squares form a complete square without internal holes.
2. **Side Length vs Area:** The recurrence tracks the *side length*. Remember to return `max_side * max_side` (the area).
3. **Relation to Count Square Submatrices (LC 1277):** In LC 1277, the number of squares with bottom-right corner at $(i, j)$ is exactly $dp[i][j]$. Summing $\sum_{i,j} dp[i][j]$ solves LC 1277 directly!