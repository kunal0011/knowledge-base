---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 120: Triangle"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - amazon
  - apple
  - google
  - bloomberg
---

# LeetCode 120: Triangle

**Target Companies:** Amazon, Apple, Google, Bloomberg, Microsoft  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Array  

---

### Problem Statement

Given a `triangle` array, return the minimum path sum from top to bottom.

For each step, you may move to an **adjacent number** of the row below. More formally, if you are on index `i` on the current row, you may move to either index `i` or index `i + 1` on the next row.

---

### Input & Output Formats & Constraints

- **Input:** A triangular 2D array of integers `triangle` where $triangle[i]$ contains $i + 1$ integers.
- **Output:** An integer representing the minimum path sum from top to bottom.
- **Constraints:**
  - `1 <= triangle.length <= 200`
  - `triangle[0].length == 1`
  - `triangle[i].length == triangle[i - 1].length + 1`
  - `-10^4 <= triangle[i][j] <= 10^4`
  - **Follow up:** Could you do this using only $\mathcal{O}(n)$ extra space, where $n$ is the total number of rows in the triangle?

---

### Key Idea & Intuition

#### Top-Down vs. Bottom-Up DP
- **Top-Down Approach:**
  Moving from $(0, 0)$ down to row $n - 1$ requires handling special edge cases for the leftmost column ($j = 0$) and rightmost column ($j = i$). Furthermore, after reaching row $n - 1$, we must scan the entire bottom row to find $\min_{0 \le j < n} dp[n-1][j]$.
- **Bottom-Up (Inverted) Approach:**
  Starting at the bottom row $n - 1$ and moving **upward** to $(0, 0)$ is dramatically simpler:
  1. For any element at row $r$, column $c$, the only two choices to continue downward were $(r+1, c)$ and $(r+1, c+1)$.
  2. Therefore, the minimum path sum starting from $(r, c)$ down to the base is:
     $$dp[r][c] = triangle[r][c] + \min(dp[r+1][c], dp[r+1][c+1])$$
  3. Notice that there are **no boundary conditions or edge cases** when working upward; every cell $(r, c)$ has valid transitions to both $(r+1, c)$ and $(r+1, c+1)$.
  4. At the end of the calculation, the answer naturally converges to the single apex cell:
     $$\text{Answer} = dp[0][0]$$

#### Space Optimization to $\mathcal{O}(n)$
Calculating row $r$ only requires the values of row $r + 1$. We can maintain a single 1D array of size $n$, initialized with the bottom row of the triangle, and overwrite values in-place from bottom to top.

---

### Solution Approach (Step-by-Step)

1. **Initialize 1D DP Array:**
   - Let $n = len(triangle)$.
   - Initialize `dp` as a copy of the bottom row: `dp = list(triangle[-1])`.
2. **Bottom-Up Transitions:**
   - For row $r$ from $n - 2$ down to 0:
     - For col $c$ from 0 to $r$:
       - $dp[c] = triangle[r][c] + \min(dp[c], dp[c + 1])$.
3. **Return Apex Value:**
   - Return `dp[0]`.

---

### Visual Algorithm Walkthrough

#### Trace for `triangle = [[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]`
```
Initial Triangle:
       [2]
      [3, 4]
    [6, 5, 7]
   [4, 1, 8, 3]

Step 1: Initialize DP with base row (r = 3):
dp = [4, 1, 8, 3]

Step 2: Move up to row r = 2 ([6, 5, 7]):
c = 0: dp[0] = 6 + min(dp[0], dp[1]) = 6 + min(4, 1) = 6 + 1 = 7
c = 1: dp[1] = 5 + min(dp[1], dp[2]) = 5 + min(1, 8) = 5 + 1 = 6
c = 2: dp[2] = 7 + min(dp[2], dp[3]) = 7 + min(8, 3) = 7 + 3 = 10
dp array state: [7, 6, 10, 3]

Step 3: Move up to row r = 1 ([3, 4]):
c = 0: dp[0] = 3 + min(dp[0], dp[1]) = 3 + min(7, 6) = 3 + 6 = 9
c = 1: dp[1] = 4 + min(dp[1], dp[2]) = 4 + min(6, 10) = 4 + 6 = 10
dp array state: [9, 10, 10, 3]

Step 4: Move up to apex r = 0 ([2]):
c = 0: dp[0] = 2 + min(dp[0], dp[1]) = 2 + min(9, 10) = 2 + 9 = 11
dp array state: [11, 10, 10, 3]

Final Minimum Path Sum: dp[0] = 11
Path taken: 2 -> 3 -> 5 -> 1 (sum = 11)
```

---

### Solved Examples with Multiple Inputs

| Triangle Input | Bottom-Up Compression Stages | Final Apex `dp[0]` | Optimal Path |
|---|---|---|---|
| `[[2],[3,4],[6,5,7],[4,1,8,3]]` | `[4,1,8,3]` $\to$ `[7,6,10]` $\to$ `[9,10]` $\to$ `[11]` | `11` | $2 \to 3 \to 5 \to 1$ |
| `[[-10]]` | `[-10]` | `-10` | Single element |
| `[[-1],[2,3],[1,-1,-3]]` | `[1,-1,-3]` $\to$ `[1,0]` $\to$ `[-1]` | `-1` | $-1 \to 3 \to -3$ |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def minimumTotal(self, triangle: list[list[int]]) -> int:
        n = len(triangle)
        # Initialize 1D DP table with the bottom-most row
        dp = list(triangle[-1])
        
        # Traverse from second-to-last row up to the top
        for r in range(n - 2, -1, -1):
            for c in range(r + 1):
                dp[c] = triangle[r][c] + min(dp[c], dp[c + 1])
                
        return dp[0]
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int minimumTotal(std::vector<std::vector<int>>& triangle) {
        int n = static_cast<int>(triangle.size());
        // Initialize DP array with bottom row
        std::vector<int> dp = triangle.back();
        
        // Bottom-up DP: move upwards to apex
        for (int r = n - 2; r >= 0; --r) {
            for (int c = 0; c <= r; ++c) {
                dp[c] = triangle[r][c] + std::min(dp[c], dp[c + 1]);
            }
        }
        
        return dp[0];
    }
};
```

#### Java 17
```java
class Solution {
    public int minimumTotal(java.util.List<java.util.List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n];
        
        // Copy bottom row into 1D DP array
        for (int c = 0; c < n; c++) {
            dp[c] = triangle.get(n - 1).get(c);
        }
        
        // Bottom-up upward aggregation
        for (int r = n - 2; r >= 0; r--) {
            for (int c = 0; c <= r; c++) {
                dp[c] = triangle.get(r).get(c) + Math.min(dp[c], dp[c + 1]);
            }
        }
        
        return dp[0];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^2)$, where $n$ is the number of rows in `triangle`. The total number of elements processed is $\frac{n(n + 1)}{2}$. Each cell does an $\mathcal{O}(1)$ minimum and addition operation.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the 1D DP array representing a single row of size at most $n$.

---

### Takeaway Pattern & Interview Traps

1. **Why Bottom-Up Dominates Top-Down:** In top-down, boundary elements ($c = 0$ and $c = r$) only have one parent, requiring branching logic. Moving bottom-up guarantees that every cell $(r, c)$ has exactly two children $(r+1, c)$ and $(r+1, c+1)$, completely removing edge cases.
2. **No Post-Processing Search:** Top-down requires scanning all $n$ cells of the bottom row with $\min$ at the end. Bottom-up naturally condenses the optimal answer into $dp[0]$.
3. **In-Place Modification Consideration:** While you can modify `triangle` in-place for $\mathcal{O}(1)$ space, mutating input parameters is often considered bad practice in production systems. The $\mathcal{O}(n)$ 1D array is the clean, non-destructive standard.