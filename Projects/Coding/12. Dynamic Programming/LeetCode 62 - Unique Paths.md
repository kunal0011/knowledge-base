---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 62: Unique Paths"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - grid
  - combinatorics
  - amazon
  - google
  - meta
---

# LeetCode 62: Unique Paths

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** 2D/1D Dynamic Programming / Grid Traversal / Combinatorics  

---

### Problem Statement

There is a robot on an `m x n` grid. The robot is initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either **down** or **right** at any point in time.

Given the two integers `m` and `n`, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to $2 \times 10^9$.

---

### Input & Output Formats & Constraints

- **Input:** `m: int, n: int` — Grid row and column counts.
- **Output:** `int` — Total count of unique paths.
- **Constraints:**
  - $1 \le m, n \le 100$
  - Answer fits in a signed 32-bit integer.

---

### Key Idea & Intuition

1. **Optimal Substructure:**
   - At any cell $(i, j)$, the robot could have only entered from two cells:
     - Directly from above: cell $(i - 1, j)$
     - Directly from the left: cell $(i, j - 1)$
   - By the sum rule of counting:
     $$\text{dp}[i][j] = \text{dp}[i - 1][j] + \text{dp}[i][j - 1]$$
   - Base Cases:
     - First row ($\text{dp}[0][j] = 1$) and first column ($\text{dp}[i][0] = 1$) because there is only 1 way to move strictly right or strictly down.

2. **1D Space Optimization:**
   - In row $i$, the value $\text{dp}[j]$ needs only the old value from the same column above (which is already `dp[j]`) and the newly computed value from the cell to its left (`dp[j - 1]`).
   - Transition collapses into a single 1D array:
     $$\text{dp}[j] = \text{dp}[j] + \text{dp}[j - 1]$$
   - Space complexity reduces from $\mathcal{O}(m \times n)$ to $\mathcal{O}(n)$.

3. **Mathematical Solution ($\mathcal{O}(\min(m, n))$ Time, $\mathcal{O}(1)$ Space):**
   - The robot must make a total of $(m - 1)$ Down steps and $(n - 1)$ Right steps, for a total of $(m + n - 2)$ moves.
   - Any path is uniquely determined by choosing which $(m - 1)$ steps are Down:
     $$\text{Total Paths} = \binom{m + n - 2}{m - 1} = \frac{(m + n - 2)!}{(m - 1)! (n - 1)!}$$

---

### Solution Approach (Step-by-Step)

1. **Initialize 1D DP Array:**
   - Create array `dp` of length $n$ filled with $1$ (representing the base case for the first row).
2. **Iterate Through Rows:**
   - For $i$ from $1$ to $m - 1$:
     - For $j$ from $1$ to $n - 1$:
       - `dp[j] += dp[j - 1]`.
3. **Return:**
   - Return `dp[n - 1]`.

---

### Visual Algorithm Walkthrough

For $m = 3$, $n = 4$:

```
Row 0: [ 1,  1,  1,  1 ]
Row 1: [ 1,  2,  3,  4 ]  (dp[1] = 1+1=2, dp[2] = 2+1=3, dp[3] = 3+1=4)
Row 2: [ 1,  3,  6, 10 ]  (dp[1] = 2+1=3, dp[2] = 3+3=6, dp[3] = 6+4=10)

Result: dp[3] = 10 unique paths.
```

---

### Solved Examples with Multiple Inputs

| Case | `m, n` | Formula / Table Calculation | Result | Explanation |
|---|---|---|---|---|
| **Standard 3x4** | `3, 4` | Row 2: `[1, 3, 6, 10]` | `10` | 2 Down moves, 3 Right moves |
| **Standard 3x2** | `3, 2` | $\binom{3}{1} = 3$ | `3` | `[R, D, D], [D, R, D], [D, D, R]` |
| **Single Row** | `1, 10` | Only 1 path (all Right) | `1` | Cannot move Down |
| **Single Column** | `7, 1` | Only 1 path (all Down) | `1` | Cannot move Right |
| **Single Cell** | `1, 1` | Already at destination | `1` | 0 moves needed |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — 1D Space Optimized)
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
                
        return dp[n - 1]
```

#### 2. C++ (C++17 / STL — 1D Space Optimized)
```cpp
#include <vector>

class Solution {
public:
    int uniquePaths(int m, int n) {
        std::vector<int> dp(n, 1);

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[j] += dp[j - 1];
            }
        }

        return dp[n - 1];
    }
};
```

#### 3. Java (Modern, Typed — 1D Space Optimized)
```java
import java.util.Arrays;

class Solution {
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        Arrays.fill(dp, 1);

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }

        return dp[n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$  
  The nested loops run $(m - 1) \times (n - 1)$ times, each taking $\mathcal{O}(1)$ time. With $m, n \le 100$, operations $\le 10^4$ ($< 0.5$ ms).
- **Space Complexity:** $\mathcal{O}(n)$  
  Only a 1D array of length $n$ is maintained in memory.

---

### Takeaway Pattern & Interview Traps

1. **Pascal's Triangle Connection:**
   - The entries of the grid are rotated diagonals of Pascal's Triangle. That is why the combinatorial identity $\binom{N}{K} = \binom{N-1}{K-1} + \binom{N-1}{K}$ directly mirrors the DP relation $\text{dp}[i][j] = \text{dp}[i-1][j] + \text{dp}[i][j-1]$.
2. **Combinatorial Multiplication Overflow:**
   - If using the combinatorial formula $\binom{m+n-2}{m-1}$, calculate factors iteratively: `ans = ans * (n - 1 + i) / i` to avoid intermediate integer overflow.