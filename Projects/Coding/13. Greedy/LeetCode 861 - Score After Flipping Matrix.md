---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 861: Score After Flipping Matrix"
tags:
  - leetcode
  - coding
  - greedy
  - bit-manipulation
  - matrix
  - google
  - amazon
  - microsoft
---

# LeetCode 861: Score After Flipping Matrix

**Target Companies:** Google, Amazon, Microsoft  
**Difficulty:** Medium  
**Topic:** Greedy / Bit Manipulation / Matrix  

---

### Problem Statement

You are given an `m x n` binary matrix `grid`.

A **move** consists of choosing any row or column and toggling each value in that row or column (i.e., changing all `0`'s to `1`'s, and all `1`'s to `0`'s).

Every row of the matrix is interpreted as a binary number, and the **score** of the matrix is the sum of these numbers.

Return *the highest possible score after making any number of moves (including zero moves)*.

---

### Input & Output Formats & Constraints

- **Input:** An `m x n` binary matrix `grid`.
- **Output:** An integer representing the maximum possible score.
- **Constraints:**
  - `m == grid.length`
  - `n == grid[i].length`
  - `1 <= m, n <= 20`
  - `grid[i][j]` is either `0` or `1`.

---

### Key Idea & Intuition

#### 1. Most Significant Bit (MSB) Dominance
In any binary number of length $n$, the most significant bit (column 0) carries weight $2^{n-1}$. Notice that:
$$2^{n-1} > \sum_{k=0}^{n-2} 2^k = 2^{n-1} - 1$$
Having a `1` at column 0 is strictly more valuable than having `1`s at *all* subsequent columns combined. Therefore, to maximize the row sum, **every single row must start with a `1` at column 0**.

#### 2. Row Flips Determined First
If `grid[r][0] == 0`, we are forced to toggle row $r$ so that its MSB becomes `1`. If `grid[r][0] == 1`, we must keep row $r$ unflipped.
This completely fixes all row operations!
Notice that we do not even need to mutate the matrix in memory: after row orientation, the effective value at $(r, c)$ is:
$$\text{effective\_val}(r, c) = \text{grid}[r][c] \oplus (\text{grid}[r][0] == 0) = \begin{cases} \text{grid}[r][c] & \text{if } \text{grid}[r][0] == 1 \\ 1 - \text{grid}[r][c] & \text{if } \text{grid}[r][0] == 0 \end{cases}$$

#### 3. Independent Column Flips
With all row orientations fixed, each column $c \in [1, n-1]$ can be flipped independently:
- Let $k$ be the count of rows with a `1` in column $c$ after row normalization.
- If we do not flip column $c$, we have $k$ ones.
- If we flip column $c$, we invert all bits, giving $m - k$ ones.
- Greedily, we choose whichever orientation gives more `1`s:
  $$\max(k, m - k)$$
- Column $c$ contributes $\max(k, m - k) \times 2^{n - 1 - c}$ to the total score.

---

### Solution Approach (Step-by-Step)

1. **Calculate Column 0 Contribution:**
   - Every row's MSB is guaranteed to be `1`.
   - Contribution of column 0 is $m \times 2^{n-1} = m \ll (n - 1)$.
2. **Evaluate Columns $1$ to $n - 1$:**
   - For each column $c$ from $1$ to $n - 1$:
     - Count $ones = \sum_{r=0}^{m-1} (\text{grid}[r][c] == \text{grid}[r][0])$.
       *(Note: if $\text{grid}[r][c] == \text{grid}[r][0]$, after flipping row $r$ to make $\text{grid}[r][0]=1$, $\text{grid}[r][c]$ will also be $1$!)*
     - The maximum number of $1$s we can achieve in column $c$ is $\max(ones, m - ones)$.
     - Add $\max(ones, m - ones) \times (1 \ll (n - 1 - c))$ to the running total.
3. **Return Output:**
   - Return the accumulated total score.

---

### Visual Algorithm Walkthrough

#### Trace for `grid = [[0,0,1,1],[1,0,1,0],[1,1,0,0]]` ($m = 3, n = 4$)
```
Initial Matrix:
Row 0: [0, 0, 1, 1]  (starts with 0 -> MUST FLIP ROW)
Row 1: [1, 0, 1, 0]  (starts with 1 -> KEEP)
Row 2: [1, 1, 0, 0]  (starts with 1 -> KEEP)

After Row Normalization:
Row 0: [1, 1, 0, 0]
Row 1: [1, 0, 1, 0]
Row 2: [1, 1, 0, 0]

Col 0 (weight 2^3 = 8):
All 3 rows have 1 -> Contribution = 3 * 8 = 24

Col 1 (weight 2^2 = 4):
Bits: [1, 0, 1] -> Count of 1s = 2, Count of 0s = 1.
max(2, 1) = 2 -> Keep column.
Contribution = 2 * 4 = 8

Col 2 (weight 2^1 = 2):
Bits: [0, 1, 0] -> Count of 1s = 1, Count of 0s = 2.
max(1, 2) = 2 -> Flip column! (Bits become [1, 0, 1])
Contribution = 2 * 2 = 4

Col 3 (weight 2^0 = 1):
Bits: [0, 0, 0] -> Count of 1s = 0, Count of 0s = 3.
max(0, 3) = 3 -> Flip column! (Bits become [1, 1, 1])
Contribution = 3 * 1 = 3

Total Max Score = 24 + 8 + 4 + 3 = 39.
```

---

### Solved Examples with Multiple Inputs

| Input `grid` | Row Flips Required | Column 1..n-1 Optimized Counts | Math Sum | Output |
|---|---|---|---|---|
| `[[0,0,1,1],[1,0,1,0],[1,1,0,0]]` | Flip row 0 | Col 1: 2, Col 2: 2, Col 3: 3 | $24 + 8 + 4 + 3 = 39$ | `39` |
| `[[0]]` | Flip row 0 | None ($n=1$) | $1 \times 1 = 1$ | `1` |
| `[[1]]` | None | None ($n=1$) | $1 \times 1 = 1$ | `1` |
| `[[0,1],[1,1]]` | Flip row 0 $\to$ `[[1,0],[1,1]]` | Col 1: bits are `0, 1` $\to \max(1, 1)=1$ | $2 \times 2^1 + 1 \times 2^0 = 5$ | `5` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def matrixScore(self, grid: list[list[int]]) -> int:
        m: int = len(grid)
        n: int = len(grid[0])
        
        # Column 0: every row is forced to have MSB = 1
        total_score: int = m * (1 << (n - 1))
        
        # Evaluate columns 1 to n - 1
        for c in range(1, n):
            # A bit at (r, c) becomes 1 if grid[r][c] == grid[r][0]
            same_as_msb = sum(1 for r in range(m) if grid[r][c] == grid[r][0])
            max_ones = max(same_as_msb, m - same_as_msb)
            total_score += max_ones * (1 << (n - 1 - c))
            
        return total_score
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int matrixScore(const std::vector<std::vector<int>>& grid) {
        int m = static_cast<int>(grid.size());
        int n = static_cast<int>(grid[0].size());
        
        // MSB (column 0) contribution: all m rows start with 1
        int total_score = m * (1 << (n - 1));
        
        // Optimize columns 1 to n - 1
        for (int c = 1; c < n; ++c) {
            int same_as_msb = 0;
            for (int r = 0; r < m; ++r) {
                // If grid[r][c] == grid[r][0], after row flip it becomes 1
                if (grid[r][c] == grid[r][0]) {
                    same_as_msb++;
                }
            }
            int max_ones = std::max(same_as_msb, m - same_as_msb);
            total_score += max_ones * (1 << (n - 1 - c));
        }
        
        return total_score;
    }
};
```

#### Java 17
```java
class Solution {
    public int matrixScore(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        
        // Force column 0 MSB to 1 for all m rows
        int totalScore = m * (1 << (n - 1));
        
        // Greedily maximize 1s in columns 1 to n - 1
        for (int c = 1; c < n; c++) {
            int sameAsMsb = 0;
            for (int r = 0; r < m; r++) {
                if (grid[r][c] == grid[r][0]) {
                    sameAsMsb++;
                }
            }
            int maxOnes = Math.max(sameAsMsb, m - sameAsMsb);
            totalScore += maxOnes * (1 << (n - 1 - c));
        }
        
        return totalScore;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$, where $m$ is the number of rows and $n$ is the number of columns. We iterate through every cell in the matrix exactly once to check the condition `grid[r][c] == grid[r][0]`.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. We compute the score directly on-the-fly without copying or modifying the matrix.

---

### Takeaway Pattern & Interview Traps

1. **Virtual Row Normalization:** The identity `(grid[r][c] == grid[r][0])` elegantly encapsulates whether the bit will be `1` after row $r$ is normalized, eliminating the need to physically mutate the grid.
2. **MSB Strict Dominance:** Always explain *why* row flips take precedence over column flips: $2^{n-1} > \sum_{i=0}^{n-2} 2^i$. There is no scenario where leaving an MSB as `0` to get more `1`s downstream results in a larger sum.
3. **Bit-Shift Safety:** With $n \le 20$, `1 << (n - 1)` easily fits within a standard 32-bit signed integer without risk of overflow.