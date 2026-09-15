---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 542: 01 Matrix"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - matrix
  - bfs
  - google
  - amazon
  - meta
---

# LeetCode 542: 01 Matrix

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, Uber  
**Difficulty:** Medium  
**Topic:** 2-Pass Grid Dynamic Programming / Multi-Source BFS / Manhattan Distance  

---

### Problem Statement

Given an `m x n` binary matrix `mat`, return the distance of the nearest `0` for each cell.

The distance between two adjacent cells is `1`.

---

### Input & Output Formats & Constraints

- **Input:** `mat: List[List[int]]` — Binary matrix where `mat[i][j]` is either `0` or `1`.
- **Output:** `List[List[int]]` — Matrix of same dimensions containing nearest distance to a `0`.
- **Constraints:**
  - $m == \text{mat.length}$
  - $n == \text{mat}[i].\text{length}$
  - $1 \le m, n \le 10^4$
  - $1 \le m \times n \le 10^4$
  - There is at least one `0` in `mat`.

---

### Key Idea & Intuition

1. **Why Single-Pass DP Fails:**
   - A cell's distance depends on all $4$ orthogonal neighbors: Top, Bottom, Left, and Right.
   - Standard grid DP traverses in a single direction (e.g. top-to-bottom, left-to-right), meaning bottom and right neighbors have not yet been evaluated.

2. **The 2-Pass Dynamic Programming Technique:**
   - Instead of using a queue (Multi-Source BFS), we decompose the $4$-directional propagation into two independent directional scans:
     - **Pass 1 (Top-Left $\to$ Bottom-Right):**
       - Propagates distances from cells above and to the left:
         $$\text{dist}[r][c] = \min(\text{dist}[r][c], \ \text{dist}[r-1][c] + 1, \ \text{dist}[r][c-1] + 1)$$
     - **Pass 2 (Bottom-Right $\to$ Top-Left):**
       - Propagates distances from cells below and to the right:
         $$\text{dist}[r][c] = \min(\text{dist}[r][c], \ \text{dist}[r+1][c] + 1, \ \text{dist}[r][c+1] + 1)$$
   - Combining both passes guarantees that shortest path paths from any direction (including diagonal Manhattan zig-zags) are completely propagated!

3. **In-Place Memory Optimization:**
   - Cells with `mat[r][c] == 0` have distance `0`.
   - Cells with `mat[r][c] == 1` are initialized to a sentinel infinity $\infty = m + n$.
   - The matrix can be updated directly in-place with zero queue overhead.

---

### Solution Approach (Step-by-Step)

1. **Initialize Distances:**
   - Allocate `dist` of size $m \times n$.
   - For all cells:
     - If `mat[r][c] == 0`: `dist[r][c] = 0`.
     - Else: `dist[r][c] = m + n` (sentinel upper bound).
2. **Pass 1 (Top-Left to Bottom-Right):**
   - For $r$ from $0$ to $m - 1$:
     - For $c$ from $0$ to $n - 1$:
       - If $r > 0$: `dist[r][c] = min(dist[r][c], dist[r-1][c] + 1)`.
       - If $c > 0$: `dist[r][c] = min(dist[r][c], dist[r][c-1] + 1)`.
3. **Pass 2 (Bottom-Right to Top-Left):**
   - For $r$ from $m - 1$ down to $0$:
     - For $c$ from $n - 1$ down to $0$:
       - If $r < m - 1$: `dist[r][c] = min(dist[r][c], dist[r+1][c] + 1)`.
       - If $c < n - 1$: `dist[r][c] = min(dist[r][c], dist[r][c+1] + 1)`.
4. **Return:**
   - Return `dist`.

---

### Visual Algorithm Walkthrough

For input matrix:
```
[ 0, 0, 0 ]
[ 0, 1, 0 ]
[ 1, 1, 1 ]
```

**Initialization (Sentinel $\infty = 100$ for 1s):**
```
[ 0,   0,   0   ]
[ 0, 100,   0   ]
[100, 100, 100  ]
```

**Pass 1: Top-Left to Bottom-Right:**
```
r=1, c=1: min(100, dist[0][1]+1, dist[1][0]+1) = min(100, 0+1, 0+1) = 1
r=2, c=0: min(100, dist[1][0]+1) = 0 + 1 = 1
r=2, c=1: min(100, dist[1][1]+1, dist[2][0]+1) = min(100, 1+1, 1+1) = 2
r=2, c=2: min(100, dist[1][2]+1, dist[2][1]+1) = min(100, 0+1, 2+1) = 1

After Pass 1:
[ 0, 0, 0 ]
[ 0, 1, 0 ]
[ 1, 2, 1 ]
```

**Pass 2: Bottom-Right to Top-Left:**
```
r=2, c=2: 1
r=2, c=1: min(2, dist[2][2]+1) = min(2, 1+1) = 2
r=2, c=0: min(1, dist[2][1]+1) = 1
All other cells already optimal!

Final Result:
[ 0, 0, 0 ]
[ 0, 1, 0 ]
[ 1, 2, 1 ]
```

---

### Solved Examples with Multiple Inputs

| Case | `mat` | Pass 1 | Pass 2 (Final) | Explanation |
|---|---|---|---|---|
| **Center 1** | `[[0,0,0],[0,1,0],[0,0,0]]` | `dist[1][1] = 1` | `[[0,0,0],[0,1,0],[0,0,0]]` | Distance to adjacent 0 is 1 |
| **All Zeros** | `[[0,0],[0,0]]` | Unchanged | `[[0,0],[0,0]]` | All distances are 0 |
| **Single Zero in Corner** | `[[0,1],[1,1]]` | `[[0,1],[1,2]]` | `[[0,1],[1,2]]` | Manhattan distance propagates outwards |
| **Long Line** | `[[0,1,1,1,0]]` | `[0,1,2,3,0]` | `[0,1,2,1,0]` | Pass 2 fixes right-to-left distance |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — 2-Pass DP)
```python
from typing import List

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        INF = m + n
        
        dist = [[0 if mat[r][c] == 0 else INF for c in range(n)] for r in range(m)]
        
        # Pass 1: Top-Left to Bottom-Right
        for r in range(m):
            for c in range(n):
                if r > 0:
                    dist[r][c] = min(dist[r][c], dist[r - 1][c] + 1)
                if c > 0:
                    dist[r][c] = min(dist[r][c], dist[r][c - 1] + 1)
                    
        # Pass 2: Bottom-Right to Top-Left
        for r in range(m - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                if r < m - 1:
                    dist[r][c] = min(dist[r][c], dist[r + 1][c] + 1)
                if c < n - 1:
                    dist[r][c] = min(dist[r][c], dist[r][c + 1] + 1)
                    
        return dist
```

#### 2. C++ (C++17 / STL — 2-Pass DP)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> updateMatrix(std::vector<std::vector<int>>& mat) {
        int m = mat.size();
        int n = mat[0].size();
        const int INF = m + n;

        std::vector<std::vector<int>> dist(m, std::vector<int>(n, INF));

        // Initialize zeros
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (mat[r][c] == 0) {
                    dist[r][c] = 0;
                }
            }
        }

        // Pass 1: Top-Left -> Bottom-Right
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (r > 0) dist[r][c] = std::min(dist[r][c], dist[r - 1][c] + 1);
                if (c > 0) dist[r][c] = std::min(dist[r][c], dist[r][c - 1] + 1);
            }
        }

        // Pass 2: Bottom-Right -> Top-Left
        for (int r = m - 1; r >= 0; --r) {
            for (int c = n - 1; c >= 0; --c) {
                if (r < m - 1) dist[r][c] = std::min(dist[r][c], dist[r + 1][c] + 1);
                if (c < n - 1) dist[r][c] = std::min(dist[r][c], dist[r][c + 1] + 1);
            }
        }

        return dist;
    }
};
```

#### 3. Java (Modern, Typed — 2-Pass DP)
```java
class Solution {
    public int[][] updateMatrix(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int inf = m + n;

        int[][] dist = new int[m][n];

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (mat[r][c] != 0) {
                    dist[r][c] = inf;
                }
            }
        }

        // Pass 1: Top-Left -> Bottom-Right
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (r > 0) dist[r][c] = Math.min(dist[r][c], dist[r - 1][c] + 1);
                if (c > 0) dist[r][c] = Math.min(dist[r][c], dist[r][c - 1] + 1);
            }
        }

        // Pass 2: Bottom-Right -> Top-Left
        for (int r = m - 1; r >= 0; r--) {
            for (int c = n - 1; c >= 0; c--) {
                if (r < m - 1) dist[r][c] = Math.min(dist[r][c], dist[r + 1][c] + 1);
                if (c < n - 1) dist[r][c] = Math.min(dist[r][c], dist[r][c + 1] + 1);
            }
        }

        return dist;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$  
  Exactly two linear sweeps through the $m \times n$ grid. Each cell examines at most two neighbors per pass in $\mathcal{O}(1)$ time. Total operations $\approx 2 \times 10^4$, executing in under $10$ ms.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space  
  Modifies the allocated return matrix directly without using BFS queues or recursion call stacks.

---

### Takeaway Pattern & Interview Traps

1. **Why Sentinel Infinity Must Be Bounded:**
   - Using `INT_MAX` can lead to signed 32-bit integer overflow when evaluating `dist[r - 1][c] + 1`.
   - The maximum possible Manhattan distance across an $m \times n$ matrix is $(m - 1) + (n - 1) < m + n$. Setting $\text{INF} = m + n$ is safe, fits easily within standard integer types, and eliminates any possibility of overflow.
2. **2-Pass DP vs. Multi-Source BFS:**
   - While Multi-Source BFS is also $\mathcal{O}(m \times n)$, it requires queue memory allocations and hash set tracking. 2-Pass DP is cache-friendly, performs sequential memory access, and requires zero dynamic container allocations.