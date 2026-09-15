---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 174: Dungeon Game"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - matrix
  - amazon
  - google
  - microsoft
---

# LeetCode 174: Dungeon Game

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Array / Matrix  

---

### Problem Statement

The demons had captured the princess and imprisoned her in the **bottom-right corner** of a `dungeon`. The dungeon consists of `m x n` rooms laid out in a 2D grid. Our valiant knight was initially positioned in the **top-left room** and must fight his way through the dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health point drops to `0` or below, he dies immediately.

Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health upon entering these rooms; other rooms are either empty (represented as `0`) or contain magic orbs that increase the knight's health (represented by positive integers).

To reach the princess as quickly as possible, the knight decides to move only **rightward** or **downward** in each step.

Return *the knight's minimum initial health so that he is able to rescue the princess*.

---

### Input & Output Formats & Constraints

- **Input:** An $m \times n$ integer matrix `dungeon`.
- **Output:** An integer representing the minimum positive initial health required.
- **Constraints:**
  - `m == dungeon.length`
  - `n == dungeon[i].length`
  - `1 <= m, n <= 200`
  - `-1000 <= dungeon[i][j] <= 1000`

---

### Key Idea & Intuition

#### Why Forward DP Fails
In a forward DP approach starting from $(0, 0)$, we would want to maximize remaining health. However, a path with higher health at $(i, j)$ might have absorbed an orb too late, while a path with lower health earlier might fail an immediate obstacle. Future survival demands depend strictly on **upcoming** hazards. Because future requirements dictate present prerequisites, the problem exhibits optimal substructure in the **reverse direction**!

#### Backward Dynamic Programming
We define $dp[i][j]$ as the **minimum health required upon entering cell $(i, j)$** such that the knight can successfully reach the destination $(m - 1, n - 1)$ alive.

1. From cell $(i, j)$, the knight can only move **Right** to $(i, j + 1)$ or **Down** to $(i + 1, j)$.
2. To minimize the initial requirement, the knight will choose the branch that requires **less health to survive**:
   $$\text{min\_needed\_next} = \min(dp[i + 1][j], dp[i][j + 1])$$
3. The health required upon *entering* $(i, j)$ must be sufficient such that after adding `dungeon[i][j]`, the knight has at least $\text{min\_needed\_next}$:
   $$\text{health\_before} + dungeon[i][j] \ge \text{min\_needed\_next} \implies \text{health\_before} \ge \text{min\_needed\_next} - dungeon[i][j]$$
4. Crucially, the knight must remain alive at all times, meaning health upon entering $(i, j)$ can **never drop below 1**:
   $$dp[i][j] = \max(1, \, \text{min\_needed\_next} - dungeon[i][j])$$

#### Sentinel Boundary Initialization
To eliminate out-of-bounds branching for edge cells, we pad the table with an extra row and column of size $(m + 1) \times (n + 1)$ filled with $\infty$.
By setting:
$$dp[m][n - 1] = 1 \quad \text{and} \quad dp[m - 1][n] = 1$$
the base cell $(m - 1, n - 1)$ naturally evaluates:
$$dp[m - 1][n - 1] = \max(1, \min(1, 1) - dungeon[m - 1][n - 1]) = \max(1, 1 - dungeon[m - 1][n - 1])$$

---

### Solution Approach (Step-by-Step)

1. **Table Dimensions & Sentinels:**
   - Initialize 2D array `dp` of size $(m + 1) \times (n + 1)$ filled with $\infty$.
   - Set $dp[m][n - 1] = 1$ and $dp[m - 1][n] = 1$.
2. **Reverse Traversal:**
   - Iterate row $i$ from $m - 1$ down to 0:
     - Iterate col $j$ from $n - 1$ down to 0:
       - $need = \min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]$.
       - $dp[i][j] = \max(1, need)$.
3. **Return:**
   - Return $dp[0][0]$.

---

### Visual Algorithm Walkthrough

#### Trace for `dungeon = [[-2, -3, 3], [-5, -10, 1], [10, 30, -5]]`
```
Dungeon Matrix:
  [-2,  -3,   3]
  [-5, -10,   1]
  [10,  30,  -5]

Backward DP Table filling from bottom-right (2, 2) to top-left (0, 0):

Row 2:
(2, 2): need = min(1, 1) - (-5) = 6 -> dp[2][2] = max(1, 6) = 6
(2, 1): need = min(dp[2][2]=6, inf) - 30 = -24 -> dp[2][1] = max(1, -24) = 1
(2, 0): need = min(dp[2][1]=1, inf) - 10 = -9 -> dp[2][0] = max(1, -9) = 1

Row 1:
(1, 2): need = min(dp[2][2]=6, inf) - 1 = 5 -> dp[1][2] = max(1, 5) = 5
(1, 1): need = min(dp[2][1]=1, dp[1][2]=5) - (-10) = 1 + 10 = 11 -> dp[1][1] = 11
(1, 0): need = min(dp[2][0]=1, dp[1][1]=11) - (-5) = 1 + 5 = 6 -> dp[1][0] = 6

Row 0:
(0, 2): need = min(dp[1][2]=5, inf) - 3 = 2 -> dp[0][2] = max(1, 2) = 2
(0, 1): need = min(dp[1][1]=11, dp[0][2]=2) - (-3) = 2 + 3 = 5 -> dp[0][1] = 5
(0, 0): need = min(dp[1][0]=6, dp[0][1]=5) - (-2) = 5 + 2 = 7 -> dp[0][0] = 7

Final DP Grid:
  [ 7,   5,   2 ]
  [ 6,  11,   5 ]
  [ 1,   1,   6 ]

Minimum Initial Health required: dp[0][0] = 7.
Path: (0,0)[HP:7-2=5] -> (0,1)[HP:5-3=2] -> (0,2)[HP:2+3=5] -> (1,2)[HP:5+1=6] -> (2,2)[HP:6-5=1] (Alive!).
```

---

### Solved Examples with Multiple Inputs

| `dungeon` Grid | Backward Target Need | Optimal Path Steps | Output |
|---|---|---|---|
| `[[-2,-3,3],[-5,-10,1],[10,30,-5]]` | Target cell needs $6$ HP | $(0,0) \to (0,1) \to (0,2) \to (1,2) \to (2,2)$ | `7` |
| `[[0]]` | Cell gives 0 | Knight needs 1 HP | `1` |
| `[[100]]` | Cell gives +100 | Knight still must start alive with at least 1 HP | `1` |
| `[[-20]]` | Cell takes 20 | Needs $1 - (-20) = 21$ HP | `21` |

---

### Multi-Language Implementations

#### Python 3
```python
import math

class Solution:
    def calculateMinimumHP(self, dungeon: list[list[int]]) -> int:
        m: int = len(dungeon)
        n: int = len(dungeon[0])
        
        # DP table with infinity padding
        dp: list[list[float]] = [[math.inf] * (n + 1) for _ in range(m + 1)]
        dp[m][n - 1] = 1
        dp[m - 1][n] = 1
        
        # Backward DP from bottom-right to top-left
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                min_next = min(dp[i + 1][j], dp[i][j + 1])
                dp[i][j] = max(1, min_next - dungeon[i][j])
                
        return int(dp[0][0])
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int calculateMinimumHP(const std::vector<std::vector<int>>& dungeon) {
        int m = static_cast<int>(dungeon.size());
        int n = static_cast<int>(dungeon[0].size());

        // Sentinel grid initialized to INT_MAX
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, INT_MAX));
        dp[m][n - 1] = 1;
        dp[m - 1][n] = 1;

        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                int min_next = std::min(dp[i + 1][j], dp[i][j + 1]);
                dp[i][j] = std::max(1, min_next - dungeon[i][j]);
            }
        }

        return dp[0][0];
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        int n = dungeon[0].length;

        int[][] dp = new int[m + 1][n + 1];
        for (int[] row : dp) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        dp[m][n - 1] = 1;
        dp[m - 1][n] = 1;

        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int minNext = Math.min(dp[i + 1][j], dp[i][j + 1]);
                dp[i][j] = Math.max(1, minNext - dungeon[i][j]);
            }
        }

        return dp[0][0];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$, where $m$ is the number of rows and $n$ is the number of columns. Each room is evaluated once in reverse with $\mathcal{O}(1)$ operations.
- **Space Complexity:** $\mathcal{O}(m \times n)$ auxiliary space for the DP table (can be optimized to $\mathcal{O}(n)$ using a 1D rolling array).

---

### Takeaway Pattern & Interview Traps

1. **Why Reverse Traversal is Essential:** When a path constraint depends on surviving unknown future requirements, formulating dynamic programming in reverse eliminates branching ambiguity.
2. **The `max(1, ...)` Floor:** Even if an orb gives $+100$ health, entering that room with $\le 0$ health causes instant death. The required entry health can never be less than $1$.
3. **Sentinel Initialization:** Setting $dp[m][n - 1] = 1$ and $dp[m - 1][n] = 1$ while keeping all other out-of-bound cells as $\infty$ allows writing a single unified loop without special checks for the bottom row, right column, or destination cell.