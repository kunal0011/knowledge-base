---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 864: Shortest Path to Get All Keys"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - bitmask
  - state-space
  - amazon
  - google
---

# LeetCode 864: Shortest Path to Get All Keys

**Target Companies:** Google (Signature Multi-Dimensional State BFS), Amazon  
**Difficulty:** Hard  
**Topic:** State-Space Graph BFS / Bitmask Representation / Shortest Path on Grid

---

### Problem Statement

You are given an `m x n` grid `grid` where:
- `'.'` is an empty cell.
- `'#'` is a wall.
- `'@'` is the starting point.
- Lowercase letters `'a'`, `'b'`, `'c'`, `'d'`, `'e'`, `'f'` represent keys.
- Uppercase letters `'A'`, `'B'`, `'C'`, `'D'`, `'E'`, `'F'` represent locks.

You start at the starting point and one move consists of walking one space in one of the four cardinal directions. You cannot walk outside the grid, or walk into a wall.

If you walk over a key, you can pick it up and you cannot carry a key that you have already picked up.

If you walk over a lock and have the corresponding lowercase key, you can open the lock.

Return the **lowest number of moves to acquire all keys**. If it is impossible, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:** `grid: List[str]`
- **Output:** `int` — Minimum number of moves to collect all keys, or `-1`.
- **Constraints:**
  - $m == \text{grid.length}$
  - $n == \text{grid}[i].\text{length}$
  - $1 \le m, n \le 30$
  - `grid[i][j]` is either a letter from `'a'` to `'f'`, `'A'` to `'F'`, `'.'`. `'#'`, or `'@'`.
  - The number of keys is in the range $[1, 6]$.
  - Each key in the grid has a unique letter.
  - Each key in the grid has a corresponding lock.
  - There is exactly one starting cell `'@'`.

---

### Key Idea & Intuition

- **Why a 2D Visited Array Fails:**
  - To pick up a key, you may need to navigate through a corridor, collect a key, and **backtrack through the exact same cells** with the newly acquired key to unlock a door.
  - A traditional `visited[r][c]` prevents backtracking, rendering the problem unsolvable.
- **State-Space Expansion via Bitmask:**
  - Because there are at most 6 keys ($k \le 6$), the set of keys currently in possession can be represented by a **6-bit integer (bitmask)**:
    $$\text{mask} \in [0, 2^6 - 1] = [0, 63]$$
    - Possessing key `'a'` corresponds to bit 0: `mask & (1 << 0) != 0`.
    - Possessing key `'b'` corresponds to bit 1: `mask & (1 << 1) != 0`.
  - A valid state in BFS is the 3-tuple:
    $$(r, c, \text{key\_mask})$$
  - The total number of states in the state space is bounded by:
    $$M \times N \times 2^k \le 30 \times 30 \times 64 = 57,600$$
- **Transition Rules:**
  - Empty cell `.` or start `@`: transition to `(nr, nc, key_mask)`.
  - Key `'a'` - `'f'`: update mask `next_mask = key_mask | (1 << (ord(char) - ord('a')))`. If `next_mask == (1 << total_keys) - 1`, all keys are acquired $\implies$ return `steps + 1`!
  - Lock `'A'` - `'F'`: passable only if `key_mask & (1 << (ord(char) - ord('A'))) != 0`.
  - Wall `#`: cannot enter.

---

### Solution Approach (Step-by-Step)

1. Parse grid: find starting coordinate `(start_r, start_c)` and count `total_keys`.
2. Target bitmask: `target_mask = (1 << total_keys) - 1`.
3. Initialize `queue = deque([(start_r, start_c, 0, 0)])` storing `(r, c, key_mask, steps)`.
4. Initialize `visited = {(start_r, start_c, 0)}`.
5. While `queue` is not empty:
   - Pop `r, c, mask, steps`.
   - If `mask == target_mask`: return `steps`.
   - For `(dr, dc)` in `[(0, 1), (0, -1), (1, 0), (-1, 0)]`:
     - Let `nr = r + dr, nc = c + dc`.
     - Check bounds $0 \le nr < m$ and $0 \le nc < n$.
     - Let `ch = grid[nr][nc]`.
     - If `ch == '#'`: continue (wall).
     - If `ch.isupper()`:
       - Lock check: `if not (mask & (1 << (ord(ch) - ord('A')))): continue`.
     - Compute `next_mask = mask`:
       - If `ch.islower()`: `next_mask |= (1 << (ord(ch) - ord('a')))`.
     - If `(nr, nc, next_mask)` not in `visited`:
       - `visited.add((nr, nc, next_mask))`
       - `queue.append((nr, nc, next_mask, steps + 1))`
6. Return `-1`.

---

### Visual Algorithm Walkthrough

```
Grid (3x3):
@ . a
# # #
b 2 B

Total keys: 'a', 'b' -> total_keys = 2, target_mask = (1 << 2) - 1 = 3 (binary '11')

Start: (0, 0) with mask = 0 (binary '00')
Queue: [(0, 0, mask=0, steps=0)]

Step 1:
Move (0, 1), empty: state (0, 1, mask=0)

Step 2:
Move (0, 2), key 'a':
  next_mask = 0 | (1 << 0) = 1 (binary '01')
  state (0, 2, mask=1, steps=2)

Backtracking is now permitted because mask changed from 0 to 1!
Move (0, 1) with mask=1 (NOT in visited because previously visited with mask=0)
Move (0, 0) with mask=1

State-space graph cleanly decouples cycles from key-dependent progress.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Key & Lock Corridor
- **Input:** `grid = ["@.a..","###.#","b.A.B"]`
- **Output:** `8`

#### Example 2: Re-entering Area with New Key
- **Input:** `grid = ["@..aA","..B#.","....b"]`
- **Output:** `6`

#### Example 3: Unreachable Key
- **Input:** `grid = ["@Aa"]`
- **Analysis:** Lock 'A' blocks key 'a'. Key 'a' cannot be reached to open 'A'.
- **Output:** `-1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        m, n = len(grid), len(grid[0])
        start_r = start_c = total_keys = 0
        
        for r in range(m):
            for c in range(n):
                ch = grid[r][c]
                if ch == '@':
                    start_r, start_c = r, c
                elif ch.islower():
                    total_keys += 1
                    
        target_mask = (1 << total_keys) - 1
        queue = deque([(start_r, start_c, 0, 0)])  # (r, c, key_mask, steps)
        visited = {(start_r, start_c, 0)}
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            r, c, mask, steps = queue.popleft()
            
            if mask == target_mask:
                return steps
                
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    ch = grid[nr][nc]
                    if ch == '#':
                        continue
                        
                    # Lock verification
                    if ch.isupper() and not (mask & (1 << (ord(ch) - ord('A')))):
                        continue
                        
                    next_mask = mask
                    if ch.islower():
                        next_mask |= (1 << (ord(ch) - ord('a')))
                        
                    if (nr, nc, next_mask) not in visited:
                        visited.add((nr, nc, next_mask))
                        queue.append((nr, nc, next_mask, steps + 1))
                        
        return -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>
#include <queue>
#include <tuple>

class Solution {
public:
    int shortestPathAllKeys(std::vector<std::string>& grid) {
        int m = grid.size(), n = grid[0].size();
        int startR = 0, startC = 0, totalKeys = 0;

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                char ch = grid[r][c];
                if (ch == '@') {
                    startR = r;
                    startC = c;
                } else if (ch >= 'a' && ch <= 'f') {
                    totalKeys++;
                }
            }
        }

        int targetMask = (1 << totalKeys) - 1;
        // visited[r][c][mask]
        std::vector<std::vector<std::vector<bool>>> visited(
            m, std::vector<std::vector<bool>>(n, std::vector<bool>(1 << totalKeys, false)));

        std::queue<std::tuple<int, int, int, int>> q; // {r, c, mask, steps}
        q.push({startR, startC, 0, 0});
        visited[startR][startC][0] = true;

        int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!q.empty()) {
            auto [r, c, mask, steps] = q.front();
            q.pop();

            if (mask == targetMask) {
                return steps;
            }

            for (auto& d : dirs) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                    char ch = grid[nr][nc];
                    if (ch == '#') continue;

                    // If lock, verify key possession
                    if (ch >= 'A' && ch <= 'F') {
                        if (!(mask & (1 << (ch - 'A')))) continue;
                    }

                    int nextMask = mask;
                    if (ch >= 'a' && ch <= 'f') {
                        nextMask |= (1 << (ch - 'a'));
                    }

                    if (!visited[nr][nc][nextMask]) {
                        visited[nr][nc][nextMask] = true;
                        q.push({nr, nc, nextMask, steps + 1});
                    }
                }
            }
        }

        return -1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Queue;

class Solution {
    public int shortestPathAllKeys(String[] grid) {
        int m = grid.length, n = grid[0].length();
        int startR = 0, startC = 0, totalKeys = 0;

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                char ch = grid[r].charAt(c);
                if (ch == '@') {
                    startR = r;
                    startC = c;
                } else if (ch >= 'a' && ch <= 'f') {
                    totalKeys++;
                }
            }
        }

        int targetMask = (1 << totalKeys) - 1;
        boolean[][][] visited = new boolean[m][n][1 << totalKeys];
        Queue<int[]> queue = new ArrayDeque<>(); // {r, c, mask, steps}

        queue.offer(new int[]{startR, startC, 0, 0});
        visited[startR][startC][0] = true;

        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!queue.isEmpty()) {
            int[] curr = queue.poll();
            int r = curr[0], c = curr[1], mask = curr[2], steps = curr[3];

            if (mask == targetMask) {
                return steps;
            }

            for (int[] d : dirs) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                    char ch = grid[nr].charAt(nc);
                    if (ch == '#') continue;

                    // If lock, verify key possession
                    if (ch >= 'A' && ch <= 'F') {
                        if ((mask & (1 << (ch - 'A'))) == 0) continue;
                    }

                    int nextMask = mask;
                    if (ch >= 'a' && ch <= 'f') {
                        nextMask |= (1 << (ch - 'a'));
                    }

                    if (!visited[nr][nc][nextMask]) {
                        visited[nr][nc][nextMask] = true;
                        queue.offer(new int[]{nr, nc, nextMask, steps + 1});
                    }
                }
            }
        }

        return -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M \times N \times 2^k)$ where $M, N \le 30$ and $k \le 6$ (total keys). The maximum number of states is $30 \times 30 \times 64 \approx 5.76 \times 10^4$. From each state, at most 4 cardinal moves are tested in $\mathcal{O}(1)$ time. Overall time is comfortably $\le 2.3 \times 10^5$ operations ($< 30$ ms).
- **Space Complexity:** $\mathcal{O}(M \times N \times 2^k)$ — 3D boolean array `visited[m][n][1 << k]` and the BFS queue.

---

### Takeaway Pattern & Interview Traps

1. **State Space Augmentation:** Whenever a problem permits revisiting nodes upon acquiring a resource (keys, fuel, coins), append a state representation (bitmask) to the spatial coordinates `(r, c, state)`.
2. **Bitwise Precedence:** In C++/Java/Python, bitwise operators (`&`, `|`, `^`) have lower precedence than relational operators (`==`, `!=`). Always wrap bitwise checks in parentheses: `if ((mask & (1 << k)) == 0)`.
3. **Continuous Target Check:** Target check `mask == target_mask` immediately identifies the goal at the earliest BFS level.