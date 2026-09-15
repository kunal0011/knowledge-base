---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 1334: Find the City With the Smallest Number of Neighbors at a Threshold Distance"
tags:
  - leetcode
  - coding
  - graphs
  - shortest-path
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 1334: Find the City With the Smallest Number of Neighbors at a Threshold Distance

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** All-Pairs Shortest Path / Floyd-Warshall Algorithm / Dijkstra from All Nodes

---

### Problem Statement

There are `n` cities numbered from `0` to `n - 1`. Given the array `edges` where `edges[i] = [from_i, to_i, weight_i]` represents a bidirectional and weighted edge between cities `from_i` and `to_i`, and given the integer `distanceThreshold`.

Return the city with the smallest number of cities that are reachable through some path and whose distance is **at most** `distanceThreshold`. If there are multiple such cities, return the city with the **greatest number**.

Notice that the distance of a path connecting cities `i` and `j` is equal to the sum of the edges' weights along that path.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`, `edges: List[List[int]]`, `distanceThreshold: int`
- **Output:** `int` — City index with the minimum reachable neighbors at or below threshold (break ties by largest index).
- **Constraints:**
  - $2 \le n \le 100$
  - $1 \le \text{edges.length} \le \frac{n(n - 1)}{2}$
  - $\text{edges}[i].\text{length} == 3$
  - $0 \le \text{from}_i < \text{to}_i < n$
  - $1 \le \text{weight}_i, \text{distanceThreshold} \le 10^4$
  - All pairs $(\text{from}_i, \text{to}_i)$ are distinct.

---

### Key Idea & Intuition

- **All-Pairs Shortest Path (APSP):**
  - We must determine the shortest distance between **every pair of cities** $(i, j)$ in the graph.
  - Given the constraint $n \le 100$, an $\mathcal{O}(n^3)$ algorithm performs at most $100^3 = 10^6$ operations, which executes in a few milliseconds.
- **The Floyd-Warshall Algorithm:**
  - Dynamic programming on intermediate vertices:
    Let $\text{dist}[i][j]$ be the shortest distance between city $i$ and city $j$ using only intermediate vertices from the set $\{0, 1, \dots, k\}$.
  - State Transition:
    $$\text{dist}[i][j] = \min(\text{dist}[i][j], \text{dist}[i][k] + \text{dist}[k][j])$$
  - **Critical Invariant:** The intermediate vertex loop $k$ **MUST be the outermost loop**!
- **Tie-Breaking Rule:**
  - If city $A$ and city $B$ have the same minimal count of reachable neighbors, the problem asks for $\max(A, B)$.
  - By tracking `if count <= min_count: min_count = count; best_city = i`, iterating $i$ from $0$ up to $n - 1$ automatically favors larger city indices!

---

### Solution Approach (Step-by-Step)

1. Initialize an $n \times n$ 2D matrix `dist` filled with $\infty$.
2. For each city $i$: `dist[i][i] = 0`.
3. For each edge `[u, v, w]`:
   - `dist[u][v] = w`
   - `dist[v][u] = w`
4. **Floyd-Warshall 3-nested loops:**
   - Loop intermediate vertex $k$ from $0$ to $n - 1$:
     - Loop source vertex $i$ from $0$ to $n - 1$:
       - Loop destination vertex $j$ from $0$ to $n - 1$:
         - `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`
5. Initialize `min_reachable = infinity`, `best_city = -1`.
6. For city $i$ from $0$ to $n - 1$:
   - Count how many cities $j$ ($j \ne i$) have `dist[i][j] <= distanceThreshold`.
   - If `count <= min_reachable`:
     - `min_reachable = count`
     - `best_city = i`
7. Return `best_city`.

---

### Visual Algorithm Walkthrough

```
Example: n = 4, distanceThreshold = 4
Edges: [0, 1, 3], [1, 2, 1], [1, 3, 4], [2, 3, 1]

Graph Layout:
  (0) ---3--- (1)
               | \
               1   4
               |     \
              (2) ---1--- (3)

Shortest Paths Matrix after Floyd-Warshall:
       0    1    2    3
0 [    0,   3,   4,   5 ]  -> Within threshold 4: {1, 2} (count = 2)
1 [    3,   0,   1,   2 ]  -> Within threshold 4: {0, 2, 3} (count = 3)
2 [    4,   1,   0,   1 ]  -> Within threshold 4: {0, 1, 3} (count = 3)
3 [    5,   2,   1,   0 ]  -> Within threshold 4: {1, 2} (count = 2)

Candidates with min count = 2: City 0 and City 3.
Tie-break: choose city with greatest index -> City 3!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Graph
- **Input:** `n = 4, edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold = 4`
- **Tracing Table:**
  | City $i$ | Reachable Cities within $\le 4$ | Count | Is New Best? |
  | :--- | :--- | :--- | :--- |
  | 0 | `{1 (dist 3), 2 (dist 4)}` | 2 | Yes (`best = 0`, `min = 2`) |
  | 1 | `{0 (3), 2 (1), 3 (2)}` | 3 | No |
  | 2 | `{0 (4), 1 (1), 3 (1)}` | 3 | No |
  | 3 | `{1 (2), 2 (1)}` | 2 | Yes ($2 \le 2 \implies$ `best = 3`) |
- **Output:** `3`

#### Example 2: Disconnected Component
- **Input:** `n = 5, edges = [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]], distanceThreshold = 2`
- **Output:** `0` (City 0 has only 1 neighbor within distance 2: City 1).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        # Initialize distance matrix
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
            
        for u, v, w in edges:
            dist[u][v] = w
            dist[v][u] = w
            
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        
        min_reachable = float('inf')
        best_city = -1
        
        # Count reachable cities for each city
        for i in range(n):
            reachable = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)
            # '<=' condition ensures greatest index wins ties
            if reachable <= min_reachable:
                min_reachable = reachable
                best_city = i
                
        return best_city
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int findTheCity(int n, std::vector<std::vector<int>>& edges, int distanceThreshold) {
        const int INF = 1e9;
        std::vector<std::vector<int>> dist(n, std::vector<int>(n, INF));

        for (int i = 0; i < n; ++i) {
            dist[i][i] = 0;
        }

        for (const auto& edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            dist[u][v] = w;
            dist[v][u] = w;
        }

        // Floyd-Warshall APSP
        for (int k = 0; k < n; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (dist[i][k] != INF && dist[k][j] != INF) {
                        dist[i][j] = std::min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }

        int minReachable = n + 1;
        int bestCity = -1;

        for (int i = 0; i < n; ++i) {
            int count = 0;
            for (int j = 0; j < n; ++j) {
                if (i != j && dist[i][j] <= distanceThreshold) {
                    count++;
                }
            }

            if (count <= minReachable) {
                minReachable = count;
                bestCity = i;
            }
        }

        return bestCity;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int findTheCity(int n, int[][] edges, int distanceThreshold) {
        int INF = 1_000_000_000;
        int[][] dist = new int[n][n];

        for (int i = 0; i < n; i++) {
            Arrays.fill(dist[i], INF);
            dist[i][i] = 0;
        }

        for (int[] edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            dist[u][v] = w;
            dist[v][u] = w;
        }

        // Floyd-Warshall
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (dist[i][k] != INF && dist[k][j] != INF) {
                        dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }

        int minReachable = n + 1;
        int bestCity = -1;

        for (int i = 0; i < n; i++) {
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (i != j && dist[i][j] <= distanceThreshold) {
                    count++;
                }
            }

            if (count <= minReachable) {
                minReachable = count;
                bestCity = i;
            }
        }

        return bestCity;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^3)$ — Floyd-Warshall executes 3 tightly nested loops from $0$ to $n - 1$, taking $\mathcal{O}(n^3)$ operations. Counting neighbors takes $\mathcal{O}(n^2)$. For $n = 100$, $100^3 = 10^6$ operations ($\approx 5$ ms).
  *(Alternative: Running Dijkstra $n$ times from each vertex with a min-heap takes $\mathcal{O}(n \cdot (E + n \log n))$, which is faster on sparse graphs).*
- **Space Complexity:** $\mathcal{O}(n^2)$ — Distance matrix of size $n \times n$.

---

### Takeaway Pattern & Interview Traps

1. **Loop Ordering Invariant:** In Floyd-Warshall, the intermediate node $k$ MUST be the outermost loop. If $i$ or $j$ is outermost, subpath distances will not be fully relaxed before being reused, leading to incorrect distances.
2. **Integer Overflow with Infinity:** Avoid using `Integer.MAX_VALUE` in C++/Java because `dist[i][k] + dist[k][j]` will overflow into negative values. Use a large safe sentinel like `1e9` or check `!= INF`.
3. **Tie-Breaking with `<=":** Using `count <= min_reachable` rather than `<` naturally updates `best_city` to the larger index on tie values because `i` ascends from $0$ to $n - 1$.