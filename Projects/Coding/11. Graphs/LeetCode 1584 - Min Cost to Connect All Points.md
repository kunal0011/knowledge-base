---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 1584: Min Cost to Connect All Points"
tags:
  - leetcode
  - coding
  - graphs
  - minimum-spanning-tree
  - prim
  - kruskal
  - amazon
  - google
---

# LeetCode 1584: Min Cost to Connect All Points

**Target Companies:** Amazon, Google, Microsoft, Meta  
**Difficulty:** Medium  
**Topic:** Minimum Spanning Tree (MST) / Prim's Algorithm / Manhattan Distance Metric

---

### Problem Statement

You are given an array `points` representing integer coordinates of some points on a 2D-plane, where `points[i] = [x_i, y_i]`.

The cost of connecting two points `[x_i, y_i]` and `[x_j, y_j]` is the **Manhattan distance** between them:
$$|x_i - x_j| + |y_i - y_j|$$

Return the **minimum cost** to make all points connected. All points are connected if there is **exactly one simple path** between any two points (i.e., a Spanning Tree).

---

### Input & Output Formats & Constraints

- **Input:** `points: List[List[int]]`
- **Output:** `int` — Minimum total edge cost to span all points.
- **Constraints:**
  - $1 \le \text{points.length} \le 1000$
  - $-10^6 \le x_i, y_i \le 10^6$
  - All pairs $(x_i, y_i)$ are distinct.

---

### Key Idea & Intuition

- **Complete Graph MST:**
  - Between every pair of points $u$ and $v$, an edge exists with weight $|x_u - x_v| + |y_u - y_v|$.
  - This forms a **dense complete graph** with $V = n$ vertices and $E = \frac{n(n - 1)}{2}$ edges.
  - Connecting all points with minimum total cost is the textbook definition of a **Minimum Spanning Tree (MST)**.
- **Why $\mathcal{O}(V^2)$ Prim's is Optimal for Dense Graphs:**
  - Kruskal's or Heap-based Prim's has complexity $\mathcal{O}(E \log E) = \mathcal{O}(V^2 \log V)$, allocating up to $10^6$ edges in memory.
  - Because the graph is complete ($E = \Theta(V^2)$), **Array-based Prim's Algorithm** runs in strictly:
    $$\mathcal{O}(V^2) \text{ time and } \mathcal{O}(V) \text{ space!}$$
  - **Algorithm Flow:**
    1. Maintain `min_dist = [infinity] * n` storing the shortest distance from the growing MST tree to each unvisited node.
    2. Set `min_dist[0] = 0`.
    3. Repeat $n$ times:
       - Find the unvisited node $u$ with minimum `min_dist[u]`.
       - Mark $u$ as visited, add `min_dist[u]` to `total_cost`.
       - For all remaining unvisited nodes $v$, update:
         $$\text{min\_dist}[v] = \min(\text{min\_dist}[v], \text{manhattan\_dist}(u, v))$$

---

### Solution Approach (Step-by-Step)

1. Let $n = \text{len}(points)$.
2. Initialize `min_dist = [infinity] * n` with `min_dist[0] = 0`.
3. Initialize `visited = [False] * n` and `total_cost = 0`.
4. Loop $n$ times:
   - Find unvisited node `curr` with smallest `min_dist[curr]`.
   - Mark `visited[curr] = True`.
   - Add `min_dist[curr]` to `total_cost`.
   - For each unvisited node `nxt` from $0$ to $n - 1$:
     - Compute `cost = abs(points[curr][0] - points[nxt][0]) + abs(points[curr][1] - points[nxt][1])`.
     - `min_dist[nxt] = min(min_dist[nxt], cost)`.
5. Return `total_cost`.

---

### Visual Algorithm Walkthrough

```
Points: [0, 0], [2, 2], [3, 10], [5, 2], [7, 0]
Let points be 0, 1, 2, 3, 4.

Step 1: Start at node 0. Cost = 0.
  Distances from 0 to unvisited:
  - to 1: |0-2| + |0-2| = 4
  - to 2: |0-3| + |0-10| = 13
  - to 3: |0-5| + |0-2| = 7
  - to 4: |0-7| + |0-0| = 7
  Minimum is Node 1 (dist = 4).

Step 2: Add Node 1. Total cost = 0 + 4 = 4.
  Update distances from Node 1:
  - to 2: min(13, |2-3|+|2-10| = 9) = 9
  - to 3: min(7, |2-5|+|2-2| = 3) = 3
  - to 4: min(7, |2-7|+|2-0| = 7) = 7
  Minimum is Node 3 (dist = 3).

Step 3: Add Node 3. Total cost = 4 + 3 = 7.
  Update distances from Node 3:
  - to 2: min(9, |5-3|+|2-10| = 10) = 9
  - to 4: min(7, |5-7|+|2-0| = 4) = 4
  Minimum is Node 4 (dist = 4).

Step 4: Add Node 4. Total cost = 7 + 4 = 11.
  Update distances from Node 4:
  - to 2: min(9, |7-3|+|0-10| = 14) = 9
  Minimum is Node 2 (dist = 9).

Step 5: Add Node 2. Total cost = 11 + 9 = 20.
All points connected! Total MST Cost = 20.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: 5 Scattered Points
- **Input:** `points = [[0,0],[2,2],[3,10],[5,2],[7,0]]`
- **Tracing Table:**
  | Iteration | Node Added | Edge Weight Added | MST Total Cost |
  | :--- | :--- | :--- | :--- |
  | 1 | 0 `[0, 0]` | 0 | 0 |
  | 2 | 1 `[2, 2]` | 4 | 4 |
  | 3 | 3 `[5, 2]` | 3 | 7 |
  | 4 | 4 `[7, 0]` | 4 | 11 |
  | 5 | 2 `[3, 10]` | 9 | **20** |
- **Output:** `20`

#### Example 2: Three Collinear Points
- **Input:** `points = [[3,12],[-2,5],[-4,1]]`
- **Edges:**
  - $0 \to 1$: $|3 - (-2)| + |12 - 5| = 5 + 7 = 12$
  - $1 \to 2$: $|-2 - (-4)| + |5 - 1| = 2 + 4 = 6$
  - $0 \to 2$: $|3 - (-4)| + |12 - 1| = 7 + 11 = 18$
- **MST Edges:** $6 + 12 = 18$.
- **Output:** `18`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed - $\mathcal{O}(V^2)$ Array Prim)
```python
from typing import List

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        min_dist = [float('inf')] * n
        min_dist[0] = 0
        visited = [False] * n
        total_cost = 0
        
        for _ in range(n):
            # 1. Pick unvisited node with minimum distance to MST
            curr = -1
            curr_min = float('inf')
            for i in range(n):
                if not visited[i] and min_dist[i] < curr_min:
                    curr_min = min_dist[i]
                    curr = i
                    
            visited[curr] = True
            total_cost += curr_min
            
            # 2. Relax distances to remaining unvisited nodes
            x1, y1 = points[curr]
            for nxt in range(n):
                if not visited[nxt]:
                    dist = abs(x1 - points[nxt][0]) + abs(y1 - points[nxt][1])
                    if dist < min_dist[nxt]:
                        min_dist[nxt] = dist
                        
        return total_cost
```

#### 2. C++ (C++17 / STL - $\mathcal{O}(V^2)$ Dense Prim)
```cpp
#include <vector>
#include <cmath>
#include <climits>
#include <algorithm>

class Solution {
public:
    int minCostConnectPoints(std::vector<std::vector<int>>& points) {
        int n = points.size();
        std::vector<int> minDist(n, INT_MAX);
        std::vector<bool> visited(n, false);
        minDist[0] = 0;
        int totalCost = 0;

        for (int step = 0; step < n; ++step) {
            int curr = -1;
            int currMin = INT_MAX;

            for (int i = 0; i < n; ++i) {
                if (!visited[i] && minDist[i] < currMin) {
                    currMin = minDist[i];
                    curr = i;
                }
            }

            visited[curr] = true;
            totalCost += currMin;

            int x1 = points[curr][0], y1 = points[curr][1];
            for (int nxt = 0; nxt < n; ++nxt) {
                if (!visited[nxt]) {
                    int dist = std::abs(x1 - points[nxt][0]) + std::abs(y1 - points[nxt][1]);
                    if (dist < minDist[nxt]) {
                        minDist[nxt] = dist;
                    }
                }
            }
        }

        return totalCost;
    }
};
```

#### 3. Java (Modern, Typed - $\mathcal{O}(V^2)$ Dense Prim)
```java
import java.util.Arrays;

class Solution {
    public int minCostConnectPoints(int[][] points) {
        int n = points.length;
        int[] minDist = new int[n];
        Arrays.fill(minDist, Integer.MAX_VALUE);
        minDist[0] = 0;
        boolean[] visited = new boolean[n];
        int totalCost = 0;

        for (int step = 0; step < n; step++) {
            int curr = -1;
            int currMin = Integer.MAX_VALUE;

            for (int i = 0; i < n; i++) {
                if (!visited[i] && minDist[i] < currMin) {
                    currMin = minDist[i];
                    curr = i;
                }
            }

            visited[curr] = true;
            totalCost += currMin;

            int x1 = points[curr][0], y1 = points[curr][1];
            for (int nxt = 0; nxt < n; nxt++) {
                if (!visited[nxt]) {
                    int dist = Math.abs(x1 - points[nxt][0]) + Math.abs(y1 - points[nxt][1]);
                    if (dist < minDist[nxt]) {
                        minDist[nxt] = dist;
                    }
                }
            }
        }

        return totalCost;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(V^2)$ — Finding the minimum node across $n$ vertices takes $\mathcal{O}(V)$ time, done $V$ times $\implies \mathcal{O}(V^2)$. Relaxing edges takes $\mathcal{O}(V)$ time per vertex, totaling $\mathcal{O}(V^2)$. For $V = 1000$, $V^2 = 10^6$ operations, which executes in $< 15$ ms (much faster than heap-based Kruskal's $\mathcal{O}(V^2 \log V)$ which requires generating and sorting $5 \times 10^5$ edges).
- **Space Complexity:** $\mathcal{O}(V)$ — Only `min_dist` and `visited` arrays of size $n$ are maintained. No edge lists or priority queue allocations needed.

---

### Takeaway Pattern & Interview Traps

1. **Algorithm Choice for Complete Graphs:**
   - On **sparse graphs** ($E \approx V$), Kruskal's Algorithm with DSU or Heap Prim is optimal ($\mathcal{O}(E \log V)$).
   - On **dense / complete graphs** ($E \approx V^2$), standard array-based Prim's algorithm runs in $\mathcal{O}(V^2)$ time with strictly $\mathcal{O}(V)$ space, outperforming heap-based implementations.
2. **Manhattan Distance Formula:** Ensure absolute values are applied: $|x_1 - x_2| + |y_1 - y_2|$.