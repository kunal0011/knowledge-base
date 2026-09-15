---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 785: Is Graph Bipartite?"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - dfs
  - bipartite
  - amazon
  - google
---

# LeetCode 785: Is Graph Bipartite?

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Bipartite Graph Verification / 2-Coloring Algorithm / BFS & DFS Traversal

---

### Problem Statement

There is an **undirected** graph with `n` nodes, where each node is numbered between `0` and `n - 1`. You are given a 2D array `graph`, where `graph[u]` is an array of nodes that node `u` is adjacent to. More formally, for each `v` in `graph[u]`, there is an undirected edge between node `u` and node `v`. The graph has the following properties:
- There are no self-edges (`graph[u]` does not contain `u`).
- There are no parallel edges (`graph[u]` does not contain duplicate values).
- If `v` is in `graph[u]`, then `u` is in `graph[v]` (the graph is undirected).
- The graph may not be connected, meaning there may be two nodes `u` and `v` such that there is no path between them.

A graph is **bipartite** if the nodes can be partitioned into two independent sets `A` and `B` such that every edge in the graph connects a node in set `A` and a node in set `B`.

Return `true` if and only if it is **bipartite**.

---

### Input & Output Formats & Constraints

- **Input:** `graph: List[List[int]]`
- **Output:** `bool` — `true` if graph is bipartite, `false` otherwise.
- **Constraints:**
  - `graph.length == n`
  - $1 \le n \le 100$
  - $0 \le \text{graph}[u].\text{length} < n$
  - $0 \le \text{graph}[u][i] \le n - 1$
  - `graph[u]` does not contain `u`.
  - All values of `graph[u]` are unique.
  - If `graph[u]` contains `v`, then `graph[v]` contains `u`.

---

### Key Idea & Intuition

- **Bipartiteness and Odd Cycles:**
  - In graph theory, a graph is bipartite **if and only if it contains no odd-length cycles**.
  - This property is tested using **2-Coloring**:
    - Assign every node one of two colors (e.g., `1` for Red and `-1` for Blue).
    - For every edge connecting $u$ and $v$, $v$ must have the opposite color of $u$:
      $$\text{color}[v] = -\text{color}[u]$$
    - If at any point we encounter an adjacent neighbor $v$ that has *already* been assigned the **same** color as $u$ ($\text{color}[v] == \text{color}[u]$), the graph contains an odd-length cycle and cannot be bipartite!
- **Handling Disconnected Components:**
  - Because the graph may consist of multiple disjoint components, we must loop through all vertices $0 \dots n - 1$.
  - Whenever an uncolored node is found (`color[i] == 0`), initiate a new 2-coloring traversal.

---

### Solution Approach (Step-by-Step)

1. Let $n = \text{len}(graph)$.
2. Initialize array `color = [0] * n` where `0` means unvisited, `1` is Red, and `-1` is Blue.
3. For each node `start` from $0$ to $n - 1$:
   - If `color[start] != 0`: continue (already validated).
   - Set `color[start] = 1`.
   - Initialize `queue = deque([start])`.
   - While `queue` is not empty:
     - Pop `curr = queue.popleft()`.
     - For `neighbor` in `graph[curr]`:
       - If `color[neighbor] == 0`:
         - Assign opposite color: `color[neighbor] = -color[curr]`.
         - `queue.append(neighbor)`.
       - Else if `color[neighbor] == color[curr]`:
         - Color conflict! Return `False`.
4. If all components are colored without conflicts, return `True`.

---

### Visual Algorithm Walkthrough

```
Case 1: Valid Bipartite (Even Cycle / Tree)
Graph: 0 - 1 - 2 - 3 - 0 (4-cycle)
- Color 0 with Red (+1)
  - Neighbors 1 and 3 colored Blue (-1)
- Pop 1 (Blue): neighbor 2 colored Red (+1)
- Pop 3 (Blue): neighbor 2 is already Red (+1) -> Valid!
Return True.

----------------------------------------------------

Case 2: Non-Bipartite (Odd Cycle)
Graph: 0 - 1 - 2 - 0 (3-cycle)
- Color 0 with Red (+1)
  - Color 1 with Blue (-1)
  - Color 2 with Blue (-1)
- Now inspect edge (1, 2):
  - Node 1 is Blue (-1)
  - Node 2 is Blue (-1)
  - color[1] == color[2] -> CONFLICT! Both endpoints have same color.
Return False.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Odd Cycle Present (False)
- **Input:** `graph = [[1,2,3],[0,2],[0,1,3],[0,2]]`
- **Tracing:** Nodes 0, 1, 2 form a triangle ($0-1-2-0$).
- **Trace Table:**
  | Node Visited | Color Assigned | Neighbor Checked | Neighbor Color | Conflict? |
  | :--- | :--- | :--- | :--- | :--- |
  | 0 | `+1` (Red) | 1, 2, 3 | Assigned `-1` (Blue) | No |
  | 1 | `-1` (Blue) | 2 | Already `-1` (Blue) | **Yes! `color[1] == color[2]`** |
- **Output:** `false`

#### Example 2: Even Cycle (True)
- **Input:** `graph = [[1,3],[0,2],[1,3],[0,2]]`
- **Output:** `true` (Square 4-cycle can be 2-colored cleanly).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        color = [0] * n  # 0: uncolored, 1: red, -1: blue
        
        for start in range(n):
            if color[start] != 0:
                continue
                
            color[start] = 1
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in graph[u]:
                    if color[v] == 0:
                        color[v] = -color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
                        
        return True
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

class Solution {
public:
    bool isBipartite(std::vector<std::vector<int>>& graph) {
        int n = graph.size();
        std::vector<int> color(n, 0); // 0: unvisited, 1: red, -1: blue

        for (int start = 0; start < n; ++start) {
            if (color[start] != 0) continue;

            color[start] = 1;
            std::queue<int> q;
            q.push(start);

            while (!q.empty()) {
                int u = q.front();
                q.pop();

                for (int v : graph[u]) {
                    if (color[v] == 0) {
                        color[v] = -color[u];
                        q.push(v);
                    } else if (color[v] == color[u]) {
                        return false;
                    }
                }
            }
        }

        return true;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Queue;

class Solution {
    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        int[] color = new int[n]; // 0: unvisited, 1: red, -1: blue

        for (int start = 0; start < n; start++) {
            if (color[start] != 0) continue;

            color[start] = 1;
            Queue<Integer> queue = new ArrayDeque<>();
            queue.offer(start);

            while (!queue.isEmpty()) {
                int u = queue.poll();

                for (int v : graph[u]) {
                    if (color[v] == 0) {
                        color[v] = -color[u];
                        queue.offer(v);
                    } else if (color[v] == color[u]) {
                        return false;
                    }
                }
            }
        }

        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(V + E)$ — Every vertex is enqueued and dequeued once. Every undirected edge is checked twice. Overall runtime is strictly linear $\mathcal{O}(V + E)$.
- **Space Complexity:** $\mathcal{O}(V)$ — The `color` array of size $n$ and the BFS queue together require $\mathcal{O}(V)$ space.

---

### Takeaway Pattern & Interview Traps

1. **Disconnected Components:** Never assume the graph is connected. Always iterate over all vertices `0` to `n - 1` with `if color[i] == 0` to check every component.
2. **Opposite Color Arithmetic:** Using `1` and `-1` allows setting the neighbor's color via negation (`-color[u]`) with zero branching.
3. **Equivalence with Bipartite Matching:** This 2-coloring verification is the foundational first step in bipartite graph matching algorithms (Hopcroft-Karp / Hungarian method).