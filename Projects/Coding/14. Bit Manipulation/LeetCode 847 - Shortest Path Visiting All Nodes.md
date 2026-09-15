---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 847: Shortest Path Visiting All Nodes"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - bitmask
  - bfs
  - graph
  - google
  - amazon
---

# LeetCode 847: Shortest Path Visiting All Nodes

**Target Companies:** Google (Signature Classic Hard), Amazon, Meta  
**Difficulty:** Hard  
**Topic:** Bitmask Multi-Source BFS on State Space $(u, \text{mask})$  

---

### Problem Statement

You have an undirected, connected graph of $n$ nodes labeled from $0$ to $n - 1$. You are given an array `graph` where `graph[i]` is a list of all the nodes connected with node $i$ by an edge.

Return the **length of the shortest path** that visits every node. You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.

---

### Input & Output Formats & Constraints

- **Input:** `graph: List[List[int]]`
- **Output:** `int` (minimum number of edge traversals)
- **Constraints:**
  - $n == \text{graph.length}$
  - $1 \le n \le 12$
  - $0 \le \text{graph}[i].\text{length} < n$
  - `graph[i]` does not contain $i$.
  - If `graph[a]` contains `b`, then `graph[b]` contains `a`.
  - The input graph is guaranteed to be connected.

---

### Key Idea & Intuition

Because nodes and edges can be revisited multiple times, a standard graph vertex $u$ is insufficient to define our state. We must also record **which subset of nodes has been visited so far**.

Notice the constraint $n \le 12$:
- A subset of visited nodes can be encoded as a **bitmask** of length $n$:
  $$\text{mask} \in [0, 2^n - 1]$$
  If the $i$-th bit of `mask` is $1$, node $i$ has been visited.
- Total State Space:
  $$\text{State} = (u, \text{mask}) \quad \text{where } u \in [0, n - 1], \text{mask} \in [0, 2^n - 1]$$
  There are exactly $n \cdot 2^n$ possible states (for $n = 12$, $12 \times 4096 = 49,152$ states, which is extremely compact!).
- **Multi-Source BFS:**
  Because the path may start at *any* node, we initialize the BFS queue with all $n$ possible starting states $(i, 1 \ll i, 0)$ at distance 0.
- Since edge weights are uniform (1), standard BFS guarantees that the first state popped with $\text{mask} == (1 \ll n) - 1$ represents the **shortest path length**!

---

### Solution Approach (Step-by-Step)

1. Base case: if $n == 1$, return `0`.
2. Target mask: `target_mask = (1 << n) - 1`.
3. Initialize `q = deque()` and `visited = set()`.
4. Multi-Source Enqueue: For each node $i \in [0, n - 1]$:
   - `mask = 1 << i`.
   - `q.append((i, mask, 0))`.
   - `visited.add((i, mask))`.
5. While `q` is not empty:
   - Pop `(node, mask, dist) = q.popleft()`.
   - If `mask == target_mask`:
     - Return `dist`.
   - For `neighbor` in `graph[node]`:
     - `next_mask = mask | (1 << neighbor)`.
     - If `(neighbor, next_mask)` not in `visited`:
       - `visited.add((neighbor, next_mask))`
       - `q.append((neighbor, next_mask, dist + 1))`
6. Return `0`.

---

### Visual Algorithm Walkthrough

```
graph = [[1, 2, 3], [0], [0], [0]], n = 4
Graph shape: Star graph centered at 0!
     (1)
      |
(2)--(0)--(3)

Target Mask: 1111_2 (15)

Level 0: Start at all nodes:
  Queue = [ (0, 0001, d=0), (1, 0010, d=0), (2, 0100, d=0), (3, 1000, d=0) ]

Optimal Traversal:
  Start at leaf (e.g. 1): (1, 0010, dist=0)
  Step 1: Jump 1 -> 0:    (0, 0011, dist=1)
  Step 2: Jump 0 -> 2:    (2, 0111, dist=2)
  Step 3: Backtrack 2 -> 0:(0, 0111, dist=3) (node 0 revisited!)
  Step 4: Jump 0 -> 3:    (3, 1111, dist=4) (target mask 1111 reached!)

Shortest Path: 1 -> 0 -> 2 -> 0 -> 3
Total distance = 4.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Star Graph
- **Input:** `graph = [[1, 2, 3], [0], [0], [0]]`
- **Output:** `4`

#### Example 2: Complete Linear Chain
- **Input:** `graph = [[1], [0, 2], [1, 3], [2]]`
- **Trace:**
  - $0 \to 1 \to 2 \to 3$ (Traverses $n - 1 = 3$ edges).
- **Output:** `3`

#### Example 3: Cycle Graph
- **Input:** `graph = [[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]]` ($n = 5$)
- **Output:** `4`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        if n == 1:
            return 0
            
        target_mask = (1 << n) - 1
        q = deque()  # stores (node, mask, dist)
        visited = set()  # stores (node, mask)
        
        # Multi-source initialization: start at any node
        for i in range(n):
            mask = 1 << i
            q.append((i, mask, 0))
            visited.add((i, mask))
            
        while q:
            node, mask, dist = q.popleft()
            
            if mask == target_mask:
                return dist
                
            for neighbor in graph[node]:
                next_mask = mask | (1 << neighbor)
                if (neighbor, next_mask) not in visited:
                    visited.add((neighbor, next_mask))
                    q.append((neighbor, next_mask, dist + 1))
                    
        return 0
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

class Solution {
public:
    int shortestPathLength(std::vector<std::vector<int>>& graph) {
        int n = graph.size();
        if (n == 1) return 0;

        int targetMask = (1 << n) - 1;
        // visited[node][mask]
        std::vector<std::vector<bool>> visited(n, std::vector<bool>(1 << n, false));
        // Queue stores {node, mask, dist}
        std::queue<std::vector<int>> q;

        for (int i = 0; i < n; ++i) {
            int mask = 1 << i;
            q.push({i, mask, 0});
            visited[i][mask] = true;
        }

        while (!q.empty()) {
            auto curr = q.front();
            q.pop();

            int node = curr[0];
            int mask = curr[1];
            int dist = curr[2];

            if (mask == targetMask) {
                return dist;
            }

            for (int neighbor : graph[node]) {
                int nextMask = mask | (1 << neighbor);
                if (!visited[neighbor][nextMask]) {
                    visited[neighbor][nextMask] = true;
                    q.push({neighbor, nextMask, dist + 1});
                }
            }
        }

        return 0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Queue;

class Solution {
    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        if (n == 1) return 0;

        int targetMask = (1 << n) - 1;
        boolean[][] visited = new boolean[n][1 << n];
        Queue<int[]> q = new ArrayDeque<>(); // int[]{node, mask, dist}

        for (int i = 0; i < n; i++) {
            int mask = 1 << i;
            q.offer(new int[]{i, mask, 0});
            visited[i][mask] = true;
        }

        while (!q.isEmpty()) {
            int[] curr = q.poll();
            int node = curr[0];
            int mask = curr[1];
            int dist = curr[2];

            if (mask == targetMask) {
                return dist;
            }

            for (int neighbor : graph[node]) {
                int nextMask = mask | (1 << neighbor);
                if (!visited[neighbor][nextMask]) {
                    visited[neighbor][nextMask] = true;
                    q.offer(new int[]{neighbor, nextMask, dist + 1});
                }
            }
        }

        return 0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \cdot 2^N)$ — The state space has $N \cdot 2^N$ vertices. Each vertex has at most $N$ edges. Total traversal time is $O(N^2 \cdot 2^N)$ in the worst case; for $N \le 12$, $12^2 \times 4096 \approx 5.8 \times 10^5$ operations, which finishes in $< 15$ ms.
- **Space Complexity:** $O(N \cdot 2^N)$ — To store the `visited[node][mask]` boolean matrix and the BFS queue.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** State-Space Augmentation with Bitmask Subsets + Multi-Source BFS.
- **Trap:** Forgetting that multiple starting points are allowed: starting from node 0 only will miss optimal paths that must start at an isolated leaf node. Multi-source initialization is essential.