---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 1192: Critical Connections in a Network"
tags:
  - leetcode
  - coding
  - graphs
  - dfs
  - tarjan
  - amazon
  - google
---

# LeetCode 1192: Critical Connections in a Network

**Target Companies:** Amazon (Signature All-Time Top #1 Hard Graph Problem), Google, Meta, Microsoft  
**Difficulty:** Hard  
**Topic:** Tarjan's Bridge-Finding Algorithm / DFS Discovery & Low-Link Numbers / Graph Biconnectivity

---

### Problem Statement

There are `n` servers numbered from `0` to `n - 1` connected by undirected server-to-server `connections` forming a network where `connections[i] = [a_i, b_i]` represents a connection between servers `a_i` and `b_i`. Any server can reach other servers directly or indirectly through the network.

A **critical connection** is a connection that, if removed, will make some servers unable to reach some other server (i.e., a **bridge** in graph theory).

Return all critical connections in the network in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`, `connections: List[List[int]]`
- **Output:** `List[List[int]]` — List of critical connections `[u, v]`.
- **Constraints:**
  - $2 \le n \le 10^5$
  - $n - 1 \le \text{connections.length} \le 10^5$
  - $0 \le a_i, b_i < n$
  - $a_i \ne b_i$
  - There are no repeated connections.
  - The graph is connected.

---

### Key Idea & Intuition

- **Bridges vs. Cycles:**
  - In an undirected connected graph, an edge $(u, v)$ is a **bridge** if and only if it does **not** belong to any cycle.
  - If $(u, v)$ is part of a cycle, removing $(u, v)$ leaves alternative pathways connecting all nodes in that cycle.
- **Tarjan's Discovery and Low-Link Values:**
  - Perform a DFS traversal while assigning two timestamps to each node $u$:
    1. `disc[u]`: The discovery time (tick count) when node $u$ was first visited.
    2. `low[u]`: The earliest discovery time reachable from $u$ by traversing zero or more tree edges followed by at most one back-edge.
  - **Tree Edge vs. Back-Edge:**
    - For edge $(u, v)$:
      - If $v == \text{parent}$: Skip (it's the direct undirected edge back to the immediate caller).
      - If $v$ is already visited: $(u, v)$ is a **back-edge** to an ancestor! Update `low[u] = min(low[u], disc[v])`.
      - If $v$ is unvisited: Recurse `dfs(v, u)`, then update `low[u] = min(low[u], low[v])`.
        - **Bridge Condition:** If `low[v] > disc[u]`, then $(u, v)$ is a **critical connection**!
        - Why? Because the lowest node reachable from $v$'s subtree is strictly discovered *after* $u$. Thus, $v$ has zero alternative paths back to $u$ or any ancestor of $u$.

---

### Solution Approach (Step-by-Step)

1. Build an undirected adjacency list `adj` from `connections`.
2. Initialize:
   - `disc = [-1] * n` (unvisited indicator and discovery timestamps).
   - `low = [-1] * n`.
   - `bridges = []`.
   - `time = 0`.
3. Define `dfs(u, parent)`:
   - Record `disc[u] = low[u] = time; time += 1`.
   - For neighbor $v$ in `adj[u]`:
     - If $v == parent$: continue.
     - If `disc[v] != -1`: (Back-edge to already discovered ancestor)
       - `low[u] = min(low[u], disc[v])`
     - Else: (Tree edge to unvisited child)
       - Recurse `dfs(v, u)`.
       - Update `low[u] = min(low[u], low[v])`.
       - If `low[v] > disc[u]`:
         - `bridges.append([u, v])`.
4. Call `dfs(0, -1)` (since the graph is connected, a single DFS covers all vertices).
5. Return `bridges`.

---

### Visual Algorithm Walkthrough

```
Network Graph:
    0 ----- 1
    |     / |
    |   /   |
    | /     |
    2 ----- 3
            |
            4 (Server 4 is connected only via edge (3, 4))

DFS Execution Trace:
1. Start at 0: disc[0] = low[0] = 0
2. Visit 1:   disc[1] = low[1] = 1
3. Visit 2:   disc[2] = low[2] = 2
   - Neighbor 0 of 2 is visited (back-edge to ancestor 0!):
     low[2] = min(low[2], disc[0]) = min(2, 0) = 0
4. Backtrack to 1: low[1] = min(1, low[2]) = min(1, 0) = 0
5. Visit 3 from 1: disc[3] = low[3] = 3
   - Neighbor 2 of 3 is visited (back-edge to 2!):
     low[3] = min(3, disc[2]) = min(3, 2) = 2
6. Visit 4 from 3: disc[4] = low[4] = 4
   - 4 has no unvisited neighbors (only parent 3).
   - Check bridge for edge (3, 4):
     low[4] = 4 > disc[3] = 3 -> BRIDGE FOUND: [3, 4]!
   - Backtrack to 3: low[3] = min(low[3], low[4]) = min(2, 4) = 2
7. For edge (1, 3): low[3] = 2 > disc[1] = 1 -> No bridge (cycle present).
8. For edges in triangle (0, 1, 2): all low values are 0 <= disc -> No bridges.

Result: [[3, 4]]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Graph with One Cycle and One Bridge
- **Input:** `n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]`
- **Tracing Table:**
  | Edge Traversed | `disc[u]` | `low[v]` | `low[v] > disc[u]`? | Is Bridge? |
  | :--- | :--- | :--- | :--- | :--- |
  | `(0, 1)` | `disc[0] = 0` | `low[1] = 0` | $0 > 0$ (False) | No (part of 0-1-2 cycle) |
  | `(1, 2)` | `disc[1] = 1` | `low[2] = 0` | $0 > 1$ (False) | No (part of 0-1-2 cycle) |
  | `(2, 0)` | Back-edge to 0 | - | - | Back-edge updates `low[2] = 0` |
  | `(1, 3)` | `disc[1] = 1` | `low[3] = 3` | $3 > 1$ (**True**) | **Yes! `[1, 3]`** |
- **Output:** `[[1, 3]]`

#### Example 2: Pure Tree (All Edges are Bridges)
- **Input:** `n = 3, connections = [[0,1],[1,2]]`
- **Output:** `[[1, 2], [0, 1]]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict

class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        adj = defaultdict(list)
        for u, v in connections:
            adj[u].append(v)
            adj[v].append(u)
            
        disc = [-1] * n
        low = [-1] * n
        bridges = []
        time = 0
        
        def dfs(u: int, parent: int) -> None:
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            
            for v in adj[u]:
                if v == parent:
                    continue
                    
                if disc[v] != -1:
                    # Back-edge to an ancestor
                    low[u] = min(low[u], disc[v])
                else:
                    # Tree edge: recurse on child
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    
                    # Bridge condition
                    if low[v] > disc[u]:
                        bridges.append([u, v])
                        
        dfs(0, -1)
        return bridges
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> criticalConnections(int n, std::vector<std::vector<int>>& connections) {
        std::vector<std::vector<int>> adj(n);
        for (const auto& edge : connections) {
            adj[edge[0]].push_back(edge[1]);
            adj[edge[1]].push_back(edge[0]);
        }

        std::vector<int> disc(n, -1);
        std::vector<int> low(n, -1);
        std::vector<std::vector<int>> bridges;
        int time = 0;

        dfs(0, -1, time, adj, disc, low, bridges);
        return bridges;
    }

private:
    void dfs(int u, int parent, int& time, const std::vector<std::vector<int>>& adj,
             std::vector<int>& disc, std::vector<int>& low, std::vector<std::vector<int>>& bridges) {
        disc[u] = low[u] = time++;

        for (int v : adj[u]) {
            if (v == parent) continue;

            if (disc[v] != -1) {
                // Back-edge
                low[u] = std::min(low[u], disc[v]);
            } else {
                // Forward tree-edge
                dfs(v, u, time, adj, disc, low, bridges);
                low[u] = std::min(low[u], low[v]);

                if (low[v] > disc[u]) {
                    bridges.push_back({u, v});
                }
            }
        }
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    private int time = 0;

    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
        }
        for (List<Integer> edge : connections) {
            int u = edge.get(0), v = edge.get(1);
            adj.get(u).add(v);
            adj.get(v).add(u);
        }

        int[] disc = new int[n];
        int[] low = new int[n];
        Arrays.fill(disc, -1);
        Arrays.fill(low, -1);

        List<List<Integer>> bridges = new ArrayList<>();
        dfs(0, -1, adj, disc, low, bridges);
        return bridges;
    }

    private void dfs(int u, int parent, List<List<Integer>> adj, int[] disc, int[] low, List<List<Integer>> bridges) {
        disc[u] = low[u] = time++;

        for (int v : adj.get(u)) {
            if (v == parent) {
                continue;
            }

            if (disc[v] != -1) {
                // Back-edge
                low[u] = Math.min(low[u], disc[v]);
            } else {
                // Tree edge
                dfs(v, u, adj, disc, low, bridges);
                low[u] = Math.min(low[u], low[v]);

                if (low[v] > disc[u]) {
                    bridges.add(Arrays.asList(u, v));
                }
            }
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(V + E)$ — Building the adjacency list takes $\mathcal{O}(V + E)$ time. Tarjan's DFS visits every vertex once and checks every edge twice (once from each endpoint), yielding optimal $\mathcal{O}(V + E)$ time.
- **Space Complexity:** $\mathcal{O}(V + E)$ — The adjacency list requires $\mathcal{O}(V + E)$ memory. The `disc`, `low`, and recursion call stack each use $\mathcal{O}(V)$ memory.

---

### Takeaway Pattern & Interview Traps

1. **Strict Inequality (`low[v] > disc[u]`):**
   - If `low[v] == disc[u]`, node $v$ can reach $u$ through an alternate path (forming a simple cycle). It is NOT a bridge.
   - Only when `low[v] > disc[u]` can $v$ NEVER reach $u$ or any earlier node without edge $(u, v)$.
2. **Back-Edge Update Uses `disc[v]`:** When encountering an already visited node $v$ that is not the immediate parent, update `low[u] = min(low[u], disc[v])` (not `low[v]`), adhering strictly to Tarjan's bridge-finding definition.
3. **Handling Disconnected Graphs:** While the problem guarantees connectivity, in general graph problems where the graph may be disconnected, wrap the DFS call inside a loop: `for i in range(n): if disc[i] == -1: dfs(i, -1)`.