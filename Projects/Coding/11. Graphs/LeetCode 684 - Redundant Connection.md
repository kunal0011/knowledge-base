---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 684: Redundant Connection"
tags:
  - leetcode
  - coding
  - graphs
  - union-find
  - dsu
  - amazon
  - google
---

# LeetCode 684: Redundant Connection

**Target Companies:** Google, Amazon, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Disjoint Set Union (DSU / Union-Find) / Undirected Graph Cycle Detection

---

### Problem Statement

In this problem, a tree is an **undirected graph** that is connected and has no cycles.

You are given a graph that started as a tree with `n` nodes labeled from `1` to `n`, with one additional edge added. The added edge has two different vertices chosen from `1` to `n`, and was not an edge that already existed. The graph is represented as an array `edges` of length `n` where `edges[i] = [a_i, b_i]` indicates that there is an edge between nodes `a_i` and `b_i` in the graph.

Return an edge that can be removed so that the resulting graph is a tree of `n` nodes. If there are multiple answers, return the answer that occurs **last in the input**.

---

### Input & Output Formats & Constraints

- **Input:** `edges: List[List[int]]`
- **Output:** `List[int]` — The redundant edge `[u, v]`.
- **Constraints:**
  - $n == \text{edges.length}$
  - $3 \le n \le 1000$
  - $\text{edges}[i].\text{length} == 2$
  - $1 \le a_i < b_i \le \text{edges.length}$
  - $a_i \ne b_i$
  - There are no repeated edges.
  - The given graph is connected.

---

### Key Idea & Intuition

- **Tree Properties:**
  - A connected undirected tree with $n$ nodes has exactly $n - 1$ edges and contains **no cycles**.
  - Given $n$ edges, exactly **one** edge is redundant and creates a cycle.
- **Disjoint Set Union (DSU / Union-Find):**
  - Initialize each vertex $1 \dots n$ into its own independent component.
  - Iterate through `edges` in the order they are provided:
    - For edge `[u, v]`:
      - Find the root representatives of $u$ and $v$: `root_u = find(u)`, `root_v = find(v)`.
      - **If `root_u == root_v`:** $u$ and $v$ are **already connected** by a path in the tree! Adding this edge forms a cycle. Because we are processing edges in order, this edge is the one that creates the cycle and occurs last in the input. Return `[u, v]`.
      - **If `root_u != root_v`:** Merge the two components via `union(root_u, root_v)` and continue.
  - With **path compression** and **union by rank**, each DSU operation runs in nearly constant $\mathcal{O}(\alpha(n))$ time (Inverse Ackermann function).

---

### Solution Approach (Step-by-Step)

1. Initialize `parent = [i for i in range(n + 1)]` and `rank = [1] * (n + 1)`.
2. Define `find(i)` with path compression:
   - If `parent[i] != i`: `parent[i] = find(parent[i])`.
   - Return `parent[i]`.
3. Define `union(i, j)` with rank balancing:
   - `root_i = find(i)`, `root_j = find(j)`.
   - If `root_i == root_j`: return `False` (already connected $\implies$ cycle detected).
   - If `rank[root_i] < rank[root_j]`: `root_i, root_j = root_j, root_i`.
   - `parent[root_j] = root_i`
   - `rank[root_i] += rank[root_j]`
   - Return `True`.
4. Iterate through each edge `[u, v]` in `edges`:
   - If `not union(u, v)`: return `[u, v]`.
5. Return `[]` (fallback).

---

### Visual Algorithm Walkthrough

```
Edges: [[1, 2], [1, 3], [2, 3]]

Initial Components: {1}, {2}, {3}

Edge 1: [1, 2]
  - find(1) = 1, find(2) = 2 (Different!)
  - union(1, 2) -> Component: {1, 2}

Edge 2: [1, 3]
  - find(1) = 1, find(3) = 3 (Different!)
  - union(1, 3) -> Component: {1, 2, 3}

Edge 3: [2, 3]
  - find(2) = 1, find(3) = 1 (SAME ROOT: 1!)
  - Adding [2, 3] creates a cycle between 1, 2, and 3.
  - Return [2, 3] as the redundant connection!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: 3-Node Cycle
- **Input:** `edges = [[1,2],[1,3],[2,3]]`
- **Step Trace:**
  | Step | Edge `[u, v]` | `find(u)` | `find(v)` | Action |
  | :--- | :--- | :--- | :--- | :--- |
  | 1 | `[1, 2]` | 1 | 2 | Union components $\{1\}$ and $\{2\}$ |
  | 2 | `[1, 3]` | 1 | 3 | Union components $\{1, 2\}$ and $\{3\}$ |
  | 3 | `[2, 3]` | 1 | 1 | **Same Root! Cycle detected $\implies$ Return `[2, 3]`** |
- **Output:** `[2, 3]`

#### Example 2: 5-Node Graph
- **Input:** `edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]`
- **Trace:**
  - `[1, 2]`, `[2, 3]`, `[3, 4]` connect nodes 1, 2, 3, 4 into component $\{1, 2, 3, 4\}$.
  - `[1, 4]` tests nodes 1 and 4, which already share root 1 $\implies$ cycle detected!
- **Output:** `[1, 4]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        parent = list(range(n + 1))
        rank = [1] * (n + 1)
        
        def find(node: int) -> int:
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]
            
        def union(u: int, v: int) -> bool:
            root_u = find(u)
            root_v = find(v)
            
            if root_u == root_v:
                return False  # Cycle detected
                
            # Union by rank
            if rank[root_u] < rank[root_v]:
                root_u, root_v = root_v, root_u
            parent[root_v] = root_u
            rank[root_u] += rank[root_v]
            return True
            
        for u, v in edges:
            if not union(u, v):
                return [u, v]
                
        return []
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>

class Solution {
public:
    std::vector<int> findRedundantConnection(std::vector<std::vector<int>>& edges) {
        int n = edges.size();
        std::vector<int> parent(n + 1);
        std::iota(parent.begin(), parent.end(), 0);
        std::vector<int> rank(n + 1, 1);

        for (const auto& edge : edges) {
            int u = edge[0], v = edge[1];
            if (!unionNodes(u, v, parent, rank)) {
                return edge;
            }
        }
        return {};
    }

private:
    int find(int node, std::vector<int>& parent) {
        if (parent[node] != node) {
            parent[node] = find(parent[node], parent); // Path compression
        }
        return parent[node];
    }

    bool unionNodes(int u, int v, std::vector<int>& parent, std::vector<int>& rank) {
        int rootU = find(u, parent);
        int rootV = find(v, parent);

        if (rootU == rootV) {
            return false; // Cycle detected
        }

        if (rank[rootU] < rank[rootV]) {
            std::swap(rootU, rootV);
        }
        parent[rootV] = rootU;
        rank[rootU] += rank[rootV];
        return true;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        int n = edges.length;
        int[] parent = new int[n + 1];
        int[] rank = new int[n + 1];

        for (int i = 1; i <= n; i++) {
            parent[i] = i;
            rank[i] = 1;
        }

        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            if (!union(u, v, parent, rank)) {
                return edge;
            }
        }

        return new int[0];
    }

    private int find(int node, int[] parent) {
        if (parent[node] != node) {
            parent[node] = find(parent[node], parent); // Path compression
        }
        return parent[node];
    }

    private boolean union(int u, int v, int[] parent, int[] rank) {
        int rootU = find(u, parent);
        int rootV = find(v, parent);

        if (rootU == rootV) {
            return false; // Cycle detected
        }

        if (rank[rootU] < rank[rootV]) {
            int temp = rootU;
            rootU = rootV;
            rootV = temp;
        }
        parent[rootV] = rootU;
        rank[rootU] += rank[rootV];
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \cdot \alpha(N)) \approx \mathcal{O}(N)$ — Where $\alpha$ is the Inverse Ackermann function ($\alpha(N) < 5$ for all realistic values of $N$). Finding roots with path compression and uniting sets with rank executes in amortized $\mathcal{O}(\alpha(N))$ per edge.
- **Space Complexity:** $\mathcal{O}(N)$ — The `parent` and `rank` arrays each take $N + 1$ space.

---

### Takeaway Pattern & Interview Traps

1. **1-Indexed Vertices:** Graph nodes are labeled $1$ to $n$. Initialize `parent` and `rank` arrays with size $n + 1$ to avoid index-out-of-bounds errors.
2. **First Cycle Encountered:** Because the problem requires returning the edge that occurs *last in the input*, iterating from start to end and returning the *first* edge that closes an existing component guarantees the correct edge is returned.
3. **Directed Variant (LeetCode 685):** In LeetCode 685 (*Redundant Connection II*), edges are directed. Cycle detection alone is insufficient because a node might have two parents. DSU alone handles undirected graphs (LC 684); directed graphs require tracking in-degrees and potential dual-parent branches.