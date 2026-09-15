---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 1377: Frog Position After T Seconds"
tags:
  - leetcode
  - coding
  - trees
  - dfs
  - bfs
  - probability
  - google
  - amazon
---

# LeetCode 1377: Frog Position After T Seconds

**Target Companies:** Google (Signature Probability on Trees Problem), Amazon  
**Difficulty:** Hard  
**Topic:** Tree DFS / BFS / Probability Distribution / Uniform Random Walk

---

### Problem Statement

Given an undirected tree consisting of `n` vertices numbered from `1` to `n`. A frog starts jumping from vertex `1`. In one second, the frog jumps from its current vertex to another **unvisited** vertex that is connected to it by an edge (if vertices are available). The frog cannot jump back to already visited vertices.

When the frog can jump to several vertices, each choice has an **equal probability**. If the frog cannot jump to any unvisited vertex, it **stays on the same vertex forever**.

Given `n`, `edges` of the tree, time `t` (in seconds), and a `target` vertex, return the probability that after `t` seconds the frog is on the `target` vertex. Answers within $10^{-5}$ of the actual answer will be accepted.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`, `edges: List[List[int]]`, `t: int`, `target: int`
- **Output:** `float` — Probability of frog being at `target` at time `t`.
- **Constraints:**
  - $1 \le n \le 100$
  - $\text{edges.length} == n - 1$
  - $\text{edges}[i].\text{length} == 2$
  - $1 \le a_i, b_i \le n$
  - $1 \le t \le 50$
  - $1 \le \text{target} \le n$

---

### Key Idea & Intuition

- **Tree Path Uniqueness:**
  - In any connected tree, there exists **exactly one unique simple path** from vertex `1` to `target`.
  - Let the length (number of edges) of this path be $d$.
  - To be at `target` after $t$ seconds:
    1. $t < d$: Impossible, probability is $0$ (the frog cannot reach target in time).
    2. $t == d$: The frog reaches `target` precisely at second $t$. The probability is the product of the branching probabilities along the path.
    3. $t > d$: The frog reaches `target` earlier. The frog can only stay at `target` if `target` is a **leaf** (i.e. has no unvisited children). If `target` has unvisited neighbors, the frog is forced to jump away, making the final probability at second $t$ equal to $0$!
- **Branching Factor:**
  - For vertex `u` with parent `p`, the number of unvisited children is:
    $$\text{children\_count} = \begin{cases} \text{degree}(u) & \text{if } u == 1 \\ \text{degree}(u) - 1 & \text{if } u \ne 1 \end{cases}$$
  - Each transition from $u$ to an unvisited child occurs with probability $\frac{1}{\text{children\_count}}$.

---

### Solution Approach (Step-by-Step)

1. Build an adjacency list representation `adj` for the undirected tree.
2. Define a DFS helper `dfs(curr, parent, time_left)` returning `float`:
   - Compute number of unvisited children:
     - If `curr == 1`: `children_count = len(adj[curr])`
     - Else: `children_count = len(adj[curr]) - 1`
   - If `curr == target`:
     - If `time_left == 0` or (`time_left > 0 and children_count == 0`): return `1.0` (frog stays here).
     - Else (`time_left > 0 and children_count > 0`): return `0.0` (frog must jump away).
   - If `time_left == 0` or `children_count == 0`: return `0.0` (ran out of time or stuck before reaching target).
   - Loop over all neighbors `nxt` of `curr`:
     - If `nxt != parent`:
       - Recursively search: `res = dfs(nxt, curr, time_left - 1)`
       - If `res > 0`: return `res / children_count`.
   - If target was not found in subtrees, return `0.0`.
3. Call `dfs(1, 0, t)` and return the probability.

---

### Visual Algorithm Walkthrough

```
Tree:
         1 (Root)
       /   \
      2     3
     / \     \
    4   5     6
             /
            7 (Target)

Path from 1 to Target 7: 1 -> 3 -> 6 -> 7 (3 edges, depth d = 3)

Case 1: t = 3, target = 7
  - At 1: children = {2, 3} (count = 2) -> P(1->3) = 1/2
  - At 3: children = {6}    (count = 1) -> P(3->6) = 1/1 = 1
  - At 6: children = {7}    (count = 1) -> P(6->7) = 1/1 = 1
  - At 7: time_left = 0, curr == target -> Valid!
  Probability = 1/2 * 1 * 1 = 0.5

Case 2: t = 2, target = 7
  - Cannot reach depth 3 in 2 seconds -> Probability = 0.0

Case 3: t = 4, target = 6 (degree = 2, parent 3, child 7)
  - Frog reaches node 6 at second 2.
  - Remaining time = 2.
  - Node 6 has unvisited child 7, so frog CANNOT stay at 6!
  - It jumps to 7 at second 3, staying at 7 at second 4.
  - Probability of being at 6 at t=4 is 0.0!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Search
- **Input:** `n = 7, edges = [[1,2],[1,3],[1,7],[2,4],[2,6],[3,5]], t = 2, target = 4`
- **Path to 4:** `1 -> 2 -> 4` (2 edges).
- **Step Trace:**
  | Node | Parent | `time_left` | Children Count | Probability Transition |
  | :--- | :--- | :--- | :--- | :--- |
  | 1 | 0 | 2 | 3 (nodes 2, 3, 7) | $\frac{1}{3}$ |
  | 2 | 1 | 1 | 2 (nodes 4, 6) | $\frac{1}{2}$ |
  | 4 | 2 | 0 | 0 | $1.0$ (target reached at $t=0$) |
- **Total Probability:** $\frac{1}{3} \times \frac{1}{2} = \frac{1}{6} \approx 0.166666...$

#### Example 2: Target with Remaining Time and No Children (Leaf)
- **Input:** `n = 7, edges = [[1,2],[1,3],[1,7],[2,4],[2,6],[3,5]], t = 20, target = 6`
- **Path to 6:** `1 -> 2 -> 6` (2 edges).
- **Analysis:** Node 6 has no children. After arriving at second 2, it has 18 seconds left, but since `children_count == 0`, it stays in place forever.
- **Output:** $\frac{1}{3} \times \frac{1}{2} = \frac{1}{6} \approx 0.166666...$

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def frogPosition(self, n: int, edges: List[List[int]], t: int, target: int) -> float:
        # Build adjacency graph
        adj = [[] for _ in range(n + 1)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        def dfs(curr: int, parent: int, time_left: int) -> float:
            # Count unvisited neighbors (excluding parent)
            children_count = len(adj[curr]) - (1 if parent != 0 else 0)
            
            # If target reached
            if curr == target:
                # Valid if time ran out or frog is stuck with no children to jump to
                if time_left == 0 or children_count == 0:
                    return 1.0
                return 0.0
                
            # If time expired or dead end reached before reaching target
            if time_left == 0 or children_count == 0:
                return 0.0
                
            # Search children
            for nxt in adj[curr]:
                if nxt != parent:
                    prob = dfs(nxt, curr, time_left - 1)
                    if prob > 0:
                        return prob / children_count
                        
            return 0.0

        return dfs(1, 0, t)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    double frogPosition(int n, std::vector<std::vector<int>>& edges, int t, int target) {
        std::vector<std::vector<int>> adj(n + 1);
        for (const auto& edge : edges) {
            adj[edge[0]].push_back(edge[1]);
            adj[edge[1]].push_back(edge[0]);
        }

        return dfs(1, 0, t, target, adj);
    }

private:
    double dfs(int curr, int parent, int timeLeft, int target, const std::vector<std::vector<int>>& adj) {
        int childrenCount = adj[curr].size() - (parent != 0 ? 1 : 0);

        if (curr == target) {
            if (timeLeft == 0 || childrenCount == 0) {
                return 1.0;
            }
            return 0.0;
        }

        if (timeLeft == 0 || childrenCount == 0) {
            return 0.0;
        }

        for (int nxt : adj[curr]) {
            if (nxt != parent) {
                double prob = dfs(nxt, curr, timeLeft - 1, target, adj);
                if (prob > 0.0) {
                    return prob / childrenCount;
                }
            }
        }

        return 0.0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public double frogPosition(int n, int[][] edges, int t, int target) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            adj.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }

        return dfs(1, 0, t, target, adj);
    }

    private double dfs(int curr, int parent, int timeLeft, int target, List<List<Integer>> adj) {
        int childrenCount = adj.get(curr).size() - (parent != 0 ? 1 : 0);

        if (curr == target) {
            if (timeLeft == 0 || childrenCount == 0) {
                return 1.0;
            }
            return 0.0;
        }

        if (timeLeft == 0 || childrenCount == 0) {
            return 0.0;
        }

        for (int nxt : adj.get(curr)) {
            if (nxt != parent) {
                double prob = dfs(nxt, curr, timeLeft - 1, target, adj);
                if (prob > 0.0) {
                    return prob / childrenCount;
                }
            }
        }

        return 0.0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — The tree has $n$ vertices and $n - 1$ edges. Since there is a unique path from vertex 1 to target, the DFS only explores the tree until the unique path is located, traversing at most $N$ nodes.
- **Space Complexity:** $\mathcal{O}(N)$ — Adjacency list requires $\mathcal{O}(N)$ space, and the recursion stack takes $\mathcal{O}(H) \le \mathcal{O}(N)$ space.

---

### Takeaway Pattern & Interview Traps

1. **The "Forced to Jump" Trap:** If the frog reaches `target` with $t > 0$ remaining, it DOES NOT automatically stay at `target`. It is forced to jump to an unvisited child if one exists! Only if `children_count == 0` does it stay at `target`.
2. **Root Degree Count:** Node 1 has no parent, so all its neighbors are unvisited children. For all other nodes, one neighbor is the parent, so `children_count = len(adj[curr]) - 1`.
3. **Floating Point Accuracy:** Returning `prob / children_count` backwards along the unique path preserves high double-precision accuracy without intermediate rounding issues.