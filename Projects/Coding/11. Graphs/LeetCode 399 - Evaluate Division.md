---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 399: Evaluate Division"
tags:
  - leetcode
  - coding
  - graphs
  - dfs
  - union-find
  - amazon
  - google
---

# LeetCode 399: Evaluate Division

**Target Companies:** Google (All-Time Signature Variable Relationship Question), Amazon, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Weighted Directed Graph / Multiplicative Path Traversal / DFS Search

---

### Problem Statement

You are given an array of variable pairs `equations` and an array of real numbers `values`, where `equations[i] = [A_i, B_i]` and `values[i]` represent the equation $A_i / B_i = \text{values}[i]$. Each $A_i$ or $B_i$ is a string representing a single variable.

You are also given some `queries`, where `queries[j] = [C_j, D_j]` represents the $j$-th query where you must find the answer for $C_j / D_j = ?$.

Return the answers to all queries. If a single answer cannot be determined, return `-1.0`.

Note: The input is always valid. You may assume that evaluating the queries will not result in division by zero and that there is no contradiction.

---

### Input & Output Formats & Constraints

- **Input:** `equations: List[List[str]]`, `values: List[float]`, `queries: List[List[str]]`
- **Output:** `List[float]` — Evaluated answers to queries ($-1.0$ if undetermined).
- **Constraints:**
  - $1 \le \text{equations.length} \le 20$
  - $\text{values.length} == \text{equations.length}$
  - $0.0 < \text{values}[i] \le 20.0$
  - $1 \le \text{queries.length} \le 20$
  - $\text{queries}[i].\text{length} == 2$
  - $1 \le A_i, B_i, C_j, D_j \le 5$
  - Variables consist of lowercase English letters and digits.

---

### Key Idea & Intuition

- **Graph Formulation with Reciprocal Weights:**
  - An equation $\frac{u}{v} = k$ defines two directed edges in a graph:
    1. Forward edge: $u \xrightarrow{k} v$
    2. Reciprocal edge: $v \xrightarrow{1/k} u$
  - Evaluating $\frac{A}{C}$ reduces to finding a path from $A$ to $C$ and **multiplying the edge weights along the path**:
    $$\frac{A}{B} \times \frac{B}{C} = \frac{A}{C} \implies \text{weight}(A \to B) \times \text{weight}(B \to C)$$
- **Undefined Query Handling:**
  - If either $C_j$ or $D_j$ does not exist in the graph, the answer is immediately `-1.0`.
  - If $C_j == D_j$ and $C_j$ exists in the graph, the answer is `1.0`.
  - If no path exists from $C_j$ to $D_j$ (disconnected components), the answer is `-1.0`.
- **Search Strategy (DFS):**
  - For each query $(src, dst)$, run a DFS from $src$ with a `visited` set to avoid infinite cycles.
  - Return the product of weights upon reaching $dst$.

---

### Solution Approach (Step-by-Step)

1. Build adjacency map `adj = defaultdict(dict)`:
   - For `(u, v), val` in `zip(equations, values)`:
     - `adj[u][v] = val`
     - `adj[v][u] = 1.0 / val`
2. Define `dfs(curr, target, visited)`:
   - If `curr == target`: return `1.0`.
   - `visited.add(curr)`
   - For neighbor $nxt$ and edge weight $w$ in `adj[curr].items()`:
     - If $nxt$ not in `visited`:
       - `sub_prod = dfs(nxt, target, visited)`
       - If `sub_prod != -1.0`:
         - Return `w * sub_prod`.
   - Return `-1.0`.
3. For each `(src, dst)` in `queries`:
   - If `src not in adj` or `dst not in adj`: append `-1.0`.
   - Else: append `dfs(src, dst, set())`.
4. Return results.

---

### Visual Algorithm Walkthrough

```
Equations: [["a", "b"], ["b", "c"]], Values: [2.0, 3.0]
Queries: [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]

Graph:
  a  <--- 2.0 --->  b  <--- 3.0 --->  c
(a->b: 2.0, b->a: 0.5) (b->c: 3.0, c->b: 1/3)

Query 1: ["a", "c"]
  - Path: a -> b (cost 2.0) -> c (cost 3.0)
  - Result: 2.0 * 3.0 = 6.0

Query 2: ["b", "a"]
  - Direct edge b -> a (cost 0.5)
  - Result: 0.5

Query 3: ["a", "e"]
  - 'e' not in graph!
  - Result: -1.0

Query 4: ["a", "a"]
  - 'a' in graph, src == dst
  - Result: 1.0

Query 5: ["x", "x"]
  - 'x' not in graph!
  - Result: -1.0
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Chain
- **Input:** `equations = [["a","b"],["b","c"]], values = [2.0,3.0], queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]`
- **Output:** `[6.0, 0.5, -1.0, 1.0, -1.0]`

#### Example 2: Multiple Components
- **Input:** `equations = [["a","b"],["c","d"]], values = [1.5, 2.5], queries = [["a","d"],["c","a"]]`
- **Trace:** $\{a, b\}$ and $\{c, d\}$ are disjoint components. No path connects $a$ to $d$.
- **Output:** `[-1.0, -1.0]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Set
from collections import defaultdict

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        adj = defaultdict(dict)
        for (u, v), val in zip(equations, values):
            adj[u][v] = val
            adj[v][u] = 1.0 / val
            
        def dfs(curr: str, target: str, visited: Set[str]) -> float:
            if curr == target:
                return 1.0
                
            visited.add(curr)
            for neighbor, weight in adj[curr].items():
                if neighbor not in visited:
                    res = dfs(neighbor, target, visited)
                    if res != -1.0:
                        return weight * res
                        
            return -1.0
            
        results = []
        for src, dst in queries:
            if src not in adj or dst not in adj:
                results.append(-1.0)
            elif src == dst:
                results.append(1.0)
            else:
                results.append(dfs(src, dst, set()))
                
        return results
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

class Solution {
public:
    std::vector<double> calcEquation(std::vector<std::vector<std::string>>& equations, 
                                     std::vector<double>& values, 
                                     std::vector<std::vector<std::string>>& queries) {
        std::unordered_map<std::string, std::unordered_map<std::string, double>> adj;
        for (size_t i = 0; i < equations.size(); ++i) {
            const std::string& u = equations[i][0];
            const std::string& v = equations[i][1];
            double val = values[i];
            adj[u][v] = val;
            adj[v][u] = 1.0 / val;
        }

        std::vector<double> results;
        for (const auto& q : queries) {
            const std::string& src = q[0];
            const std::string& dst = q[1];

            if (adj.find(src) == adj.end() || adj.find(dst) == adj.end()) {
                results.push_back(-1.0);
            } else if (src == dst) {
                results.push_back(1.0);
            } else {
                std::unordered_set<std::string> visited;
                results.push_back(dfs(src, dst, visited, adj));
            }
        }

        return results;
    }

private:
    double dfs(const std::string& curr, const std::string& target, 
               std::unordered_set<std::string>& visited,
               std::unordered_map<std::string, std::unordered_map<std::string, double>>& adj) {
        if (curr == target) return 1.0;

        visited.insert(curr);
        for (const auto& [neighbor, weight] : adj[curr]) {
            if (visited.find(neighbor) == visited.end()) {
                double res = dfs(neighbor, target, visited, adj);
                if (res != -1.0) {
                    return weight * res;
                }
            }
        }

        return -1.0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String, Map<String, Double>> adj = new HashMap<>();

        for (int i = 0; i < equations.size(); i++) {
            String u = equations.get(i).get(0);
            String v = equations.get(i).get(1);
            double val = values[i];

            adj.putIfAbsent(u, new HashMap<>());
            adj.putIfAbsent(v, new HashMap<>());
            adj.get(u).put(v, val);
            adj.get(v).put(u, 1.0 / val);
        }

        double[] results = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            String src = queries.get(i).get(0);
            String dst = queries.get(i).get(1);

            if (!adj.containsKey(src) || !adj.containsKey(dst)) {
                results[i] = -1.0;
            } else if (src.equals(dst)) {
                results[i] = 1.0;
            } else {
                Set<String> visited = new HashSet<>();
                results[i] = dfs(src, dst, visited, adj);
            }
        }

        return results;
    }

    private double dfs(String curr, String target, Set<String> visited, Map<String, Map<String, Double>> adj) {
        if (curr.equals(target)) {
            return 1.0;
        }

        visited.add(curr);
        for (Map.Entry<String, Double> entry : adj.get(curr).entrySet()) {
            String neighbor = entry.getKey();
            double weight = entry.getValue();

            if (!visited.contains(neighbor)) {
                double res = dfs(neighbor, target, visited, adj);
                if (res != -1.0) {
                    return weight * res;
                }
            }
        }

        return -1.0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(Q \times (V + E))$ where $Q = \text{len}(queries)$, $V$ is the number of distinct variables, and $E = \text{len}(equations)$. Building the graph takes $\mathcal{O}(E)$. For each query, DFS traverses at most $V$ vertices and $E$ edges. Given $Q, E \le 20$, total operations are $\le 20 \times 40 = 800$, running in under $1$ ms.
- **Space Complexity:** $\mathcal{O}(V + E)$ — The adjacency map stores $V$ variables and $2E$ directed edges. The `visited` set and recursion stack take $\mathcal{O}(V)$ memory per query.

---

### Takeaway Pattern & Interview Traps

1. **Missing Variables Querying Themselves:** If $x$ is not present anywhere in `equations`, the query `["x", "x"]` MUST return `-1.0` (not `1.0`), because $x$ is undefined. Always check existence in `adj` before checking `src == dst`.
2. **Path Weight Multiplication:** Notice that edge values are multiplied together (not added), because ratios chain multiplicatively: $\frac{a}{b} \times \frac{b}{c} = \frac{a}{c}$.
3. **Disjoint Set Union (DSU) with Weights:** While DFS is clean and optimal for small constraints, this problem can also be solved with Weighted Union-Find in $\mathcal{O}(E + Q \cdot \alpha(V))$ time, where each node stores a ratio relative to its component root.