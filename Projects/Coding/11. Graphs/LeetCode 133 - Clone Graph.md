---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 133: Clone Graph"
tags:
  - leetcode
  - coding
  - graphs
  - dfs
  - bfs
  - hash-table
  - amazon
  - google
---

# LeetCode 133: Clone Graph

**Target Companies:** Meta (Signature Graph Deep Copy Question), Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Graph Traversal / Deep Copy / Hash Map Visited Mapping / DFS & BFS

---

### Problem Statement

Given a reference of a node in a **connected** undirected graph.

Return a **deep copy** (clone) of the graph.

Each node in the graph contains a value (`int`) and a list (`List[Node]`) of its neighbors.

```java
class Node {
    public int val;
    public List<Node> neighbors;
}
```

---

### Input & Output Formats & Constraints

- **Input:** `node: Optional[Node]`
- **Output:** `Optional[Node]` (Deep copy of the given graph)
- **Constraints:**
  - The number of nodes in the graph is in the range $[0, 100]$.
  - $1 \le \text{Node.val} \le 100$
  - `Node.val` is **unique** for each node.
  - There are no repeated edges and no self-loops in the graph.
  - The graph is connected and all nodes can be visited starting from the given node.

---

### Key Idea & Intuition

- **Cycle Prevention via Lookup Map:**
  - A graph may contain cycles (e.g. Node 1 connects to Node 2, and Node 2 connects back to Node 1).
  - Attempting to clone neighbors naively will cause an infinite recursive loop or duplicate object allocations.
  - Maintain a hash map `cloned` mapping:
    $$\text{original\_node} \implies \text{cloned\_node}$$
- **Registration Invariant:**
  - When visiting a node `curr`:
    1. If `curr` is already present in `cloned`: return `cloned[curr]` immediately.
    2. Otherwise, instantiate `copy = Node(curr.val)`.
    3. **Crucial:** Add `cloned[curr] = copy` **immediately** *before* recursing into `curr.neighbors`!
    4. For each `neighbor` in `curr.neighbors`:
       $$\text{copy.neighbors.append}(\text{dfs}(\text{neighbor}))$$
    5. Return `copy`.

---

### Solution Approach (Step-by-Step)

1. Edge case: If `node is None`, return `None`.
2. Initialize dictionary `clones = {}`.
3. Define `dfs(curr)`:
   - If `curr in clones`: return `clones[curr]`.
   - Create `copy = Node(curr.val)`.
   - Register: `clones[curr] = copy`.
   - Loop over `neighbor` in `curr.neighbors`:
     - `copy.neighbors.append(dfs(neighbor))`
   - Return `copy`.
4. Return `dfs(node)`.

---

### Visual Algorithm Walkthrough

```
Original Graph:
  1 ----- 2
  |       |
  |       |
  4 ----- 3

Clone Progression (DFS):
1. dfs(1):
   - Create Copy(1). clones = {1: Copy(1)}
   - Neighbors of 1: [2, 4]
   - Recurse dfs(2):
     - Create Copy(2). clones = {1: Copy(1), 2: Copy(2)}
     - Neighbors of 2: [1, 3]
       - dfs(1): 1 already in clones! Return Copy(1).
         (Copy(2).neighbors receives Copy(1))
       - Recurse dfs(3):
         - Create Copy(3). clones = {1: Copy(1), 2: Copy(2), 3: Copy(3)}
         - Neighbors of 3: [2, 4]
           - dfs(2): 2 in clones! Return Copy(2).
           - Recurse dfs(4):
             - Create Copy(4). clones = {1: Copy(1), 2: Copy(2), 3: Copy(3), 4: Copy(4)}
             - Neighbors of 4: [1, 3]
               - dfs(1): in clones! Return Copy(1).
               - dfs(3): in clones! Return Copy(3).
             - Returns Copy(4).
         - Returns Copy(3).
     - Returns Copy(2).
   - Back to 1: neighbor 4 -> dfs(4) already in clones! Returns Copy(4).
   - Returns Copy(1).

Deep copy complete. All edges and node instances fully decoupled.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: 4-Node Cycle
- **Input:** `adjList = [[2,4],[1,3],[2,4],[1,3]]`
- **Tracing Table:**
  | Call | Target Node | Action | `clones` Map Size |
  | :--- | :--- | :--- | :--- |
  | 1 | Node 1 | New node created | 1 |
  | 2 | Node 2 | New node created | 2 |
  | 3 | Node 1 | Exists in `clones` | Returns `Copy(1)` |
  | 4 | Node 3 | New node created | 3 |
  | 5 | Node 4 | New node created | 4 |
- **Output:** Cloned graph with identical topology and distinct object references.

#### Example 2: Empty Graph
- **Input:** `node = None`
- **Output:** `None`

#### Example 3: Single Node Without Edges
- **Input:** `adjList = [[]]`
- **Trace:** Creates `Node(1)` with `neighbors = []`.
- **Output:** `[[]]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
            
        clones = {}
        
        def dfs(curr: 'Node') -> 'Node':
            if curr in clones:
                return clones[curr]
                
            # Create copy and register immediately to prevent cycles
            copy = Node(curr.val)
            clones[curr] = copy
            
            for neighbor in curr.neighbors:
                copy.neighbors.append(dfs(neighbor))
                
            return copy
            
        return dfs(node)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Node {
public:
    int val;
    std::vector<Node*> neighbors;
    Node() : val(0), neighbors() {}
    Node(int _val) : val(_val), neighbors() {}
    Node(int _val, std::vector<Node*> _neighbors) : val(_val), neighbors(_neighbors) {}
};

class Solution {
public:
    Node* cloneGraph(Node* node) {
        if (!node) return nullptr;
        std::unordered_map<Node*, Node*> clones;
        return dfs(node, clones);
    }

private:
    Node* dfs(Node* curr, std::unordered_map<Node*, Node*>& clones) {
        if (clones.find(curr) != clones.end()) {
            return clones[curr];
        }

        Node* copy = new Node(curr->val);
        clones[curr] = copy;

        for (Node* neighbor : curr->neighbors) {
            copy->neighbors.push_back(dfs(neighbor, clones));
        }

        return copy;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Node {
    public int val;
    public List<Node> neighbors;
    public Node() {
        val = 0;
        neighbors = new ArrayList<Node>();
    }
    public Node(int _val) {
        val = _val;
        neighbors = new ArrayList<Node>();
    }
    public Node(int _val, ArrayList<Node> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
}

class Solution {
    private Map<Node, Node> clones = new HashMap<>();

    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }

        if (clones.containsKey(node)) {
            return clones.get(node);
        }

        Node copy = new Node(node.val);
        clones.put(node, copy);

        for (Node neighbor : node.neighbors) {
            copy.neighbors.add(cloneGraph(neighbor));
        }

        return copy;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(V + E)$ — Every node $V$ is cloned once, and every edge $E$ is examined twice (once from each neighbor list).
- **Space Complexity:** $\mathcal{O}(V)$ — The `clones` hash map stores $V$ key-value pairs. The recursion stack depth is bounded by $V$ in the worst case (linear chain graph).

---

### Takeaway Pattern & Interview Traps

1. **Premature Recursion Trap:** Storing `clones[curr] = copy` *after* looping over neighbors will fail on any graph with cycles, causing a `RecursionError` / `StackOverflowError`. Always register the copy in the map immediately upon instantiation.
2. **Deep Copy vs Shallow Copy:** Creating a new node but copying the original neighbor list directly (`copy.neighbors = curr.neighbors`) is a shallow copy and violates the problem requirement. Every connected node and neighbor list must be new memory allocations.
3. **BFS Implementation Option:** BFS is equally viable using a queue. When popping `curr`, instantiate unvisited neighbors, register them in `cloned`, and push them to the queue.