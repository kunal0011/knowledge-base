---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 210: Course Schedule II"
tags:
  - leetcode
  - coding
  - graphs
  - topological-sort
  - amazon
  - google
---

# LeetCode 210: Course Schedule II

**Target Companies:** Amazon (Top #1 Graph Question), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Topological Sort (Kahn's Algorithm / In-Degree BFS)

---

### Problem Statement

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [a_i, b_i]` indicates that you **must** take course $b_i$ first if you want to take course $a_i$.

- For example, the pair `[0, 1]` indicates that to take course `0` you have to first take course `1`.

Return the **ordering of courses** you should take to finish all courses. If there are many valid answers, return **any** of them. If it is impossible to finish all courses, return an **empty array**.

---

### Input & Output Formats & Constraints

- **Input:** `numCourses: int`, `prerequisites: List[List[int]]`
- **Output:** `List[int]` (topological ordering or `[]` if cycle detected)
- **Constraints:**
  - $1 \le \text{numCourses} \le 2000$
  - $0 \le \text{prerequisites.length} \le \text{numCourses} \times (\text{numCourses} - 1)$
  - `prerequisites[i].length == 2`
  - $0 \le a_i, b_i < \text{numCourses}$
  - $a_i \ne b_i$
  - All the pairs `[a_i, b_i]` are distinct.

---

### Key Idea & Intuition

- **Directed Acyclic Graph (DAG) Modeling:**
  - An edge $b \to a$ represents dependency ($b$ must be taken before $a$).
  - A valid course order is a **Topological Ordering** of this directed graph.
  - If the graph contains a **cycle**, no valid ordering exists!
- **Kahn's Algorithm (BFS with In-Degrees):**
  1. Calculate the in-degree (number of prerequisite dependencies) for each node.
  2. Nodes with `in_degree == 0` have no prerequisites satisfied and can be taken immediately. Push them all into a queue.
  3. While queue is not empty:
     - Pop node `u`, append `u` to `order`.
     - For each neighbor `v` of `u` (edges $u \to v$):
       - Decrement `in_degree[v] -= 1`.
       - If `in_degree[v] == 0`, all prerequisites for `v` are completed! Push `v` to queue.
  4. If `len(order) == numCourses`, return `order`; otherwise, a cycle exists, return `[]`.

---

### Solution Approach (Step-by-Step)

1. **Build Adjacency List & Compute In-Degrees:**
   - Initialize `adj = [[] for _ in range(numCourses)]` and `in_degree = [0] * numCourses`.
   - For each `[dest, src]` in `prerequisites`:
     - Add directed edge `src -> dest`: `adj[src].append(dest)`.
     - Increment `in_degree[dest] += 1`.
2. **Initialize Queue with In-Degree 0 Nodes:**
   - Enqueue all course IDs $i$ with `in_degree[i] == 0`.
3. **Process Courses via BFS:**
   - Initialize `order = []`.
   - While queue is not empty:
     - Dequeue course `curr`, append to `order`.
     - For each prerequisite-dependent course `neighbor` in `adj[curr]`:
       - Decrement `in_degree[neighbor] -= 1`.
       - If `in_degree[neighbor] == 0`, enqueue `neighbor`.
4. **Cycle Detection Check:**
   - If `len(order) == numCourses`, return `order`.
   - Else (cycle detected, cannot complete all courses), return `[]`.

---

### Visual Algorithm Walkthrough

```
Prerequisites: [1, 0], [2, 0], [3, 1], [3, 2]
Edges: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3

In-degrees:
0: 0 (No prerequisites)
1: 1 (Needs 0)
2: 1 (Needs 0)
3: 2 (Needs 1 and 2)

Step 1: Queue = [0], Order = []
Step 2: Pop 0 -> Order = [0]. Decrement in-degree of 1 and 2:
        in_degree[1] = 0 -> Queue.push(1)
        in_degree[2] = 0 -> Queue.push(2)
        Queue = [1, 2]
Step 3: Pop 1 -> Order = [0, 1]. in_degree[3] becomes 2 - 1 = 1.
Step 4: Pop 2 -> Order = [0, 1, 2]. in_degree[3] becomes 1 - 1 = 0 -> Queue.push(3).
Step 5: Pop 3 -> Order = [0, 1, 2, 3].

Result: [0, 1, 2, 3] (or [0, 2, 1, 3])
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Multi-Prerequisite DAG
- **Input:** `numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]`
- **Output:** `[0, 1, 2, 3]`

#### Example 2: Cycle Present (Impossible)
- **Input:** `numCourses = 2, prerequisites = [[1,0],[0,1]]`
- **Trace:** Both nodes have in-degree 1. Queue starts empty. `len(order) = 0 != 2`.
- **Output:** `[]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque, defaultdict

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj = defaultdict(list)
        in_degree = [0] * numCourses
        
        for dest, src in prerequisites:
            adj[src].append(dest)
            in_degree[dest] += 1
            
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        order = []
        
        while queue:
            curr = queue.popleft()
            order.append(curr)
            
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return order if len(order) == numCourses else []
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

class Solution {
public:
    std::vector<int> findOrder(int numCourses, std::vector<std::vector<int>>& prerequisites) {
        std::vector<std::vector<int>> adj(numCourses);
        std::vector<int> inDegree(numCourses, 0);

        for (const auto& pre : prerequisites) {
            int dest = pre[0], src = pre[1];
            adj[src].push_back(dest);
            inDegree[dest]++;
        }

        std::queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (inDegree[i] == 0) q.push(i);
        }

        std::vector<int> order;
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            order.push_back(curr);

            for (int next : adj[curr]) {
                if (--inDegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        return order.size() == numCourses ? order : std::vector<int>();
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) adj.add(new ArrayList<>());
        
        int[] inDegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            int dest = pre[0], src = pre[1];
            adj.get(src).add(dest);
            inDegree[dest]++;
        }

        Queue<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) queue.offer(i);
        }

        int[] order = new int[numCourses];
        int idx = 0;

        while (!queue.isEmpty()) {
            int curr = queue.poll();
            order[idx++] = curr;

            for (int neighbor : adj.get(curr)) {
                if (--inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        return idx == numCourses ? order : new int[0];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(V + E)$ where $V = \text{numCourses}$ and $E = \text{prerequisites.length}$. Every vertex and directed edge is traversed exactly once.
- **Space Complexity:** $\mathcal{O}(V + E)$ for the adjacency list and queue.

---

### Takeaway Pattern & Interview Traps

1. **Topological Order Uniqueness:** A directed acyclic graph can have **multiple valid topological orderings** (e.g. $[0, 1, 2, 3]$ and $[0, 2, 1, 3]$). The problem statement allows returning any valid order.
2. **Cycle Returns Empty List:** If `len(order) != numCourses`, there is at least one cycle. Returning `[]` is strictly required.
3. **Difference from LeetCode 207:** LeetCode 207 tests if a topological sort is possible (`bool`), while LeetCode 210 asks for the actual topological ordering array (`List[int]`).

