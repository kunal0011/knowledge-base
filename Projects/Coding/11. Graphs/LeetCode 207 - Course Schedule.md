---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 207: Course Schedule"
tags:
  - leetcode
  - coding
  - graphs
  - topological-sort
  - bfs
  - dfs
  - amazon
  - google
---

# LeetCode 207: Course Schedule

**Target Companies:** Amazon (All-Time Top #1 Directed Graph Problem), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Topological Sort / Kahn's Algorithm / Cycle Detection in Directed Graphs

---

### Problem Statement

There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [a_i, b_i]` indicates that you **must** take course $b_i$ first if you want to take course $a_i$.

- For example, the pair `[0, 1]` indicates that to take course `0` you have to first take course `1`.

Return `true` if you can finish all courses. Otherwise, return `false`.

---

### Input & Output Formats & Constraints

- **Input:** `numCourses: int`, `prerequisites: List[List[int]]`
- **Output:** `bool` — `true` if all courses can be completed, `false` otherwise.
- **Constraints:**
  - $1 \le \text{numCourses} \le 2000$
  - $0 \le \text{prerequisites.length} \le 5000$
  - $\text{prerequisites}[i].\text{length} == 2$
  - $0 \le a_i, b_i < \text{numCourses}$
  - All the pairs `prerequisites[i]` are **unique**.

---

### Key Idea & Intuition

- **Directed Acyclic Graph (DAG) Invariant:**
  - Taking course $b$ before $a$ translates to a directed edge:
    $$b \longrightarrow a$$
  - Finishing all courses is possible **if and only if** there are no circular dependencies (i.e. the directed graph is **acyclic**).
  - If a cycle exists (e.g. $A \to B \to A$), neither course can ever be started.
- **Kahn's Algorithm (BFS with In-Degrees):**
  1. Calculate `in_degree[i]` for every course $i$ (number of incoming prerequisite edges).
  2. Courses with `in_degree[i] == 0` have no prerequisites and can be taken immediately. Push them all into a FIFO `queue`.
  3. While `queue` is not empty:
     - Pop course $u$, increment `courses_completed += 1`.
     - For each course $v$ dependent on $u$ (edge $u \to v$):
       - Decrement `in_degree[v] -= 1`.
       - If `in_degree[v] == 0`: course $v$'s prerequisites are fully satisfied $\implies$ push $v$ into `queue`.
  4. If `courses_completed == numCourses`, all courses were processed without deadlocks $\implies$ return `True`. Otherwise, a cycle prevented some courses from reaching in-degree 0 $\implies$ return `False`.

---

### Solution Approach (Step-by-Step)

1. Build adjacency list `adj` where `adj[b].append(a)` for each `[a, b]` in `prerequisites`.
2. Compute `indegree = [0] * numCourses`. Increment `indegree[a] += 1`.
3. Initialize `queue = deque([i for i in range(numCourses) if indegree[i] == 0])`.
4. Initialize `completed = 0`.
5. While `queue` is not empty:
   - Pop `curr = queue.popleft()`.
   - `completed += 1`.
   - For `nxt` in `adj[curr]`:
     - `indegree[nxt] -= 1`.
     - If `indegree[nxt] == 0`:
       - `queue.append(nxt)`.
6. Return `completed == numCourses`.

---

### Visual Algorithm Walkthrough

```
Case 1: No Cycle (Can Finish)
numCourses = 4, prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
Dependencies: 0 -> 1 -> 3
              0 -> 2 -> 3

In-degrees:
0: 0, 1: 1, 2: 1, 3: 2

Step 1: Queue = [0]
Pop 0 -> completed = 1
  Decrement indegree[1] -> 0 (enqueue 1)
  Decrement indegree[2] -> 0 (enqueue 2)
  Queue = [1, 2]

Step 2: Pop 1 -> completed = 2
  Decrement indegree[3] -> 1 (not 0 yet)
  Queue = [2]

Step 3: Pop 2 -> completed = 3
  Decrement indegree[3] -> 0 (enqueue 3)
  Queue = [3]

Step 4: Pop 3 -> completed = 4
completed (4) == numCourses (4) -> Return True!

----------------------------------------------------

Case 2: Cycle Present (Cannot Finish)
numCourses = 2, prerequisites = [[1, 0], [0, 1]]
0 <---> 1 (Mutual dependency cycle)
In-degrees: 0: 1, 1: 1
Queue = [] (No node has indegree 0!)
Loop terminates immediately -> completed = 0 != 2 -> Return False!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Valid Path
- **Input:** `numCourses = 2, prerequisites = [[1, 0]]`
- **In-degrees:** `node 0: 0`, `node 1: 1`.
- **Trace:**
  - Queue begins with `[0]`.
  - Pop 0 $\implies$ completed = 1. `indegree[1]` becomes 0 $\implies$ enqueue 1.
  - Pop 1 $\implies$ completed = 2.
- **Output:** `true`

#### Example 2: Direct Cycle
- **Input:** `numCourses = 2, prerequisites = [[1, 0], [0, 1]]`
- **Output:** `false`

#### Example 3: Disconnected Independent Courses
- **Input:** `numCourses = 3, prerequisites = []`
- **Trace:** All 3 courses have in-degree 0. All 3 processed.
- **Output:** `true`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque, defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = defaultdict(list)
        indegree = [0] * numCourses
        
        # Build graph: prereq (b) -> course (a)
        for course, prereq in prerequisites:
            adj[prereq].append(course)
            indegree[course] += 1
            
        # Queue all nodes with 0 in-degree
        queue = deque([i for i in range(numCourses) if indegree[i] == 0])
        completed = 0
        
        while queue:
            curr = queue.popleft()
            completed += 1
            
            for neighbor in adj[curr]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return completed == numCourses
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

class Solution {
public:
    bool canFinish(int numCourses, std::vector<std::vector<int>>& prerequisites) {
        std::vector<std::vector<int>> adj(numCourses);
        std::vector<int> indegree(numCourses, 0);

        for (const auto& pre : prerequisites) {
            int course = pre[0], prereq = pre[1];
            adj[prereq].push_back(course);
            indegree[course]++;
        }

        std::queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (indegree[i] == 0) {
                q.push(i);
            }
        }

        int completed = 0;
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            completed++;

            for (int neighbor : adj[curr]) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }

        return completed == numCourses;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            adj.add(new ArrayList<>());
        }

        int[] indegree = new int[numCourses];
        for (int[] pre : prerequisites) {
            int course = pre[0], prereq = pre[1];
            adj.get(prereq).add(course);
            indegree[course]++;
        }

        Queue<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        int completed = 0;
        while (!queue.isEmpty()) {
            int curr = queue.poll();
            completed++;

            for (int neighbor : adj.get(curr)) {
                indegree[neighbor]--;
                if (indegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        return completed == numCourses;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(V + E)$ where $V = \text{numCourses}$ and $E = \text{prerequisites.length}$. Building the graph takes $\mathcal{O}(V + E)$ time. In Kahn's BFS, each node is enqueued and dequeued once, and each edge is traversed once.
- **Space Complexity:** $\mathcal{O}(V + E)$ — The adjacency list stores $V$ vertices and $E$ edges. The `indegree` array and BFS queue each require $\mathcal{O}(V)$ memory.

---

### Takeaway Pattern & Interview Traps

1. **Reversed Dependency Edge Direction:** In `[a, b]`, $b$ is the prerequisite for $a$. The edge must be directed $b \to a$ (not $a \to b$). Inverting the edge direction causes incorrect in-degree counts.
2. **Cycle vs Visited Count:** A common mistake is assuming that if the queue becomes empty, the answer is `True`. If there is a cycle, the loop finishes prematurely with `completed < numCourses`. The check `completed == numCourses` is essential.
3. **Equivalence with LeetCode 210:** LeetCode 207 asks for feasibility (`bool`), whereas LeetCode 210 asks for the actual topological order (`List[int]`).