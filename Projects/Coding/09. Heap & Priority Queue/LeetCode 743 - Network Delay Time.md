---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 743: Network Delay Time"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 743: Network Delay Time

Below is a structured, interview-grade explanation of **LeetCode 743 – Network Delay Time**, aligned with how this problem is typically discussed in system design and algorithmic interviews.

---

## 1. Problem Statement

You are given a directed weighted graph representing a network of `n` nodes, labeled from `1` to `n`.

* `times[i] = [u, v, w]` means there is a directed edge from node `u` to node `v` with travel time `w`.
* A signal is sent from a starting node `k`.

**Objective:**  
Determine the minimum time required for the signal to reach **all** nodes in the network.  
If it is impossible to reach all nodes, return `-1`.

---

## 2. Key Observations

1. **Shortest path from a single source**

   * We need the shortest time from node `k` to every other node.
   * Edge weights are **non-negative**.
2. **Directed weighted graph**

   * Simple BFS does not work because weights are not uniform.
   * Bellman–Ford is possible but inefficient for large graphs.
3. **Classic use case for Dijkstra’s Algorithm**

   * Single-source shortest path
   * Non-negative edge weights
   * Efficient using a **priority queue (min-heap)**
4. **Final answer is not a single distance**

   * We need the **maximum** shortest distance among all nodes.
   * If any node is unreachable → return `-1`.

---

## 3. Priority Queue Technique (Dijkstra)

### Why Priority Queue?

* Always expand the node with the **minimum known distance**.
* Guarantees optimal shortest paths when edge weights are non-negative.

### Core Idea

1. Build an adjacency list.
2. Maintain a distance array `dist[]`, initialized to infinity.
3. Use a min-heap storing `(current_time, node)`.
4. Repeatedly:

   * Pop the node with the smallest time.
   * Relax its outgoing edges.
5. Ignore stale entries (when popped distance > recorded distance).

---

## 4. Algorithm Steps

1. Build adjacency list from `times`.
2. Initialize:

   * `dist[k] = 0`
   * All other distances = `∞`
3. Push `(0, k)` into the priority queue.
4. While the heap is not empty:

   * Pop `(time, node)`
   * If `time > dist[node]`, skip
   * For each neighbor:

     * If `time + weight < dist[neighbor]`, update and push
5. If any node has distance `∞`, return `-1`
6. Otherwise, return `max(dist[1..n])`

---

## 5. Python 3 Solution (with typing)

```python
from typing import List
import heapq
from collections import defaultdict

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Step 1: Build adjacency list
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))

        # Step 2: Distance array
        INF = float('inf')
        dist = [INF] * (n + 1)
        dist[k] = 0

        # Step 3: Min-heap (time, node)
        min_heap = [(0, k)]

        # Step 4: Dijkstra
        while min_heap:
            curr_time, node = heapq.heappop(min_heap)

            if curr_time > dist[node]:
                continue

            for neighbor, weight in graph[node]:
                new_time = curr_time + weight
                if new_time < dist[neighbor]:
                    dist[neighbor] = new_time
                    heapq.heappush(min_heap, (new_time, neighbor))

        # Step 5: Check reachability
        max_time = max(dist[1:])
        return -1 if max_time == INF else max_time
```

---

## 6. Worked Example 1

### Input

```
times = [[2,1,1], [2,3,1], [3,4,1]]
n = 4
k = 2
```

### Graph

```python
2 -> 1 (1)
2 -> 3 (1)
3 -> 4 (1)
```

### Step-by-Step Execution

| Step | Heap | dist[] (1..4) |
| --- | --- | --- |
| Init | (0,2) | [∞, 0, ∞, ∞] |
| Pop | (0,2) |  |
| Relax | 1=1, 3=1 | [1, 0, 1, ∞] |
| Pop | (1,1) | no edges |
| Pop | (1,3) | relax 4=2 |
| Pop | (2,4) | no edges |

### Final Distances

```
Node 1 → 1
Node 2 → 0
Node 3 → 1
Node 4 → 2
```

### Answer

```
max = 2
```

---

## 7. Worked Example 2 (Unreachable Node)

### Input

```
times = [[1,2,1]]
n = 2
k = 2
```

### Analysis

* Node `2` can reach no one.
* Node `1` remains unreachable.

### dist[]

```
[∞, ∞, 0]
```

### Answer

```
-1
```

---

## 8. Time and Space Complexity

### Time Complexity

* **O(E log V)**

  * Each edge relaxation involves heap operations.

### Space Complexity

* **O(V + E)**

  * Graph storage + distance array + heap.

---

## 9. Common Interview Pitfalls

1. Using BFS instead of Dijkstra (incorrect for weighted graphs).
2. Forgetting to skip stale heap entries.
3. Returning `dist[n]` instead of `max(dist)`.
4. Not handling unreachable nodes.

---

## 10. Pattern Recognition

This problem is a **canonical example** of:

> **Single Source Shortest Path (SSSP) with non-negative weights**

Closely related problems:

* LeetCode 787 (Cheapest Flights Within K Stops)
* LeetCode 1514 (Path with Maximum Probability)
* LeetCode 1631 (Minimum Effort Path)

---

If you want, I can also provide:

* Dry run with a heap state diagram
* Comparison with Bellman–Ford
* How this maps to real systems (network propagation, distributed systems)