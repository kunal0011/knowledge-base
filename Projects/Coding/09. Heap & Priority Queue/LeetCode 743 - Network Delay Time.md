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
  - graph
  - shortest-path
  - amazon
  - google
---

# LeetCode 743: Network Delay Time

**Target Companies:** Google, Amazon, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Priority Queue (Min-Heap) / Dijkstra's Algorithm / Single-Source Shortest Path

---

### Problem Statement

You are given a network of `n` nodes, labeled from `1` to `n`. You are also given `times`, a list of travel times as directed edges `times[i] = (u_i, v_i, w_i)`, where `u_i` is the source node, `v_i` is the target node, and `w_i` is the time it takes for a signal to travel from source to target.

We will send a signal from a given node `k`. Return *the **minimum** time it takes for all the `n` nodes to receive the signal*. If it is impossible for all the `n` nodes to receive the signal, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `times`: `List[List[int]]`, where each element is `[u, v, w]`. $1 \le \text{times.length} \le 6000$.
  - `n`: `int`, total nodes labeled $1$ to $n$ ($1 \le n \le 100$).
  - `k`: `int`, start node ($1 \le k \le n$).
  - $0 \le w_i \le 100$.
  - All pairs $(u_i, v_i)$ are unique (no multi-edges in the same direction).
- **Output:**
  - `int`: Maximum shortest-path distance from node `k` across all $n$ nodes, or `-1` if any node remains unreachable.
- **Constraints:**
  - Edge weights are non-negative ($w_i \ge 0$), perfectly enabling Dijkstra's greedy priority queue strategy.

---

### Key Idea & Intuition

Because the signal propagates concurrently across all outgoing channels, the time when node $x$ receives the signal is the **shortest path distance** from source node $k$ to node $x$.
For all nodes to receive the signal, the total time required is the **maximum** of these shortest paths:
$$\text{Total Time} = \max_{1 \le i \le n}(\text{dist}(k, i))$$
If any node $i$ has $\text{dist}(k, i) = \infty$, the graph is not strongly connected from $k$, and we must return `-1`.

#### Dijkstra's Algorithm with a Min-Heap:
Because all edge weights are non-negative ($w \ge 0$):
1. Maintain `dist[1...n]`, initialized to $\infty$, with `dist[k] = 0`.
2. Maintain a **min-heap** of pairs: `(time_so_far, node)`.
3. In each step, pop the node with the minimum `time_so_far`:
   - If `time_so_far > dist[node]`: this is a stale heap entry from an earlier relaxation; discard it.
   - Otherwise, iterate through all outgoing edges `(neighbor, weight)` from `node`:
     - If `time_so_far + weight < dist[neighbor]`:
       - `dist[neighbor] = time_so_far + weight`
       - Push `(dist[neighbor], neighbor)` into the min-heap.
4. After the heap empties, find $ans = \max_{1 \le i \le n}(dist[i])$. Return $-1$ if $ans == \infty$, otherwise return $ans$.

---

### Solution Approach (Step-by-Step)

1. **Construct Graph:**
   - Adjacency list: `graph = defaultdict(list)` where `graph[u].append((v, w))`.
2. **Initialize Distance Table & Min-Heap:**
   - `dist = {i: float('inf') for i in range(1, n + 1)}`
   - `dist[k] = 0`
   - `min_heap = [(0, k)]`
3. **Execute Dijkstra:**
   - While `min_heap` is non-empty:
     - Pop `(d, u) = heappop(min_heap)`.
     - If `d > dist[u]`: continue (skip stale entry).
     - For each `(v, w)` in `graph[u]`:
       - If `d + w < dist[v]`:
         - `dist[v] = d + w`
         - `heappush(min_heap, (dist[v], v))`
4. **Determine Result:**
   - `max_dist = max(dist.values())`
   - Return `max_dist if max_dist < float('inf') else -1`.

---

### Visual Algorithm Walkthrough

Let `times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]`, $n = 4$, $k = 2$:

```
Graph:
    (1) --[1]--> (1)
     ^
     | [1]
    (2) --[1]--> (3) --[1]--> (4)

Initialization:
  dist = {1: inf, 2: 0, 3: inf, 4: inf}
  min_heap = [(0, 2)]

Step 1:
  Pop (0, 2).
  Explore neighbors of 2:
    - Node 1: new dist = 0 + 1 = 1 < inf -> dist[1] = 1, push (1, 1)
    - Node 3: new dist = 0 + 1 = 1 < inf -> dist[3] = 1, push (1, 3)
  Heap: [ (1, 1), (1, 3) ]

Step 2:
  Pop (1, 1).
  Node 1 has no outgoing edges.
  Heap: [ (1, 3) ]

Step 3:
  Pop (1, 3).
  Explore neighbors of 3:
    - Node 4: new dist = 1 + 1 = 2 < inf -> dist[4] = 2, push (2, 4)
  Heap: [ (2, 4) ]

Step 4:
  Pop (2, 4).
  Node 4 has no outgoing edges.
  Heap empty.

Final Distances:
  Node 1: 1
  Node 2: 0
  Node 3: 1
  Node 4: 2

Max distance = max(1, 0, 1, 2) = 2.
Output: 2
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Fully Connected Network

- **Input:** `times = [[2,1,1],[2,3,1],[3,4,1]]`, `n = 4`, `k = 2`
- **Output:** `2`

#### Example 2: Simple 2-Node Graph

- **Input:** `times = [[1,2,1]]`, `n = 2`, `k = 1`
- **Output:** `1`

#### Example 3: Unreachable Node

- **Input:** `times = [[1,2,1]]`, `n = 2`, `k = 2`
- **Tracing:** Starting from 2, node 1 cannot be reached ($\text{dist}[1] = \infty$).
- **Output:** `-1`

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from collections import defaultdict
from typing import List

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Step 1: Build adjacency list
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))

        # Step 2: Distance array
        dist = [float('inf')] * (n + 1)
        dist[k] = 0

        # Step 3: Priority Queue storing (travel_time, node)
        min_heap = [(0, k)]

        while min_heap:
            d, u = heapq.heappop(min_heap)

            # Skip stale entry
            if d > dist[u]:
                continue

            for v, w in graph[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(min_heap, (dist[v], v))

        # Step 4: Minimum time for all nodes to receive signal
        max_time = max(dist[1:])
        return -1 if max_time == float('inf') else max_time
```

#### C++17

```cpp
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>

class Solution {
public:
    int networkDelayTime(const std::vector<std::vector<int>>& times, int n, int k) {
        // Build graph
        std::vector<std::vector<std::pair<int, int>>> graph(n + 1);
        for (const auto& edge : times) {
            graph[edge[0]].emplace_back(edge[1], edge[2]);
        }

        std::vector<int> dist(n + 1, INT_MAX);
        dist[k] = 0;

        // Min-heap storing pair of (distance, node)
        using NodeDist = std::pair<int, int>;
        std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> min_heap;
        min_heap.emplace(0, k);

        while (!min_heap.empty()) {
            auto [d, u] = min_heap.top();
            min_heap.pop();

            if (d > dist[u]) continue;

            for (const auto& [v, w] : graph[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    min_heap.emplace(dist[v], v);
                }
            }
        }

        int max_time = 0;
        for (int i = 1; i <= n; ++i) {
            if (dist[i] == INT_MAX) return -1;
            max_time = std::max(max_time, dist[i]);
        }

        return max_time;
    }
};
```

#### Java

```java
import java.util.*;

public class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        List<List<int[]>> graph = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            graph.add(new ArrayList<>());
        }

        for (int[] edge : times) {
            graph.get(edge[0]).add(new int[]{edge[1], edge[2]});
        }

        int[] dist = new int[n + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[k] = 0;

        // Min-heap storing [distance, node]
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(a[0], b[0])
        );
        minHeap.offer(new int[]{0, k});

        while (!minHeap.isEmpty()) {
            int[] top = minHeap.poll();
            int d = top[0];
            int u = top[1];

            if (d > dist[u]) continue;

            for (int[] edge : graph.get(u)) {
                int v = edge[0];
                int w = edge[1];

                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    minHeap.offer(new int[]{dist[v], v});
                }
            }
        }

        int maxTime = 0;
        for (int i = 1; i <= n; i++) {
            if (dist[i] == Integer.MAX_VALUE) return -1;
            maxTime = Math.max(maxTime, dist[i]);
        }

        return maxTime;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(E \log V)$
  - In the worst case, every directed edge relaxation pushes a new element into the priority queue, where $E$ is the number of edges ($E \le 6000$) and $V = n \le 100$.
  - Total time: $\mathcal{O}(E \log V)$, which runs in $< 5\text{ ms}$.
- **Space Complexity:** $\mathcal{O}(V + E)$
  - The adjacency list consumes $\mathcal{O}(V + E)$ space, and the heap contains at most $E$ entries.

---

### Takeaway Pattern & Interview Traps

1. **Skipping Stale Entries:**
   - Always check `if d > dist[u]: continue`. Without this check, multiple outdated paths pushed to the heap will wastefully re-examine neighbors, turning the Dijkstra implementation into a slow exponential exploration.
2. **Returning `max(dist)` vs `dist[dest]`:**
   - The question asks for the time until **all** nodes receive the signal. Many candidates mistakenly return the distance to node $n$ or the sum of distances. The correct answer is the **maximum** of all shortest paths.