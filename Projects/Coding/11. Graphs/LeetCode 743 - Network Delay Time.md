---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 743: Network Delay Time"
tags:
  - leetcode
  - coding
  - graphs
  - dijkstra
  - shortest-path
  - heap
  - amazon
  - google
---

# LeetCode 743: Network Delay Time

**Target Companies:** Amazon (Top Tier Dijkstra Classic), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Single-Source Shortest Path / Dijkstra's Algorithm / Min-Heap Priority Queue

---

### Problem Statement

You are given a network of `n` nodes, labeled from `1` to `n`. You are also given `times`, a list of travel times as directed edges `times[i] = (u_i, v_i, w_i)`, where `u_i` is the source node, `v_i` is the target node, and `w_i` is the time it takes for a signal to travel from source to target.

We will send a signal from a given node `k`. Return the **minimum time** it takes for all the `n` nodes to receive the signal. If it is impossible for all the `n` nodes to receive the signal, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:** `times: List[List[int]]`, `n: int`, `k: int`
- **Output:** `int` — Minimum time for all nodes to receive signal, or `-1`.
- **Constraints:**
  - $1 \le k \le n \le 100$
  - $1 \le \text{times.length} \le 6000$
  - $\text{times}[i].\text{length} == 3$
  - $1 \le u_i, v_i \le n$
  - $u_i \ne v_i$
  - $0 \le w_i \le 100$
  - All pairs $(u_i, v_i)$ are unique (no multiple edges in the same direction).

---

### Key Idea & Intuition

- **Single-Source Shortest Path Formulation:**
  - Since edge weights represent signal transmission times ($w_i \ge 0$), the time a node $v$ receives the signal from source $k$ is precisely the **shortest path distance** from $k$ to $v$.
  - The entire network receives the signal when the **last (farthest) node** receives it:
    $$\text{Total Time} = \max_{v \in \{1, \dots, n\}} \text{dist}(k, v)$$
  - If any node cannot be reached from $k$, return `-1`.
- **Dijkstra's Algorithm with Min-Heap:**
  - Initialize a priority queue (min-heap) storing `(distance, node)` pairs, starting with `(0, k)`.
  - Maintain a hash map or array `settled` of shortest known distances.
  - While the heap is not empty:
    - Pop `(d, u)` with the smallest distance.
    - If `u` is already settled, discard it (stale entry).
    - Mark `u` settled with distance `d`.
    - For each outgoing edge $u \xrightarrow{w} v$:
      - If $v$ is not yet settled, push `(d + w, v)` into the heap.
  - When the heap is empty, if the number of settled nodes is $n$, the answer is $\max(\text{settled distances})$. Otherwise, return `-1`.

---

### Solution Approach (Step-by-Step)

1. Build directed adjacency list `adj` mapping `u -> [(v, w)]`.
2. Initialize `min_heap = [(0, k)]` and `settled = {}`.
3. While `min_heap` is not empty:
   - Pop `d, u = heapq.heappop(min_heap)`.
   - If `u in settled`: continue.
   - `settled[u] = d`.
   - For neighbor $v$ and weight $w$ in `adj[u]`:
     - If $v$ not in `settled`:
       - `heapq.heappush(min_heap, (d + w, v))`.
4. If `len(settled) == n`: return `max(settled.values())`.
5. Else return `-1`.

---

### Visual Algorithm Walkthrough

```
Times: [[2,1,1], [2,3,1], [3,4,1]], n = 4, k = 2

Network Topology:
       (1)
      ^
     / 1
   (2) ---> 1 ---> (3) ---> 1 ---> (4)

Dijkstra Step-by-Step:
1. Start at k = 2: Heap = [(0, 2)]
2. Pop (0, 2) -> Settled: {2: 0}
   - Edge 2 -> 1 (w=1): Push (1, 1)
   - Edge 2 -> 3 (w=1): Push (1, 3)
   Heap = [(1, 1), (1, 3)]

3. Pop (1, 1) -> Settled: {2: 0, 1: 1}
   - No outgoing edges from 1.
   Heap = [(1, 3)]

4. Pop (1, 3) -> Settled: {2: 0, 1: 1, 3: 1}
   - Edge 3 -> 4 (w=1): Push (1 + 1 = 2, 4)
   Heap = [(2, 4)]

5. Pop (2, 4) -> Settled: {2: 0, 1: 1, 3: 1, 4: 2}

All 4 nodes settled!
Final distances: {1: 1, 2: 0, 3: 1, 4: 2}
Max distance = 2.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Connected Network
- **Input:** `times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2`
- **Tracing Table:**
  | Iteration | Node Popped | Distance Settled | Neighbors Added |
  | :--- | :--- | :--- | :--- |
  | 1 | 2 | 0 | `(1, 1), (1, 3)` |
  | 2 | 1 | 1 | None |
  | 3 | 3 | 1 | `(2, 4)` |
  | 4 | 4 | 2 | None |
- **Output:** `2`

#### Example 2: Unreachable Node
- **Input:** `times = [[1,2,1]], n = 2, k = 2`
- **Analysis:** Node 2 has no outgoing edges to reach node 1. Settled nodes count $= 1 \ne 2$.
- **Output:** `-1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
import heapq
from typing import List
from collections import defaultdict

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        adj = defaultdict(list)
        for u, v, w in times:
            adj[u].append((v, w))
            
        min_heap = [(0, k)]  # (dist, node)
        settled = {}
        
        while min_heap:
            d, u = heapq.heappop(min_heap)
            if u in settled:
                continue
            settled[u] = d
            
            for v, w in adj[u]:
                if v not in settled:
                    heapq.heappush(min_heap, (d + w, v))
                    
        return max(settled.values()) if len(settled) == n else -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>

class Solution {
public:
    int networkDelayTime(std::vector<std::vector<int>>& times, int n, int k) {
        std::vector<std::vector<std::pair<int, int>>> adj(n + 1);
        for (const auto& edge : times) {
            adj[edge[0]].push_back({edge[1], edge[2]});
        }

        // Min-heap storing {distance, node}
        std::priority_queue<std::pair<int, int>, 
                            std::vector<std::pair<int, int>>, 
                            std::greater<std::pair<int, int>>> pq;

        std::vector<int> dist(n + 1, INT_MAX);
        dist[k] = 0;
        pq.push({0, k});

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            if (d > dist[u]) continue;

            for (const auto& [v, w] : adj[u]) {
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pq.push({dist[v], v});
                }
            }
        }

        int maxTime = 0;
        for (int i = 1; i <= n; ++i) {
            if (dist[i] == INT_MAX) return -1;
            maxTime = std::max(maxTime, dist[i]);
        }

        return maxTime;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        List<List<int[]>> adj = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            adj.add(new ArrayList<>());
        }

        for (int[] edge : times) {
            adj.get(edge[0]).add(new int[]{edge[1], edge[2]});
        }

        // Min-heap comparing by distance
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        int[] dist = new int[n + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);

        dist[k] = 0;
        pq.offer(new int[]{0, k});

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int d = curr[0], u = curr[1];

            if (d > dist[u]) continue;

            for (int[] edge : adj.get(u)) {
                int v = edge[0], w = edge[1];
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    pq.offer(new int[]{dist[v], v});
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

- **Time Complexity:** $\mathcal{O}((V + E) \log V)$ where $V = n$ and $E = \text{times.length}$. Inserting and extracting from the binary min-heap takes $\mathcal{O}(\log V)$ time for each of the $E$ edge relaxations.
- **Space Complexity:** $\mathcal{O}(V + E)$ — The adjacency list stores $V$ vertices and $E$ edges. The distance table and min-heap take $\mathcal{O}(V + E)$ space.

---

### Takeaway Pattern & Interview Traps

1. **Non-Negative Weights Prerequisite:** Dijkstra’s algorithm is only correct when all edge weights are non-negative ($w \ge 0$). If negative weights existed, Bellman-Ford or SPFA would be required.
2. **Discard Stale Heap Elements:** Always check `if d > dist[u]: continue`. When a node receives a shorter distance later, earlier worse entries remain in the priority queue. Skipping them prevents redundant neighbor evaluations.
3. **1-Indexed Nodes:** Remember that nodes are numbered $1$ to $n$. Allocate size $n + 1$ for arrays in C++ and Java.