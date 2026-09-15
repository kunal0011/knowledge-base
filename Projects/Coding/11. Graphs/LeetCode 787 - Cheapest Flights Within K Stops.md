---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 787: Cheapest Flights Within K Stops"
tags:
  - leetcode
  - coding
  - graphs
  - bellman-ford
  - shortest-path
  - amazon
  - google
---

# LeetCode 787: Cheapest Flights Within K Stops

**Target Companies:** Google, Amazon, Meta, Airbnb  
**Difficulty:** Medium  
**Topic:** Bellman-Ford Algorithm / Level-Restricted Edge Relaxation / Shortest Path with Hop Constraint

---

### Problem Statement

There are `n` cities connected by some number of flights. You are given an array `flights` where `flights[i] = [from_i, to_i, price_i]` indicates that there is a flight from city `from_i` to city `to_i` with cost `price_i`.

You are also given three integers `src`, `dst`, and `k`, return the **cheapest price** from `src` to `dst` with at most `k` stops. If there is no such route, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`, `flights: List[List[int]]`, `src: int`, `dst: int`, `k: int`
- **Output:** `int` — Minimum cost to reach `dst` from `src` with $\le k$ stops, or `-1`.
- **Constraints:**
  - $1 \le n \le 100$
  - $0 \le \text{flights.length} \le \frac{n(n - 1)}{2}$
  - $\text{flights}[i].\text{length} == 3$
  - $0 \le \text{from}_i, \text{to}_i < n$
  - $\text{from}_i \ne \text{to}_i$
  - $1 \le \text{price}_i \le 10^4$
  - There will not be any multiple flights between the same two cities.
  - $0 \le \text{src}, \text{dst} < n$
  - $\text{src} \ne \text{dst}$
  - $0 \le k \le n - 1$

---

### Key Idea & Intuition

- **Stops vs. Edges Relationship:**
  - At most $k$ intermediate stops means the path can use at most **$k + 1$ flight edges**:
    $$\text{src} \xrightarrow{\text{edge 1}} \text{stop 1} \xrightarrow{\text{edge 2}} \dots \xrightarrow{\text{edge } k} \text{stop } k \xrightarrow{\text{edge } k+1} \text{dst}$$
- **The Bellman-Ford Algorithm Invariant:**
  - Standard Bellman-Ford computes shortest paths by relaxing all edges $V - 1$ times.
  - After round $i$, `prices[v]` holds the shortest path from `src` to `v` that uses at most **$i$ edges**!
  - Therefore, running Bellman-Ford for exactly **$k + 1$ iterations** directly solves the problem.
- **The "Cascading Update" Trap (Snapshot Array):**
  - If we update the `prices` array directly in-place during an iteration, relaxing edge $A \to B$ could immediately enable relaxing $B \to C$ in the *very same round*, effectively using 2 edges in 1 step!
  - To prevent this, create a clone `tmp_prices = prices.copy()` at the start of each round. Read source costs from `prices`, and write relaxed costs to `tmp_prices`.

---

### Solution Approach (Step-by-Step)

1. Initialize `prices = [float('inf')] * n` with `prices[src] = 0`.
2. Loop $k + 1$ times:
   - Create a snapshot: `tmp_prices = prices.copy()`.
   - For each flight `[u, v, p]` in `flights`:
     - If `prices[u] == float('inf')`: continue (cannot depart from an unreachable city).
     - If `prices[u] + p < tmp_prices[v]`:
       - `tmp_prices[v] = prices[u] + p`.
   - Update: `prices = tmp_prices`.
3. Return `prices[dst]` if `prices[dst] != float('inf')` else `-1`.

---

### Visual Algorithm Walkthrough

```
n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1

Hop Limit: k = 1 stop -> at most 2 edges (k + 1 = 2 rounds)

Initial Prices: [0, inf, inf, inf]

Round 1 (Paths of <= 1 edge):
- Flight 0 -> 1: cost = 0 + 100 = 100 -> tmp[1] = 100
- Other flights cannot depart from 0 yet.
Prices after Round 1: [0, 100, inf, inf]

Round 2 (Paths of <= 2 edges):
- Flight 0 -> 1: cost 100
- Flight 1 -> 2: cost = 100 + 100 = 200 -> tmp[2] = 200
- Flight 1 -> 3: cost = 100 + 600 = 700 -> tmp[3] = 700
- Flight 2 -> 3: 2 was inf in 'prices', cannot be used in this round!
Prices after Round 2: [0, 100, 200, 700]

Finished 2 rounds (k = 1 stop limit reached).
Destination price for 3 = 700.
(Notice: path 0 -> 1 -> 2 -> 3 costs 400, but takes 2 stops! Bounded by k=1, so 700 is correct).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Bounded Stops Beats Globally Cheaper Path
- **Input:** `n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1`
- **Output:** `700`

#### Example 2: Globally Cheaper Path Allowed with Higher $K$
- **Input:** `n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 2`
- **Trace:** Round 3 allows using edge $2 \to 3$ with cost $200 + 200 = 400$.
- **Output:** `400`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float('inf')] * n
        prices[src] = 0
        
        # At most k stops means at most k + 1 edges
        for _ in range(k + 1):
            tmp_prices = prices.copy()
            for u, v, p in flights:
                if prices[u] == float('inf'):
                    continue
                if prices[u] + p < tmp_prices[v]:
                    tmp_prices[v] = prices[u] + p
            prices = tmp_prices
            
        return int(prices[dst]) if prices[dst] != float('inf') else -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int findCheapestPrice(int n, std::vector<std::vector<int>>& flights, int src, int dst, int k) {
        const int INF = 1e9;
        std::vector<int> prices(n, INF);
        prices[src] = 0;

        for (int i = 0; i <= k; ++i) {
            std::vector<int> tmpPrices = prices;
            for (const auto& flight : flights) {
                int u = flight[0], v = flight[1], p = flight[2];
                if (prices[u] == INF) continue;

                if (prices[u] + p < tmpPrices[v]) {
                    tmpPrices[v] = prices[u] + p;
                }
            }
            prices = tmpPrices;
        }

        return prices[dst] == INF ? -1 : prices[dst];
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        int INF = 1_000_000_000;
        int[] prices = new int[n];
        Arrays.fill(prices, INF);
        prices[src] = 0;

        for (int i = 0; i <= k; i++) {
            int[] tmpPrices = Arrays.copyOf(prices, n);
            for (int[] flight : flights) {
                int u = flight[0], v = flight[1], p = flight[2];
                if (prices[u] == INF) continue;

                if (prices[u] + p < tmpPrices[v]) {
                    tmpPrices[v] = prices[u] + p;
                }
            }
            prices = tmpPrices;
        }

        return prices[dst] == INF ? -1 : prices[dst];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}((K + 1) \times E)$ — The outer loop runs $K + 1$ times. The inner loop iterates through all $E$ flight edges. For $K \le 100$ and $E \le 5000$, total operations are $\le 100 \times 5000 = 5 \times 10^5$, running in $< 10$ ms.
- **Space Complexity:** $\mathcal{O}(V)$ — Only two arrays of size $n$ (`prices` and `tmp_prices`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Why Dijkstra can fail or require modifications:** Standard Dijkstra greedily explores the globally lowest cost node. However, a cheaper route that consumes too many stops might be chosen over a slightly more expensive route with fewer stops! Standard Dijkstra must be modified with state `(cost, node, stops)` to handle this, whereas Bellman-Ford handles the stop constraint naturally and flawlessly.
2. **The Snapshot Array is Non-Negotiable:** Updating `prices[v]` directly without `tmp_prices` will propagate updates across multiple edges in a single iteration, violating the $K$ stop limit.