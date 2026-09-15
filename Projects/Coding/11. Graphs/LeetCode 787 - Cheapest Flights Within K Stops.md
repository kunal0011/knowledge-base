---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 787: Cheapest Flights Within K Stops"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 787: Cheapest Flights Within K Stops

---

### Problem Statement

Given `n` cities and `flights` with prices, find cheapest price from `src` to `dst` with at most `k` stops.

---

### Key Observation

* Shortest path with edge count constraint $
  ightarrow$ **Bellman-Ford Algorithm** executed for exactly `k + 1` iterations.
* Use a temporary copy of previous distance array to prevent cascaded updates within the same round.

---

### Core Technique: Level-Restricted Bellman-Ford / BFS Price Propagation

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float('inf')] * n
        prices[src] = 0
        
        for _ in range(k + 1):
            tmp_prices = prices.copy()
            for u, v, p in flights:
                if prices[u] == float('inf'):
                    continue
                if prices[u] + p < tmp_prices[v]:
                    tmp_prices[v] = prices[u] + p
            prices = tmp_prices
            
        return prices[dst] if prices[dst] != float('inf') else -1
```

---

### Worked-Out Example

```python
n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1
Iter 0 (0 stops): 0->1 costs 100
Iter 1 (1 stop): 1->3 costs 100+600=700, 1->2 costs 100+100=200
Result = 700 (2-stop route 0->1->2->3 is not allowed for k=1)
```

---

### Complexity Analysis

* **Time Complexity:** `O((k + 1) * E)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Copy the previous distance array in Bellman-Ford to strictly enforce the k-hops constraint.