---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 815: Bus Routes"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 815: Bus Routes

**Target Companies:** Google (Signature Hard), Amazon

---

### Problem Statement

Given array `routes` where `routes[i]` is the bus route of the i-th bus. Return the least number of buses you must take to travel from `source` to `target` (-1 if impossible).

---

### Key Observation

* Instead of BFS from bus stop to bus stop (which has millions of edges), perform **BFS on Bus Routes directly**!
* Build a map: `stop -> list_of_buses_passing_through_it`.
* Queue stores current stops. When taking a bus, add all its unvisited stops to queue and mark the entire bus route as visited.

---

### Core Technique: Route-Level Breadth-First Search (Dual Indexing)

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict, deque

class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        if source == target:
            return 0
            
        stop_to_buses = defaultdict(list)
        for bus, route in enumerate(routes):
            for stop in route:
                stop_to_buses[stop].append(bus)
                
        queue = deque([source])
        visited_stops = {source}
        visited_buses = set()
        buses_taken = 0
        
        while queue:
            buses_taken += 1
            for _ in range(len(queue)):
                curr_stop = queue.popleft()
                for bus in stop_to_buses[curr_stop]:
                    if bus in visited_buses:
                        continue
                    visited_buses.add(bus)
                    
                    for next_stop in routes[bus]:
                        if next_stop == target:
                            return buses_taken
                        if next_stop not in visited_stops:
                            visited_stops.add(next_stop)
                            queue.append(next_stop)
                            
        return -1
```

---

### Worked-Out Example

```python
routes = [[1, 2, 7], [3, 6, 7]], source = 1, target = 6
stop 1 -> take Bus 0 -> reaches stop 7
stop 7 -> take Bus 1 -> reaches stop 6 (target reached!)
Buses taken = 2
```

---

### Complexity Analysis

* **Time Complexity:** `O(total_stops_across_all_routes)`
* **Space Complexity:** `O(total_stops_across_all_routes)`

---

### Takeaway Pattern

Model public transit graphs at route level (`visited\_buses`) rather than stop level to prevent exponential branching.