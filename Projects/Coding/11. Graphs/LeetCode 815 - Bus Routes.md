---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 815: Bus Routes"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - hash-table
  - amazon
  - google
---

# LeetCode 815: Bus Routes

**Target Companies:** Google (All-Time Signature Transit Graph Problem), Amazon, Uber  
**Difficulty:** Hard  
**Topic:** Route-Level Breadth-First Search (Dual Indexing) / Bipartite Stop-Route Graph

---

### Problem Statement

You are given an array `routes` representing bus routes where `routes[i]` is a bus route that the $i$-th bus repeats forever.
- For example, if `routes[0] = [1, 5, 7]`, this means that the $0$-th bus travels in the sequence `1 -> 5 -> 7 -> 1 -> 5 -> 7 -> ...` forever.

You will start at the bus stop `source` (You are not on any bus initially), and you want to go to the bus stop `target`. You can travel between bus stops by buses only.

Return the **least number of buses** you must take to travel from `source` to `target`. Return `-1` if it is not possible.

---

### Input & Output Formats & Constraints

- **Input:** `routes: List[List[int]]`, `source: int`, `target: int`
- **Output:** `int` — Minimum number of distinct buses boarded, or `-1`.
- **Constraints:**
  - $1 \le \text{routes.length} \le 500$
  - $1 \le \text{routes}[i].\text{length} \le 10^5$
  - $1 \le \sum \text{routes}[i].\text{length} \le 10^5$
  - $0 \le \text{routes}[i][j] < 10^6$
  - All values of `routes[i]` are unique.
  - $0 \le \text{source}, \text{target} < 10^6$

---

### Key Idea & Intuition

- **The Stop-to-Stop Clique Explosion:**
  - A bus with $L$ stops connects every pair of stops. Building a direct graph from stop to stop would create complete subgraphs (cliques) with $\mathcal{O}(L^2)$ edges per bus, causing Memory Limit Exceeded and Time Limit Exceeded.
  - Furthermore, the question asks to minimize the **number of buses boarded**, not the number of stops visited!
- **Route-Level BFS via Stop-to-Bus Inverted Index:**
  - Build an inverted index mapping each stop to all bus routes that pass through it:
    $$\text{stop\_to\_buses}[\text{stop}] = [\text{bus}_0, \text{bus}_1, \dots]$$
  - Treat the problem as a BFS where each level corresponds to **boarding $+1$ additional bus**:
    1. Start BFS with `queue = [source]`, `buses_taken = 0`.
    2. Pop all stops at the current distance tier.
    3. For each stop, look up all buses servicing that stop.
    4. If a bus has **not yet been boarded** (`bus not in visited_buses`):
       - Mark `visited_buses.add(bus)`.
       - Add all unvisited stops along that bus route to the queue.
       - If any stop is `target`, return `buses_taken + 1`.
  - Because each bus route is explored at most once, every stop on every route is traversed at most once $\implies$ **strict linear time** $\mathcal{O}(\sum \text{route lengths})$!

---

### Solution Approach (Step-by-Step)

1. Edge case: If `source == target`, return `0` (no bus needed).
2. Build `stop_to_buses = defaultdict(list)`:
   - For bus index `bus, route` in `enumerate(routes)`:
     - For `stop` in `route`:
       - `stop_to_buses[stop].append(bus)`.
3. If `source not in stop_to_buses` or `target not in stop_to_buses`: return `-1`.
4. Initialize:
   - `queue = deque([source])`
   - `visited_stops = {source}`
   - `visited_buses = set()`
   - `buses_taken = 0`
5. While `queue` is not empty:
   - `buses_taken += 1`
   - For `_` in range(`len(queue)`):
     - `curr_stop = queue.popleft()`
     - For `bus` in `stop_to_buses[curr_stop]`:
       - If `bus in visited_buses`: continue
       - `visited_buses.add(bus)`
       - For `next_stop` in `routes[bus]`:
         - If `next_stop == target`: return `buses_taken`.
         - If `next_stop not in visited_stops`:
           - `visited_stops.add(next_stop)`
           - `queue.append(next_stop)`
6. Return `-1`.

---

### Visual Algorithm Walkthrough

```
routes = [[1, 2, 7], [3, 6, 7]], source = 1, target = 6

Stop-to-Bus Mapping:
1 -> [Bus 0]
2 -> [Bus 0]
7 -> [Bus 0, Bus 1]  (Transfer Hub!)
3 -> [Bus 1]
6 -> [Bus 1]

BFS Level 1 (buses_taken = 1):
Queue = [1]
- Pop stop 1 -> servicing bus: Bus 0 (unvisited)
- Board Bus 0:
  - Add stops: {2, 7} to queue
  - Visited Buses = {0}
Queue for next level: [2, 7]

BFS Level 2 (buses_taken = 2):
- Pop stop 2: Bus 0 already visited.
- Pop stop 7: Servicing buses: [Bus 0 (visited), Bus 1 (unvisited)]
- Board Bus 1:
  - Check stops on Bus 1: {3, 6, 7}
  - Stop 6 matches target!
Return buses_taken = 2!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Transfer
- **Input:** `routes = [[1,2,7],[3,6,7]], source = 1, target = 6`
- **Output:** `2` (Bus 0 to stop 7, transfer to Bus 1 to stop 6).

#### Example 2: Already at Target
- **Input:** `routes = [[1, 2, 7]], source = 7, target = 7`
- **Output:** `0`

#### Example 3: Disconnected Routes
- **Input:** `routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]], source = 15, target = 12`
- **Trace:** Component with stop 15 cannot connect to stop 12.
- **Output:** `-1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
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
                
        if source not in stop_to_buses or target not in stop_to_buses:
            return -1
            
        queue = deque([source])
        visited_stops = {source}
        visited_buses = set()
        buses_taken = 0
        
        while queue:
            buses_taken += 1
            level_size = len(queue)
            
            for _ in range(level_size):
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

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

class Solution {
public:
    int numBusesToDestination(std::vector<std::vector<int>>& routes, int source, int target) {
        if (source == target) return 0;

        std::unordered_map<int, std::vector<int>> stopToBuses;
        for (int bus = 0; bus < routes.size(); ++bus) {
            for (int stop : routes[bus]) {
                stopToBuses[stop].push_back(bus);
            }
        }

        if (stopToBuses.find(source) == stopToBuses.end() || 
            stopToBuses.find(target) == stopToBuses.end()) {
            return -1;
        }

        std::queue<int> q;
        std::unordered_set<int> visitedStops;
        std::vector<bool> visitedBuses(routes.size(), false);

        q.push(source);
        visitedStops.insert(source);
        int busesTaken = 0;

        while (!q.empty()) {
            busesTaken++;
            int levelSize = q.size();

            for (int i = 0; i < levelSize; ++i) {
                int currStop = q.front();
                q.pop();

                for (int bus : stopToBuses[currStop]) {
                    if (visitedBuses[bus]) continue;
                    visitedBuses[bus] = true;

                    for (int nextStop : routes[bus]) {
                        if (nextStop == target) {
                            return busesTaken;
                        }
                        if (visitedStops.find(nextStop) == visitedStops.end()) {
                            visitedStops.insert(nextStop);
                            q.push(nextStop);
                        }
                    }
                }
            }
        }

        return -1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int numBusesToDestination(int[][] routes, int source, int target) {
        if (source == target) {
            return 0;
        }

        Map<Integer, List<Integer>> stopToBuses = new HashMap<>();
        for (int bus = 0; bus < routes.length; bus++) {
            for (int stop : routes[bus]) {
                stopToBuses.computeIfAbsent(stop, k -> new ArrayList<>()).add(bus);
            }
        }

        if (!stopToBuses.containsKey(source) || !stopToBuses.containsKey(target)) {
            return -1;
        }

        Queue<Integer> queue = new ArrayDeque<>();
        Set<Integer> visitedStops = new HashSet<>();
        boolean[] visitedBuses = new boolean[routes.length];

        queue.offer(source);
        visitedStops.add(source);
        int busesTaken = 0;

        while (!queue.isEmpty()) {
            busesTaken++;
            int levelSize = queue.size();

            for (int i = 0; i < levelSize; i++) {
                int currStop = queue.poll();

                for (int bus : stopToBuses.get(currStop)) {
                    if (visitedBuses[bus]) {
                        continue;
                    }
                    visitedBuses[bus] = true;

                    for (int nextStop : routes[bus]) {
                        if (nextStop == target) {
                            return busesTaken;
                        }
                        if (!visitedStops.contains(nextStop)) {
                            visitedStops.add(nextStop);
                            queue.offer(nextStop);
                        }
                    }
                }
            }
        }

        return -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\sum \text{routes}[i].\text{length})$ — Building `stop_to_buses` takes time linear in the total number of stops across all routes. Because each bus route is marked in `visited_buses` and processed at most once, every stop entry across all routes is inspected at most once in the BFS loop.
- **Space Complexity:** $\mathcal{O}(\sum \text{routes}[i].\text{length})$ — Storing the inverted index, the `visited_stops` set, and the BFS queue requires memory proportional to the total number of stops in all routes.

---

### Takeaway Pattern & Interview Traps

1. **`source == target` Edge Case:** If already at the destination, 0 buses are needed. Returning 0 at the start prevents the algorithm from attempting to board an unnecessary bus.
2. **Dual Visited Tracking:** You MUST track both `visited_buses` and `visited_stops`. If you only track `visited_stops`, you will re-traverse the same bus routes from multiple stops, causing TLE. Marking the whole bus as visited once boarded avoids redundant scans.