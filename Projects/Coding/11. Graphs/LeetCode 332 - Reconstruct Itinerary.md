---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 332: Reconstruct Itinerary"
tags:
  - leetcode
  - coding
  - graphs
  - dfs
  - eulerian-path
  - hierarchical-search
  - amazon
  - google
---

# LeetCode 332: Reconstruct Itinerary

**Target Companies:** Google (Signature Eulerian Path Hard), Amazon, Meta  
**Difficulty:** Hard  
**Topic:** Hierholzer's Algorithm / Eulerian Path in Directed Graph / Post-Order DFS

---

### Problem Statement

You are given a list of airline `tickets` where `tickets[i] = [from_i, to_i]` represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from `"JFK"`, thus, the itinerary must begin with `"JFK"`. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

- For example, the itinerary `["JFK", "LGA"]` has a smaller lexical order than `["JFK", "LGB"]`.

You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.

---

### Input & Output Formats & Constraints

- **Input:** `tickets: List[List[str]]`
- **Output:** `List[str]` — Sequence of airport codes in the valid Eulerian path.
- **Constraints:**
  - $1 \le \text{tickets.length} \le 300$
  - $\text{tickets}[i].\text{length} == 2$
  - $\text{from}_i.\text{length} == 3, \text{to}_i.\text{length} == 3$
  - $\text{from}_i$ and $\text{to}_i$ consist of uppercase English letters.
  - $\text{from}_i \ne \text{to}_i$

---

### Key Idea & Intuition

- **Eulerian Path in a Directed Graph:**
  - An **Eulerian path** is a trail in a finite graph that visits every **edge** exactly once.
  - Airports are vertices; tickets are directed edges. Multiple tickets between the same two airports make this a directed multigraph.
- **Why Naive Greedy DFS Fails:**
  - If we greedily choose the lexicographically smallest airport and commit to it in pre-order, we might enter a subtree or dead-end cycle too early, stranding remaining tickets.
- **Hierholzer's Algorithm (Post-Order Backtracking):**
  - Sort outgoing edges from each airport in lexicographical order.
  - Traverse edges recursively. Whenever we reach an airport with **no remaining outgoing edges**, we have reached the terminal dead-end of the current path.
  - Append this airport to our itinerary list **upon backtracking (post-order)**.
  - Once the DFS completes, the collected list represents the Eulerian path in **reverse order**.
  - Reversing this list yields the correct, complete itinerary!

---

### Solution Approach (Step-by-Step)

1. Build an adjacency list `adj` mapping departure airports to a list of destination airports.
2. Sort tickets in reverse lexicographical order so that `adj[airport].pop()` removes the smallest lexical destination in $\mathcal{O}(1)$ time.
3. Initialize empty list `itinerary = []`.
4. Define recursive function `dfs(airport)`:
   - While `adj[airport]` is not empty:
     - `next_dest = adj[airport].pop()`.
     - `dfs(next_dest)`.
   - `itinerary.append(airport)` (post-order insertion).
5. Start recursion: `dfs("JFK")`.
6. Return `itinerary[::-1]`.

---

### Visual Algorithm Walkthrough

```
Tickets: [["JFK","SFO"], ["JFK","ATL"], ["SFO","ATL"], ["ATL","JFK"], ["ATL","SFO"]]

Graph Edges:
JFK -> ATL, SFO
ATL -> JFK, SFO
SFO -> ATL

Lexicographical Sorted Adjacency (popping smallest first):
JFK: [SFO, ATL]   (pop ATL first)
ATL: [SFO, JFK]   (pop JFK first)
SFO: [ATL]        (pop ATL first)

Hierholzer DFS Walkthrough:
1. dfs("JFK") -> pops "ATL"
2.   dfs("ATL") -> pops "JFK"
3.     dfs("JFK") -> pops "SFO"
4.       dfs("SFO") -> pops "ATL"
5.         dfs("ATL") -> pops "SFO"
6.           dfs("SFO") -> no outgoing edges!
             itinerary.append("SFO")   [Dead-end reached]
5.         Backtrack to "ATL" -> no edges left -> itinerary.append("ATL")
4.       Backtrack to "SFO" -> no edges left -> itinerary.append("SFO")
3.     Backtrack to "JFK" -> no edges left -> itinerary.append("JFK")
2.   Backtrack to "ATL" -> no edges left -> itinerary.append("ATL")
1. Backtrack to "JFK" -> no edges left -> itinerary.append("JFK")

Itinerary (post-order): ["SFO", "ATL", "SFO", "JFK", "ATL", "JFK"]
Reversed: ["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"] (Uses all 5 tickets!)
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Multi-Edge Eulerian Path
- **Input:** `tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]`
- **Linear Path:** `JFK -> MUC -> LHR -> SFO -> SJC`
- **Output:** `["JFK", "MUC", "LHR", "SFO", "SJC"]`

#### Example 2: Premature Dead-End Requiring Hierholzer Resolution
- **Input:** `tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]`
- **Analysis:**
  - Lexicographically, "KUL" comes before "NRT".
  - If we greedily flew `JFK -> KUL`, we are stuck with tickets `[JFK, NRT]` and `[NRT, JFK]` unused!
  - Hierholzer's DFS: `JFK -> KUL` hits dead-end, so `"KUL"` is added to post-order first.
  - Remaining cycle `JFK -> NRT -> JFK` finishes.
  - Post-order: `["KUL", "JFK", "NRT", "JFK"]` $\implies$ Reversed: `["JFK", "NRT", "JFK", "KUL"]`.
- **Output:** `["JFK", "NRT", "JFK", "KUL"]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        adj = defaultdict(list)
        # Sort reverse lexicographically so popping from list end is O(1)
        for src, dst in sorted(tickets, reverse=True):
            adj[src].append(dst)
            
        itinerary = []
        
        def dfs(airport: str) -> None:
            while adj[airport]:
                next_dest = adj[airport].pop()
                dfs(next_dest)
            # Post-order addition
            itinerary.append(airport)
            
        dfs("JFK")
        return itinerary[::-1]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>

class Solution {
public:
    std::vector<std::string> findItinerary(std::vector<std::vector<std::string>>& tickets) {
        // Min-heap priority_queue maintains lexicographical order
        std::unordered_map<std::string, std::priority_queue<std::string, 
                           std::vector<std::string>, std::greater<std::string>>> adj;

        for (const auto& ticket : tickets) {
            adj[ticket[0]].push(ticket[1]);
        }

        std::vector<std::string> itinerary;
        dfs("JFK", adj, itinerary);

        std::reverse(itinerary.begin(), itinerary.end());
        return itinerary;
    }

private:
    void dfs(const std::string& airport,
             std::unordered_map<std::string, std::priority_queue<std::string, 
             std::vector<std::string>, std::greater<std::string>>>& adj,
             std::vector<std::string>& itinerary) {
        auto& pq = adj[airport];
        while (!pq.empty()) {
            std::string nextDest = pq.top();
            pq.pop();
            dfs(nextDest, adj, itinerary);
        }
        itinerary.push_back(airport);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    private Map<String, PriorityQueue<String>> adj = new HashMap<>();
    private List<String> itinerary = new LinkedList<>();

    public List<String> findItinerary(List<List<String>> tickets) {
        for (List<String> ticket : tickets) {
            adj.putIfAbsent(ticket.get(0), new PriorityQueue<>());
            adj.get(ticket.get(0)).offer(ticket.get(1));
        }

        dfs("JFK");
        return itinerary;
    }

    private void dfs(String airport) {
        PriorityQueue<String> pq = adj.get(airport);
        while (pq != null && !pq.isEmpty()) {
            String nextDest = pq.poll();
            dfs(nextDest);
        }
        // Insert at beginning of linked list (equivalent to reversing post-order)
        itinerary.add(0, airport);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(E \log E)$ where $E$ is the number of tickets. Sorting edges for each airport takes $\mathcal{O}(E \log E)$. Each edge is traversed and removed from the adjacency list exactly once in $\mathcal{O}(1)$ time. Overall time is $\mathcal{O}(E \log E)$.
- **Space Complexity:** $\mathcal{O}(V + E)$ — The adjacency list stores all $V$ airports and $E$ tickets. The recursion stack and result list hold $\mathcal{O}(E)$ elements.

---

### Takeaway Pattern & Interview Traps

1. **Pre-Order vs Post-Order Pitfall:** Appending to the itinerary on entry (pre-order) causes premature dead-end failures whenever a branch doesn't loop back. Appending on exit (post-order) and reversing guarantees that dead-ends are placed at the end of the journey.
2. **`pop()` from Back of List:** Sorting tickets in reverse order initially allows Python's `list.pop()` to run in $\mathcal{O}(1)$, avoiding $\mathcal{O}(N)$ deletions from the front of lists.
3. **Multigraph Support:** Tickets can have duplicate `[from, to]` pairs. Priority queues or lists naturally support repeated edges without special deduplication.