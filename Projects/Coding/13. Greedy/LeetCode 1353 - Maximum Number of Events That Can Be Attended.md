---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1353: Maximum Number of Events That Can Be Attended"
tags:
  - leetcode
  - coding
  - greedy
  - heap
  - sorting
  - amazon
  - google
---

# LeetCode 1353: Maximum Number of Events That Can Be Attended

**Target Companies:** Google, Amazon, Microsoft, Meta, Uber  
**Difficulty:** Medium  
**Topic:** Greedy / Priority Queue / Earliest Deadline First (EDF)  

---

### Problem Statement

You are given an array of `events` where `events[i] = [startDay_i, endDay_i]`. Every event $i$ starts at `startDay_i` and ends at `endDay_i`.

You can attend an event $i$ at any day $d$ where $d \in [\text{startDay}_i, \text{endDay}_i]$. You can only attend **one event at any time $d$** (i.e., at most one event per day).

Return the **maximum number of events** you can attend.

---

### Input & Output Formats & Constraints

- **Input:**
  - `events`: `List[List[int]]` / `vector<vector<int>>` / `int[][]` ($1 \le \text{events.length} \le 10^5$).
- **Output:**
  - `int` — the maximum number of unique events attended.
- **Constraints:**
  - $1 \le \text{events.length} \le 10^5$
  - `events[i].length == 2`
  - $1 \le \text{startDay}_i \le \text{endDay}_i \le 10^5$

---

### Key Idea & Intuition

On any given day $d$:
- Multiple events might already be active (their `startDay <= d` and `endDay >= d`).
- You can attend at most **one** of these active events today.

#### Which event should you attend on day $d$?
Suppose event $A$ ends on day 3, and event $B$ ends on day 10.
- If you choose to attend $B$ today, event $A$ might expire before tomorrow, leaving you unable to attend $A$.
- If you choose to attend $A$ today, you preserve the option to attend $B$ anytime up to day 10!
- Therefore, by the **Earliest Deadline First (EDF)** principle, the optimal choice on any day is always the event with the **earliest end day**.

#### Algorithm Structure:
1. Sort all events by `startDay` ascending.
2. Maintain a min-heap storing the `endDay` of all currently open events.
3. Simulate each day $d$:
   - **Time Skip Optimization:** If the heap is empty and there are remaining events, jump $d$ directly to `events[i].startDay` instead of iterating day by day through empty gaps.
   - Add the `endDay` of all events that start on or before day $d$ into the min-heap.
   - Evict any events from the heap whose `endDay < d` (they expired before we could attend them).
   - If the heap is not empty, pop the minimum `endDay` (attend this event today) and increment our count of attended events.
   - Advance to day $d + 1$.

---

### Solution Approach (Step-by-Step)

1. Sort `events` by `startDay`.
2. Initialize `min_heap = []`, `event_idx = 0`, `attended = 0`, and `d = 0`.
3. While `event_idx < len(events)` or `min_heap`:
   - If `min_heap` is empty:
     - Fast-forward `d = events[event_idx][0]`.
   - While `event_idx < len(events)` and `events[event_idx][0] <= d`:
     - Push `events[event_idx][1]` into `min_heap`.
     - `event_idx += 1`
   - While `min_heap` and `min_heap[0] < d`:
     - Pop expired event from `min_heap`.
   - If `min_heap`:
     - Pop the earliest deadline event: `heapq.heappop(min_heap)`.
     - Increment `attended += 1`.
     - Advance `d += 1`.
4. Return `attended`.

---

### Visual Algorithm Walkthrough

For `events = [[1,2],[2,3],[3,4],[1,2]]`:
Sorted: `[[1,2], [1,2], [2,3], [3,4]]`

```
Day d = 1:
  Events starting on day 1: [1,2], [1,2]
  min_heap: [2, 2]
  Attend event with end 2: pop 2. attended = 1
  Advance to d = 2.

Day d = 2:
  Events starting on day 2: [2,3]
  min_heap currently: [2, 3]
  Attend event with end 2: pop 2. attended = 2
  Advance to d = 3.

Day d = 3:
  Events starting on day 3: [3,4]
  min_heap currently: [3, 4]
  Attend event with end 3: pop 3. attended = 3
  Advance to d = 4.

Day d = 4:
  No new events.
  min_heap currently: [4]
  Attend event with end 4: pop 4. attended = 4
  Advance to d = 5.

Day d = 5:
  min_heap empty, all events processed.

Total events attended = 4.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `events = [[1,2],[2,3],[3,4]]`
- **Output:** `3` (Day 1 attend [1,2], Day 2 attend [2,3], Day 3 attend [3,4])

#### Example 2:
- **Input:** `events = [[1,2],[2,3],[3,4],[1,2]]`
- **Output:** `4`

#### Example 3 (Overlapping with Expiration):
- **Input:** `events = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7]]`
- **Tracing:** 7 events start on day 1. We can attend one per day on days 1, 2, 3, 4, 5, 6, 7.
- **Output:** `7`

#### Example 4 (Large Time Gap):
- **Input:** `events = [[1,4],[4,4],[2,2],[3,4],[1,1]]`, sorted start: `[[1,4],[1,1],[2,2],[3,4],[4,4]]`
- **Output:** `4`

---

### Multi-Language Implementations

#### Python 3
```python
import heapq
from typing import List

class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        # Sort events primarily by start day
        events.sort(key=lambda x: x[0])
        
        min_heap: List[int] = []  # Stores end days of currently available events
        event_idx = 0
        n = len(events)
        d = 0
        attended = 0
        
        while event_idx < n or min_heap:
            # Fast forward day if no active events in queue
            if not min_heap:
                d = events[event_idx][0]
                
            # Add all events that start today or earlier
            while event_idx < n and events[event_idx][0] <= d:
                heapq.heappush(min_heap, events[event_idx][1])
                event_idx += 1
                
            # Discard all expired events
            while min_heap and min_heap[0] < d:
                heapq.heappop(min_heap)
                
            # Attend the event that closes earliest
            if min_heap:
                heapq.heappop(min_heap)
                attended += 1
                d += 1
                
        return attended
```

#### C++17
```cpp
#include <vector>
#include <queue>
#include <algorithm>

class Solution {
public:
    int maxEvents(std::vector<std::vector<int>>& events) {
        std::sort(events.begin(), events.end());
        
        std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;
        int n = static_cast<int>(events.size());
        int event_idx = 0;
        int d = 0;
        int attended = 0;
        
        while (event_idx < n || !min_heap.empty()) {
            if (min_heap.empty()) {
                d = events[event_idx][0];
            }
            
            while (event_idx < n && events[event_idx][0] <= d) {
                min_heap.push(events[event_idx][1]);
                event_idx++;
            }
            
            while (!min_heap.empty() && min_heap.top() < d) {
                min_heap.pop();
            }
            
            if (!min_heap.empty()) {
                min_heap.pop();
                attended++;
                d++;
            }
        }
        
        return attended;
    }
};
```

#### Java 17
```java
import java.util.Arrays;
import java.util.PriorityQueue;

class Solution {
    public int maxEvents(int[][] events) {
        Arrays.sort(events, (a, b) -> Integer.compare(a[0], b[0]));
        
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        int n = events.length;
        int eventIdx = 0;
        int d = 0;
        int attended = 0;
        
        while (eventIdx < n || !minHeap.isEmpty()) {
            if (minHeap.isEmpty()) {
                d = events[eventIdx][0];
            }
            
            while (eventIdx < n && events[eventIdx][0] <= d) {
                minHeap.offer(events[eventIdx][1]);
                eventIdx++;
            }
            
            while (!minHeap.isEmpty() && minHeap.peek() < d) {
                minHeap.poll();
            }
            
            if (!minHeap.isEmpty()) {
                minHeap.poll();
                attended++;
                d++;
            }
        }
        
        return attended;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n + D \log n)$ where $D \le \max(\text{endDay}) \le 10^5$
  - Sorting the $n$ events takes $\mathcal{O}(n \log n)$.
  - Each event enters the min-heap once and is popped at most once, taking $\mathcal{O}(n \log n)$ total heap operations.
  - The fast-forward step guarantees that $d$ only increments when an event is attended or skips to the next event start, bounding the loop iterations by $\mathcal{O}(n + D)$.
  - Total time: $\mathcal{O}(n \log n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space
  - The priority queue holds at most $n$ event deadlines simultaneously.

---

### Takeaway Pattern & Interview Traps

- **Earliest Deadline First (EDF) Optimality:** Whenever you can only process one job per unit of time, picking the job with the earliest deadline always leaves maximum slack for remaining jobs.
- **The Fast-Forward Optimization:** Without `if not min_heap: d = events[event_idx][0]`, an input with events like `[1, 2]` and `[100000, 100000]` would increment $d$ day-by-day 100,000 times, causing Time Limit Exceeded! Fast-forwarding guarantees zero wasted cycles across sparse timelines.