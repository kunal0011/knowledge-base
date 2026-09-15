---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1353: Maximum Number of Events That Can Be Attended"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1353: Maximum Number of Events That Can Be Attended

Below is a **complete, structured explanation** of **LeetCode 1353 – Maximum Number of Events That Can Be Attended**, tailored for algorithmic clarity and greedy reasoning.

---

## 1. Problem Statement

You are given an array `events`, where:

```
events[i] = [startDayi, endDayi]
```

Each event lasts **from `startDay` to `endDay` inclusive**.

Rules:

* You can attend **at most one event per day**
* If you attend an event, you must attend it on **any single day** within its range

### Objective

Return the **maximum number of events** you can attend.

---

## 2. Key Observation

This is a **scheduling problem** under constraints → classic **greedy + priority queue**.

Important insights:

1. **Attending an event earlier gives more room for future events**
2. If multiple events are available on a day, you should attend:

   * the one that **ends the earliest**

Why?

* An event with an earlier end date has **less flexibility**
* If you skip it, it may expire before you get another chance

This aligns with **Earliest Deadline First (EDF)** scheduling.

---

## 3. Greedy Strategy (Core Trick)

### High-level idea:

1. Sort events by **start day**
2. Iterate **day by day**
3. Maintain a **min-heap of end days** for all events that:

   * have started
   * have not yet expired
4. Each day:

   * Add all events starting today to the heap
   * Remove expired events
   * Attend the event with the **earliest end day**

---

## 4. Algorithm Breakdown

### Step-by-step Logic

1. **Sort events by start day**
2. Initialize:

   * `min_heap` → stores event end days
   * `day = 1`
   * `i = 0` → pointer over sorted events
3. Loop while:

   * there are unprocessed events OR
   * heap is not empty
4. On each day:

   * Add all events whose `startDay == day`
   * Remove all events whose `endDay < day`
   * If heap not empty:

     * Attend event with smallest `endDay`
     * Increment answer
5. Increment `day`

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List
import heapq

class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        # Sort events by start day
        events.sort(key=lambda x: x[0])

        min_heap: List[int] = []
        day = 1
        i = 0
        n = len(events)
        attended = 0

        # Continue while events remain or heap has active events
        while i < n or min_heap:
            # If no active events, jump to next event start day
            if not min_heap and i < n:
                day = events[i][0]

            # Add all events starting today
            while i < n and events[i][0] <= day:
                heapq.heappush(min_heap, events[i][1])
                i += 1

            # Remove expired events
            while min_heap and min_heap[0] < day:
                heapq.heappop(min_heap)

            # Attend the event that ends earliest
            if min_heap:
                heapq.heappop(min_heap)
                attended += 1
                day += 1

        return attended
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
events = [[1,2], [2,3], [3,4]]
```

### Step 1: Sort

```
[[1,2], [2,3], [3,4]]
```

---

### Day 1

* Add events starting ≤ day:

  * Add `[1,2]`
* Heap: `[2]`
* Attend event ending at day `2`
* Attended count = 1
* Move to Day 2

---

### Day 2

* Add events starting ≤ day:

  * Add `[2,3]`
* Heap: `[3]`
* Attend event ending at day `3`
* Attended count = 2
* Move to Day 3

---

### Day 3

* Add events starting ≤ day:

  * Add `[3,4]`
* Heap: `[4]`
* Attend event ending at day `4`
* Attended count = 3
* Move to Day 4

---

### End

* No events left
* Heap empty

### Final Answer

```
3
```

---

## 7. Why This Greedy Approach Works

| Principle | Explanation |
| --- | --- |
| Earliest Deadline First | Events with earlier end days have less flexibility |
| Local optimal → global optimal | Choosing shortest deadline preserves future options |
| Heap efficiency | Quickly picks best event each day |

---

## 8. Complexity Analysis

* **Sorting**: `O(n log n)`
* **Heap operations**: Each event pushed & popped once → `O(n log n)`
* **Total Time**: `O(n log n)`
* **Space**: `O(n)` for heap

---

## 9. Pattern Recognition

This problem belongs to the **Greedy + Min Heap Scheduling** pattern.

Related problems:

* LeetCode 630 – Course Schedule III
* LeetCode 502 – IPO
* Interval Scheduling with Deadlines

---

If you want, I can also:

* Prove correctness formally
* Show a **counterexample** for wrong greedy choices
* Draw a **timeline visualization**
* Compare with brute force / DP approaches

Just tell me.