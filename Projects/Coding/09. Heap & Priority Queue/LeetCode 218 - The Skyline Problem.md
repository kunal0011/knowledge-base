---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 218: The Skyline Problem"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 218: The Skyline Problem

Below is a **complete, structured explanation of LeetCode 218 – The Skyline Problem**, aligned with your usual learning pattern: problem statement → key observation → priority queue technique → Python 3 solution with typing → fully worked example.

---

## 1. Problem Statement (LeetCode 218 – The Skyline Problem)

You are given an array of buildings, where each building is represented as:

```
[left, right, height]
```

* `left`: x-coordinate of the left edge
* `right`: x-coordinate of the right edge
* `height`: height of the building

All buildings are rectangles grounded on the x-axis.

### Objective

Return the **skyline** formed by these buildings as a list of **key points**:

```
[x, height]
```

Rules:

* The skyline is the outer contour of the buildings when viewed from a distance.
* The skyline must be **sorted by x-coordinate**.
* Consecutive points must not have the same height.
* The skyline must end with height `0`.

---

## 2. Key Observations

1. **Skyline changes only at building boundaries**

   * Either at a building **start (left)** or **end (right)**.
2. **At any x-coordinate, the skyline height is**

   ```
   max(height of all active buildings)
   ```
3. Buildings overlap → we need to **track the current maximum height dynamically**.
4. When a taller building starts, skyline goes up.  
   When the tallest building ends, skyline goes down to the next tallest.

👉 This naturally leads to:

* **Sweep Line Algorithm**
* **Max Heap / Priority Queue**

---

## 3. Priority Queue (Max Heap) Technique

### Step 1: Convert buildings into events

For each building `[L, R, H]`:

* Start event: `(L, -H, R)` → negative height to indicate entering
* End event is implicit when `R` is passed

Why negative height?

* So that when sorting, **start events are processed before end events** at the same x.

### Step 2: Sort all events

Sort by:

1. x-coordinate
2. height (start before end)

---

### Step 3: Use a Max Heap

Heap elements:

```
(-height, right)
```

Why store `right`?

* To remove buildings that are no longer active when `x >= right`.

---

### Step 4: Track skyline changes

* Maintain `prev_max_height`
* At each event:

  * Remove expired buildings
  * Add new buildings
  * Compare current max height with previous
  * If changed → add a skyline point

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List
import heapq

class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        events = []

        # Step 1: Create events
        for left, right, height in buildings:
            events.append((left, -height, right))  # building starts

        # Step 2: Sort events
        events.sort()

        result: List[List[int]] = []

        # Max heap: (-height, right)
        heap = [(0, float("inf"))]  # ground level
        prev_max_height = 0

        # Step 3: Process events
        for x, neg_height, right in events:
            # Remove inactive buildings
            while heap[0][1] <= x:
                heapq.heappop(heap)

            # If it's a start event
            if neg_height != 0:
                heapq.heappush(heap, (neg_height, right))

            # Current max height
            current_height = -heap[0][0]

            # If skyline changes
            if current_height != prev_max_height:
                result.append([x, current_height])
                prev_max_height = current_height

        return result
```

---

## 5. Worked Example (Step-by-Step)

### Input

```
buildings = [
    [2, 9, 10],
    [3, 7, 15],
    [5, 12, 12],
    [15, 20, 10],
    [19, 24, 8]
]
```

---

### Step 1: Events after conversion & sorting

```
(2, -10, 9)
(3, -15, 7)
(5, -12, 12)
(15, -10, 20)
(19, -8, 24)
```

---

### Step 2: Process Events

| x | Active Buildings (heights) | Max Height | Skyline Change |
| --- | --- | --- | --- |
| 2 | [10] | 10 | [2,10] |
| 3 | [10,15] | 15 | [3,15] |
| 5 | [10,15,12] | 15 | — |
| 7 | [10,12] | 12 | [7,12] |
| 9 | [12] | 12 | — |
| 12 | [] | 0 | [12,0] |
| 15 | [10] | 10 | [15,10] |
| 19 | [10,8] | 10 | — |
| 20 | [8] | 8 | [20,8] |
| 24 | [] | 0 | [24,0] |

---

### Final Output

```
[
    [2,10],
    [3,15],
    [7,12],
    [12,0],
    [15,10],
    [20,8],
    [24,0]
]
```

---

## 6. Complexity Analysis

* **Time Complexity:**  
  `O(N log N)`

  * Sorting events: `O(N log N)`
  * Heap operations: `O(log N)` per event
* **Space Complexity:**  
  `O(N)` for events and heap

---

## 7. Key Takeaways (Interview-Ready)

* This is a **Sweep Line + Max Heap** problem.
* Skyline height = **maximum active building height at x**.
* Use **negative heights** to prioritize start events.
* Always remove expired buildings using their `right` boundary.
* Track height transitions only.

If you want, next I can:

* Draw the **sweep line diagram**
* Show a **heap state trace**
* Compare with a **segment tree approach**
* Explain **why lazy deletion is critical here**

Just tell me.