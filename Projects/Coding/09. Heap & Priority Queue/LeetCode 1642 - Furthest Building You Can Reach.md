---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 1642: Furthest Building You Can Reach"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 1642: Furthest Building You Can Reach

Below is a complete, structured explanation of **LeetCode 1642 – Furthest Building You Can Reach**, aligned with how this problem is typically reasoned about in interviews and competitive programming.

---

## 1. Problem Statement

You are given:

* An integer array `heights`, where `heights[i]` is the height of the `i-th` building.
* An integer `bricks`, representing the number of bricks you have.
* An integer `ladders`, representing the number of ladders you have.

You start at building `0` and want to move to the right.  
To move from building `i` to `i + 1`:

* If `heights[i + 1] <= heights[i]`, no resources are needed.
* If `heights[i + 1] > heights[i]`, you must either:

  * Use **bricks** equal to the height difference, or
  * Use **one ladder** (which can cover any height difference).

**Goal:**  
Return the index of the **furthest building** you can reach.

---

## 2. Key Observation

The crucial insight is:

> **Ladders should be used for the largest climbs, and bricks for smaller climbs.**

Why?

* A ladder can cover **any** height difference at a fixed cost (1 ladder).
* Bricks have a **variable cost** equal to the height difference.

Therefore, to minimize brick usage:

* Always reserve ladders for the **largest upward jumps**.
* Use bricks for the smaller ones.

However, while traversing left to right, you do not know in advance which climbs will be the largest.  
This naturally leads to a **priority queue (min-heap)** strategy.

---

## 3. Priority Queue Technique (Greedy)

### Core Idea

1. For every upward climb (`diff > 0`):

   * Assume we use a ladder and push `diff` into a **min-heap**.
2. If the number of climbs exceeds the number of ladders:

   * Convert the **smallest climb** (from the heap) to bricks.
   * Subtract that value from `bricks`.
3. If at any point `bricks < 0`, you cannot proceed further.

### Why a Min-Heap?

* The heap keeps track of all climbs where we tentatively used ladders.
* When ladders run out, we “reassign” the smallest ladder usage to bricks.
* This guarantees ladders are used for the largest climbs.

---

## 4. Algorithm Steps

1. Initialize an empty min-heap.
2. Iterate from building `0` to `n - 2`:

   * Compute `diff = heights[i+1] - heights[i]`.
   * If `diff <= 0`, continue.
   * Push `diff` into the heap.
   * If `heap size > ladders`:

     * Pop the smallest `diff` and subtract it from `bricks`.
     * If `bricks < 0`, return `i`.
3. If the loop completes, return `n - 1`.

---

## 5. Python 3 Solution (with Typing)

```python
from typing import List
import heapq

class Solution:
    def furthestBuilding(
        self, heights: List[int], bricks: int, ladders: int
    ) -> int:
        min_heap: List[int] = []

        for i in range(len(heights) - 1):
            diff = heights[i + 1] - heights[i]

            if diff > 0:
                heapq.heappush(min_heap, diff)

                # If we used more ladders than available,
                # convert the smallest climb to bricks
                if len(min_heap) > ladders:
                    bricks -= heapq.heappop(min_heap)

                    if bricks < 0:
                        return i

        return len(heights) - 1
```

---

## 6. Worked Example 1

### Input

```
heights = [4, 2, 7, 6, 9, 14, 12]
bricks = 5
ladders = 1
```

### Step-by-Step Execution

| Move | Height Diff | Heap (ladder usage) | Bricks | Action |
| --- | --- | --- | --- | --- |
| 4 → 2 | -2 | [] | 5 | No cost |
| 2 → 7 | +5 | [5] | 5 | Ladder used |
| 7 → 6 | -1 | [5] | 5 | No cost |
| 6 → 9 | +3 | [3, 5] | 5 | Too many climbs |
|  |  | [5] | 2 | Use bricks for 3 |
| 9 → 14 | +5 | [5, 5] | 2 | Too many climbs |
|  |  | [5] | -3 | Bricks exhausted |

### Result

You fail at building index **4**.

**Output:**

```
4
```

---

## 7. Worked Example 2

### Input

```
heights = [1, 2, 3, 4]
bricks = 0
ladders = 1
```

### Execution

| Move | Diff | Heap | Bricks |
| --- | --- | --- | --- |
| 1 → 2 | 1 | [1] | 0 |
| 2 → 3 | 1 | [1, 1] | 0 → -1 (pop 1) |

Bricks go negative at index `1`.

**Output:**

```
1
```

---

## 8. Complexity Analysis

* **Time Complexity:**  
  `O(n log n)` in the worst case (heap operations per building).
* **Space Complexity:**  
  `O(ladders)` effectively, since the heap size is bounded by ladder usage.

---

## 9. Summary

* This is a **greedy + priority queue** problem.
* Always:

  * Use ladders for the **largest climbs**.
  * Use bricks for the **smallest climbs**.
* A **min-heap** allows dynamic reassignment of ladders to bricks optimally.
* This approach is both optimal and interview-friendly.

If you want, I can also provide:

* A dry-run diagram visualization
* A comparison with a “wrong greedy” approach
* Variants of this pattern in other LeetCode problems