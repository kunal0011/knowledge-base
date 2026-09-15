---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 435: Non-overlapping Intervals"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 435: Non-overlapping Intervals

**LeetCode 435 – Non-overlapping Intervals**, structured exactly as requested.

---

## LeetCode 435: Non-overlapping Intervals

### Problem Statement

You are given an array of intervals, where  
`intervals[i] = [start_i, end_i]`.

An interval `[a, b]` **overlaps** with `[c, d]` if  
`min(b, d) > max(a, c)`.

Return the **minimum number of intervals you need to remove** so that the remaining intervals are **non-overlapping**.

---

### Key Observation (Core Insight)

Instead of thinking **“which intervals to remove”**, think:

> **“What is the maximum number of non-overlapping intervals I can keep?”**

Once you know that:

```
intervals_to_remove = total_intervals − max_non_overlapping_intervals
```

This transforms the problem into a **classic interval scheduling problem**.

---

### Why Greedy Works Here

To **keep as many intervals as possible**, we should:

* Always pick the interval that **ends earliest**
* This leaves **maximum room** for future intervals

This is the same greedy principle used in:

* Activity Selection
* Interval Scheduling Maximum Compatibility

---

### Greedy Strategy (Critical Trick)

1. **Sort intervals by end time**
2. Track the `end` of the last selected interval
3. If the current interval starts **before** the last end → overlap → must remove it
4. Otherwise → keep it and update `end`

---

### Algorithm Steps

1. Sort intervals by `end`
2. Initialize:

   * `count_removed = 0`
   * `prev_end = end of first interval`
3. Iterate through remaining intervals:

   * If `start < prev_end` → overlap → increment removal count
   * Else → update `prev_end`

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        
        # Step 1: Sort by end time
        intervals.sort(key=lambda x: x[1])
        
        removed = 0
        prev_end = intervals[0][1]
        
        # Step 2: Process intervals greedily
        for start, end in intervals[1:]:
            if start < prev_end:
                # Overlap detected → remove current interval
                removed += 1
            else:
                # No overlap → keep interval
                prev_end = end
        
        return removed
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```
intervals = [[1,2], [2,3], [3,4], [1,3]]
```

---

### Step 1: Sort by End Time

```
Sorted intervals:
[1,2], [1,3], [2,3], [3,4]
```

---

### Step 2: Initialize

```
prev_end = 2   (from [1,2])
removed = 0
```

---

### Step 3: Iterate Through Intervals

| Current Interval | Check (`start < prev_end`) | Action | prev\_end | removed |
| --- | --- | --- | --- | --- |
| [1,3] | 1 < 2 → YES | Remove | 2 | 1 |
| [2,3] | 2 < 2 → NO | Keep | 3 | 1 |
| [3,4] | 3 < 3 → NO | Keep | 4 | 1 |

---

### Final Result

```
Minimum intervals to remove = 1
```

---

### Why Removing the Later-Ending Interval Is Optimal

When overlap occurs:

* Removing the interval with **larger end time** keeps the schedule more flexible
* That is exactly what sorting by end time guarantees

---

### Time and Space Complexity

* **Time:** `O(n log n)` (sorting)
* **Space:** `O(1)` extra space (in-place processing)

---

### Interview Takeaway

If you remember only one line:

> **Sort by end time, not start time — greedy interval problems almost always depend on minimizing future conflicts.**

If you want, I can also:

* Convert this into a **visual interval timeline**
* Compare with **DP approach (why greedy dominates here)**
* Show **edge cases that break incorrect greedy strategies**