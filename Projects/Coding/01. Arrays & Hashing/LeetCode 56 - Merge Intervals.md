---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 56: Merge Intervals"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 56: Merge Intervals

---

### Problem Statement

Given an array of `intervals` where `intervals[i] = [start_i, end_i]`, merge all overlapping intervals, and return an array of the non-overlapping intervals.

---

### Key Observation

* Sort intervals by their start times: `intervals.sort(key=lambda x: x[0])`.
* Two intervals overlap if `curr_start <= prev_end`.
* If overlapping, merge by setting `prev_end = max(prev_end, curr_end)`. Otherwise, append new interval.

---

### Core Technique: Interval Sorting and Greedy Merging

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        merged = []
        
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
                
        return merged
```

---

### Worked-Out Example

```python
intervals = [[1,3],[2,6],[8,10],[15,18]]
[1,3]: append -> [[1,3]]
[2,6]: 2 <= 3 (overlap!) -> merge: max(3, 6)=6 -> [[1,6]]
[8,10]: 8 > 6 (no overlap) -> append -> [[1,6],[8,10]]
[15,18]: 15 > 10 (no overlap) -> append -> [[1,6],[8,10],[15,18]]
Result = [[1,6],[8,10],[15,18]]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n log n) for sorting`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Always sort intervals by start time to turn overlap checks into a single linear scan.