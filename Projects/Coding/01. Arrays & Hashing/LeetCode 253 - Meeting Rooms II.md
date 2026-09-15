---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 253: Meeting Rooms II"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 253: Meeting Rooms II

**Target Companies:** Amazon, Google, Meta

---

### Problem Statement

Given an array of meeting time intervals `intervals` where `intervals[i] = [start_i, end_i]`, return the minimum number of conference rooms required.

---

### Key Observation

* Separate all start times and end times into two sorted arrays.
* A new room is needed whenever a meeting starts before the earliest ending meeting finishes (`start[s] < end[e]`).
* Alternatively, maintain a Min-Heap of meeting end times: if `start >= min_heap[0]`, reuse room by popping.

---

### Core Technique: Two Pointers on Chronological Events / Min-Heap Concurrency

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        starts = sorted([i[0] for i in intervals])
        ends = sorted([i[1] for i in intervals])
        
        s, e = 0, 0
        used_rooms = 0
        max_rooms = 0
        
        while s < len(intervals):
            if starts[s] < ends[e]:
                used_rooms += 1
                s += 1
            else:
                used_rooms -= 1
                e += 1
            max_rooms = max(max_rooms, used_rooms)
            
        return max_rooms
```

---

### Worked-Out Example

```python
intervals = [[0, 30], [5, 10], [15, 20]]
starts = [0, 5, 15], ends = [10, 20, 30]
start=0 < end=10 -> rooms = 1, s=1
start=5 < end=10 -> rooms = 2, s=2
start=15 >= end=10 -> rooms = 1, e=1
start=15 < end=20 -> rooms = 2, s=3
Max rooms needed = 2
```

---

### Complexity Analysis

* **Time Complexity:** `O(n log n) for sorting`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Deconstruct intervals into independent start and end event timelines to track peak concurrency.