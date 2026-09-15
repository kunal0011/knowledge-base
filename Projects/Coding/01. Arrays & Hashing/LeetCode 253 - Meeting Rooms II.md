---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 253: Meeting Rooms II"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - heap
  - sorting
  - intervals
  - amazon
  - google
  - meta
---

# LeetCode 253: Meeting Rooms II

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Chronological Event Sweep / Min-Heap Room Allocation

---

### Problem Statement

Given an array of meeting time intervals `intervals` where `intervals[i] = [start_i, end_i]`, return the **minimum number of conference rooms required**.

---

### Input & Output Formats & Constraints

- **Input:** `intervals: List[List[int]]`
- **Output:** `int` (minimum conference rooms)
- **Constraints:**
  - $1 \le \text{intervals.length} \le 10^4$
  - $0 \le start_i < end_i \le 10^6$

---

### Key Idea & Intuition

The minimum number of rooms required equals the **maximum number of overlapping meetings at any single point in time**.

We can solve this using two primary paradigms:

#### 1. Chronological Two-Pointer Sweep:
- A room is needed when a meeting starts.
- A room is freed when a meeting ends.
- The identity of which specific room hosts which meeting does not matter—only the total pool of occupied rooms matters.
- Separate all `start` times and all `end` times into two independently sorted arrays:
  - `starts = sorted([i[0] for i in intervals])`
  - `ends = sorted([i[1] for i in intervals])`
- Traverse each `start` time:
  - If `start < ends[end_ptr]`: A new meeting starts before the earliest ending meeting finishes. We must allocate an additional room (`rooms += 1`).
  - Else (`start >= ends[end_ptr]`): The earliest ending meeting has finished and its room is recycled. Advance `end_ptr += 1`.
- Total rooms allocated across the sweep is our answer.

#### 2. Min-Heap (Priority Queue) Approach:
- Sort intervals by start time.
- Maintain a min-heap of meeting end times.
- For each interval, if `curr.start >= heap.top()`, pop the earliest ending meeting (recycling that room).
- Push `curr.end` onto the heap.
- The heap size at the end is the number of rooms needed.

Both approaches have $\mathcal{O}(n \log n)$ time complexity. The two-pointer sweep uses simple arrays without heap overhead.

---

### Solution Approach (Step-by-Step)

1. **Extract and Sort Boundaries:**
   - Extract all start times into `starts` and sort them ascending.
   - Extract all end times into `ends` and sort them ascending.
2. **Initialize Pointers and Counters:**
   - `rooms = 0`: Total rooms needed.
   - `end_ptr = 0`: Index of the earliest meeting that will end next.
3. **Chronological Sweep:**
   - For each `start` in `starts`:
     - If `start < ends[end_ptr]`:
       - No existing room has become free. Increment `rooms += 1`.
     - Else:
       - An existing room has freed up! Advance `end_ptr += 1` to indicate the next earliest end time.
4. **Return Result:**
   - Return `rooms`.

---

### Visual Algorithm Walkthrough

#### Example: `intervals = [[0, 30], [5, 10], [15, 20]]`

```
starts = [0, 5, 15]
ends   = [10, 20, 30]
rooms  = 0, end_ptr = 0

Start 0:
  0 < ends[0] (10) -> Need a room! rooms = 1.

Start 5:
  5 < ends[0] (10) -> Need another room! rooms = 2.

Start 15:
  15 >= ends[0] (10) -> Meeting ended at 10! Reuse that room.
  Advance end_ptr = 1 (next ending is 20).
  rooms remains 2.

All start events processed.
Result = 2 rooms.
```

---

### Solved Examples with Multiple Inputs

| Input `intervals` | Sorted `starts` | Sorted `ends` | Max Concurrent Overlap | Output |
| :--- | :--- | :--- | :--- | :--- |
| `[[0, 30], [5, 10], [15, 20]]` | `[0, 5, 15]` | `[10, 20, 30]` | 2 meetings concurrent at $t = 5$ | `2` |
| `[[7, 10], [2, 4]]` | `[2, 7]` | `[4, 10]` | No overlap ($7 \ge 4$) | `1` |
| `[[1, 5], [5, 10]]` | `[1, 5]` | `[5, 10]` | Touching at $t = 5 \implies$ room freed and reused | `1` |
| `[[1, 10], [2, 7], [3, 19], [8, 12], [10, 20], [11, 30]]` | Multiple overlapping | Multiple ending | 4 overlapping meetings | `4` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        starts = sorted([i[0] for i in intervals])
        ends = sorted([i[1] for i in intervals])
        
        rooms = 0
        end_ptr = 0
        
        for start in starts:
            if start < ends[end_ptr]:
                rooms += 1
            else:
                end_ptr += 1
                
        return rooms
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int minMeetingRooms(std::vector<std::vector<int>>& intervals) {
        std::vector<int> starts;
        std::vector<int> ends;
        starts.reserve(intervals.size());
        ends.reserve(intervals.size());

        for (const auto& interval : intervals) {
            starts.push_back(interval[0]);
            ends.push_back(interval[1]);
        }

        std::sort(starts.begin(), starts.end());
        std::sort(ends.begin(), ends.end());

        int rooms = 0;
        int endPtr = 0;

        for (int start : starts) {
            if (start < ends[endPtr]) {
                rooms++;
            } else {
                endPtr++;
            }
        }

        return rooms;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int minMeetingRooms(int[][] intervals) {
        int n = intervals.length;
        int[] starts = new int[n];
        int[] ends = new int[n];

        for (int i = 0; i < n; i++) {
            starts[i] = intervals[i][0];
            ends[i] = intervals[i][1];
        }

        Arrays.sort(starts);
        Arrays.sort(ends);

        int rooms = 0;
        int endPtr = 0;

        for (int start : starts) {
            if (start < ends[endPtr]) {
                rooms++;
            } else {
                endPtr++;
            }
        }

        return rooms;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$, where $n$ is the number of meeting intervals. Sorting `starts` and `ends` takes $\mathcal{O}(n \log n)$ time. The two-pointer sweep processes each element in $\mathcal{O}(1)$ time, taking $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the extracted `starts` and `ends` arrays.

---

### Takeaway Pattern & Interview Traps

1. **Simultaneous Start and End Times:**
   - If a meeting ends at $t = 5$ and another starts at $t = 5$, the condition `start < ends[end_ptr]` evaluates to `False` (since $5 < 5$ is false). The branch triggers room reuse (`end_ptr++`), correctly handling the requirement that back-to-back meetings do not conflict.
2. **Min-Heap vs Two-Pointer Sweep:**
   - Min-heap requires $\mathcal{O}(n \log n)$ with continuous heap rebalancing. The two-pointer sweep uses only two cache-friendly contiguous arrays, achieving better constant-factor performance.
