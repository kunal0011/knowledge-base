---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 435: Non-overlapping Intervals"
tags:
  - leetcode
  - coding
  - greedy
  - interval-scheduling
  - sorting
  - array
  - amazon
  - google
---

# LeetCode 435: Non-overlapping Intervals

**Target Companies:** Amazon (Signature Classic), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Interval Scheduling / Sorting  

---

### Problem Statement

Given an array of intervals `intervals` where `intervals[i] = [start_i, end_i]`, return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Note that intervals which touch at a single point (such as `[1, 2]` and `[2, 3]`) are non-overlapping.

---

### Input & Output Formats & Constraints

- **Input:**
  - `intervals`: `List[List[int]]` / `vector<vector<int>>` / `int[][]` ($1 \le \text{intervals.length} \le 10^5$).
- **Output:**
  - `int` — minimum number of intervals to remove.
- **Constraints:**
  - $1 \le \text{intervals.length} \le 10^5$
  - `intervals[i].length == 2`
  - $-5 \times 10^4 \le \text{start}_i < \text{end}_i \le 5 \times 10^4$

---

### Key Idea & Intuition

The problem asks to **minimize the number of removed intervals**.
By the complementary counting principle:
$$\text{Min Removals} = N - \text{Max Non-Overlapping Intervals Kept}$$
This immediately transforms the problem into the textbook **Interval Scheduling Maximization Problem (ISMP)**.

#### The Earliest End Time Greedy Invariant:
To maximize the number of non-overlapping intervals we can retain:
- We should sort intervals by their **finish / end time** ascending.
- If we must choose between two overlapping intervals, picking the one with the **earlier end time** is always optimal:
  - An interval that finishes earlier frees up the timeline as soon as possible.
  - It leaves the maximum possible slack for future non-overlapping intervals to be selected.

#### Alternative: Sorting by Start Time
If sorted by `start_time` ascending:
- Maintain `prev_end`.
- If current interval `start < prev_end` (overlap detected!):
  - Increment removals `removals += 1`.
  - Greedily eliminate the interval with the later end: `prev_end = min(prev_end, end)`.
- Else (no overlap):
  - `prev_end = end`.

Both approaches yield the exact same optimal answer in $\mathcal{O}(N \log N)$ time.

---

### Solution Approach (Step-by-Step: Earliest End Time)

1. If `len(intervals) <= 1`, return `0`.
2. Sort `intervals` in ascending order of their end times: `intervals.sort(key=lambda x: x[1])`.
3. Initialize `kept_count = 1` and `last_end = intervals[0][1]`.
4. Iterate through `intervals` from index $1$ to $N - 1$:
   - If `interval[0] >= last_end`:
     - This interval does not overlap with our last kept interval.
     - Keep it: `kept_count += 1`, `last_end = interval[1]`.
5. Return `len(intervals) - kept_count`.

---

### Visual Algorithm Walkthrough

For `intervals = [[1,2], [2,3], [3,4], [1,3]]`:

```
1. Sorted by end time:
   [1, 2], [2, 3], [1, 3], [3, 4]

Timeline:
[1, 2] : |---|
[2, 3] :     |---|
[1, 3] : |-------|   (Conflicts with [1,2] and [2,3])
[3, 4] :         |---|

Trace:
- Interval 0: [1, 2] -> Keep! last_end = 2, kept = 1
- Interval 1: [2, 3] -> start(2) >= last_end(2) -> Keep! last_end = 3, kept = 2
- Interval 2: [1, 3] -> start(1) < last_end(3) -> Conflict! Do not keep.
- Interval 3: [3, 4] -> start(3) >= last_end(3) -> Keep! last_end = 4, kept = 3

Total kept = 3.
Removals needed = Total (4) - Kept (3) = 1 (remove [1, 3]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `intervals = [[1,2],[2,3],[3,4],[1,3]]`
- **Output:** `1`

#### Example 2 (Already Non-Overlapping):
- **Input:** `intervals = [[1,2],[2,3]]`
- **Output:** `0`

#### Example 3 (Identical Overlapping Intervals):
- **Input:** `intervals = [[1,2],[1,2],[1,2]]`
- **Tracing:** Only one `[1,2]` can be kept. Remove the other 2.
- **Output:** `2`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
            
        # Sort intervals by end time ascending
        intervals.sort(key=lambda x: x[1])
        
        kept = 1
        last_end = intervals[0][1]
        
        for i in range(1, len(intervals)):
            if intervals[i][0] >= last_end:
                kept += 1
                last_end = intervals[i][1]
                
        return len(intervals) - kept
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int eraseOverlapIntervals(std::vector<std::vector<int>>& intervals) {
        if (intervals.empty()) return 0;
        
        // Sort by end time
        std::sort(intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
            return a[1] < b[1];
        });
        
        int kept = 1;
        int last_end = intervals[0][1];
        int n = static_cast<int>(intervals.size());
        
        for (int i = 1; i < n; ++i) {
            if (intervals[i][0] >= last_end) {
                kept++;
                last_end = intervals[i][1];
            }
        }
        
        return n - kept;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        }
        
        // Sort by end coordinate ascending
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[1], b[1]));
        
        int kept = 1;
        int lastEnd = intervals[0][1];
        
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= lastEnd) {
                kept++;
                lastEnd = intervals[i][1];
            }
        }
        
        return intervals.length - kept;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$
  - Sorting $n$ intervals takes $\mathcal{O}(n \log n)$.
  - A single linear scan takes $\mathcal{O}(n)$.
  - Total time: $\mathcal{O}(n \log n)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - In-place sorting and scalar end tracking.

---

### Takeaway Pattern & Interview Traps

- **Complementary Inversion:** Min removals $\iff$ Max non-overlapping intervals. Always solve the maximum compatibility problem with earliest end time sorting.
- **Adjacent Contact Is Non-Overlapping:** `intervals[i][0] >= last_end` uses `>=` because the problem explicitly specifies touching endpoints `[1, 2]` and `[2, 3]` are non-overlapping.
- **Twin Problem:** LeetCode 452 (Minimum Number of Arrows to Burst Balloons) is structurally identical, except touching points `[1, 2]` and `[2, 3]` DO overlap for arrow hits (`>`).