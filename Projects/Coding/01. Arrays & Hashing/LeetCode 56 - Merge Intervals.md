---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 56: Merge Intervals"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - intervals
  - sorting
  - greedy
  - amazon
  - google
  - meta
---

# LeetCode 56: Merge Intervals

**Target Companies:** Amazon (Top #1 Interval Problem), Google, Meta, Microsoft, Bloomberg, Apple  
**Difficulty:** Medium  
**Topic:** Interval Sorting + Greedy Merging

---

### Problem Statement

Given an array of `intervals` where `intervals[i] = [start_i, end_i]`, merge all overlapping intervals, and return *an array of the non-overlapping intervals that cover all the intervals in the input*.

---

### Input & Output Formats & Constraints

- **Input:** `intervals: List[List[int]]`
- **Output:** `List[List[int]]` (merged non-overlapping intervals)
- **Constraints:**
  - $1 \le \text{intervals.length} \le 10^4$
  - `intervals[i].length == 2`
  - $0 \le start_i \le end_i \le 10^4$

---

### Key Idea & Intuition

#### 1. Why Sort by Start Time?
If intervals are sorted by start time $start_i$, any interval that can possibly overlap with an earlier interval must appear immediately following it. We never need to look back multiple intervals or re-check already merged segments.

#### 2. Greedy Overlap Invariant:
Let the current interval being examined be $[curr.start, curr.end]$ and the last merged interval in our result list be $[last.start, last.end]$:
- **Case 1: Overlap Occurs ($curr.start \le last.end$)**
  - Because intervals are sorted by start time, we know $last.start \le curr.start$.
  - Since $curr.start \le last.end$, the two intervals overlap or touch at an endpoint.
  - They merge into a single interval: $[last.start, \max(last.end, curr.end)]$.
  - We update $last.end = \max(last.end, curr.end)$ in-place.
- **Case 2: No Overlap ($curr.start > last.end$)**
  - The current interval starts strictly after the previous interval has ended.
  - Since all subsequent intervals will have start times $\ge curr.start > last.end$, the previous interval is permanently closed and will never merge with anything else.
  - Append $curr$ as a new interval to the merged list.

---

### Solution Approach (Step-by-Step)

1. **Edge Case Check:**
   - If `intervals` has length $\le 1$, return `intervals` directly.
2. **Sort by Start Time:**
   - Sort the list in ascending order of `interval[0]`.
3. **Initialize Result:**
   - Create `merged = [intervals[0]]`.
4. **Linear Scan:**
   - For each interval `curr` from index 1 to $N-1$:
     - Let `last = merged[-1]`.
     - If `curr[0] <= last[1]`:
       - Update `last[1] = max(last[1], curr[1])`.
     - Else:
       - Append `curr` to `merged`.
5. **Return Result:**
   - Return `merged`.

---

### Visual Algorithm Walkthrough

#### Example: `intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]`

```
After Sorting: [[1, 3], [2, 6], [8, 10], [15, 18]]

merged = [[1, 3]]

Interval [2, 6]:
  curr.start (2) <= last.end (3) -> OVERLAP!
  last.end = max(3, 6) = 6
  merged = [[1, 6]]

Interval [8, 10]:
  curr.start (8) > last.end (6) -> NO OVERLAP!
  Append [8, 10]
  merged = [[1, 6], [8, 10]]

Interval [15, 18]:
  curr.start (15) > last.end (10) -> NO OVERLAP!
  Append [15, 18]
  merged = [[1, 6], [8, 10], [15, 18]]

Final Result: [[1, 6], [8, 10], [15, 18]]
```

---

### Solved Examples with Multiple Inputs

| Input `intervals` | Sorted Order | Merged Result | Reason / Boundary Behavior |
| :--- | :--- | :--- | :--- |
| `[[1,3],[2,6],[8,10],[15,18]]` | `[[1,3],[2,6],[8,10],[15,18]]` | `[[1,6],[8,10],[15,18]]` | Partial overlap between [1,3] and [2,6] |
| `[[1,4],[4,5]]` | `[[1,4],[4,5]]` | `[[1,5]]` | Touching endpoints $4 \le 4$ merge |
| `[[1,4],[2,3]]` | `[[1,4],[2,3]]` | `[[1,4]]` | Complete containment: $\max(4, 3) = 4$ |
| `[[1,4],[0,4]]` | `[[0,4],[1,4]]` | `[[0,4]]` | Out of order input gets properly sorted |
| `[[1,4],[0,0]]` | `[[0,0],[1,4]]` | `[[0,0],[1,4]]` | Zero-length interval $[0,0]$ precedes $[1,4]$ |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
            
        intervals.sort(key=lambda x: x[0])
        merged: List[List[int]] = [intervals[0]]
        
        for curr in intervals[1:]:
            last = merged[-1]
            if curr[0] <= last[1]:
                last[1] = max(last[1], curr[1])
            else:
                merged.append(curr)
                
        return merged
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> merge(std::vector<std::vector<int>>& intervals) {
        if (intervals.empty()) return {};

        std::sort(intervals.begin(), intervals.end());
        std::vector<std::vector<int>> merged;
        merged.push_back(intervals[0]);

        for (size_t i = 1; i < intervals.size(); ++i) {
            auto& last = merged.back();
            if (intervals[i][0] <= last[1]) {
                last[1] = std::max(last[1], intervals[i][1]);
            } else {
                merged.push_back(intervals[i]);
            }
        }
        return merged;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[0][0];
        }

        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        List<int[]> merged = new ArrayList<>();
        merged.add(intervals[0]);

        for (int i = 1; i < intervals.length; i++) {
            int[] last = merged.get(merged.size() - 1);
            if (intervals[i][0] <= last[1]) {
                last[1] = Math.max(last[1], intervals[i][1]);
            } else {
                merged.add(intervals[i]);
            }
        }

        return merged.toArray(new int[merged.size()][]);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$, where $n$ is the number of intervals. Sorting the intervals by start time takes $\mathcal{O}(n \log n)$. The subsequent single linear pass takes $\mathcal{O}(n)$ time.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the output merged intervals (and $\mathcal{O}(\log n)$ stack space for sorting).

---

### Takeaway Pattern & Interview Traps

1. **Interval Complete Containment:**
   - When an interval is entirely contained within the previous one (e.g., $[1, 10]$ and $[2, 5]$), `curr.end` may be strictly smaller than `last.end`. Using `last.end = max(last.end, curr.end)` correctly avoids shrinking the merged window.
2. **Boundary Touching (`curr.start == last.end`):**
   - Intervals $[1, 4]$ and $[4, 5]$ overlap because $4 \le 4$. Remember to use $\le$ rather than $<$.
3. **In-Place Modification Consideration:**
   - In languages like C++, you can overwrite intervals in-place using two pointers to achieve $\mathcal{O}(1)$ additional memory beyond sorting.
