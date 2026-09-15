---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 435: Non-overlapping Intervals"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - greedy
  - interval-scheduling
  - sorting
  - amazon
  - meta
  - google
---

# LeetCode 435: Non-overlapping Intervals

**Target Companies:** Meta, Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Interval Scheduling / Dynamic Programming (Interval LIS)  

---

### Problem Statement

Given an array of intervals `intervals` where `intervals[i] = [start_i, end_i]`, return the **minimum number of intervals you need to remove** to make the rest of the intervals non-overlapping.

**Note:** Intervals that touch at a single point (e.g., `[1, 2]` and `[2, 3]`) are considered **non-overlapping**.

---

### Input & Output Formats & Constraints

- **Input:** `intervals: List[List[int]]` — 2D array of interval coordinates.
- **Output:** `int` — Minimum removals needed to eliminate all overlaps.
- **Constraints:**
  - $1 \le \text{intervals.length} \le 10^5$
  - $\text{intervals}[i].\text{length} == 2$
  - $-5 \times 10^4 \le \text{start}_i < \text{end}_i \le 5 \times 10^4$

---

### Key Idea & Intuition

1. **Complementary Problem Formulation:**
   - Minimizing removals is strictly equivalent to **maximizing the number of non-overlapping intervals kept**:
     $$\text{min\_removals} = n - \text{max\_non\_overlapping}$$
   - This is the classic **Interval Scheduling Problem**.

2. **Greedy Choice Property (Sort by End Time):**
   - Which interval should we select first to maximize future options?
   - The interval that **finishes earliest** leaves the maximum possible remaining time for subsequent intervals to fit without overlapping.
   - If we sort intervals by their **end time** $\text{end}_i$ ascending:
     - Iterate through intervals in sorted order.
     - If the current interval starts after or at the previous kept interval's end ($\text{start} \ge \text{last\_end}$), keep it and update $\text{last\_end} = \text{end}$.
     - Otherwise, the current interval overlaps with the previous one. Because the previous one finishes earlier, it is always strictly superior to discard the current interval ($\text{removals} += 1$).

3. **Why DP is Quadratic ($O(N^2)$) and Greedy is Optimal ($O(N \log N)$):**
   - In DP, $\text{dp}[i] = 1 + \max_{j < i, \text{end}_j \le \text{start}_i} \text{dp}[j]$ takes $\mathcal{O}(N^2)$ time.
   - With $N = 10^5$, an $\mathcal{O}(N^2)$ DP will result in Time Limit Exceeded (TLE). The greedy property proven above allows an $\mathcal{O}(N \log N)$ single-pass scan after sorting.

---

### Solution Approach (Step-by-Step)

1. **Base Case:**
   - If `len(intervals) <= 1`, return `0`.
2. **Sort by End Time:**
   - Sort `intervals` by `x[1]` (the end coordinate) in ascending order.
3. **Iterate & Count Removals:**
   - Initialize `last_end = intervals[0][1]` and `removals = 0`.
   - For $i$ from $1$ to $n - 1$:
     - If `intervals[i][0] >= last_end`:
       - No overlap; update `last_end = intervals[i][1]`.
     - Else:
       - Overlap detected; increment `removals += 1`.
4. **Return:**
   - Return `removals`.

---

### Visual Algorithm Walkthrough

Given `intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]`:

**Step 1: Sort by End Time**
```
Index 0: [1, 2] (ends at 2)
Index 1: [2, 3] (ends at 3)
Index 2: [1, 3] (ends at 3)
Index 3: [3, 4] (ends at 4)
```

**Step 2: Linear Greedy Sweep**
```
Initial: last_end = 2, removals = 0 (Kept: [1, 2])

i = 1 ([2, 3]):
  start (2) >= last_end (2) -> No overlap!
  Keep [2, 3], update last_end = 3

i = 2 ([1, 3]):
  start (1) < last_end (3) -> Overlap!
  Remove [1, 3], removals = 1
  last_end remains 3

i = 3 ([3, 4]):
  start (3) >= last_end (3) -> No overlap!
  Keep [3, 4], update last_end = 4

Final Removals = 1 (Removed [1, 3])
Kept non-overlapping intervals: [1, 2], [2, 3], [3, 4] (Total 3)
```

---

### Solved Examples with Multiple Inputs

| Case | `intervals` | Sorted by End Time | Removed Intervals | Result | Explanation |
|---|---|---|---|---|---|
| **Single Overlap** | `[[1,2],[2,3],[3,4],[1,3]]` | `[[1,2],[2,3],[1,3],[3,4]]` | `[1,3]` | `1` | Removing `[1,3]` leaves 3 non-overlapping |
| **All Identical** | `[[1,2],[1,2],[1,2]]` | `[[1,2],[1,2],[1,2]]` | Two of `[1,2]` | `2` | Must remove 2 to keep only 1 |
| **Already Non-Overlapping** | `[[1,2],[2,3]]` | `[[1,2],[2,3]]` | None | `0` | Touching at boundaries is allowed |
| **Enclosed Interval** | `[[1,100],[11,22],[1,11],[2,12]]` | `[[1,11],[2,12],[11,22],[1,100]]` | `[2,12]`, `[1,100]` | `2` | Keep `[1,11]` and `[11,22]` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
            
        # Sort intervals by their end coordinates ascending
        intervals.sort(key=lambda x: x[1])
        
        removals = 0
        last_end = intervals[0][1]
        
        for i in range(1, len(intervals)):
            if intervals[i][0] >= last_end:
                # No overlap: extend compatible set
                last_end = intervals[i][1]
            else:
                # Overlap: greedily drop the one that ends later
                removals += 1
                
        return removals
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int eraseOverlapIntervals(std::vector<std::vector<int>>& intervals) {
        if (intervals.empty()) return 0;

        // Sort by end time
        std::sort(intervals.begin(), intervals.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
            return a[1] < b[1];
        });

        int removals = 0;
        int last_end = intervals[0][1];

        for (size_t i = 1; i < intervals.size(); ++i) {
            if (intervals[i][0] >= last_end) {
                last_end = intervals[i][1];
            } else {
                removals++;
            }
        }

        return removals;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;
import java.util.Comparator;

class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals == null || intervals.length == 0) return 0;

        // Sort by end time ascending
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[1]));

        int removals = 0;
        int lastEnd = intervals[0][1];

        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] >= lastEnd) {
                lastEnd = intervals[i][1];
            } else {
                removals++;
            }
        }

        return removals;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N)$  
  Sorting $N$ intervals by end time takes $\mathcal{O}(N \log N)$. The subsequent greedy traversal performs a single linear pass in $\mathcal{O}(N)$ time. Total time is $\mathcal{O}(N \log N)$, finishing within $15$ ms for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ (ignoring sorting stack)  
  Only a few scalar tracking variables are used.

---

### Takeaway Pattern & Interview Traps

1. **Sorting Key: End Time vs Start Time:**
   - In interval merging problems (e.g. LeetCode 56), we sort by **start time**.
   - In interval scheduling / removal minimization problems (e.g. LeetCode 435, 452), we sort by **end time** to maximize available room for future candidates.
2. **Touching Endpoints Rule:**
   - Check the problem statement carefully: `[1, 2]` and `[2, 3]` are explicitly allowed and do not overlap. Hence the non-overlap condition is `start >= last_end` (strict inequality `start > last_end` would wrongly classify touching as overlapping).