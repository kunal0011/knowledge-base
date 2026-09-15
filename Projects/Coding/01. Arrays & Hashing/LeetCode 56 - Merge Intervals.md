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
  - amazon
  - google
---

# LeetCode 56: Merge Intervals

**Target Companies:** Amazon (Top #1 Interval Problem), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Interval Sorting + Greedy Merging

---

### Problem Statement

Given an array of `intervals` where `intervals[i] = [start_i, end_i]`, merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

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

- **Why Sort by Start Time?**
  - If intervals are sorted by $start_i$, any interval that can possibly overlap with interval $i$ must appear immediately adjacent to it in the sorted sequence!
- **Greedy Merge Rule:**
  - Maintain `merged = [intervals[0]]`.
  - For each subsequent interval `curr = [start, end]`:
    - Let `last = merged[-1]`.
    - If `curr.start <= last.end`: They overlap! Extend `last.end = max(last.end, curr.end)`.
    - Else: No overlap. Append `curr` to `merged`.

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
        merged = [intervals[0]]
        
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
        if (intervals == null || intervals.length == 0) return new int[0][0];

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

- **Time Complexity:** $O(N \log N)$ for sorting the intervals + $O(N)$ linear merge sweep.
- **Space Complexity:** $O(N)$ for the merged result array.
