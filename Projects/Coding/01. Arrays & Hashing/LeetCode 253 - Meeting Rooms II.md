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
  - amazon
  - google
---

# LeetCode 253: Meeting Rooms II

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Chronological Event Sweep / Min-Heap Allocation

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

- **Chronological Separation:**
  - A meeting room is needed when a meeting begins.
  - A meeting room is freed when a meeting ends.
  - If we separate all `starts` and `ends` into two sorted arrays:
    - When `starts[s] < ends[e]`: A meeting starts before the earliest room is freed $\implies$ increment `rooms += 1` and `s += 1`.
    - When `starts[s] >= ends[e]`: A meeting ended and freed its room $\implies$ reuse room (`e += 1` and `s += 1`).

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
        std::vector<int> starts, ends;
        for (const auto& i : intervals) {
            starts.push_back(i[0]);
            ends.push_back(i[1]);
        }

        std::sort(starts.begin(), starts.end());
        std::sort(ends.begin(), ends.end());

        int rooms = 0, endPtr = 0;
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

- **Time Complexity:** $O(N \log N)$ for sorting start and end times.
- **Space Complexity:** $O(N)$ for the separated arrays.
