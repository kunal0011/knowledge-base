---
date: "2026-08-29"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 295: Find Median from Data Stream"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 295: Find Median from Data Stream

**Target Companies:** Amazon (Top #1), Google (Top #1), Apple

---

### Problem Statement

Design a data structure that supports adding integers from a data stream and finding the median of all elements seen so far in `O(1)` time.

---

### Key Observation

* Split numbers into two halves: Left half (smaller numbers) in a **Max-Heap** (simulated by storing negatives in Python) and Right half (larger numbers) in a **Min-Heap**.
* Balance heaps such that `len(small) == len(large)` or `len(small) == len(large) + 1`.
* Ensure `max(small) <= min(large)`.
* Median is either `max(small)` (if odd) or average of heap tops (if even).

---

### Core Technique: Two-Heap Dynamic Balancing (Max-Heap + Min-Heap)

---

### Python 3 Solution (with typing)

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max-heap (invert values)
        self.large = []  # min-heap

    def addNum(self, num: int) -> None:
        # Push to small (max-heap)
        heapq.heappush(self.small, -num)
        
        # Ensure max(small) <= min(large)
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
            
        # Balance sizes: len(small) >= len(large)
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2.0
```

---

### Worked-Out Example

```python
addNum(1): small=[-1], large=[] -> median = 1.0
addNum(2): small=[-1], large=[2] -> median = (1 + 2)/2 = 1.5
addNum(3): small=[-2, -1], large=[3] -> median = 2.0
```

---

### Complexity Analysis

* **Time Complexity:** `O(log n) for addNum, O(1) for findMedian`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

The classic two-heap pattern dynamically tracks median by maintaining balanced upper and lower halves in log time.