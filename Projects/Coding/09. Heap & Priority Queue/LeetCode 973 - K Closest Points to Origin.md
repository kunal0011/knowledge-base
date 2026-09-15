---
date: "2026-09-15"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 973: K Closest Points to Origin"
tags:
  - leetcode
  - coding
  - heap
  - amazon
  - google
---

# LeetCode 973: K Closest Points to Origin

**Target Companies:** Amazon (All-time Top #1 Heap Problem), Google, Meta  
**Difficulty:** Medium  
**Topic:** Max-Heap of Size K / Quickselect

---

### Problem Statement

Given an array of `points` where `points[i] = [x_i, y_i]` represents a point on the **X-Y** plane and an integer `k`, return the `k` closest points to the origin `(0, 0)`.

The distance between two points on the **X-Y** plane is the Euclidean distance:
$$\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$

You may return the answer in **any order**. The answer is guaranteed to be **unique** (except for the order that it is in).

---

### Input & Output Formats & Constraints

- **Input:** `points: List[List[int]]`, `k: int`
- **Output:** `List[List[int]]` of length `k`
- **Constraints:**
  - $1 \le k \le \text{points.length} \le 10^4$
  - $-10^4 \le x_i, y_i \le 10^4$

---

### Key Idea & Intuition

- **Metric Simplification:**
  - Since $\sqrt{d_1} < \sqrt{d_2} \iff d_1 < d_2$, we can avoid costly floating-point square root operations and compare squared Euclidean distances directly:
    $$dist = x^2 + y^2$$
- **Heap Selection Strategy:**
  - To find $K$ **smallest** elements in a stream or large list, maintain a **Max-Heap** of size $K$.
  - The root of the Max-Heap contains the *farthest* of the current $K$ closest points.
  - For each point:
    - Push `(-dist, point)` into the heap.
    - If `heap.size() > k`, pop the largest distance point.
  - Final heap contains the $K$ closest points in $O(N \log K)$ time and $O(K)$ space.

---

### Solution Approach (Step-by-Step)

1. Maintain `max_heap = []`.
2. For each point `[x, y]` in `points`:
   - Compute `dist = x*x + y*y`.
   - Push `(-dist, x, y)` onto `max_heap`.
   - If `len(max_heap) > k`:
     - `heapq.heappop(max_heap)`.
3. Extract points from `max_heap`: return `[[x, y] for (_, x, y) in max_heap]`.

---

### Visual Algorithm Walkthrough

```
Points: [[1,3], [-2,2], [5,-1]], K = 2

Distances:
[1, 3]  -> 1^2 + 3^2 = 10
[-2, 2] -> (-2)^2 + 2^2 = 8
[5, -1] -> 5^2 + (-1)^2 = 26

Max-Heap Size 2 Evolution:
Add (10, [1,3])   -> Heap: [10]
Add (8, [-2,2])   -> Heap: [10 (root), 8]
Add (26, [5,-1])  -> Push 26, Pop root 26 -> Heap remains [10, 8]

Final closest points: [[1, 3], [-2, 2]]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Case
- **Input:** `points = [[1,3],[-2,2]], k = 1`
- **Tracing Table:**
  | Point `[x, y]` | $x^2 + y^2$ | Max-Heap State (size $\le 1$) | Action / Note |
  | :--- | :--- | :--- | :--- |
  | Start | - | `[]` | Init heap |
  | `[1, 3]` | $1 + 9 = 10$ | `[(10, [1, 3])]` | Push |
  | `[-2, 2]` | $4 + 4 = 8$ | `[(8, [-2, 2])]` | Push 8, pop 10 (as $10 > 8$) |
- **Output:** `[[-2, 2]]`

#### Example 2: Tie-breaking & Mixed Coordinates
- **Input:** `points = [[3,3],[5,-1],[-2,4]], k = 2`
- **Distances:**
  - `[3, 3]`: $9 + 9 = 18$
  - `[5, -1]`: $25 + 1 = 26$
  - `[-2, 4]`: $4 + 16 = 20$
- **Tracing Table:**
  | Point `[x, y]` | $x^2 + y^2$ | Max-Heap State (size $\le 2$) | Action / Note |
  | :--- | :--- | :--- | :--- |
  | `[3, 3]` | 18 | `[18]` | Push |
  | `[5, -1]` | 26 | `[26 (root), 18]` | Push |
  | `[-2, 4]` | 20 | `[20 (root), 18]` | Push 20, pop 26 (max element) |
- **Output:** `[[3, 3], [-2, 4]]` (order does not matter)

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
import heapq

class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        max_heap = []  # (-dist, x, y)
        
        for x, y in points:
            dist = x * x + y * y
            heapq.heappush(max_heap, (-dist, x, y))
            if len(max_heap) > k:
                heapq.heappop(max_heap)
                
        return [[x, y] for (_, x, y) in max_heap]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

class Solution {
public:
    std::vector<std::vector<int>> kClosest(std::vector<std::vector<int>>& points, int k) {
        // Max-heap storing pair<dist, point>
        auto comp = [](const std::pair<int, std::vector<int>>& a, 
                       const std::pair<int, std::vector<int>>& b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<int, std::vector<int>>, 
                            std::vector<std::pair<int, std::vector<int>>>, 
                            decltype(comp)> maxHeap(comp);
                            
        for (const auto& pt : points) {
            int dist = pt[0] * pt[0] + pt[1] * pt[1];
            maxHeap.push({dist, pt});
            if (maxHeap.size() > k) {
                maxHeap.pop();
            }
        }
        
        std::vector<std::vector<int>> result;
        while (!maxHeap.empty()) {
            result.push_back(maxHeap.top().second);
            maxHeap.pop();
        }
        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.PriorityQueue;

class Solution {
    public int[][] kClosest(int[][] points, int k) {
        // Max-heap comparing squared distances
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(b[0] * b[0] + b[1] * b[1], a[0] * a[0] + a[1] * a[1])
        );

        for (int[] pt : points) {
            maxHeap.offer(pt);
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
        }

        int[][] result = new int[k][2];
        for (int i = 0; i < k; i++) {
            result[i] = maxHeap.poll();
        }
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log K)$ — Inserting into a heap of size $K$ costs $\mathcal{O}(\log K)$, performed $N$ times. (Optimal with Quickselect in $\mathcal{O}(N)$ average, but Max-Heap is optimal for streaming data when points cannot be buffered).
- **Space Complexity:** $\mathcal{O}(K)$ auxiliary space for the heap storing $K$ elements.

---

### Takeaway Pattern & Interview Traps

1. **Squared Distance Avoids Precision Errors:** Never take the square root $\sqrt{x^2 + y^2}$; calculating $x^2 + y^2$ directly prevents floating point inaccuracies and execution overhead.
2. **K Smallest vs K Largest Rule:**
   - Finding the **$K$ smallest** $\to$ maintain a **Max-Heap** of size $K$ (so the largest of the candidate set is constantly pruned).
   - Finding the **$K$ largest** $\to$ maintain a **Min-Heap** of size $K$.
3. **Integer Overflow Guard:**
   - Here coordinates are up to $10^4$, so $x^2 + y^2 \le 2 \times 10^8$, which fits safely in a standard signed 32-bit integer ($2.14 \times 10^9$). If coordinates could be $10^5$ or larger, use 64-bit integers (`long` in Java / `long long` in C++).
4. **Streaming Scalability:**
   - In an Amazon or Google system design follow-up: *"What if the points don't fit in memory or arrive as an infinite stream?"*
   - Sorting or Quickselect requires all points in memory $\mathcal{O}(N)$ space.
   - The Max-Heap approach only retains $K$ elements in memory $\mathcal{O}(K)$, making it ideal for distributed/streaming pipelines.
