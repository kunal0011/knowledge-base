---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 218: The Skyline Problem"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - sweep-line
  - amazon
  - google
---

# LeetCode 218: The Skyline Problem

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, Uber  
**Difficulty:** Hard  
**Topic:** Priority Queue (Max-Heap) / Sweep-Line Algorithm

---

### Problem Statement

A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Given the locations and heights of all the buildings, return *the skyline formed by these buildings collectively*.

The geometric information of each building is given in the array `buildings` where `buildings[i] = [left_i, right_i, height_i]`:
- `left_i` is the x-coordinate of the left edge of the $i$-th building.
- `right_i` is the x-coordinate of the right edge of the $i$-th building.
- `height_i` is the height of the $i$-th building.

You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height 0.

The skyline should be represented as a list of "key points" `[x, y]` sorted by their x-coordinate in the form `[[x1, y1], [x2, y2], ...]`. Each key point is the left endpoint of some horizontal segment in the skyline, except the last point in the sketch, which always has a y-coordinate `0` and marks the termination of the skyline on the rightmost building.

There must be no consecutive horizontal lines of equal height in the output skyline.

---

### Input & Output Formats & Constraints

- **Input:**
  - `buildings`: `List[List[int]]`, where $1 \le \text{buildings.length} \le 10^4$.
  - $0 \le left_i < right_i \le 2^{31} - 1$.
  - $1 \le height_i \le 2^{31} - 1$.
  - `buildings` is sorted by `left_i` in non-decreasing order.
- **Output:**
  - `List[List[int]]`: A list of key points `[x, height]`.
- **Constraints:**
  - No consecutive points with the same height.
  - Sorted primarily by x-coordinate.

---

### Key Idea & Intuition

A key point on the skyline occurs **only** when the maximum active height changes at some x-coordinate:
- When a building starts, the skyline may rise.
- When a building ends, the skyline may drop.

This behavior naturally suggests a **Sweep-Line Algorithm**:
1. Imagine an imaginary vertical line sweeping across the x-axis from left to right.
2. At any position $x$, the skyline height is the **maximum height among all buildings that currently overlap $x$**:
   $$\text{current\_height}(x) = \max(\{h \mid L \le x < R\} \cup \{0\})$$
3. A key point `[x, current_height]` is emitted if and only if $\text{current\_height}(x) \ne \text{previous\_max\_height}$.

#### Event Transformation & Priority Queue:
- For each building `[L, R, H]`:
  - **Start Event:** at $x = L$, building enters with height $H$ and expiry $R$. Represented as `(L, -H, R)` (negative height so taller buildings are processed first at the same $x$).
  - **End Event:** at $x = R$, height drops. Represented as `(R, 0, 0)` (or simply handled via lazy heap eviction when current $x \ge \text{expiry}$).
- Maintain a **Max-Heap** of active buildings storing `(-height, expiry_R)`:
  - At each event $x$:
    - Evict all expired buildings from the heap: while `heap[0].expiry <= x`, pop.
    - If it is a start event ($H > 0$), push `(-H, R)` into the heap.
    - Inspect the current maximum height: `curr_max = -heap[0].height`.
    - If `curr_max != prev_max`:
      - Append `[x, curr_max]` to result.
      - `prev_max = curr_max`.

---

### Solution Approach (Step-by-Step)

1. **Construct Events:**
   - For `[l, r, h]` in `buildings`:
     - Append `(l, -h, r)` (start event).
     - Append `(r, 0, 0)` (end event to trigger height re-evaluation at $x = r$).
2. **Sort Events:**
   - Sort primarily by $x$.
   - For ties at the same $x$:
     - A start event before end event (negative height $< 0$).
     - Taller building before shorter building (e.g. $-15 < -10$).
3. **Process Events with Max-Heap:**
   - Initialize `heap = [(0, inf)]` (ground level sentinel) and `prev_max = 0`.
   - For `(x, neg_h, r)` in `events`:
     - Pop expired buildings: while `heap[0][1] <= x`: `heappop(heap)`.
     - If `neg_h < 0`: `heappush(heap, (neg_h, r))`.
     - `curr_max = -heap[0][0]`.
     - If `curr_max != prev_max`:
       - `result.append([x, curr_max])`.
       - `prev_max = curr_max`.
4. Return `result`.

---

### Visual Algorithm Walkthrough

Let `buildings = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]`:

```
Active heights across time:
At x = 2:  Start [2, 9, 10] -> Max height changes 0 -> 10. Key point [2, 10].
At x = 3:  Start [3, 7, 15] -> Max height changes 10 -> 15. Key point [3, 15].
At x = 5:  Start [5, 12, 12] -> Height 12 < 15. Max stays 15. No point.
At x = 7:  End of [3, 7, 15] -> Max height drops 15 -> 12. Key point [7, 12].
At x = 9:  End of [2, 9, 10] -> Max height is still 12. No point.
At x = 12: End of [5, 12, 12] -> All active ended! Height drops 12 -> 0. Key point [12, 0].
At x = 15: Start [15, 20, 10] -> Max height rises 0 -> 10. Key point [15, 10].
At x = 19: Start [19, 24, 8]  -> Height 8 < 10. Max stays 10. No point.
At x = 20: End of [15, 20, 10] -> Height drops 10 -> 8. Key point [20, 8].
At x = 24: End of [19, 24, 8]  -> Height drops 8 -> 0. Key point [24, 0].

Result:
[[2,10], [3,15], [7,12], [12,0], [15,10], [20,8], [24,0]]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Overlapping Buildings

- **Input:** `buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]`
- **Output:** `[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]`

#### Example 2: Adjacent Non-Overlapping Buildings

- **Input:** `buildings = [[0,2,3],[2,5,3]]`
- **Output:** `[[0,3],[5,0]]` (Height remains 3 continuously between x=0 and x=5).

#### Example 3: Completely Nested Buildings

- **Input:** `buildings = [[1,10,5],[2,5,3]]`
- **Output:** `[[1,5],[10,0]]` (Inner building is obscured completely).

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from typing import List

class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        events = []
        for l, r, h in buildings:
            events.append((l, -h, r))  # Start event
            events.append((r, 0, 0))   # End event

        # Sort primarily by x, then by height (neg_h ensures proper tie-breaking)
        events.sort()

        result = []
        # Max-heap storing (-height, right_endpoint)
        heap = [(0, float('inf'))]
        prev_max = 0

        for x, neg_h, r in events:
            # Lazily remove expired buildings
            while heap[0][1] <= x:
                heapq.heappop(heap)

            # Insert new active building
            if neg_h < 0:
                heapq.heappush(heap, (neg_h, r))

            curr_max = -heap[0][0]
            if curr_max != prev_max:
                result.append([x, curr_max])
                prev_max = curr_max

        return result
```

#### C++17

```cpp
#include <vector>
#include <queue>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> getSkyline(const std::vector<std::vector<int>>& buildings) {
        // Event format: (x, -height) for start, (x, height) for end
        std::vector<std::pair<int, int>> events;
        for (const auto& b : buildings) {
            events.emplace_back(b[0], -b[2]); // Start
            events.emplace_back(b[1], b[2]);  // End
        }

        // Sorting correctly handles all corner cases:
        // 1. Different x -> ascending x
        // 2. Same x, both start -> higher building first (-h1 < -h2)
        // 3. Same x, both end -> lower building first (h1 < h2)
        // 4. Same x, one start one end -> start first (-h < +h)
        std::sort(events.begin(), events.end());

        // Max-heap storing (height, right_boundary)
        std::priority_queue<std::pair<int, int>> max_heap;
        std::vector<std::vector<int>> result;
        int prev_max = 0;

        for (const auto& [x, h] : events) {
            if (h < 0) {
                max_heap.emplace(-h, x); // Start event
            }

            // In C++, we can also use multiset or store (height, right_x) with lazy eviction
            // Alternatively, multiset<int> heights:
        }

        // Standard Multiset implementation for O(N log N) without lazy deletion:
        std::vector<std::vector<int>> ans;
        std::multiset<int> live_heights = {0};
        prev_max = 0;

        for (const auto& [x, h] : events) {
            if (h < 0) {
                live_heights.insert(-h);
            } else {
                live_heights.erase(live_heights.find(h));
            }

            int curr_max = *live_heights.rbegin();
            if (curr_max != prev_max) {
                ans.push_back({x, curr_max});
                prev_max = curr_max;
            }
        }

        return ans;
    }
};
```

#### Java

```java
import java.util.*;

public class Solution {
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<int[]> events = new ArrayList<>();
        for (int[] b : buildings) {
            events.add(new int[]{b[0], -b[2]}); // Start: negative height
            events.add(new int[]{b[1], b[2]});  // End: positive height
        }

        // Sort events
        events.sort((a, b) -> {
            if (a[0] != b[0]) return Integer.compare(a[0], b[0]);
            return Integer.compare(a[1], b[1]);
        });

        // Max-heap tracking height frequencies using TreeMap
        TreeMap<Integer, Integer> heightCount = new TreeMap<>();
        heightCount.put(0, 1); // Ground level
        int prevMax = 0;

        List<List<Integer>> result = new ArrayList<>();

        for (int[] ev : events) {
            int x = ev[0];
            int h = ev[1];

            if (h < 0) {
                heightCount.put(-h, heightCount.getOrDefault(-h, 0) + 1);
            } else {
                int count = heightCount.get(h);
                if (count == 1) {
                    heightCount.remove(h);
                } else {
                    heightCount.put(h, count - 1);
                }
            }

            int currMax = heightCount.lastKey();
            if (currMax != prevMax) {
                result.add(Arrays.asList(x, currMax));
                prevMax = currMax;
            }
        }

        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N)$
  - Generating $2N$ events takes $\mathcal{O}(N)$ time.
  - Sorting $2N$ events takes $\mathcal{O}(N \log N)$ time.
  - Each event performs at most one heap / multiset insertion and deletion taking $\mathcal{O}(\log N)$ time.
  - Total Time: $\mathcal{O}(N \log N)$.
- **Space Complexity:** $\mathcal{O}(N)$
  - Storing the events and the active buildings in the heap/multiset requires $\mathcal{O}(N)$ auxiliary space.

---

### Takeaway Pattern & Interview Traps

1. **Tie-Breaking Order for Identical X-Coordinates:**
   - Both start: taller building must be processed first to avoid redundant points.
   - Both end: lower building must be processed first.
   - One start, one end: start must be processed before end to avoid a spurious drop to 0 between adjacent buildings.
   - Encoding start height as negative and end height as positive naturally accomplishes all three tie-breaking rules via standard ascending order sort!
2. **C++ `std::multiset::erase` Trap:**
   - In C++, `multiset.erase(val)` removes **all** copies of `val`. To remove only a single building, you must call `multiset.erase(multiset.find(val))`.