---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 1642: Furthest Building You Can Reach"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - greedy
  - amazon
  - google
---

# LeetCode 1642: Furthest Building You Can Reach

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Priority Queue (Min-Heap) / Greedy Resource Allocation

---

### Problem Statement

You are given an integer array `heights` representing the heights of buildings, some `bricks`, and some `ladders`.

You start your journey from building `0` and move to the next building by possibly using bricks or ladders.

While moving from building `i` to building `i+1` (0-indexed):
- If the current building's height is **greater than or equal** to the next building's height, you do **not** need a ladder or bricks.
- If the current building's height is **less than** the next building's height, you can either use **one ladder** or `(heights[i+1] - heights[i])` **bricks**.

Return *the furthest building index (0-indexed) you can reach if you use the given ladders and bricks optimally*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `heights`: `List[int]`, where $1 \le \text{heights.length} \le 10^5$. $1 \le heights[i] \le 10^6$.
  - `bricks`: `int`, where $0 \le bricks \le 10^9$.
  - `ladders`: `int`, where $0 \le ladders \le \text{heights.length}$.
- **Output:**
  - `int`: Maximum 0-based building index reachable.
- **Constraints:**
  - Downward or level transitions cost zero resources.
  - Climb cost equals height difference in bricks OR 1 ladder.

---

### Key Idea & Intuition

A ladder covers an arbitrarily large climb at the fixed cost of $1$ ladder, whereas bricks scale linearly with the climb height ($diff = heights[i+1] - heights[i]$).

Therefore, the greedy principle dictates:
$$\text{Allocate ladders to the LARGEST upward climbs, and bricks to the SMALLEST upward climbs.}$$

However, in an online left-to-right traversal, we do not know future climb heights in advance. A climb of height 5 today might seem large, but later climbs might be 50 or 100.

#### Min-Heap Dynamic Ladder Reassignment:
1. Maintain a **min-heap** of climbs for which we tentatively assign ladders.
2. For each upward step $diff > 0$:
   - Tentatively use a ladder: push $diff$ into the min-heap.
   - If the number of climbs in the heap exceeds our available `ladders`:
     - We can no longer afford to use a ladder on all of these climbs.
     - We must revoke the ladder from the **smallest climb** in the heap: `smallest_diff = heappop(min_heap)`.
     - Pay for that smallest climb using bricks: `bricks -= smallest_diff`.
   - If `bricks < 0`:
     - We cannot cover this climb with our remaining bricks, nor do we have any smaller climb to convert. We cannot reach building $i + 1$.
     - We return index $i$.
3. If we finish the entire array without exhausting bricks, return $n - 1$.

---

### Solution Approach (Step-by-Step)

1. Initialize `min_heap = []` (stores climbs covered by ladders).
2. Loop $i$ from $0$ to $n - 2$:
   - `diff = heights[i + 1] - heights[i]`.
   - If `diff <= 0`: continue (free step).
   - `heappush(min_heap, diff)`.
   - If `len(min_heap) > ladders`:
     - `bricks -= heappop(min_heap)`
     - If `bricks < 0`:
       - Return `i` (furthest building reached).
3. If the loop completes successfully, return $n - 1$.

---

### Visual Algorithm Walkthrough

Let `heights = [4, 2, 7, 6, 9, 14, 12]`, `bricks = 5`, `ladders = 1`:

```
Indices:   0   1   2   3   4   5   6
Heights:   4   2   7   6   9  14  12

Building 0 -> 1: diff = 2 - 4 = -2 (downward) -> Free.
Building 1 -> 2: diff = 7 - 2 = +5
                 Tentatively use ladder: min_heap = [5].
                 len(min_heap) = 1 <= ladders (1) -> OK. Bricks = 5.

Building 2 -> 3: diff = 6 - 7 = -1 (downward) -> Free.
Building 3 -> 4: diff = 9 - 6 = +3
                 Tentatively use ladder: min_heap = [3, 5].
                 len(min_heap) = 2 > ladders (1)!
                 Evict smallest: pop 3. Pay with bricks!
                 bricks = 5 - 3 = 2 >= 0. min_heap = [5].

Building 4 -> 5: diff = 14 - 9 = +5
                 Tentatively use ladder: min_heap = [5, 5].
                 len(min_heap) = 2 > ladders (1)!
                 Evict smallest: pop 5. Pay with bricks!
                 bricks = 2 - 5 = -3 < 0!
                 Bricks exhausted! Cannot reach building 5.
                 Return index 4.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Ladders and Bricks

- **Input:** `heights = [4, 2, 7, 6, 9, 14, 12]`, `bricks = 5`, `ladders = 1`
- **Output:** `4`

#### Example 2: Abundant Ladders

- **Input:** `heights = [4, 12, 2, 7, 3, 18, 20, 3, 19]`, `bricks = 10`, `ladders = 2`
- **Output:** `7`

#### Example 3: Zero Ladders (Pure Bricks)

- **Input:** `heights = [14, 3, 19, 3]`, `bricks = 17`, `ladders = 0`
- **Tracing:**
  - $0 \to 1$: free
  - $1 \to 2$: climb 16. With 0 ladders, pop 16 $\implies bricks = 17 - 16 = 1$.
  - $2 \to 3$: free.
  - Reached end index 3.
- **Output:** `3`

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from typing import List

class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        min_heap: List[int] = []  # Tracks climbs covered by ladders
        n = len(heights)

        for i in range(n - 1):
            diff = heights[i + 1] - heights[i]
            if diff <= 0:
                continue

            heapq.heappush(min_heap, diff)

            # If we've used more ladders than available, convert the smallest climb to bricks
            if len(min_heap) > ladders:
                smallest_climb = heapq.heappop(min_heap)
                bricks -= smallest_climb
                if bricks < 0:
                    return i

        return n - 1
```

#### C++17

```cpp
#include <vector>
#include <queue>

class Solution {
public:
    int furthestBuilding(const std::vector<int>& heights, int bricks, int ladders) {
        // Min-heap storing the largest climbs assigned to ladders
        std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;
        int n = static_cast<int>(heights.size());

        for (int i = 0; i < n - 1; ++i) {
            int diff = heights[i + 1] - heights[i];
            if (diff <= 0) continue;

            min_heap.push(diff);

            if (static_cast<int>(min_heap.size()) > ladders) {
                int smallest_climb = min_heap.top();
                min_heap.pop();
                bricks -= smallest_climb;
                if (bricks < 0) {
                    return i;
                }
            }
        }

        return n - 1;
    }
};
```

#### Java

```java
import java.util.PriorityQueue;

public class Solution {
    public int furthestBuilding(int[] heights, int bricks, int ladders) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        int n = heights.length;

        for (int i = 0; i < n - 1; i++) {
            int diff = heights[i + 1] - heights[i];
            if (diff <= 0) {
                continue;
            }

            minHeap.offer(diff);

            if (minHeap.size() > ladders) {
                int smallestClimb = minHeap.poll();
                bricks -= smallestClimb;
                if (bricks < 0) {
                    return i;
                }
            }
        }

        return n - 1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log L)$
  - Where $N$ is the number of buildings and $L$ is the number of ladders ($L \le N$).
  - In each step, we insert into a min-heap of size at most $L + 1$. Each heap operation takes $\mathcal{O}(\log L)$ time.
  - Total Time: $\mathcal{O}(N \log L)$.
- **Space Complexity:** $\mathcal{O}(L)$
  - The min-heap stores at most $L + 1$ elements at any moment.

---

### Takeaway Pattern & Interview Traps

1. **Greedy with Delayed Commitment:**
   - Instead of deciding irrevocably whether a climb gets a ladder or bricks when first encountering it, we tentatively use a ladder and revoke it (swapping to bricks) only when a better candidate demands it.
2. **Space Bounded by Ladders:**
   - Notice that the heap size is strictly bounded by $ladders + 1$, NOT $N$. This makes the algorithm space complexity $\mathcal{O}(L)$ instead of $\mathcal{O}(N)$, which is optimal when $L \ll N$.