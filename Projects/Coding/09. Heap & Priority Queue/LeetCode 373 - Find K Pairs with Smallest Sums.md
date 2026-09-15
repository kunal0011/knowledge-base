---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 373: Find K Pairs with Smallest Sums"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - amazon
  - google
---

# LeetCode 373: Find K Pairs with Smallest Sums

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Priority Queue (Min-Heap) / K-Way Merge / Frontier Search

---

### Problem Statement

You are given two integer arrays `nums1` and `nums2` sorted in **non-decreasing order** and an integer `k`.

Define a pair `(u, v)` which consists of one element from the first array and one element from the second array.

Return *the `k` pairs `(u_1, v_1), (u_2, v_2), ..., (u_k, v_k)` with the smallest sums*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums1`: `List[int]`, sorted in non-decreasing order ($1 \le \text{nums1.length} \le 10^5$).
  - `nums2`: `List[int]`, sorted in non-decreasing order ($1 \le \text{nums2.length} \le 10^5$).
  - `k`: `int`, where $1 \le k \le 10^4$.
  - $-10^9 \le nums1[i], nums2[i] \le 10^9$.
- **Output:**
  - `List[List[int]]`: The $k$ pairs with the smallest sums, ordered non-decreasingly by sum.
- **Constraints:**
  - Total possible pairs is $N_1 \times N_2 \le 10^{10}$, so generating all pairs will exceed memory and time limits.

---

### Key Idea & Intuition

Consider an implicit 2D matrix $M$ where $M[i][j] = nums1[i] + nums2[j]$:
- Since `nums1` and `nums2` are both sorted in non-decreasing order:
  - Each row $i$ is sorted: $M[i][0] \le M[i][1] \le M[i][2] \le \dots$
  - Each column $j$ is sorted: $M[0][j] \le M[1][j] \le M[2][j] \le \dots$
- Therefore, the smallest sum in row $i$ is always $M[i][0]$.
- Once $M[i][j]$ has been chosen, the next smallest candidate from row $i$ is guaranteed to be $M[i][j+1]$.

This reduces the problem to **merging $K$ sorted rows** using a **Min-Heap**:
1. Initialize the min-heap with the first column of the first $\min(k, N_1)$ rows:
   $$(nums1[i] + nums2[0], i, 0) \quad \text{for } 0 \le i < \min(k, N_1)$$
2. Pop the smallest element `(sum, i, j)` from the heap.
3. Add `[nums1[i], nums2[j]]` to the result.
4. If $j + 1 < N_2$, push the next candidate from the same row:
   $$(nums1[i] + nums2[j + 1], i, j + 1)$$
5. Repeat until $k$ pairs are extracted or the heap becomes empty.

---

### Solution Approach (Step-by-Step)

1. **Guard Clauses:**
   - If `nums1` is empty, `nums2` is empty, or $k == 0$, return `[]`.
2. **Min-Heap Initialization:**
   - Push `(nums1[i] + nums2[0], i, 0)` for $i \in [0, \min(k, \text{len}(nums1)) - 1]$.
   - Heap size is at most $\min(k, N_1)$.
3. **Extract $k$ Smallest Pairs:**
   - While `min_heap` is non-empty and `len(result) < k`:
     - Pop `(sum, i, j) = heappop(min_heap)`.
     - Append `[nums1[i], nums2[j]]` to `result`.
     - If $j + 1 < \text{len}(nums2)$:
       - `heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))`.
4. Return `result`.

---

### Visual Algorithm Walkthrough

Let `nums1 = [1, 7, 11]`, `nums2 = [2, 4, 6]`, $k = 3$:

```
Implicit Matrix of Sums:
        nums2[0]=2  nums2[1]=4  nums2[2]=6
nums1[0]=1:    3           5           7
nums1[1]=7:    9          11          13
nums1[2]=11:  13          15          17

Step 1: Initialize heap with column 0 of all valid rows
  Row 0: (1+2=3, i=0, j=0)
  Row 1: (7+2=9, i=1, j=0)
  Row 2: (11+2=13, i=2, j=0)
  Heap: [ (3, 0, 0), (9, 1, 0), (13, 2, 0) ]

Step 2: Pop #1
  Pop (3, 0, 0) -> Pair [1, 2]
  Advance row 0: Push (1+4=5, i=0, j=1)
  Heap: [ (5, 0, 1), (9, 1, 0), (13, 2, 0) ]
  Result: [[1, 2]]

Step 3: Pop #2
  Pop (5, 0, 1) -> Pair [1, 4]
  Advance row 0: Push (1+6=7, i=0, j=2)
  Heap: [ (7, 0, 2), (9, 1, 0), (13, 2, 0) ]
  Result: [[1, 2], [1, 4]]

Step 4: Pop #3
  Pop (7, 0, 2) -> Pair [1, 6]
  Row 0 exhausted (j=2 is end of nums2). No push.
  Result: [[1, 2], [1, 4], [1, 6]]

Extracted k=3 pairs. Terminate!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Disjoint Array

- **Input:** `nums1 = [1, 7, 11]`, `nums2 = [2, 4, 6]`, `k = 3`
- **Output:** `[[1, 2], [1, 4], [1, 6]]`

#### Example 2: Negative Numbers

- **Input:** `nums1 = [-10, -4, 0, 2]`, `nums2 = [-5, 1]`, `k = 3`
- **Tracing:**
  - `[-10, -5]` $\implies -15$
  - `[-10, 1]` $\implies -9$
  - `[-4, -5]` $\implies -9$
- **Output:** `[[-10, -5], [-10, 1], [-4, -5]]`

#### Example 3: $k > N_1 \times N_2$

- **Input:** `nums1 = [1, 2]`, `nums2 = [3]`, `k = 5`
- **Output:** `[[1, 3], [2, 3]]` (Returns all available pairs).

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from typing import List

class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if not nums1 or not nums2 or k == 0:
            return []

        min_heap = []
        result = []
        n1, n2 = len(nums1), len(nums2)

        # Initialize heap with the first column of each row (up to k rows)
        for i in range(min(k, n1)):
            heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))

        # Extract smallest pairs
        while min_heap and len(result) < k:
            total, i, j = heapq.heappop(min_heap)
            result.append([nums1[i], nums2[j]])

            # If there's a next element in the current row, push it
            if j + 1 < n2:
                heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))

        return result
```

#### C++17

```cpp
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> kSmallestPairs(const std::vector<int>& nums1,
                                                 const std::vector<int>& nums2,
                                                 int k) {
        if (nums1.empty() || nums2.empty() || k == 0) return {};

        // Min-heap storing tuple of (sum, i, j)
        using Element = std::tuple<long long, int, int>;
        std::priority_queue<Element, std::vector<Element>, std::greater<Element>> min_heap;

        int n1 = static_cast<int>(nums1.size());
        int n2 = static_cast<int>(nums2.size());

        for (int i = 0; i < std::min(k, n1); ++i) {
            min_heap.emplace(static_cast<long long>(nums1[i]) + nums2[0], i, 0);
        }

        std::vector<std::vector<int>> result;
        result.reserve(k);

        while (!min_heap.empty() && static_cast<int>(result.size()) < k) {
            auto [sum, i, j] = min_heap.top();
            min_heap.pop();

            result.push_back({nums1[i], nums2[j]});

            if (j + 1 < n2) {
                min_heap.emplace(static_cast<long long>(nums1[i]) + nums2[j + 1], i, j + 1);
            }
        }

        return result;
    }
};
```

#### Java

```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

public class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums1 == null || nums2 == null || nums1.length == 0 || nums2.length == 0 || k == 0) {
            return result;
        }

        int n1 = nums1.length;
        int n2 = nums2.length;

        // Min-heap storing [sum, i, j]
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(a[0], b[0])
        );

        for (int i = 0; i < Math.min(k, n1); i++) {
            minHeap.offer(new int[]{nums1[i] + nums2[0], i, 0});
        }

        while (!minHeap.isEmpty() && result.size() < k) {
            int[] top = minHeap.poll();
            int i = top[1];
            int j = top[2];

            result.add(Arrays.asList(nums1[i], nums2[j]));

            if (j + 1 < n2) {
                minHeap.offer(new int[]{nums1[i] + nums2[j + 1], i, j + 1});
            }
        }

        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(k \log (\min(k, N_1)))$
  - Initializing the heap takes $\mathcal{O}(\min(k, N_1))$.
  - In each of the $k$ iterations, we pop and push into a heap of size at most $\min(k, N_1)$, taking $\mathcal{O}(\log (\min(k, N_1)))$ time.
  - Total Time: $\mathcal{O}(k \log (\min(k, N_1)))$. With $k \le 10^4$, $k \log k \approx 1.3 \times 10^5$ operations (blazing fast $< 10\text{ ms}$).
- **Space Complexity:** $\mathcal{O}(\min(k, N_1))$
  - The heap stores at most $\min(k, N_1)$ nodes at any given time.

---

### Takeaway Pattern & Interview Traps

1. **Frontier Expansion without Visited Set:**
   - By advancing only along the second dimension ($j \to j + 1$) for each row $i$, every pair $(i, j)$ is visited along a unique path. This completely eliminates the need for an expensive `visited` hash set!
2. **Bounds on Rows Initialized:**
   - We only need to initialize $\min(k, N_1)$ rows. Initializing all $N_1$ rows when $N_1 = 10^5$ and $k = 1$ would waste unnecessary time and memory.