---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 378: Kth Smallest Element in a Sorted Matrix"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - binary-search
  - matrix
  - amazon
  - google
---

# LeetCode 378: Kth Smallest Element in a Sorted Matrix

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg, Apple  
**Difficulty:** Medium  
**Topic:** Priority Queue (Min-Heap) / Binary Search on Value Range

---

### Problem Statement

Given an `n x n` `matrix` where each of the rows and columns is sorted in ascending order, return *the $k^{\text{th}}$ smallest element in the matrix*.

Note that it is the $k^{\text{th}}$ smallest element **in the sorted order**, not the $k^{\text{th}}$ **distinct** element.

You must find a solution with a memory complexity better than $\mathcal{O}(n^2)$.

---

### Input & Output Formats & Constraints

- **Input:**
  - `matrix`: `List[List[int]]`, an $n \times n$ matrix where each row and column is sorted in ascending order.
  - `k`: `int`, where $1 \le k \le n^2$.
- **Output:**
  - `int`: The value that occupies index $k - 1$ in the sorted sequence of all $n^2$ matrix cells.
- **Constraints:**
  - $n == matrix.length == matrix[i].length$.
  - $1 \le n \le 300$.
  - $-10^9 \le matrix[i][j] \le 10^9$.
  - $1 \le k \le n^2$.

---

### Key Idea & Intuition

Flattening and sorting the entire matrix takes $\mathcal{O}(n^2 \log n^2)$ time and $\mathcal{O}(n^2)$ space, which is suboptimal.
Because each row and column is sorted, we can exploit this monotonic structure in two standard ways:

#### Paradigm 1: Min-Heap ($K$-Way Merge) ($\mathcal{O}(k \log n)$ Time, $\mathcal{O}(n)$ Space)
- Treat each of the $n$ rows as an independent sorted linked list.
- Push the first element of each of the first $\min(k, n)$ rows into a min-heap:
  $$(matrix[r][0], r, 0)$$
- Repeatedly extract the minimum element:
  - When $(val, r, c)$ is popped, the next candidate from that same row is $(matrix[r][c + 1], r, c + 1)$.
  - Push it into the heap if $c + 1 < n$.
- The $k^{\text{th}}$ popped element is the answer!

#### Paradigm 2: Binary Search on Value Space ($\mathcal{O}(n \log(\text{MAX} - \text{MIN}))$ Time, $\mathcal{O}(1)$ Space - Optimal)
- The answer lies in the range $[\text{low} = matrix[0][0], \text{high} = matrix[n-1][n-1]]$.
- For a guessed middle value $mid$:
  - How many numbers in the matrix are $\le mid$?
  - Since rows and columns are sorted, we can count the numbers $\le mid$ in **$\mathcal{O}(n)$ time** using a **Staircase Walk** starting at the bottom-left corner $(n-1, 0)$:
    - If $matrix[r][c] \le mid$: all elements in column $c$ above row $r$ are also $\le mid$. Add $r + 1$ to `count`, and move right to $c + 1$.
    - If $matrix[r][c] > mid$: move up to $r - 1$.
  - If `count < k`: $mid$ is too small $\implies low = mid + 1$.
  - If `count >= k`: $mid$ could be the answer $\implies high = mid$.

---

### Solution Approach (Step-by-Step)

#### Algorithm 1: Min-Heap
1. Initialize `min_heap = []`.
2. For $r$ from $0$ to $\min(k, n) - 1$:
   - `heappush(min_heap, (matrix[r][0], r, 0))`.
3. Loop $k$ times:
   - `val, r, c = heappop(min_heap)`.
   - If $c + 1 < n$:
     - `heappush(min_heap, (matrix[r][c + 1], r, c + 1))`.
4. Return `val`.

#### Algorithm 2: Binary Search on Value Space ($\mathcal{O}(1)$ Extra Space)
1. `low = matrix[0][0], high = matrix[n-1][n-1]`.
2. While `low < high`:
   - `mid = low + (high - low) // 2`.
   - Compute `count = countLessOrEqual(matrix, mid)`:
     - Start at $r = n - 1, c = 0, count = 0$.
     - While $r \ge 0$ and $c < n$:
       - If $matrix[r][c] \le mid$: $count += r + 1; c += 1$.
       - Else: $r -= 1$.
   - If `count < k`: `low = mid + 1`.
   - Else: `high = mid`.
3. Return `low`.

---

### Visual Algorithm Walkthrough

Let `matrix = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]`, $k = 8$:

```
Matrix:
  [  1,  5,  9 ]
  [ 10, 11, 13 ]
  [ 12, 13, 15 ]

Min-Heap Approach:
Initial Heap (col 0):
  [(1, r=0, c=0), (10, r=1, c=0), (12, r=2, c=0)]

Pop 1:  1  -> push (5, r=0, c=1)
Pop 2:  5  -> push (9, r=0, c=2)
Pop 3:  9  -> row 0 exhausted
Pop 4:  10 -> push (11, r=1, c=1)
Pop 5:  11 -> push (13, r=1, c=2)
Pop 6:  12 -> push (13, r=2, c=1)
Pop 7:  13 -> row 1 exhausted
Pop 8:  13 -> 8th pop!
Output: 13
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard $3 \times 3$ Matrix

- **Input:**
  ```python
  matrix = [
      [1, 5, 9],
      [10, 11, 13],
      [12, 13, 15]
  ]
  k = 8
  ```
- **Output:** `13`

#### Example 2: Single Element Matrix

- **Input:** `matrix = [[-5]]`, `k = 1`
- **Output:** `-5`

#### Example 3: Negative Elements with Duplicates

- **Input:**
  ```python
  matrix = [
      [-5, -4],
      [-5, -4]
  ]
  k = 2
  ```
- **Output:** `-5` (Sorted: `[-5, -5, -4, -4]`)

---

### Multi-Language Implementations

#### Python 3

##### Min-Heap Approach ($\mathcal{O}(k \log n)$ Time, $\mathcal{O}(n)$ Space)
```python
import heapq
from typing import List

class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        min_heap = []

        # Push first element of each row
        for r in range(min(k, n)):
            heapq.heappush(min_heap, (matrix[r][0], r, 0))

        val = 0
        for _ in range(k):
            val, r, c = heapq.heappop(min_heap)
            if c + 1 < n:
                heapq.heappush(min_heap, (matrix[r][c + 1], r, c + 1))

        return val
```

##### Optimal Binary Search Approach ($\mathcal{O}(n \log(\text{Range}))$ Time, $\mathcal{O}(1)$ Space)
```python
class SolutionBinarySearch:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        low, high = matrix[0][0], matrix[n - 1][n - 1]

        def count_less_or_equal(mid: int) -> int:
            count = 0
            r, c = n - 1, 0
            while r >= 0 and c < n:
                if matrix[r][c] <= mid:
                    count += r + 1
                    c += 1
                else:
                    r -= 1
            return count

        while low < high:
            mid = low + (high - low) // 2
            if count_less_or_equal(mid) < k:
                low = mid + 1
            else:
                high = mid

        return low
```

#### C++17

```cpp
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>

class Solution {
public:
    // Min-Heap Approach
    int kthSmallest(const std::vector<std::vector<int>>& matrix, int k) {
        int n = static_cast<int>(matrix.size());
        using Element = std::tuple<int, int, int>; // (val, row, col)
        std::priority_queue<Element, std::vector<Element>, std::greater<Element>> min_heap;

        for (int r = 0; r < std::min(k, n); ++r) {
            min_heap.emplace(matrix[r][0], r, 0);
        }

        int val = 0;
        for (int step = 0; step < k; ++step) {
            auto [v, r, c] = min_heap.top();
            min_heap.pop();
            val = v;

            if (c + 1 < n) {
                min_heap.emplace(matrix[r][c + 1], r, c + 1);
            }
        }

        return val;
    }
};
```

#### Java

```java
import java.util.PriorityQueue;

public class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        // Min-heap storing [val, row, col]
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(a[0], b[0])
        );

        for (int r = 0; r < Math.min(k, n); r++) {
            minHeap.offer(new int[]{matrix[r][0], r, 0});
        }

        int val = 0;
        for (int step = 0; step < k; step++) {
            int[] top = minHeap.poll();
            val = top[0];
            int r = top[1];
            int c = top[2];

            if (c + 1 < n) {
                minHeap.offer(new int[]{matrix[r][c + 1], r, c + 1});
            }
        }

        return val;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Min-Heap Approach:** $\mathcal{O}(k \log (\min(k, n)))$. We perform $k$ extractions from a heap of size at most $n$.
  - **Binary Search Approach:** $\mathcal{O}(n \log(\text{MAX} - \text{MIN}))$. Each iteration takes $\mathcal{O}(n)$ staircase traversal. With $32$ bits, $\log(\text{Range}) \approx 32 \implies 32 \times 300 \approx 9600$ operations!
- **Space Complexity:**
  - **Min-Heap Approach:** $\mathcal{O}(n)$ auxiliary space.
  - **Binary Search Approach:** $\mathcal{O}(1)$ auxiliary space.

---

### Takeaway Pattern & Interview Traps

1. **Trade-Off between Heap and Binary Search:**
   - If $k$ is small ($k \ll n^2$), the Min-Heap approach ($\mathcal{O}(k \log n)$) is faster.
   - If $k$ is close to $n^2$, the Binary Search on Value Range approach ($\mathcal{O}(n \log(\text{Range}))$) is significantly faster and uses $\mathcal{O}(1)$ space.
2. **Staircase Step Counting:**
   - Walking from bottom-left $(n-1, 0)$ gives instantaneous counting because all items above $(r, c)$ in the same column are $\le matrix[r][c] \le mid$. Thus, adding $r + 1$ accounts for the entire column prefix in $\mathcal{O}(1)$.