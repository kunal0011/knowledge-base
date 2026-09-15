---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 215: Kth Largest Element in an Array"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - quickselect
  - divide-and-conquer
  - amazon
  - google
---

# LeetCode 215: Kth Largest Element in an Array

**Target Companies:** Meta (Top Classic), Amazon, Google, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Priority Queue (Min-Heap) / Quickselect (Linear Time)

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return the $k^{\text{th}}$ largest element in the array.

Note that it is the $k^{\text{th}}$ largest element in the sorted order, not the $k^{\text{th}}$ distinct element.

Can you solve it without sorting?

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le k \le \text{nums.length} \le 10^5$.
  - $-10^4 \le nums[i] \le 10^4$.
- **Output:**
  - `int`: The value that occupies index $n - k$ in the sorted version of `nums`.
- **Constraints:**
  - Duplicates are counted individually.
  - An answer is always guaranteed to exist.

---

### Key Idea & Intuition

Sorting the entire array takes $\mathcal{O}(N \log N)$ time. We can do much better:

#### Paradigm 1: Min-Heap of Size $k$ ($\mathcal{O}(N \log k)$ Time, $\mathcal{O}(k)$ Space)
Instead of sorting all $N$ elements, maintain only the **top $k$ largest elements** seen so far:
- Insert elements into a min-heap.
- Whenever the heap size exceeds $k$, pop the root (which is the smallest of the top $k + 1$ elements).
- After processing all $N$ elements, the heap contains exactly the $k$ largest elements in `nums`.
- The root of the min-heap (`heap[0]`) is the minimum of these top $k$, which by definition is the $k^{\text{th}}$ largest element!

#### Paradigm 2: Quickselect ($\mathcal{O}(N)$ Average Time, $\mathcal{O}(1)$ Space - Optimal)
Based on the partition procedure of Quicksort:
- We want to find the element that belongs at target index `target = n - k` in 0-indexed sorted order.
- Pick a random pivot, partition the array such that all elements $\le pivot$ are to the left, and elements $> pivot$ are to the right.
- Let the pivot's final index be $p$:
  - If $p == target$: we found our element! Return $nums[p]$.
  - If $p < target$: the target element lies in the right partition. Recurse/iterate on $[p + 1, R]$.
  - If $p > target$: the target element lies in the left partition. Recurse/iterate on $[L, p - 1]$.
- Because we only recurse into **one** half rather than both, the expected work follows a geometric series:
  $$N + \frac{N}{2} + \frac{N}{4} + \dots = 2N = \mathcal{O}(N)$$

---

### Solution Approach (Step-by-Step)

#### Algorithm 1: Min-Heap
1. Initialize `min_heap = []`.
2. For each number `x` in `nums`:
   - `heappush(min_heap, x)`.
   - If `len(min_heap) > k`:
     - `heappop(min_heap)`.
3. Return `min_heap[0]`.

#### Algorithm 2: Quickselect (In-Place Iterative)
1. Let `target = len(nums) - k`.
2. Set `left = 0, right = len(nums) - 1`.
3. While `left <= right`:
   - Pick random pivot index, swap with `right`.
   - Partition array around `nums[right]`.
   - If pivot index $p == target$, return `nums[p]`.
   - If $p < target$, `left = p + 1`.
   - If $p > target$, `right = p - 1`.

---

### Visual Algorithm Walkthrough

Let `nums = [3, 2, 1, 5, 6, 4]`, $k = 2$:

```
Min-Heap Evolution (maintaining top 2 largest):

Read 3: Heap = [3]
Read 2: Heap = [2, 3]
Read 1: Push 1 -> [1, 3, 2]. Size 3 > 2! Pop 1 -> Heap = [2, 3]
Read 5: Push 5 -> [2, 3, 5]. Size 3 > 2! Pop 2 -> Heap = [3, 5]
Read 6: Push 6 -> [3, 5, 6]. Size 3 > 2! Pop 3 -> Heap = [5, 6]
Read 4: Push 4 -> [4, 6, 5]. Size 3 > 2! Pop 4 -> Heap = [5, 6]

Finished all numbers.
Heap top = heap[0] = 5.
Output: 5 (The 2nd largest element).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Array

- **Input:** `nums = [3, 2, 1, 5, 6, 4]`, `k = 2`
- **Output:** `5`

#### Example 2: Array with Duplicates

- **Input:** `nums = [3, 2, 3, 1, 2, 4, 5, 5, 6]`, `k = 4`
- **Sorted View:** `[1, 2, 2, 3, 3, 4, 5, 5, 6]`
- **Output:** `4` (4th element from the right)

#### Example 3: $k = 1$ (Maximum Element)

- **Input:** `nums = [7, 10, 4, 3, 20, 15]`, `k = 1`
- **Output:** `20`

---

### Multi-Language Implementations

#### Python 3

##### Min-Heap Approach ($\mathcal{O}(N \log k)$ Time, $\mathcal{O}(k)$ Space)
```python
import heapq
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap: List[int] = []

        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        return min_heap[0]
```

##### Optimal Quickselect Approach ($\mathcal{O}(N)$ Average Time, $\mathcal{O}(1)$ Space)
```python
import random
from typing import List

class SolutionQuickselect:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        target = len(nums) - k

        def quickselect(l: int, r: int) -> int:
            pivot_idx = random.randint(l, r)
            pivot_val = nums[pivot_idx]
            nums[pivot_idx], nums[r] = nums[r], nums[pivot_idx]

            store_idx = l
            for i in range(l, r):
                if nums[i] < pivot_val:
                    nums[store_idx], nums[i] = nums[i], nums[store_idx]
                    store_idx += 1

            nums[store_idx], nums[r] = nums[r], nums[store_idx]

            if store_idx == target:
                return nums[store_idx]
            elif store_idx < target:
                return quickselect(store_idx + 1, r)
            else:
                return quickselect(l, store_idx - 1)

        return quickselect(0, len(nums) - 1)
```

#### C++17

```cpp
#include <vector>
#include <queue>
#include <random>
#include <algorithm>

class Solution {
public:
    // Min-Heap Approach
    int findKthLargest(std::vector<int>& nums, int k) {
        std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;

        for (int num : nums) {
            min_heap.push(num);
            if (static_cast<int>(min_heap.size()) > k) {
                min_heap.pop();
            }
        }

        return min_heap.top();
    }
};
```

#### Java

```java
import java.util.PriorityQueue;

public class Solution {
    public int findKthLargest(int[] nums, int k) {
        // Min-heap keeping top k elements
        PriorityQueue<Integer> minHeap = new PriorityQueue<>(k);

        for (int num : nums) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }

        return minHeap.peek();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Min-Heap Approach:** $\mathcal{O}(N \log k)$. Inserting $N$ elements into a heap of size $k$ takes $\mathcal{O}(\log k)$ per element.
  - **Quickselect Approach:** $\mathcal{O}(N)$ average case. Worst-case is $\mathcal{O}(N^2)$ (mitigated by randomized pivot selection).
- **Space Complexity:**
  - **Min-Heap Approach:** $\mathcal{O}(k)$ auxiliary memory for the heap.
  - **Quickselect Approach:** $\mathcal{O}(1)$ auxiliary space if implemented iteratively, or $\mathcal{O}(\log N)$ call stack space if recursive.

---

### Takeaway Pattern & Interview Traps

1. **Why Min-Heap for $K$-th Largest?**
   - Counter-intuitively, finding the $K$-th **largest** uses a **min-heap** of size $k$ so the minimum of the top $k$ is easily accessible at the root.
2. **Streaming Data Applicability:**
   - The min-heap solution easily adapts to an infinite incoming stream of numbers where the total count $N$ is unknown upfront, maintaining the answer in $\mathcal{O}(\log k)$ per stream update.