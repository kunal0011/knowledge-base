---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 703: Kth Largest Element in a Stream"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 703: Kth Largest Element in a Stream

Below is a complete, structured explanation of **LeetCode 703 – Kth Largest Element in a Stream**, aligned with interview-ready reasoning and clean implementation.

---

## LeetCode 703: Kth Largest Element in a Stream

### Problem Statement

Design a class `KthLargest` that finds the **kth largest element** in a stream of integers.

You are given:

* An integer `k`
* An initial array of integers `nums`

You need to:

* Initialize the object with `k` and `nums`
* For each call to `add(val)`, insert `val` into the stream and **return the kth largest element** in the stream so far

---

### Key Observation

We do **not** need to maintain the entire stream in sorted order.

Key insights:

* The kth largest element is the **smallest element among the top k largest elements**
* If we maintain only the **k largest elements seen so far**, the kth largest is easy to retrieve

This leads directly to using a **Min Heap (Priority Queue)** of size `k`.

---

### Priority Queue Technique Used

#### Why Min Heap?

* A **min heap of size k** stores the top k largest elements
* The **root (top)** of the min heap is the kth largest element

#### Strategy

1. Initialize a min heap
2. Insert elements from `nums`:

   * Push into heap
   * If heap size exceeds `k`, pop the smallest element
3. For every `add(val)`:

   * Push `val`
   * If heap size > `k`, pop
   * Return heap[0]

#### Invariant Maintained

At all times:

* Heap size ≤ k
* Heap contains the k largest elements seen so far
* `heap[0]` = kth largest element

---

### Time and Space Complexity

| Operation | Complexity |
| --- | --- |
| Initialization | O(n log k) |
| add(val) | O(log k) |
| Space | O(k) |

This is optimal for streaming problems.

---

### Python 3 Solution (with Typing)

```python
from typing import List
import heapq

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap: List[int] = []

        for num in nums:
            heapq.heappush(self.min_heap, num)
            if len(self.min_heap) > k:
                heapq.heappop(self.min_heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]
```

---

### Worked Example (Step-by-Step)

#### Input

```text
k = 3
nums = [4, 5, 8, 2]
```

#### Initialization

| Step | Heap | Explanation |
| --- | --- | --- |
| add 4 | [4] | size < k |
| add 5 | [4, 5] | size < k |
| add 8 | [4, 5, 8] | size == k |
| add 2 | [2, 4, 8, 5] → pop 2 → [4, 5, 8] | keep only top 3 |

**Current kth largest = 4**

---

#### Stream Add Operations

##### add(3)

```
heap = [3, 4, 8, 5] → pop 3 → [4, 5, 8]
```

**Return:** `4`

---

##### add(5)

```
heap = [4, 5, 8, 5] → pop 4 → [5, 5, 8]
```

**Return:** `5`

---

##### add(10)

```
heap = [5, 5, 8, 10] → pop 5 → [5, 10, 8]
```

**Return:** `5`

---

##### add(9)

```
heap = [5, 9, 8, 10] → pop 5 → [8, 9, 10]
```

**Return:** `8`

---

##### add(4)

```
heap = [4, 8, 10, 9] → pop 4 → [8, 9, 10]
```

**Return:** `8`

---

### Final Takeaways

* This is a **classic streaming + heap problem**
* The core trick is:

  > “Maintain only the top k elements using a min heap”
* The kth largest is always at the **root of the heap**
* This pattern generalizes to many “top k in stream” problems

If you want, I can also:

* Explain **why max heap is worse here**
* Convert this into a **generic Top-K streaming template**
* Show a **dry-run visualization** using heap trees