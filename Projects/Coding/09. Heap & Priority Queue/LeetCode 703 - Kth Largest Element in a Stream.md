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
  - stream
  - design
  - amazon
  - google
---

# LeetCode 703: Kth Largest Element in a Stream

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Easy  
**Topic:** Priority Queue (Min-Heap) / Streaming Data / Top-K Design

---

### Problem Statement

Design a class to find the $k^{\text{th}}$ largest element in a stream. Note that it is the $k^{\text{th}}$ largest element in the sorted order, not the $k^{\text{th}}$ distinct element.

Implement `KthLargest` class:
- `KthLargest(int k, int[] nums)`: Initializes the object with the integer `k` and the stream of integers `nums`.
- `int add(int val)`: Appends the integer `val` to the stream and returns the element representing the $k^{\text{th}}$ largest element in the stream.

---

### Input & Output Formats & Constraints

- **Input:**
  - Method calls: `["KthLargest", "add", "add", ...]` with parameters `[k, nums]` followed by integers `[val]`.
- **Output:**
  - `int`: The $k^{\text{th}}$ largest element after each `add` invocation.
- **Constraints:**
  - $1 \le k \le 10^4$.
  - $0 \le nums.length \le 10^4$.
  - $-10^4 \le nums[i], val \le 10^4$.
  - At most $10^4$ calls will be made to `add`.
  - It is guaranteed that there will be at least $k$ elements in the array when you search for the $k^{\text{th}}$ element.

---

### Key Idea & Intuition

Sorting the entire stream upon every new insertion takes $\mathcal{O}(N \log N)$ or $\mathcal{O}(N)$ with insertion sort, which would easily time out across $10^4$ streaming calls.

#### The Min-Heap of Size $k$ Invariant:
We only care about the **top $k$ largest elements** seen so far:
- If we store the $k$ largest elements in a **Min-Heap**:
  - The smallest among these top $k$ elements sits right at the root (`heap[0]`).
  - By definition, the smallest of the top $k$ elements IS the $k^{\text{th}}$ largest element overall!
- When a new value `val` arrives:
  - Push `val` into the min-heap.
  - If the heap size exceeds $k$, pop the root (the element smaller than the top $k$).
  - Return `heap[0]`.

This gives **$\mathcal{O}(\log k)$ time per insertion** and **$\mathcal{O}(k)$ space**.

---

### Solution Approach (Step-by-Step)

1. **Constructor (`__init__(k, nums)`):**
   - Store `self.k = k`.
   - Initialize `self.min_heap = []`.
   - For each number in `nums`, call `self.add(num)` (or push all and pop down to size $k$).
2. **`add(val)`:**
   - Push `val` to `self.min_heap`.
   - If `len(self.min_heap) > self.k`:
     - `heappop(self.min_heap)`.
   - Return `self.min_heap[0]`.

---

### Visual Algorithm Walkthrough

Initialize with $k = 3$, `nums = [4, 5, 8, 2]`:

```
1. Initialization:
   - Insert 4: Heap = [4]
   - Insert 5: Heap = [4, 5]
   - Insert 8: Heap = [4, 5, 8]  (size == 3)
   - Insert 2: Push 2 -> [2, 4, 8, 5]. Size 4 > 3! Pop 2 -> Heap = [4, 5, 8]
   Current 3rd largest = heap[0] = 4.

2. add(3):
   - Push 3 -> Heap has 4 elements.
   - Pop smallest (3).
   - Heap remains: [4, 5, 8].
   - Return heap[0] = 4.

3. add(5):
   - Push 5 -> Heap has 4 elements.
   - Pop smallest (4).
   - Heap becomes: [5, 5, 8].
   - Return heap[0] = 5.

4. add(10):
   - Push 10.
   - Pop smallest (5).
   - Heap becomes: [5, 8, 10].
   - Return heap[0] = 5.

5. add(9):
   - Push 9.
   - Pop smallest (5).
   - Heap becomes: [8, 9, 10].
   - Return heap[0] = 8.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Stream

- **Commands:** `KthLargest(3, [4, 5, 8, 2])`, `add(3)`, `add(5)`, `add(10)`, `add(9)`, `add(4)`
- **Returns:** `4`, `5`, `5`, `8`, `8`

#### Example 2: Initially Empty `nums` Array

- **Commands:** `KthLargest(1, [])`, `add(-3)`, `add(-2)`, `add(-4)`, `add(0)`, `add(4)`
- **Tracing:** $k = 1 \implies$ Always tracks the running maximum element!
  - `add(-3)` $\implies -3$
  - `add(-2)` $\implies -2$
  - `add(-4)` $\implies -2$
  - `add(0)` $\implies 0$
  - `add(4)` $\implies 4$

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from typing import List

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap: List[int] = []

        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]
```

#### C++17

```cpp
#include <vector>
#include <queue>

class KthLargest {
private:
    int k;
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;

public:
    KthLargest(int k, const std::vector<int>& nums) : k(k) {
        for (int num : nums) {
            add(num);
        }
    }

    int add(int val) {
        min_heap.push(val);
        if (static_cast<int>(min_heap.size()) > k) {
            min_heap.pop();
        }
        return min_heap.top();
    }
};
```

#### Java

```java
import java.util.PriorityQueue;

public class KthLargest {
    private final int k;
    private final PriorityQueue<Integer> minHeap;

    public KthLargest(int k, int[] nums) {
        this.k = k;
        this.minHeap = new PriorityQueue<>(k);

        for (int num : nums) {
            add(num);
        }
    }

    public int add(int val) {
        minHeap.offer(val);
        if (minHeap.size() > k) {
            minHeap.poll();
        }
        return minHeap.peek();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Constructor:** $\mathcal{O}(N \log k)$ where $N$ is the number of initial elements in `nums`.
  - **`add(val)`:** $\mathcal{O}(\log k)$ per stream insertion.
- **Space Complexity:** $\mathcal{O}(k)$
  - The heap retains at most $k$ elements, regardless of how many elements arrive over the lifetime of the stream.

---

### Takeaway Pattern & Interview Traps

1. **Why Max-Heap Fails for Streams:**
   - A max-heap would need to store **all** incoming elements ($N$), requiring $\mathcal{O}(N)$ memory and $\mathcal{O}(k \log N)$ to extract and restore the $k^{\text{th}}$ element. A bounded min-heap keeps only $k$ elements and answers in $\mathcal{O}(1)$ peek time.
2. **Initial Array Smaller than $k$:**
   - The initial `nums` array can contain fewer than $k$ elements (e.g. `nums = []` and $k = 1$). Checking `if len(heap) > k` guarantees the heap safely grows until size $k$ is achieved.