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
  - design
  - amazon
  - google
---

# LeetCode 295: Find Median from Data Stream

**Target Companies:** Amazon (Top #1), Google (Top #1), Microsoft, Meta, Apple, Bloomberg  
**Difficulty:** Hard  
**Topic:** Priority Queue (Two Heaps: Max-Heap + Min-Heap) / Dynamic Median

---

### Problem Statement

The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

- For example, for `arr = [2, 3, 4]`, the median is `3`.
- For example, for `arr = [2, 3]`, the median is `(2 + 3) / 2 = 2.5`.

Implement the `MedianFinder` class:
- `MedianFinder()`: initializes the `MedianFinder` object.
- `void addNum(int num)`: adds the integer `num` from the data stream to the data structure.
- `double findMedian()`: returns the median of all elements so far. Answers within $10^{-5}$ of the actual answer will be accepted.

---

### Input & Output Formats & Constraints

- **Input:**
  - Method calls: `["MedianFinder", "addNum", "addNum", "findMedian", ...]` with corresponding integer parameters.
- **Output:**
  - `double`: The median value.
- **Constraints:**
  - $-10^5 \le num \le 10^5$.
  - There will be at least one element in the data structure before calling `findMedian`.
  - At most $5 \times 10^4$ calls will be made to `addNum` and `findMedian`.

---

### Key Idea & Intuition

Sorting the entire list of elements takes $\mathcal{O}(N \log N)$ per query. Insertion sort into a sorted array takes $\mathcal{O}(N)$ per `addNum`.
To achieve **$\mathcal{O}(\log N)$ insertion and $\mathcal{O}(1)$ median retrieval**, we divide the data stream into two equal halves:

1. **Lower Half (`small`):** A **Max-Heap** holding the smaller half of numbers. The top of this heap is the largest number in the lower half.
2. **Upper Half (`large`):** A **Min-Heap** holding the larger half of numbers. The top of this heap is the smallest number in the upper half.

#### Invariants Maintained:
1. **Order Invariant:**
   $$\max(\text{small}) \le \min(\text{large})$$
2. **Size Balance Invariant:**
   $$\text{len}(\text{small}) = \text{len}(\text{large}) \quad \text{or} \quad \text{len}(\text{small}) = \text{len}(\text{large}) + 1$$

With these two invariants:
- If the total count of numbers is **odd**: the median is simply $\max(\text{small})$.
- If the total count of numbers is **even**: the median is $\frac{\max(\text{small}) + \min(\text{large})}{2.0}$.

Both queries execute in strictly **$\mathcal{O}(1)$ time**!

---

### Solution Approach (Step-by-Step)

1. **`addNum(num)`:**
   - First, push `num` to `small` (max-heap).
   - Ensure the order invariant: if `small` and `large` are non-empty and $\max(\text{small}) > \min(\text{large})$, pop from `small` and push to `large`.
   - Balance the sizes:
     - If `len(small) > len(large) + 1`: pop from `small` and push to `large`.
     - If `len(large) > len(small)`: pop from `large` and push to `small`.
2. **`findMedian()`:**
   - If `len(small) > len(large)`: return $\max(\text{small})$.
   - Else: return $(\max(\text{small}) + \min(\text{large})) / 2.0$.

---

### Visual Algorithm Walkthrough

Insert sequence: `[1, 2, 3, 4, 5]`

```
1. addNum(1):
   small (max-heap): [1]
   large (min-heap): []
   Sizes: 1 vs 0 (valid)
   findMedian() -> 1.0

2. addNum(2):
   Push 2 to small -> small: [2, 1]
   Order check: 2 > large (empty)
   Balance check: len(small)=2 > len(large)+1 -> move 2 to large!
   small: [1], large: [2]
   findMedian() -> (1 + 2) / 2.0 = 1.5

3. addNum(3):
   Push 3 to small -> small: [3, 1]
   Order check: max(small)=3 > min(large)=2!
   Swap: move 3 to large, move 2 to small.
   small: [2, 1], large: [3]
   Sizes: 2 vs 1 (valid)
   findMedian() -> max(small) = 2.0

4. addNum(4):
   small: [2, 1], large: [3, 4] (size 2 vs 2)
   findMedian() -> (2 + 3) / 2.0 = 2.5

5. addNum(5):
   small: [3, 2, 1], large: [4, 5] (size 3 vs 2)
   findMedian() -> max(small) = 3.0
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Stream

- **Calls:** `["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]`
- **Arguments:** `[[], [1], [2], [], [3], []]`
- **Step Tracing:**
  - `addNum(1)` $\implies [1]$, median = `1.0`
  - `addNum(2)` $\implies [1], [2]$, median = `1.5`
  - `addNum(3)` $\implies [1, 2], [3]$, median = `2.0`
- **Output:** `[null, null, null, 1.5, null, 2.0]`

#### Example 2: Negative Numbers

- **Calls:** `addNum(-1), addNum(-2), addNum(-3)`
- **Trace:**
  - `small = [-2, -3]`, `large = [-1]`
  - Median = `-2.0`
- **Output:** `-2.0`

---

### Multi-Language Implementations

#### Python 3

```python
import heapq

class MedianFinder:
    def __init__(self):
        # small: max-heap (values stored as negative numbers)
        self.small = []
        # large: min-heap (values stored as positive numbers)
        self.large = []

    def addNum(self, num: int) -> None:
        # Step 1: Add to small (max-heap)
        heapq.heappush(self.small, -num)

        # Step 2: Ensure order invariant: max(small) <= min(large)
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # Step 3: Ensure size invariant: len(small) == len(large) or len(small) == len(large) + 1
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

#### C++17

```cpp
#include <queue>
#include <vector>

class MedianFinder {
private:
    std::priority_queue<int> small; // Max-heap for lower half
    std::priority_queue<int, std::vector<int>, std::greater<int>> large; // Min-heap for upper half

public:
    MedianFinder() {}

    void addNum(int num) {
        small.push(num);

        // Ensure max(small) <= min(large)
        if (!small.empty() && !large.empty() && small.top() > large.top()) {
            large.push(small.top());
            small.pop();
        }

        // Maintain balance: small has at most 1 more element than large
        if (small.size() > large.size() + 1) {
            large.push(small.top());
            small.pop();
        } else if (large.size() > small.size()) {
            small.push(large.top());
            large.pop();
        }
    }

    double findMedian() {
        if (small.size() > large.size()) {
            return static_cast<double>(small.top());
        }
        return (static_cast<double>(small.top()) + static_cast<double>(large.top())) / 2.0;
    }
};
```

#### Java

```java
import java.util.Collections;
import java.util.PriorityQueue;

public class MedianFinder {
    private PriorityQueue<Integer> small; // Max-heap
    private PriorityQueue<Integer> large; // Min-heap

    public MedianFinder() {
        small = new PriorityQueue<>(Collections.reverseOrder());
        large = new PriorityQueue<>();
    }

    public void addNum(int num) {
        small.offer(num);

        // Ensure max(small) <= min(large)
        if (!small.isEmpty() && !large.isEmpty() && small.peek() > large.peek()) {
            large.offer(small.poll());
        }

        // Balance sizes: small can have 1 more element than large
        if (small.size() > large.size() + 1) {
            large.offer(small.poll());
        } else if (large.size() > small.size()) {
            small.offer(large.poll());
        }
    }

    public double findMedian() {
        if (small.size() > large.size()) {
            return small.peek();
        }
        return (small.peek() + large.peek()) / 2.0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `addNum(num)`: $\mathcal{O}(\log N)$ to push and pop from the two binary heaps.
  - `findMedian()`: $\mathcal{O}(1)$ strictly to inspect the tops of the two heaps.
- **Space Complexity:** $\mathcal{O}(N)$
  - Total space across both heaps is equal to the number of elements inserted so far.

---

### Takeaway Pattern & Interview Traps

1. **Two-Heap Balancing Pattern:**
   - Any continuous stream problem requiring real-time percentile, median, or running split between top/bottom $K$ elements is fundamentally solved with two inversely ordered heaps.
2. **Division by 2.0:**
   - In C++ and Java, dividing `(small.top() + large.top()) / 2` with integer division will drop the decimal portion (e.g., $(2 + 3) / 2 = 2$ instead of $2.5$). Always cast to `double` or divide by `2.0`.