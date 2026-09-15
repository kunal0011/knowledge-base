---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 346: Moving Average from Data Stream"
tags:
  - leetcode
  - coding
  - queue
  - amazon
  - google
---

# LeetCode 346: Moving Average from Data Stream

**Target Companies:** Google (Top #1 Queue Classic), Amazon, Meta  
**Difficulty:** Easy / Medium  
**Topic:** Queue / Circular Buffer / Sliding Stream

---

### Problem Statement

Given a stream of integers and a window size `size`, calculate the moving average of all integers in the sliding window.

Implement the `MovingAverage` class:
- `MovingAverage(int size)`: Initializes the object with the size of the window `size`.
- `double next(int val)`: Returns the moving average of the last `size` values of the stream.

---

### Input & Output Formats & Constraints

- **Input Operations:** `MovingAverage(size)`, `next(val)`
- **Output:** Floating point number (double precision)
- **Constraints:**
  - $1 \le 	ext{size} \le 1000$
  - $-10^5 \le 	ext{val} \le 10^5$
  - At most $10^4$ calls will be made to `next`.

---

### Key Idea & Intuition

- **Sliding Window via FIFO Queue:**
  - We need to maintain the most recent `size` elements in sequential arrival order.
  - When the queue size exceeds `size`, the oldest element (at front of queue) must be evicted.
- **Running Sum Optimization:**
  - Instead of summing all elements in the queue on each call ($O(size)$), maintain a `running_sum`.
  - When adding `val`:
    - `running_sum += val`
    - If `queue.size() > size`: `running_sum -= queue.popleft()`
  - Average is simply `running_sum / queue.size()`.
  - Achieves strict $O(1)$ time complexity per query!

---

### Solution Approach (Step-by-Step)

1. **`__init__(size)`**:
   - Store `self.size = size`.
   - `self.queue = deque()`.
   - `self.running_sum = 0.0`.
2. **`next(val)`**:
   - Append `val` to `self.queue`.
   - `self.running_sum += val`.
   - If `len(self.queue) > self.size`:
     - `self.running_sum -= self.queue.popleft()`.
   - Return `self.running_sum / len(self.queue)`.

---

### Visual Algorithm Walkthrough

```
Window Size = 3

next(1):  Queue: [1]          Sum: 1   Avg: 1 / 1 = 1.0
next(10): Queue: [1, 10]      Sum: 11  Avg: 11 / 2 = 5.5
next(3):  Queue: [1, 10, 3]   Sum: 14  Avg: 14 / 3 = 4.66667
next(5):  Evict 1 -> Queue: [10, 3, 5]
          Sum: 14 - 1 + 5 = 18. Avg: 18 / 3 = 6.0
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Window Evolution
| Operation | Val | Queue State | Window Sum | Window Size | Return Value |
|:---|:---:|:---|:---:|:---:|:---:|
| `MovingAverage(3)` | - | `[]` | `0` | `0` | `null` |
| `next(1)` | `1` | `[1]` | `1` | `1` | `1.0` |
| `next(10)` | `10` | `[1, 10]` | `11` | `2` | `5.5` |
| `next(3)` | `3` | `[1, 10, 3]` | `14` | `3` | `4.66667` |
| `next(5)` | `5` | `[10, 3, 5]` | `18` | `3` | `6.0` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from collections import deque

class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.running_sum = 0.0

    def next(self, val: int) -> float:
        self.queue.append(val)
        self.running_sum += val
        
        if len(self.queue) > self.size:
            self.running_sum -= self.queue.popleft()
            
        return self.running_sum / len(self.queue)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <queue>

class MovingAverage {
private:
    int maxSize;
    double runningSum;
    std::queue<int> q;

public:
    MovingAverage(int size) : maxSize(size), runningSum(0.0) {}

    double next(int val) {
        q.push(val);
        runningSum += val;

        if (q.size() > maxSize) {
            runningSum -= q.front();
            q.pop();
        }

        return runningSum / q.size();
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class MovingAverage {
    private final int size;
    private final Deque<Integer> queue;
    private double runningSum;

    public MovingAverage(int size) {
        this.size = size;
        this.queue = new ArrayDeque<>();
        this.runningSum = 0.0;
    }

    public double next(int val) {
        queue.offer(val);
        runningSum += val;

        if (queue.size() > size) {
            runningSum -= queue.poll();
        }

        return runningSum / queue.size();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** Strict $O(1)$ per `next()` call — One queue push, at most one pop, and constant arithmetic operations.
- **Space Complexity:** $O(	ext{size})$ — Auxiliary storage bounded strictly by the moving window capacity.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Circular Buffer / FIFO sliding queue with accumulator variable.
- **Follow-up:** Can you implement this with a fixed-size array and modular arithmetic `head = (head + 1) % size` without dynamic memory allocations? (Circular queue array optimization).
