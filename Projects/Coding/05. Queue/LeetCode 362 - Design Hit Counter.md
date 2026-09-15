---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 362: Design Hit Counter"
tags:
  - leetcode
  - coding
  - queue
  - circular-buffer
  - system-design
  - google
  - amazon
---

# LeetCode 362: Design Hit Counter

**Target Companies:** Google (Top Signature Systems Question), Amazon, Stripe, Netflix  
**Difficulty:** Medium  
**Topic:** Fixed-Size Circular Ring Buffer / Sliding Window Queue  

---

### Problem Statement

Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).

Your system should accept a `timestamp` parameter (in seconds granularity), and you may assume that calls are being made to the system in **chronological order** (i.e., `timestamp` is monotonically increasing). Several hits may arrive at roughly the same time.

Implement the `HitCounter` class:
- `HitCounter()`: Initializes the hit counter system.
- `void hit(int timestamp)`: Records a hit that happened at `timestamp` (in seconds). Several hits may happen at the same timestamp.
- `int getHits(int timestamp)`: Returns the number of hits in the past 5 minutes from `timestamp` (i.e., hits in $[	ext{timestamp} - 299, 	ext{timestamp}]$ inclusive).

---

### Input & Output Formats & Constraints

- **Input:** Sequence of operations: `["HitCounter", "hit", "hit", "hit", "getHits", "hit", "getHits", "getHits"]`
- **Output:** Returns from operations (`null`, `null`, `null`, `null`, `3`, `null`, `4`, `3`)
- **Constraints:**
  - $1 \le \text{timestamp} \le 2 \times 10^9$
  - All calls are being made in chronological order (`timestamp` increases).
  - At most $300$ seconds window is tracked.
  - At most $3 \times 10^5$ calls will be made across `hit` and `getHits`.

---

### Key Idea & Intuition

There are two primary ways to approach this:

#### Approach 1: Standard Queue / Deque
Maintain a queue of pairs `(timestamp, count)` or timestamps. In `getHits(t)`, pop elements where `t - element.timestamp >= 300`.
- **Limitation:** If 10 million hits arrive per second, storing every hit in memory causes high memory overhead.

#### Approach 2: Production Gold Standard — Fixed-Size Circular Ring Buffer
Since the time window is strictly 300 seconds:
- Allocate two fixed-size arrays of length 300:
  1. `times[300]`: records the latest timestamp mapped to this slot.
  2. `hits[300]`: records the total hit count for this timestamp.
- **Slot Mapping:** Index is computed simply as `idx = timestamp % 300`.
- **Slot Expiration:** When a hit arrives at `timestamp`:
  - If `times[idx] == timestamp`: Multiple hits at the exact same second $\to$ simply increment `hits[idx] += 1`.
  - If `times[idx] != timestamp`: The slot contains an old cycle's timestamp (from $\ge 300$ seconds ago) $\to$ overwrite `times[idx] = timestamp` and reset `hits[idx] = 1`.
- **Querying:** In `getHits(timestamp)`, loop over all 300 slots:
  - If `timestamp - times[i] < 300`, add `hits[i]` to `total`.
- **Advantage:** Space is bounded strictly to $O(300) = O(1)$ space, regardless of whether 1 hit or 10 billion hits occur!

---

### Visual Algorithm Walkthrough

```
Ring Buffer of Size 300:
Slot indices: 0, 1, 2, ..., 299

hit(1):
  idx = 1 % 300 = 1
  times[1] = 1, hits[1] = 1

hit(2), hit(3):
  times[2] = 2, hits[2] = 1
  times[3] = 3, hits[3] = 1

getHits(4):
  Look across 300 slots:
  - slot 1: 4 - 1 = 3 < 300 -> include hits[1] (1)
  - slot 2: 4 - 2 = 2 < 300 -> include hits[2] (1)
  - slot 3: 4 - 3 = 1 < 300 -> include hits[3] (1)
  Total = 3

hit(300):
  idx = 300 % 300 = 0
  times[0] = 300, hits[0] = 1

getHits(300):
  Slots 0, 1, 2, 3 all have 300 - times[i] < 300. Total = 4.

getHits(301):
  Slot 1 has times[1] = 1: 301 - 1 = 300 (not < 300, EXPIRED!).
  Slots 0, 2, 3 remain active: Total = 3.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Lifecycle
```text
Operations:
HitCounter counter = new HitCounter();
counter.hit(1);
counter.hit(2);
counter.hit(3);
counter.getHits(4);   // returns 3
counter.hit(300);
counter.getHits(300); // returns 4
counter.getHits(301); // returns 3 (hit at t=1 expired)
```

#### Example 2: Massive Bursts at Same Timestamp
```text
counter.hit(10);
counter.hit(10);
counter.hit(10);
counter.getHits(10);  // returns 3
counter.getHits(310); // 310 - 10 = 300 (expired) -> returns 0
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class HitCounter:
    def __init__(self) -> None:
        # Fixed 300-second circular ring buffer
        self.times = [0] * 300
        self.hits = [0] * 300

    def hit(self, timestamp: int) -> None:
        idx = timestamp % 300
        if self.times[idx] != timestamp:
            self.times[idx] = timestamp
            self.hits[idx] = 1
        else:
            self.hits[idx] += 1

    def getHits(self, timestamp: int) -> int:
        total = 0
        for i in range(300):
            if timestamp - self.times[i] < 300:
                total += self.hits[i]
        return total
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class HitCounter {
private:
    std::vector<int> times;
    std::vector<int> hits;

public:
    HitCounter() : times(300, 0), hits(300, 0) {}

    void hit(int timestamp) {
        int idx = timestamp % 300;
        if (times[idx] != timestamp) {
            times[idx] = timestamp;
            hits[idx] = 1;
        } else {
            hits[idx]++;
        }
    }

    int getHits(int timestamp) {
        int total = 0;
        for (int i = 0; i < 300; ++i) {
            if (timestamp - times[i] < 300) {
                total += hits[i];
            }
        }
        return total;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class HitCounter {
    private final int[] times;
    private final int[] hits;

    public HitCounter() {
        this.times = new int[300];
        this.hits = new int[300];
    }

    public void hit(int timestamp) {
        int idx = timestamp % 300;
        if (times[idx] != timestamp) {
            times[idx] = timestamp;
            hits[idx] = 1;
        } else {
            hits[idx]++;
        }
    }

    public int getHits(int timestamp) {
        int total = 0;
        for (int i = 0; i < 300; i++) {
            if (timestamp - times[i] < 300) {
                total += hits[i];
            }
        }
        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `hit(timestamp)`: $O(1)$ constant time array indexing.
  - `getHits(timestamp)`: $O(300) = O(1)$ fixed 300-iteration scan regardless of number of hits.
- **Space Complexity:** $O(1)$ strictly constant memory: two arrays of length 300 each (600 integers total).

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Circular Ring Buffer for Fixed Time-Window Rate Limiting.
- **Interview Follow-Up (Concurrency / Thread Safety):**
  - What if multiple threads call `hit()` and `getHits()` concurrently?
  - In production, use **atomic variables** (`AtomicIntegerArray` in Java or `std::atomic<int>` in C++) or read-write locks (`ReentrantReadWriteLock`), or shard the circular buffer across worker threads and sum their hit counts on `getHits()`.