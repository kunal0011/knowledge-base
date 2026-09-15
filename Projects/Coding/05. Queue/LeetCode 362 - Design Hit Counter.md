---
date: "2026-08-29"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 362: Design Hit Counter"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 362: Design Hit Counter

**Target Companies:** Google (Signature Systems Question), Amazon

---

### Problem Statement

Design a hit counter which counts the number of hits received in the past 5 minutes (300 seconds). Calls are made in chronological order.

---

### Key Observation

* Approach 1: Queue storing timestamps. On `getHits(timestamp)`, pop elements `<= timestamp - 300`.
* Approach 2 (Scalable for high concurrency): **Circular Ring Buffer** of size 300 holding `times[300]` and `hits[300]` arrays.
* Using modulo indexing `idx = timestamp % 300` eliminates memory growth under massive traffic.

---

### Core Technique: Circular Ring Buffer / Sliding Window Queue

---

### Python 3 Solution (with typing)

```python
class HitCounter:
    def __init__(self):
        # 300-second circular buffer
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

---

### Worked-Out Example

```python
hit(1), hit(2), hit(3)
getHits(4) -> checks all slots within 300s -> returns 3
hit(300)
getHits(300) -> returns 4
getHits(301) -> timestamp 1 expires (301 - 1 = 300 >= 300) -> returns 3
```

---

### Complexity Analysis

* **Time Complexity:** `O(1) for hit, O(300) = O(1) for getHits`
* **Space Complexity:** `O(1) fixed 300-element array`

---

### Takeaway Pattern

Use fixed-size modulo ring buffers to implement memory-bounded rate limiters and counters under heavy production load.