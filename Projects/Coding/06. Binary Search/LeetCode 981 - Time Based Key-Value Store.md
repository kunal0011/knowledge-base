---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 981: Time Based Key-Value Store"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 981: Time Based Key-Value Store

---

### Problem Statement

Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

---

### Key Observation

* Timestamps for each key are strictly increasing, so values for a key form a sorted array by timestamp.
* We can use Binary Search (`right\_bound` / `bisect\_right`) to find the latest timestamp `<= timestamp` in `O(log n)`.

---

### Core Technique: Binary Search on Sorted Timestamp Tuples

---

### Python 3 Solution (with typing)

```python
from collections import defaultdict

class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)  # key -> list of (timestamp, value)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        values = self.store[key]
        left, right = 0, len(values) - 1
        res = ""
        
        while left <= right:
            mid = left + (right - left) // 2
            if values[mid][0] <= timestamp:
                res = values[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        return res
```

---

### Worked-Out Example

```python
set("foo", "bar", 1) -> [ (1, "bar") ]
set("foo", "bar2", 4) -> [ (1, "bar"), (4, "bar2") ]
get("foo", 4) -> returns "bar2"
get("foo", 3) -> returns "bar" (since 1 <= 3 < 4)
```

---

### Complexity Analysis

* **Time Complexity:** `O(1) for set, O(log n) for get`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Append strictly increasing timestamps to a list and query floor matches using binary search.