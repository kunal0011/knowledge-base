---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 981: Time Based Key-Value Store"
tags:
  - leetcode
  - coding
  - binary-search
  - hash-map
  - system-design
  - google
  - amazon
---

# LeetCode 981: Time Based Key-Value Store

**Target Companies:** Google (Top Signature Systems Question), Amazon, Netflix, Meta  
**Difficulty:** Medium  
**Topic:** Hash Map with Timestamped Monotonic Vector & Floor Binary Search  

---

### Problem Statement

Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the `TimeMap` class:
- `TimeMap()`: Initializes the object of the data structure.
- `void set(String key, String value, int timestamp)`: Stores the key `key` with the value `value` at the given time `timestamp`.
- `String get(String key, int timestamp)`: Returns a value such that `set` was called previously, with `timestamp_prev <= timestamp`. If there are multiple such values, it returns the value associated with the largest `timestamp_prev`. If there are no values, it returns `""`.

---

### Input & Output Formats & Constraints

- **Input:** Sequence of operations: `["TimeMap", "set", "get", "get", "set", "get", "get"]`
- **Output:** Returns from operations (`null`, `null`, `"bar"`, `"bar"`, `null`, `"bar2"`, `"bar2"`)
- **Constraints:**
  - $1 \le \text{key.length}, \text{value.length} \le 100$
  - `key` and `value` consist of lowercase English letters and digits.
  - $1 \le \text{timestamp} \le 10^7$
  - All the timestamps `timestamp` of `set` are **strictly increasing**.
  - At most $2 \times 10^5$ calls will be made across `set` and `get`.

---

### Key Idea & Intuition

The problem states that calls to `set` for all keys are made in **strictly increasing order of timestamps**.
This means:
1. For any given `key`, appending `(timestamp, value)` to a list results in a list that is **always sorted in ascending order of timestamps** without requiring any explicit sorting!
2. Inserting takes strictly $O(1)$ time (`append`).
3. For `get(key, timestamp)`:
   - If `key` does not exist, return `""`.
   - If `key` exists, we must find the latest recorded entry with $\text{entry.timestamp} \le \text{timestamp}$ (the mathematical **Floor** element).
   - Because the list of timestamps is strictly sorted, we can use **Binary Search**:
     - If `list[mid].timestamp <= timestamp`: record `list[mid].value` as a candidate answer, and continue searching **to the right** (`left = mid + 1`) to see if an even closer timestamp exists.
     - Else (`list[mid].timestamp > timestamp`): search **to the left** (`right = mid - 1`).
   - Binary search executes in $O(\log M)$ time, where $M$ is the number of `set` calls for that specific `key`.

---

### Solution Approach (Step-by-Step)

1. `__init__()`: Initialize a hash map `store = defaultdict(list)` mapping `key` to a list of tuples `(timestamp, value)`.
2. `set(key, value, timestamp)`:
   - Append `(timestamp, value)` directly to `store[key]`.
3. `get(key, timestamp)`:
   - If `key` is not in `store`, return `""`.
   - Let `values = store[key]`.
   - Set `left = 0`, `right = len(values) - 1`, `ans = ""`.
   - While `left <= right`:
     - `mid = left + (right - left) // 2`.
     - If `values[mid][0] <= timestamp`:
       - `ans = values[mid][1]`
       - `left = mid + 1` (search for a larger valid timestamp)
     - Else:
       - `right = mid - 1`
   - Return `ans`.

---

### Visual Algorithm Walkthrough

```
Operations:
1. set("foo", "bar", 1)  -> store["foo"] = [(1, "bar")]
2. set("foo", "bar2", 4) -> store["foo"] = [(1, "bar"), (4, "bar2")]

Query 1: get("foo", 4)
  Search in [(1, "bar"), (4, "bar2")] for largest timestamp <= 4:
  - left = 0, right = 1 -> mid = 0 (timestamp 1 <= 4):
    cand = "bar", left = mid + 1 = 1
  - left = 1, right = 1 -> mid = 1 (timestamp 4 <= 4):
    cand = "bar2", left = mid + 1 = 2
  Terminated (left > right).
  Returns "bar2".

Query 2: get("foo", 3)
  Search for largest timestamp <= 3:
  - left = 0, right = 1 -> mid = 0 (timestamp 1 <= 3):
    cand = "bar", left = mid + 1 = 1
  - left = 1, right = 1 -> mid = 1 (timestamp 4 > 3):
    right = mid - 1 = 0
  Terminated.
  Returns "bar".

Query 3: get("foo", 0)
  Target 0 < smallest timestamp 1:
  All timestamps > 0.
  Returns "".
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Exact and Interior Hits
```text
TimeMap timeMap = new TimeMap();
timeMap.set("foo", "bar", 1);
timeMap.get("foo", 1);         // return "bar"
timeMap.get("foo", 3);         // return "bar" (1 <= 3)
timeMap.set("foo", "bar2", 4);
timeMap.get("foo", 4);         // return "bar2"
timeMap.get("foo", 5);         // return "bar2" (4 <= 5)
```

#### Example 2: Out of Bound Queries
```text
timeMap.get("foo", 0);         // return "" (all timestamps >= 1)
timeMap.get("nonexistent", 1); // return ""
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from collections import defaultdict
from typing import List, Tuple

class TimeMap:
    def __init__(self) -> None:
        # key -> list of (timestamp, value)
        self.store: dict[str, List[Tuple[int, str]]] = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
            
        values = self.store[key]
        left, right = 0, len(values) - 1
        ans = ""
        
        while left <= right:
            mid = left + (right - left) // 2
            if values[mid][0] <= timestamp:
                ans = values[mid][1]
                left = mid + 1  # Seek latest timestamp <= query
            else:
                right = mid - 1
                
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

class TimeMap {
private:
    std::unordered_map<std::string, std::vector<std::pair<int, std::string>>> store;

public:
    TimeMap() {}

    void set(std::string key, std::string value, int timestamp) {
        store[key].emplace_back(timestamp, value);
    }

    std::string get(std::string key, int timestamp) {
        auto it = store.find(key);
        if (it == store.end()) return "";

        const auto& entries = it->second;
        int left = 0, right = static_cast<int>(entries.size()) - 1;
        std::string ans = "";

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (entries[mid].first <= timestamp) {
                ans = entries[mid].second;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return ans;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class TimeMap {
    private static class Entry {
        int timestamp;
        String value;
        Entry(int timestamp, String value) {
            this.timestamp = timestamp;
            this.value = value;
        }
    }

    private final Map<String, List<Entry>> store;

    public TimeMap() {
        this.store = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        store.computeIfAbsent(key, k -> new ArrayList<>()).add(new Entry(timestamp, value));
    }

    public String get(String key, int timestamp) {
        List<Entry> entries = store.get(key);
        if (entries == null) {
            return "";
        }

        int left = 0, right = entries.size() - 1;
        String ans = "";

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (entries.get(mid).timestamp <= timestamp) {
                ans = entries.get(mid).value;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `set(key, value, timestamp)`: $O(1)$ amortized append to vector.
  - `get(key, timestamp)`: $O(\log M)$ where $M$ is the number of entries stored for the queried key.
- **Space Complexity:** $O(N)$ total space to store $N$ total `(timestamp, value)` records across all keys.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Temporal Key-Value Store with Monotonic Vector Floor Search.
- **Trap:** Using `TreeMap` in Java vs. `ArrayList`: While `TreeMap.floorEntry()` gives $O(\log M)$ on both `set` and `get`, `TreeMap` incurs red-black tree rebalancing and object node overhead. Using `ArrayList` gives $O(1)$ `set` and identical $O(\log M)$ binary search on `get`, which is faster and more memory-efficient.