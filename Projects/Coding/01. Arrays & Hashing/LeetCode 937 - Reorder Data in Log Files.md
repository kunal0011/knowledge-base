---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 937: Reorder Data in Log Files"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 937: Reorder Data in Log Files

**Target Companies:** Amazon (Top #1 Classic)

---

### Problem Statement

Reorder logs so letter-logs come before digit-logs. Letter-logs are sorted lexicographically by content (then identifier). Digit-logs maintain relative order.

---

### Key Observation

* Separate logs into letter-logs and digit-logs.
* Letter-logs use a composite sorting key: `(log_content, log_identifier)`.
* Digit-logs remain in original arrival order. Return `letter_logs + digit_logs`.

---

### Core Technique: Custom Tuple Key Sorting & Stable Partitioning

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def get_key(log: str):
            identifier, content = log.split(" ", 1)
            if content[0].isalpha():
                return (0, content, identifier)  # Letter-log: priority 0
            return (1,)                          # Digit-log: priority 1 (stable)
            
        return sorted(logs, key=get_key)
```

---

### Worked-Out Example

```
logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Letter logs sorted by (content, id):
1. "let1 art can" (content: "art can")
2. "let3 art zero" (content: "art zero")
3. "let2 own kit dig" (content: "own kit dig")
Digit logs appended in original order: ["dig1 8 1 5 1", "dig2 3 6"]
```

---

### Complexity Analysis

* **Time Complexity:** `O(M * N log N) where N is number of logs, M is max log length`
* **Space Complexity:** `O(M * N)`

---

### Takeaway Pattern

Use custom tuple sort keys `(is\_digit, content, identifier)` with Python's stable Timsort.