---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 128: Longest Consecutive Sequence"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 128: Longest Consecutive Sequence

---

### Problem Statement

Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence in `O(n)` time.

---

### Key Observation

* Convert array to a Hash Set for `O(1)` lookups.
* A number `x` is the **start** of a consecutive sequence if and only if `x - 1` is NOT in the set.
* Only count upwards from true sequence starts to ensure each number is visited at most twice.

---

### Core Technique: Hash Set Sequence Boundary Detection

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        longest = 0
        
        for num in num_set:
            # Check if start of sequence
            if num - 1 not in num_set:
                length = 1
                while num + length in num_set:
                    length += 1
                longest = max(longest, length)
                
        return longest
```

---

### Worked-Out Example

```python
nums = [100, 4, 200, 1, 3, 2]
num_set = {100, 4, 200, 1, 3, 2}
100: (99 not in set) -> count: 100 -> len 1
4: (3 is in set) -> skip
200: (199 not in set) -> count: 200 -> len 1
1: (0 not in set) -> count: 1, 2, 3, 4 -> len 4
Result = 4
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Filter for sequence start conditions `(x - 1 not in set)` to guarantee linear time on unsorted set operations.