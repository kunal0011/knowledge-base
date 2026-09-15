---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 347: Top K Frequent Elements"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 347: Top K Frequent Elements

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.

---

### Key Observation

* Count frequencies of all numbers in `O(n)`.
* Use Bucket Sort where bucket index represents frequency count (max count is `n`).
* Iterating backwards from bucket `n` down to `1` yields top `k` elements in `O(n)` time.

---

### Core Technique: Bucket Sort by Frequency

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        buckets = [[] for _ in range(len(nums) + 1)]
        
        for num, freq in count.items():
            buckets[freq].append(num)
            
        res = []
        for freq in range(len(buckets) - 1, 0, -1):
            for num in buckets[freq]:
                res.append(num)
                if len(res) == k:
                    return res
        return res
```

---

### Worked-Out Example

```python
nums = [1, 1, 1, 2, 2, 3], k = 2
count = {1: 3, 2: 2, 3: 1}
buckets[3] = [1], buckets[2] = [2], buckets[1] = [3]
Iterate from bucket 3 -> pick 1, iterate to bucket 2 -> pick 2
Result = [1, 2]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

When frequency values are bounded by array size n, Bucket Sort beats O(n log k) heap.