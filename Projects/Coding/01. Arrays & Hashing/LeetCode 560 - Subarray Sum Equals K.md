---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 560: Subarray Sum Equals K"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 560: Subarray Sum Equals K

---

### Problem Statement

Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals to `k`.

---

### Key Observation

* A subarray sum from index `i` to `j` is `prefix_sum[j] - prefix_sum[i - 1] = k`.
* Rearranging: `prefix_sum[i - 1] = prefix_sum[j] - k`.
* Maintain a running prefix sum and a frequency Hash Map counting occurrences of previously seen prefix sums.

---

### Core Technique: Prefix Sum with Frequency Hash Map

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_count = defaultdict(int)
        prefix_count[0] = 1  # base case: empty prefix has sum 0
        curr_sum = 0
        ans = 0
        
        for num in nums:
            curr_sum += num
            target = curr_sum - k
            if target in prefix_count:
                ans += prefix_count[target]
            prefix_count[curr_sum] += 1
            
        return ans
```

---

### Worked-Out Example

```text
nums = [1, 2, 3], k = 3
prefix_count = {0: 1}
num = 1: curr_sum = 1, target = 1 - 3 = -2 (not in map), prefix_count = {0: 1, 1: 1}
num = 2: curr_sum = 3, target = 3 - 3 = 0 (in map! count + 1), prefix_count[3] = 1
num = 3: curr_sum = 6, target = 6 - 3 = 3 (in map! count + 1), prefix_count[6] = 1
Total count = 2 ([1, 2] and [3])
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Subarray sums with negative or positive numbers cannot use sliding window; use Prefix Sum + Hash Map.