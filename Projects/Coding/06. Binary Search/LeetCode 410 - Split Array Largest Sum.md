---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 410: Split Array Largest Sum"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 410: Split Array Largest Sum

---

### Problem Statement

Given an integer array `nums` and an integer `k`, split the array into `k` non-empty subarrays such that the largest sum of any subarray is minimized.

---

### Key Observation

* The minimum possible largest sum is `max(nums)` (single element subarray).
* The maximum possible largest sum is `sum(nums)` (entire array as 1 subarray).
* Since feasibility is monotonic, binary search the answer range `[max(nums), sum(nums)]`.

---

### Core Technique: Binary Search on Minimized Maximum Subarray Sum

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        def can_split(max_sum: int) -> bool:
            subarrays = 1
            current_sum = 0
            for num in nums:
                if current_sum + num > max_sum:
                    subarrays += 1
                    current_sum = num
                else:
                    current_sum += num
            return subarrays <= k
            
        left, right = max(nums), sum(nums)
        ans = right
        
        while left <= right:
            mid = left + (right - left) // 2
            if can_split(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        return ans
```

---

### Worked-Out Example

```python
nums = [7, 2, 5, 10, 8], k = 2
Search range: [10, 32]
mid = 21: [7,2,5] (sum 14), [10,8] (sum 18) -> 2 subarrays <= 2 (valid) -> ans=21, right=20
mid = 15: [7,2,5] (14), [10] (10), [8] (8) -> 3 subarrays > 2 (invalid) -> left=16
mid = 18: [7,2,5] (14), [10,8] (18) -> 2 subarrays <= 2 (valid) -> ans=18, right=17
Result = 18
```

---

### Complexity Analysis

* **Time Complexity:** `O(n * log(sum(nums) - max(nums)))`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Painter's Partition / Book Allocation / Ship Packages all share this exact Binary Search on Sum pattern.