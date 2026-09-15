---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

**Leetcode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit**.

---

## 📌 Problem Recap

We’re given an array `nums` and an integer `limit`.  
We need the **length of the longest subarray** such that

max⁡(subarray)−min⁡(subarray)≤limitmax(subarray)−min(subarray)≤limit

---

## ✅ Approach 1: Brute Force

### Idea:

* Try all subarrays, check max & min.

### Code:

```python
def longestSubarray_brute(nums, limit):
    n = len(nums)
    ans = 0
    for i in range(n):
        curr_min = curr_max = nums[i]
        for j in range(i, n):
            curr_min = min(curr_min, nums[j])
            curr_max = max(curr_max, nums[j])
            if curr_max - curr_min <= limit:
                ans = max(ans, j - i + 1)
            else:
                break
    return ans
```

### Complexity:

* Time: **O(n²)**
* Space: **O(1)**

❌ Too slow for `n = 10^5`.

---

## ✅ Approach 2: Sliding Window + Sorted List (Binary Search)

### Idea:

* Maintain a window `[left, right]` such that `max - min ≤ limit`.
* Use `bisect.insort` to keep a sorted list of window values.
* If condition breaks, shrink from left.

### Code:

```python
import bisect

def longestSubarray_sortedlist(nums, limit):
    window = []
    ans = left = 0
    
    for right, val in enumerate(nums):
        bisect.insort(window, val)
        while window[-1] - window[0] > limit:
            window.pop(bisect.bisect_left(window, nums[left]))
            left += 1
        ans = max(ans, right - left + 1)
    return ans
```

### Complexity:

* Time: **O(n log n)** (insertion in sorted list).
* Space: **O(n)**

✅ Works but slower than optimal.

---

## ✅ Approach 3: Sliding Window + Monotonic Deques (Optimal)

### Idea:

* Maintain two deques:

  * `max_dq`: decreasing → front = max
  * `min_dq`: increasing → front = min
* Expand `right`, push new element while keeping order.
* If `max - min > limit`, shrink from `left`.
* Track max window size.

### Code:

```python
from collections import deque

def longestSubarray(nums, limit):
    max_dq, min_dq = deque(), deque()
    left = 0
    ans = 0
    
    for right, val in enumerate(nums):
        # maintain decreasing deque for max
        while max_dq and val > max_dq[-1]:
            max_dq.pop()
        max_dq.append(val)
        
        # maintain increasing deque for min
        while min_dq and val < min_dq[-1]:
            min_dq.pop()
        min_dq.append(val)
        
        # shrink window if invalid
        while max_dq[0] - min_dq[0] > limit:
            if nums[left] == max_dq[0]:
                max_dq.popleft()
            if nums[left] == min_dq[0]:
                min_dq.popleft()
            left += 1
        
        ans = max(ans, right - left + 1)
    
    return ans
```

### Complexity:

* Time: **O(n)** (each element enters/exits deque once).
* Space: **O(n)**

✅✅ Best solution

---

# 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(n²) | ❌ | Too slow |
| Sliding Window + SortedList | O(n log n), O(n) | ✅ | Acceptable |
| Sliding Window + Deque | O(n), O(n) | ✅✅ | **Best solution** |