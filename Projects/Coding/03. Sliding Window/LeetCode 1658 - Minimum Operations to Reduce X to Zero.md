---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1658: Minimum Operations to Reduce X to Zero"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1658: Minimum Operations to Reduce X to Zero

**Leetcode 1658: Minimum Operations to Reduce X to Zero**.

---

## 📌 Problem Recap

We’re given:

* An array `nums` (positive integers).
* An integer `x`.

We can only remove elements from **start** or **end** of the array.  
We want the **minimum number of operations** such that the sum of removed elements = `x`.  
If not possible → return `-1`.

---

## ✅ Approach 1: Brute Force (Try All Splits)

### Idea:

* Suppose we remove `i` elements from left and `j` from right.
* If `sum(left[:i]) + sum(right[:j]) == x`, track `i + j`.

### Code:

```python
def minOperations_bruteforce(nums, x):
    n = len(nums)
    ans = float("inf")
    
    left_sums = {0:0}
    s = 0
    for i in range(n):
        s += nums[i]
        left_sums[s] = i + 1
    
    s = 0
    for j in range(n-1, -1, -1):
        s += nums[j]
        if x - s in left_sums:
            ans = min(ans, left_sums[x - s] + (n - j))
    
    return ans if ans != float("inf") else -1
```

### Complexity:

* Time: **O(n)** with hashing, but memory overhead.
* Space: **O(n)**.

✅ Works but not the cleanest.

---

## ✅ Approach 2: Transform to Longest Subarray Problem (Optimal)

### Key Observation:

* Total sum of array = `total`.
* We need to remove elements summing to `x`.
* Equivalent to **finding the longest subarray whose sum = total - x**.
* Then answer = `n - length_of_that_subarray`.

### Example:

`nums = [1,1,4,2,3], x = 5, total = 11`.  
We want subarray sum = `11 - 5 = 6`.  
Longest subarray with sum = 6 is `[1,4,1]` (length 3).  
So answer = `5 - 3 = 2`.

---

### Code (Sliding Window since all positive):

```python
def minOperations(nums, x):
    total = sum(nums)
    target = total - x
    if target < 0: 
        return -1
    if target == 0: 
        return len(nums)
    
    n = len(nums)
    left = 0
    curr_sum = 0
    max_len = -1
    
    for right in range(n):
        curr_sum += nums[right]
        
        while curr_sum > target:
            curr_sum -= nums[left]
            left += 1
        
        if curr_sum == target:
            max_len = max(max_len, right - left + 1)
    
    return n - max_len if max_len != -1 else -1
```