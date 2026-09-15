---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2134: Minimum Swaps to Group All 1’s Together II"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 2134: Minimum Swaps to Group All 1’s Together II

**Leetcode 2134: Minimum Swaps to Group All 1’s Together II**.

---

## 📌 Problem Recap

We’re given:

* A **binary array** `nums` (only 0s and 1s).
* The array is **circular**.

We want the **minimum number of swaps** to group all `1`s together.

---

## ✅ Key Observations

1. Let `totalOnes = sum(nums)`.
2. Goal = find a **subarray of length `totalOnes`** that contains **maximum number of 1s**.
3. Minimum swaps = `totalOnes - maxOnesInWindow`.

**Circular array trick:**

* Concatenate the array with itself: `nums + nums` → simulate circular behavior.
* Then slide a window of size `totalOnes` to find the max 1s.

---

## ✅ Approach: Sliding Window

### Code:

```python
def minSwaps(nums):
    total_ones = sum(nums)
    n = len(nums)
    nums = nums * 2  # simulate circular array
    
    # initial window
    window_ones = sum(nums[:total_ones])
    max_ones = window_ones
    
    for i in range(total_ones, 2*n):
        window_ones += nums[i] - nums[i - total_ones]
        max_ones = max(max_ones, window_ones)
    
    return total_ones - max_ones
```

---

### Complexity:

* Time: **O(n)** (window slides 2n - total\_ones ≈ O(n))
* Space: **O(n)** (due to concatenation, can optimize in-place if needed)

---

### Why it works:

1. All 1s must fit in a window of length `totalOnes`.
2. The best window = window with **most 1s already there** → fewer swaps.
3. Circular array handled by concatenation → all circular windows are checked.

---

### Example:

```text
nums = [1,0,1,0,1]
total_ones = 3
Window length = 3

Windows:
[1,0,1] → 2 ones → swaps = 1
[0,1,0] → 1 ones → swaps = 2
[1,0,1] → 2 ones → swaps = 1
[0,1,1] → 2 ones → swaps = 1

Answer = 1
```

---

# 🎯 Final Summary

| Step | Complexity | Notes |
| --- | --- | --- |
| Sliding window on circular array | O(n), O(n) | Efficient, canonical solution |
| Naive: try all rotations | O(n²) | Too slow |