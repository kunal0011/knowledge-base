---
date: "2025-12-23"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 1679: Max Number of K-Sum Pairs"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 1679: Max Number of K-Sum Pairs

Below is a complete, structured explanation for **LeetCode 1679 – Max Number of K-Sum Pairs**, following your requested format.

---

## LeetCode 1679: Max Number of K-Sum Pairs

### Problem Statement

You are given an integer array `nums` and an integer `k`.

In one operation, you can pick **two numbers** from the array whose sum equals `k` and remove them from the array.

Return the **maximum number of operations** you can perform.

#### Constraints (relevant)

* `1 <= nums.length <= 10^5`
* `1 <= nums[i] <= 10^9`
* `1 <= k <= 10^9`

---

## Key Observation

* Each valid operation **consumes exactly two elements** whose sum is `k`.
* Once an element is used in a pair, it **cannot be reused**.
* To maximize the number of operations:

  * We must **form as many disjoint pairs** as possible.
* This is a classic **pairing problem**, where ordering and efficient pairing matter.

---

## Why Two Pointer Technique Works

### Core Insight

If the array is **sorted**:

* Smallest element + largest element gives the **widest range check**.
* This allows us to:

  * Increase sum by moving the left pointer.
  * Decrease sum by moving the right pointer.

### Strategy

1. **Sort** the array.
2. Use two pointers:

   * `left` at start
   * `right` at end
3. While `left < right`:

   * If `nums[left] + nums[right] == k`:

     * Valid pair found → count++
     * Move both pointers
   * If sum < k → move `left`
   * If sum > k → move `right`

This greedy approach ensures:

* Each element is used at most once.
* Maximum number of valid pairs are formed.

---

## Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        
        left: int = 0
        right: int = len(nums) - 1
        operations: int = 0
        
        while left < right:
            current_sum = nums[left] + nums[right]
            
            if current_sum == k:
                operations += 1
                left += 1
                right -= 1
            elif current_sum < k:
                left += 1
            else:
                right -= 1
        
        return operations
```

---

## Worked Out Example

### Input

```text
nums = [1, 2, 3, 4]
k = 5
```

### Step 1: Sort

```text
nums = [1, 2, 3, 4]
```

### Step 2: Two Pointer Traversal

| Left | Right | nums[left] | nums[right] | Sum | Action |
| --- | --- | --- | --- | --- | --- |
| 0 | 3 | 1 | 4 | 5 | Pair found → count = 1 |
| 1 | 2 | 2 | 3 | 5 | Pair found → count = 2 |

Pointers cross → stop.

### Output

```
2
```

---

## Complexity Analysis

* **Time Complexity**:

  * Sorting: `O(n log n)`
  * Two pointer scan: `O(n)`
  * **Overall**: `O(n log n)`
* **Space Complexity**:

  * `O(1)` extra space (ignoring sort’s internal stack)

---

## Summary

* Sorting + Two Pointer is the optimal and cleanest approach.
* Greedy pairing ensures maximum valid operations.
* This pattern is common in **pair-sum**, **array matching**, and **greedy optimization** problems.

If you want, I can also provide:

* Hash-map–based solution (O(n) time)
* Pattern classification (Two Sum variants)
* Edge-case walkthroughs