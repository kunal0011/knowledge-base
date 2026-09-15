---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 713: Subarray Product Less Than K"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 713: Subarray Product Less Than K

LeetCode 713: *Subarray Product Less Than K*

requires finding the number of contiguous subarrays where the product of all elements is less than k.

Optimal Solution (Sliding Window Technique):

from typing import List

def numSubarrayProductLessThanK(nums: List[int], k: int) -> int:

    if k <= 1:

        return 0

    product = 1

    result = 0

    left = 0

    for right in range(len(nums)):

        product \*= nums[right]

        while product >= k:

            product //= nums[left]

            left += 1

        result += right - left + 1

    return result

Explanation:

1. **Early Return:** If k <= 1, it's impossible to have any subarray with a product less than k.
2. **Sliding Window:** We expand the window by moving right and contract it by moving left when the product exceeds k.
3. **Counting:** For every valid window [left, right], the number of new subarrays ending at right is right - left + 1.

Complexity:

* **Time Complexity:** O(n) — Each element is processed at most twice.
* **Space Complexity:** O(1) — Only a few variables for tracking state.

This is the most efficient approach for this problem.