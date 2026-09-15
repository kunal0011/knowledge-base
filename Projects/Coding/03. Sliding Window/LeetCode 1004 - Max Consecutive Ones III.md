---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1004: Max Consecutive Ones III"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1004: Max Consecutive Ones III

**LeetCode 1004: Max Consecutive Ones III**

asks to find the maximum number of consecutive 1s in a binary array if you can flip at most k 0s.

Optimal Solution (Sliding Window Approach):

def longestOnes(nums, k):

    left = 0  # Left boundary of the window

    max\_length = 0

    for right in range(len(nums)):

        # Decrease k for every 0 encountered

        if nums[right] == 0:

            k -= 1

        # If k becomes negative, move the left pointer

        while k < 0:

            if nums[left] == 0:

                k += 1

            left += 1

        # Update the maximum length

        max\_length = max(max\_length, right - left + 1)

    return max\_length

Explanation:

1. **Sliding Window:** Expand the window with right, and shrink from left when flips exceed k.
2. k < 0 **Condition:** Ensures we flip at most k zeros.
3. **Time Complexity:** O(n) (each element visited at most twice).
4. **Space Complexity:** O(1) (constant extra space).

This is the most optimal solution for large input sizes.