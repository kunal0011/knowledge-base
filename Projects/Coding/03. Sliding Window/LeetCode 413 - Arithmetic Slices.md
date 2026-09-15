---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 413: Arithmetic Slices"
tags:
  - leetcode
  - coding
  - sliding-window
---

# 413. Arithmetic Slices  

# Solved

# Medium

# Topics

# premium lock icon

# Companies

# An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.

  

# For example, [1,3,5,7,9], [7,7,7,7], and [3,-1,-5,-9] are arithmetic sequences.

# Given an integer array nums, return the number of arithmetic subarrays of nums.

  

# A subarray is a contiguous subsequence of the array.

class Solution:

def numberOfArithmeticSlices(self, A: list[int]) -> int:

if len(A) < 3:

return 0

count = 0

current\_slices = 0

  

for i in range(2, len(A)):

if A[i] - A[i - 1] == A[i - 1] - A[i - 2]:

current\_slices += 1

count += current\_slices

else:

current\_slices = 0

return count

  

# Example usage:

solution = Solution()

print(solution.numberOfArithmeticSlices([1, 2, 3, 4])) # Expected output: 3

print(solution.numberOfArithmeticSlices([1])) # Expected output: 0

print(solution.numberOfArithmeticSlices([1, 3, 5, 7, 9])) # Expected output: 6