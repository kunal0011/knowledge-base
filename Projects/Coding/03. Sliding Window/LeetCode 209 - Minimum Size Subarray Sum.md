---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 209: Minimum Size Subarray Sum"
tags:
  - leetcode
  - coding
  - sliding-window
---

# 209. Minimum Size Subarray Sum

# Solved

# Medium

# Topics

# premium lock icon

# Companies

# Given an array of positive integers nums and a positive integer target, return the minimal length of a subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.

  

class Solution:

def minSubArrayLen(self, target: int, nums: List[int]) -> int:

n = len(nums)

min\_len = float('inf') # Initialize minimum length to infinity

current\_sum = 0

left = 0 # Left pointer of the sliding window

  

for right in range(n):

# Expand the window by including nums[right]

current\_sum += nums[right]

  

# Contract the window until the sum is less than target

while current\_sum >= target:

min\_len = min(min\_len, right - left + 1)

current\_sum -= nums[left]

left += 1

  

return min\_len if min\_len != float('inf') else 0