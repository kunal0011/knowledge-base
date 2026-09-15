---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 220: Contains Duplicate III"
tags:
  - leetcode
  - coding
  - sliding-window
---

# 220. Contains Duplicate III

# Solved

# Hard

# Topics

# premium lock icon

# Companies

# Hint

# You are given an integer array nums and two integers indexDiff and valueDiff.

  

# Find a pair of indices (i, j) such that:

  

# i != j,

# abs(i - j) <= indexDiff.

# abs(nums[i] - nums[j]) <= valueDiff, and

# Return true if such pair exists or false otherwise.

  
  

# Example 1:

  

# Input: nums = [1,2,3,1], indexDiff = 3, valueDiff = 0

# Output: true

# Explanation: We can choose (i, j) = (0, 3).

# We satisfy the three conditions:

# i != j --> 0 != 3

# abs(i - j) <= indexDiff --> abs(0 - 3) <= 3

# abs(nums[i] - nums[j]) <= valueDiff --> abs(1 - 1) <= 0

# Example 2:

  

# Input: nums = [1,5,9,1,5,9], indexDiff = 2, valueDiff = 3

# Output: false

# Explanation: After trying all the possible pairs (i, j), we cannot satisfy the three conditions, so we return false.

  

# Constraints:

  

# 2 <= nums.length <= 105

# -109 <= nums[i] <= 109

# 1 <= indexDiff <= nums.length

# 0 <= valueDiff <= 109

  
  
  

class Solution:

def containsNearbyAlmostDuplicate(self, nums: [int], k: int, t: int) -> bool:

if t < 0 or k < 1:

return False

bucket = {}

bucket\_size = t + 1 # Bucket size is t + 1 to handle the range of numbers

for i, num in enumerate(nums):

bucket\_id = num // bucket\_size

# If a number exists in the current bucket, return True

if bucket\_id in bucket:

return True

# Check adjacent buckets

if bucket\_id - 1 in bucket and abs(num - bucket[bucket\_id - 1]) <= t:

return True

if bucket\_id + 1 in bucket and abs(num - bucket[bucket\_id + 1]) <= t:

return True

# Insert the number into the bucket

bucket[bucket\_id] = num

# Remove elements outside the sliding window

if i >= k:

del bucket[nums[i - k] // bucket\_size]

return False