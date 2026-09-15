---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 219: Contains Duplicate II"
tags:
  - leetcode
  - coding
  - sliding-window
---

# 219. Contains Duplicate II

# Solved

# Easy

# Topics

# premium lock icon

# Companies

# Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

  
  

# Example 1:

  

# Input: nums = [1,2,3,1], k = 3

# Output: true

# Example 2:

  

# Input: nums = [1,0,1,1], k = 1

# Output: true

# Example 3:

  

# Input: nums = [1,2,3,1,2,3], k = 2

# Output: false

  

class Solution:

def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:

mp = {}

for i, v in enumerate(nums):

if v in mp and i - mp[v] <= k:

return True

mp[v] = i

return False