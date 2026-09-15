---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 567: Permutation in String"
tags:
  - leetcode
  - coding
  - sliding-window
---

# 567. Permutation in String

# Solved

# Medium

# Topics

# premium lock icon

# Companies

# Hint

# Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

  

# In other words, return true if one of s1's permutations is the substring of s2.

  
  

# Example 1:

  

# Input: s1 = "ab", s2 = "eidbaooo"

# Output: true

# Explanation: s2 contains one permutation of s1 ("ba").

# Example 2:

  

# Input: s1 = "ab", s2 = "eidboaoo"

# Output: false

  

# Constraints:

  

# 1 <= s1.length, s2.length <= 104

# s1 and s2 consist of lowercase English letters.

  

from collections import Counter

  

class Solution:

def checkInclusion(self, s1: str, s2: str) -> bool:

len\_s1, len\_s2 = len(s1), len(s2)

if len\_s1 > len\_s2:

return False

# Frequency counter for s1

s1\_count = Counter(s1)

# Initial window frequency counter

window\_count = Counter(s2[:len\_s1])

# Check the first window

if s1\_count == window\_count:

return True

# Sliding window to compare counts of each window

for i in range(len\_s1, len\_s2):

# Add the next character in the window

window\_count[s2[i]] += 1

# Remove the character that is no longer in the window

window\_count[s2[i - len\_s1]] -= 1

# Remove the count completely if it's zero to match Counter's behavior

if window\_count[s2[i - len\_s1]] == 0:

del window\_count[s2[i - len\_s1]]

# Check if the window matches s1's character count

if s1\_count == window\_count:

return True

return False