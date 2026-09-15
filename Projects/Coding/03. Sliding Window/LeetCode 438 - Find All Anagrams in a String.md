---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 438: Find All Anagrams in a String"
tags:
  - leetcode
  - coding
  - sliding-window
---

# 438. Find All Anagrams in a String

# Solved

# Medium

# Topics

# premium lock icon

# Companies

# Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order.

  
  

# Example 1:

  

# Input: s = "cbaebabacd", p = "abc"

# Output: [0,6]

# Explanation:

# The substring with start index = 0 is "cba", which is an anagram of "abc".

# The substring with start index = 6 is "bac", which is an anagram of "abc".

# Example 2:

  

# Input: s = "abab", p = "ab"

# Output: [0,1,2]

# Explanation:

# The substring with start index = 0 is "ab", which is an anagram of "ab".

# The substring with start index = 1 is "ba", which is an anagram of "ab".

# The substring with start index = 2 is "ab", which is an anagram of "ab".

  

# Constraints:

  

# 1 <= s.length, p.length <= 3 \* 104

# s and p consist of lowercase English letters.

  
  
  

from collections import Counter

  

class Solution:

def findAnagrams(self, s: str, p: str):

p\_len = len(p)

s\_len = len(s)

if p\_len > s\_len:

return []

# Frequency map of string p

p\_count = Counter(p)

# Frequency map of the first window in s

s\_count = Counter(s[:p\_len - 1])

result = []

# Start sliding the window

for i in range(p\_len - 1, s\_len):

# Include the current character in the window

s\_count[s[i]] += 1

# If the frequency maps are equal, add the starting index

if s\_count == p\_count:

result.append(i - p\_len + 1)

# Move the window: remove the character that is left out

s\_count[s[i - p\_len + 1]] -= 1

if s\_count[s[i - p\_len + 1]] == 0:

del s\_count[s[i - p\_len + 1]]

return result