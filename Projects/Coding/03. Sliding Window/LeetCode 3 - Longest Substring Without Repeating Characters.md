---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 3: Longest Substring Without Repeating Characters"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 3: Longest Substring Without Repeating Characters

Leetcode [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

class Solution:

def lengthOfLongestSubstring(self, s: str) -> int:

char\_set = set() # Set to store characters in the current window

left = 0 # Left pointer of the sliding window

max\_length = 0 # Result to store the maximum length of substring

  

for right in range(len(s)):

# Expand the window by including s[right]

while s[right] in char\_set:

char\_set.remove(s[left])

left += 1

char\_set.add(s[right])

max\_length = max(max\_length, right - left + 1)

  

return max\_length