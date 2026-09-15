---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 76: Minimum Window Substring"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 76: Minimum Window Substring

#76. Minimum Window Substring

from collections import Counter

  

class Solution:

def minWindow(self, s: str, t: str) -> str:

if not t or not s:

return ""

# Dictionary to keep a count of all the unique characters in t

dict\_t = Counter(t)

# Number of unique characters in t that need to be present in the window

required = len(dict\_t)

# Left and Right pointer

l, r = 0, 0

# Formed is the number of unique characters in the current window

# that meet the desired frequency in t

formed = 0

# Dictionary to keep a count of characters in the current window

window\_counts = {}

# (window length, left, right)

ans = float("inf"), None, None

while r < len(s):

# Add one character from the right to the window

character = s[r]

window\_counts[character] = window\_counts.get(character, 0) + 1

# If the current character's frequency matches that in t

if character in dict\_t and window\_counts[character] == dict\_t[character]:

formed += 1

# Try and contract the window till the point where it ceases to be 'desirable'

while l <= r and formed == required:

character = s[l]

# Save the smallest window until now

if r - l + 1 < ans[0]:

ans = (r - l + 1, l, r)

# The character at the position pointed by the `left` pointer is no longer a part of the window

window\_counts[character] -= 1

if character in dict\_t and window\_counts[character] < dict\_t[character]:

formed -= 1

# Move the left pointer ahead

l += 1

# Keep expanding the window once we are done contracting

r += 1

return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]