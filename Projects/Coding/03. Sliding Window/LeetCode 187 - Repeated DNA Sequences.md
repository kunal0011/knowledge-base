---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 187: Repeated DNA Sequences"
tags:
  - leetcode
  - coding
  - sliding-window
---

# 187. Repeated DNA Sequences

# Solved

# Medium

# Topics

# premium lock icon

# Companies

# The DNA sequence is composed of a series of nucleotides abbreviated as 'A', 'C', 'G', and 'T'.

  

# For example, "ACGAATTCCG" is a DNA sequence.

# When studying DNA, it is useful to identify repeated sequences within the DNA.

  

# Given a string s that represents a DNA sequence, return all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.

  
  

# Example 1:

  

# Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"

# Output: ["AAAAACCCCC","CCCCCAAAAA"]

# Example 2:

  

# Input: s = "AAAAAAAAAAAAA"

# Output: ["AAAAAAAAAA"]

  

# Constraints:

  

# 1 <= s.length <= 105

# s[i] is either 'A', 'C', 'G', or 'T'.

  

class Solution:

def findRepeatedDnaSequences(self, s: str) -> List[str]:

if len(s) < 10:

return []

  

seen, repeated = {}, set()

  

for i in range(len(s) - 9):

substring = s[i:i + 10]

if substring in seen:

repeated.add(substring)

else:

seen[substring] = 1

  

return list(repeated)