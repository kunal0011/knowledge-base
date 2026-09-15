---
date: "2026-08-29"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 1143: Longest Common Subsequence"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 1143: Longest Common Subsequence

**Target Companies:** Amazon, Google, Microsoft

---

### Problem Statement

Given two strings `text1` and `text2`, return the length of their longest common subsequence.

---

### Key Observation

* 2D DP: `dp[i][j]` is the LCS of `text1[i:]` and `text2[j:]`.
* If `text1[i] == text2[j]`: `dp[i][j] = 1 + dp[i+1][j+1]`.
* If `text1[i] != text2[j]`: `dp[i][j] = max(dp[i+1][j], dp[i][j+1])`.

---

### Core Technique: 2D Grid Subsequence Dynamic Programming

---

### Python 3 Solution (with typing)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                    
        return dp[0][0]
```

---

### Worked-Out Example

```
text1 = "abcde", text2 = "ace"
Matches at: 'a', 'c', 'e'
LCS = 3 ("ace")
```

---

### Complexity Analysis

* **Time Complexity:** `O(m * n)`
* **Space Complexity:** `O(m * n) (can be reduced to O(min(m, n)))`

---

### Takeaway Pattern

The cornerstone pattern for diff tools, DNA alignment, and edit distances.