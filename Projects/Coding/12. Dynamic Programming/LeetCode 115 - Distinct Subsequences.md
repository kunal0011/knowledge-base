---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 115: Distinct Subsequences"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 115: Distinct Subsequences

**LeetCode 115 — Distinct Subsequences**, focused on **state definition, transitions, DP table construction, and a worked example**.

---

## LeetCode 115 — Distinct Subsequences

### Problem Statement

Given two strings **s** and **t**, return the number of **distinct subsequences** of **s** which equal **t**.

A subsequence is obtained by deleting zero or more characters from **s** without changing the relative order of the remaining characters.

---

## Key Observation

* We must **count**, not just check existence.
* Order matters.
* This is a **prefix-to-prefix matching problem**, making it a classic **2D Dynamic Programming** problem.

---

## DP State Definition

Let:

```
dp[i][j] = number of distinct subsequences of s[0..i-1] that equal t[0..j-1]
```

Meaning:

* First `i` characters of `s`
* First `j` characters of `t`

---

## Base Cases

### 1. Empty target string

```
dp[i][0] = 1   for all i
```

Reason:

* There is exactly **one** way to form an empty string: delete everything.

### 2. Empty source string (non-empty target)

```
dp[0][j] = 0   for j > 0
```

Reason:

* You cannot form a non-empty string from an empty string.

---

## State Transition

We compare:

```
s[i-1] and t[j-1]
```

### Case 1: Characters match

```
s[i-1] == t[j-1]
```

We have **two choices**:

1. **Use** this character → match both prefixes  
   → `dp[i-1][j-1]`
2. **Skip** this character in `s`  
   → `dp[i-1][j]`

```
dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
```

---

### Case 2: Characters do not match

```
s[i-1] != t[j-1]
```

We can only **skip** the current character of `s`:

```
dp[i][j] = dp[i-1][j]
```

---

## Final Answer

```
dp[len(s)][len(t)]
```

---

## Example Walkthrough

### Input

```
s = "rabbbit"
t = "rabbit"
```

Lengths:

```
s → 7 characters
t → 6 characters
```

---

### DP Table Structure

Rows → `s` (including empty prefix)  
Columns → `t` (including empty prefix)

| s \ t | "" | r | a | b | b | i | t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| "" | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| r | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| a | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| b | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| b | 1 | 1 | 1 | 2 | 1 | 0 | 0 |
| b | 1 | 1 | 1 | 3 | 3 | 0 | 0 |
| i | 1 | 1 | 1 | 3 | 3 | 3 | 0 |
| t | 1 | 1 | 1 | 3 | 3 | 3 | 3 |

---

### Explanation of Key Cell

At `dp[5][3]` (matching `"rabbb"` → `"rab"`):

* `s[4] == 'b'` and `t[2] == 'b'`
* So:

```
dp[5][3] = dp[4][2] + dp[4][3]
         = 1 + 2
         = 3
```

---

### Final Result

```
dp[7][6] = 3
```

There are **3 distinct subsequences** of `"rabbbit"` that equal `"rabbit"`.

---

## Python 3 DP Implementation (With Typing)

```python
from typing import List

class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        
        # dp[i][j] = number of ways s[0..i-1] forms t[0..j-1]
        dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Base case: empty target
        for i in range(m + 1):
            dp[i][0] = 1
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        
        return dp[m][n]
```

---

## Complexity Analysis

* **Time Complexity:** `O(m × n)`
* **Space Complexity:** `O(m × n)`
* Can be optimized to `O(n)` using rolling array (optional).

---

## DP Pattern Classification

* **Pattern:** *Subsequence Counting DP*
* **Related Problems:**

  * LeetCode 392 (Is Subsequence)
  * LeetCode 583 (Delete Operation for Two Strings)
  * LeetCode 1092 (Shortest Common Supersequence)

---

If you want, I can:

* Show **1D space-optimized DP**
* Explain **why order of loops matters**
* Compare this with **LCS-style DP**
* Draw the **DP dependency graph**

Just specify.