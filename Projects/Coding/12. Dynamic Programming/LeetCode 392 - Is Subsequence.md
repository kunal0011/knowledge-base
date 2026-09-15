---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 392: Is Subsequence"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 392: Is Subsequence

**LeetCode 392 – Is Subsequence**, with **formal state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 392 — Is Subsequence

### Problem Statement

Given two strings `s` and `t`, return `true` if `s` is a subsequence of `t`, otherwise return `false`.

A **subsequence** is obtained by deleting zero or more characters from `t` **without changing the relative order** of the remaining characters.

---

## Dynamic Programming Approach

Although a greedy two-pointer solution is optimal, DP is useful to **formally model the subsequence relationship**, especially as a foundation for problems like LCS.

---

## 1. DP State Definition

Let:

```
dp[i][j] = True if the first i characters of s
           can be formed as a subsequence
           from the first j characters of t
```

### Meaning

* `i` → prefix length of `s` (`s[0...i-1]`)
* `j` → prefix length of `t` (`t[0...j-1]`)

---

## 2. Base Cases

1. **Empty `s` is a subsequence of any prefix of `t`**

```
dp[0][j] = True   for all j
```

2. **Non-empty `s` cannot be formed from empty `t`**

```
dp[i][0] = False  for all i > 0
```

---

## 3. State Transition

For `i > 0` and `j > 0`:

### Case 1: Characters match

If:

```
s[i-1] == t[j-1]
```

Then we can consume both characters:

```
dp[i][j] = dp[i-1][j-1]
```

### Case 2: Characters do not match

If:

```
s[i-1] != t[j-1]
```

Then we skip `t[j-1]`:

```
dp[i][j] = dp[i][j-1]
```

### Final Transition Formula

```
dp[i][j] =
    dp[i-1][j-1]  if s[i-1] == t[j-1]
    dp[i][j-1]    otherwise
```

---

## 4. DP Table Construction (Example)

### Input

```
s = "abc"
t = "ahbgdc"
```

### Table Layout

Rows → `s` (including empty)  
Columns → `t` (including empty)

```
      Ø  a  h  b  g  d  c
   Ø  T  T  T  T  T  T  T
   a  F  T  T  T  T  T  T
   b  F  F  F  T  T  T  T
   c  F  F  F  F  F  F  T
```

### Explanation of Key Cells

* `dp[1][1]` → `'a' == 'a'` → `dp[0][0] = True`
* `dp[2][3]` → `'b' == 'b'` → `dp[1][2] = True`
* `dp[3][6]` → `'c' == 'c'` → `dp[2][5] = True`

### Final Answer

```
dp[len(s)][len(t)] = dp[3][6] = True
```

---

## 5. Python 3 DP Implementation (with typing)

```python
from typing import List

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        m, n = len(s), len(t)

        # DP table
        dp: List[List[bool]] = [[False] * (n + 1) for _ in range(m + 1)]

        # Base case: empty s
        for j in range(n + 1):
            dp[0][j] = True

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]

        return dp[m][n]
```

---

## 6. Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | `O(m × n)` |
| Space | `O(m × n)` |

---

## 7. Important Insight

* This DP formulation is essentially a **restricted version of LCS**
* If `LCS(s, t) == len(s)` → `s` is a subsequence of `t`
* Greedy solution optimizes this DP to `O(n)` space and time

---

If you want, I can next:

* Convert this DP to **1D space optimization**
* Explain **why greedy works and DP is overkill**
* Connect this DP formally to **LCS recurrence**

Tell me how you want to proceed.