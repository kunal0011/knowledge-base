---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 10: Regular Expression Matching"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 10: Regular Expression Matching

## LeetCode 10 – Regular Expression Matching

### Problem Statement

Given a string `s` and a pattern `p`, implement regular expression matching with support for:

* `.` → matches any single character
* `*` → matches zero or more of the preceding element

The matching must cover the **entire** input string.

---

## Why Dynamic Programming?

Greedy or simple recursion fails because:

* `*` introduces **multiple choices** (use zero or many)
* Overlapping subproblems exist
* We must ensure **full match**, not partial

DP allows us to systematically explore all valid matches.

---

## 1. DP State Definition

### DP State

```
dp[i][j] = True if s[0..i-1] matches p[0..j-1]
```

* `i` → length of prefix of `s`
* `j` → length of prefix of `p`
* Final answer: `dp[len(s)][len(p)]`

---

## 2. DP Table Dimensions

```
Rows    = len(s) + 1
Columns = len(p) + 1
```

Extra row/column represent **empty string / empty pattern**

---

## 3. Base Case Initialization

### Empty string vs empty pattern

```
dp[0][0] = True
```

### Empty string vs pattern

Only valid if pattern can represent empty string (like `a*`, `a*b*`)

```
for j in range(2, len(p) + 1):
    if p[j - 1] == '*':
        dp[0][j] = dp[0][j - 2]
```

---

## 4. State Transition

### Case 1: Direct match or `.`

If:

* `s[i-1] == p[j-1]` OR
* `p[j-1] == '.'`

```
dp[i][j] = dp[i-1][j-1]
```

---

### Case 2: `*` Operator

`*` applies to **previous character** → `p[j-2]`

#### Option A: Use `*` as ZERO occurrence

```
dp[i][j] = dp[i][j-2]
```

#### Option B: Use `*` as ONE or MORE occurrences

Valid only if:

```
p[j-2] == s[i-1] OR p[j-2] == '.'
```

```
dp[i][j] = dp[i-1][j]
```

#### Final `*` transition

```
dp[i][j] = dp[i][j-2] OR dp[i-1][j]
```

---

## 5. Complete DP Transition Logic

```
if p[j - 1] == '*':
    dp[i][j] = dp[i][j - 2]
    if p[j - 2] == s[i - 1] or p[j - 2] == '.':
        dp[i][j] |= dp[i - 1][j]
elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
    dp[i][j] = dp[i - 1][j - 1]
```

---

## 6. Full Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp: List[List[bool]] = [[False] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = True

        # Initialize empty string vs pattern
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]
                    if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                        dp[i][j] |= dp[i - 1][j]
                elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]

        return dp[m][n]
```

---

## 7. Worked Example

### Input

```
s = "aab"
p = "c*a*b"
```

---

### DP Table (T = True, F = False)

| s\p | "" | c | \* | a | \* | b |
| --- | --- | --- | --- | --- | --- | --- |
| "" | T | F | T | F | T | F |
| a | F | F | F | T | T | F |
| a | F | F | F | F | T | F |
| b | F | F | F | F | F | T |

---

### Key Transitions

* `c*` → ignored (zero occurrence)
* `a*` → matches both `a`s
* `b` → exact match

Final result:

```
dp[3][5] = True
```

---

## 8. Time and Space Complexity

```
Time  : O(len(s) × len(p))
Space : O(len(s) × len(p))
```

---

## 9. Key Interview Observations

* `*` always looks **two steps back**
* DP index `(i, j)` represents prefixes, not characters
* Initialization of row `0` is crucial
* This is **NOT** wildcard matching (`?`, `*`) — different problem