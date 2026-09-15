---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 97: Interleaving String"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 97: Interleaving String

**LeetCode 97 – Interleaving String**, with **precise state definition, transition logic, DP table construction, and a worked example**.

---

## LeetCode 97 – Interleaving String

### Problem Statement

Given three strings `s1`, `s2`, and `s3`, determine whether `s3` is formed by **interleaving** `s1` and `s2`.

**Interleaving rules**

* Characters from `s1` and `s2` must appear **in order**
* You may switch between strings, but relative order must be preserved

---

## Key Observation (Feasibility Check)

If

```
len(s1) + len(s2) != len(s3)
```

→ **Impossible**, return `False` immediately.

---

## DP State Definition

### State

```
dp[i][j] = True
```

means:

> `s3[0 : i + j]` can be formed by interleaving  
> `s1[0 : i]` and `s2[0 : j]`

### Indices Meaning

* `i` → number of characters taken from `s1`
* `j` → number of characters taken from `s2`
* total characters formed = `i + j`

---

## DP Transition

To compute `dp[i][j]`, we consider **where the last character came from**.

### Option 1: Last char came from `s1`

```
if dp[i-1][j] == True
and s1[i-1] == s3[i+j-1]
→ dp[i][j] = True
```

### Option 2: Last char came from `s2`

```
if dp[i][j-1] == True
and s2[j-1] == s3[i+j-1]
→ dp[i][j] = True
```

### Final Transition

```
dp[i][j] =
   (dp[i-1][j] and s1[i-1] == s3[i+j-1])
or (dp[i][j-1] and s2[j-1] == s3[i+j-1])
```

---

## Base Cases

### 1. Empty strings

```
dp[0][0] = True
```

### 2. First column (only s1 used)

```
dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
```

### 3. First row (only s2 used)

```
dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
```

---

## DP Table Size

```
(len(s1) + 1) x (len(s2) + 1)
```

---

## Example Walkthrough

### Input

```
s1 = "ab"
s2 = "cd"
s3 = "acbd"
```

### DP Table Layout

| dp[i][j] | j=0 | j=1 | j=2 |
| --- | --- | --- | --- |
| i=0 | T | F | F |
| i=1 | T | T | F |
| i=2 | F | T | T |

---

### Step-by-step Explanation

#### dp[0][0] = True

Empty + empty → empty

---

#### First row

* `dp[0][1]`: `"c"` vs `"a"` → ❌
* `dp[0][2]`: `"cd"` vs `"ac"` → ❌

---

#### First column

* `dp[1][0]`: `"a"` matches `"a"` → ✅
* `dp[2][0]`: `"ab"` vs `"ac"` → ❌

---

#### dp[1][1]

`s3[1] = 'c'`

* From `s1`: `'a'` ≠ `'c'`
* From `s2`: `'c'` == `'c'` and `dp[1][0] == True`

→ `dp[1][1] = True`

---

#### dp[2][1]

`s3[2] = 'b'`

* From `s1`: `'b'` == `'b'` and `dp[1][1] == True`

→ `dp[2][1] = True`

---

#### dp[2][2]

`s3[3] = 'd'`

* From `s2`: `'d'` == `'d'` and `dp[2][1] == True`

→ `dp[2][2] = True`

---

### Final Answer

```
dp[len(s1)][len(s2)] = dp[2][2] = True
```

---

## Python 3 DP Implementation (Typed)

```python
from typing import List

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)

        if m + n != len(s3):
            return False

        dp: List[List[bool]] = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True

        # First column
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]

        # First row
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (
                    (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or
                    (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
                )

        return dp[m][n]
```

---

## Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time | `O(m * n)` |
| Space | `O(m * n)` (can be optimized to `O(n)`) |

---

## Conceptual Summary

* **State** tracks how many characters are consumed from each string
* **Transition** checks where the last character of `s3` came from
* This is a **2D prefix-matching DP**, not substring matching

If you want, I can also provide:

* Space-optimized 1D DP version
* Backtracking vs DP comparison
* Visual grid traversal explanation similar to path DP