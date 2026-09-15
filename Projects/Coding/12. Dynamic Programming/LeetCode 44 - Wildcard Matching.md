---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 44: Wildcard Matching"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 44: Wildcard Matching

**LeetCode 44 (Wildcard Matching)** with **precise state definition, transitions, DP table construction, and a worked example**. This is written in a way suitable for interviews and deep understanding.

---

## LeetCode 44 — Wildcard Matching

### Problem Statement

Given an input string `s` and a pattern `p`, implement wildcard matching with support for:

* `?` → matches **exactly one character**
* `*` → matches **any sequence of characters (including empty)**

The matching must cover the **entire string**.

---

## Key Observations

1. This is a **full string matching** problem → not substring.
2. `*` is the only complex character:

   * It can match **nothing**
   * Or match **one or more characters**
3. The problem naturally maps to **Dynamic Programming** because:

   * Matching prefixes of `s` and `p`
   * Optimal substructure exists

---

## DP State Definition

Let:

```
dp[i][j] = True if s[0..i-1] matches p[0..j-1]
```

* `i` → length of prefix of `s`
* `j` → length of prefix of `p`
* Final answer → `dp[len(s)][len(p)]`

---

## Base Cases

### 1. Empty string & empty pattern

```
dp[0][0] = True
```

### 2. Empty string & non-empty pattern

Only valid if **pattern consists entirely of `*`**:

```
dp[0][j] = dp[0][j-1] if p[j-1] == '*'
```

### 3. Non-empty string & empty pattern

```
dp[i][0] = False  (i > 0)
```

---

## State Transition

### Case 1: Normal character or `?`

If:

```
p[j-1] == s[i-1] OR p[j-1] == '?'
```

Then:

```
dp[i][j] = dp[i-1][j-1]
```

---

### Case 2: `*`

`*` has **two choices**:

1. Match **empty** → ignore `*`
2. Match **one or more characters**

```
dp[i][j] = dp[i][j-1]    # '*' matches empty
         OR dp[i-1][j]  # '*' consumes one char from s
```

---

### Case 3: No match

```
dp[i][j] = False
```

---

## Full DP Recurrence

```
if p[j-1] == '*':
    dp[i][j] = dp[i][j-1] or dp[i-1][j]
elif p[j-1] == '?' or p[j-1] == s[i-1]:
    dp[i][j] = dp[i-1][j-1]
else:
    dp[i][j] = False
```

---

## DP Table Creation Order

* Fill row by row
* `i = 0 → len(s)`
* `j = 0 → len(p)`

This ensures all dependencies are already computed.

---

## Worked Example

### Input

```
s = "adceb"
p = "*a*b"
```

### DP Table (T = True, F = False)

| s \ p | "" | \* | a | \* | b |
| --- | --- | --- | --- | --- | --- |
| "" | T | T | F | F | F |
| a | F | T | T | T | F |
| d | F | T | F | T | F |
| c | F | T | F | T | F |
| e | F | T | F | T | F |
| b | F | T | F | T | T |

### Final Answer

```
dp[5][4] = True
```

---

## Python 3 DP Solution (Typed)

```python
from typing import List

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        
        dp: List[List[bool]] = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Base case
        dp[0][0] = True
        
        # Pattern prefix matching empty string
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
        
        return dp[m][n]
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(m × n)** |
| Space | **O(m × n)** |
| Optimizable | Yes → 1D DP |

---

## Mental Model (Interview Tip)

* Think in terms of **prefix matching**
* `*` is the only branching decision:

  * Stay in same pattern (`dp[i-1][j]`)
  * Or move forward (`dp[i][j-1]`)

---

If you want next:

* **1D DP optimization**
* **Why greedy works for this problem**
* **Comparison: LeetCode 10 vs 44**
* **Backtracking vs DP tradeoff**

Say the word.