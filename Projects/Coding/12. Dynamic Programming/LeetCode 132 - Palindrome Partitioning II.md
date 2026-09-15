---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 132: Palindrome Partitioning II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 132: Palindrome Partitioning II

**LeetCode 132 – Palindrome Partitioning II**, with **proper state definition, transitions, DP table construction, and a worked example**.

---

## LeetCode 132 — Palindrome Partitioning II

### Problem Statement

Given a string `s`, partition `s` such that every substring of the partition is a **palindrome**.  
Return the **minimum number of cuts** needed.

---

## Key Insight

* This is an **optimization DP problem**
* Naive backtracking is exponential
* We must **precompute palindromes** and then **minimize cuts**

---

## Step 1: Palindrome Precomputation (DP over substrings)

### Palindrome DP State

```
pal[i][j] = True if s[i..j] is a palindrome
```

### Transition

```
pal[i][j] = (s[i] == s[j]) AND (j - i <= 2 OR pal[i+1][j-1])
```

### Order of Computation

* `i` from `n-1 → 0`
* `j` from `i → n-1`

### Why?

We need `pal[i+1][j-1]` computed before `pal[i][j]`.

---

## Step 2: Minimum Cut DP

### DP State Definition

```
dp[i] = minimum cuts needed for substring s[0..i]
```

### Final Answer

```
dp[n-1]
```

---

## DP Transition

### Case 1: Whole substring is palindrome

```
if pal[0][i]:
    dp[i] = 0
```

### Case 2: Try all partitions

```
dp[i] = min(dp[j-1] + 1)
for all j in [1..i] if pal[j][i] == True
```

### Why `+1`?

We make a cut **before j**, so:

* Left part: `s[0..j-1]`
* Right part: `s[j..i]` (palindrome)

---

## DP Table Creation Example

### Input

```
s = "aab"
index: 0 1 2
        a a b
```

---

### Palindrome Table (`pal[i][j]`)

| i\j | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | T | T | F |
| 1 |  | T | F |
| 2 |  |  | T |

**Explanation**

* `"a"` → T
* `"aa"` → T
* `"ab"` → F
* `"b"` → T

---

### DP Table (`dp[i]`)

| i | substring | dp[i] | Reason |
| --- | --- | --- | --- |
| 0 | `"a"` | 0 | already palindrome |
| 1 | `"aa"` | 0 | whole string palindrome |
| 2 | `"aab"` | 1 | `"aa" |

---

### How `dp[2]` is computed

```
pal[0][2] = False
Try cuts:

j = 1 → pal[1][2] = False
j = 2 → pal[2][2] = True
dp[2] = dp[1] + 1 = 0 + 1 = 1
```

---

## Final DP Visualization

```
"aab"
  |
"a" | "ab" ❌
"aa" | "b" ✅

Minimum cuts = 1
```

---

## Time & Space Complexity

| Component | Complexity |
| --- | --- |
| Palindrome DP | O(n²) |
| Cut DP | O(n²) |
| Total | **O(n²)** |
| Space | **O(n²)** |

---

## Python 3 Implementation (Typed)

```python
from typing import List

class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)

        # Step 1: Palindrome DP
        pal: List[List[bool]] = [[False] * n for _ in range(n)]

        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or pal[i + 1][j - 1]):
                    pal[i][j] = True

        # Step 2: Cut DP
        dp: List[int] = [float('inf')] * n

        for i in range(n):
            if pal[0][i]:
                dp[i] = 0
            else:
                for j in range(1, i + 1):
                    if pal[j][i]:
                        dp[i] = min(dp[i], dp[j - 1] + 1)

        return dp[n - 1]
```

---

## Why This DP Is Optimal

* Avoids recomputation of palindrome checks
* Converts exponential backtracking into polynomial DP
* Clean separation of **palindrome detection** and **cut optimization**

---

If you want next:

* Backtracking → DP intuition bridge
* Bottom-up vs Top-down (memo) comparison
* Optimization to **O(n²) space → O(n)** space
* Full **DP table trace for larger input**

Tell me how deep you want to go.