---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 516: Longest Palindromic Subsequence"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 516: Longest Palindromic Subsequence

**LeetCode 516 – Longest Palindromic Subsequence**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 516: Longest Palindromic Subsequence

### Problem Statement

Given a string `s`, return the length of the **longest subsequence** that is also a **palindrome**.

> A subsequence does **not** need to be contiguous.

---

## Key Observation

A palindrome:

* Reads the same forward and backward
* Characters at both ends must match

This problem is best solved using **Dynamic Programming on substrings**.

---

## DP State Definition

Let:

```
dp[i][j] = length of the longest palindromic subsequence
           in the substring s[i...j]
```

Where:

* `i` = starting index
* `j` = ending index
* `0 ≤ i ≤ j < n`

---

## Base Cases

1. **Single character**

```
dp[i][i] = 1
```

Every single character is a palindrome of length 1.

2. **Invalid range**

```
i > j → 0 (not stored explicitly)
```

---

## State Transition

We consider the characters at the ends of the substring: `s[i]` and `s[j]`.

### Case 1: Characters match

```
if s[i] == s[j]:
    dp[i][j] = 2 + dp[i+1][j-1]
```

Why?

* Matching ends can be included in the palindrome
* We add 2 and solve the inner substring

---

### Case 2: Characters do NOT match

```
if s[i] != s[j]:
    dp[i][j] = max(
        dp[i+1][j],
        dp[i][j-1]
    )
```

Why?

* One of the ends must be excluded
* We try both possibilities and take the maximum

---

## DP Table Construction Order

Since `dp[i][j]` depends on:

* `dp[i+1][j]`
* `dp[i][j-1]`
* `dp[i+1][j-1]`

We must fill the table **bottom-up**:

### Correct order:

1. Length = 1 substrings
2. Length = 2 substrings
3. Length = 3 …
4. Up to length = `n`

Equivalent to:

```
for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
```

---

## Example Walkthrough

### Input

```
s = "bbbab"
index: 0 1 2 3 4
        b b b a b
```

---

### Step 1: Initialize DP table

Diagonal = 1

```
dp[i][i] = 1

    b b b a b
      -----------
b |  1 0 0 0 0
b |  0 1 0 0 0
b |  0 0 1 0 0
a |  0 0 0 1 0
b |  0 0 0 0 1
```

---

### Step 2: Length = 2

* `dp[0][1]` ("bb") → match → `2`
* `dp[1][2]` ("bb") → match → `2`
* `dp[2][3]` ("ba") → no match → `1`
* `dp[3][4]` ("ab") → no match → `1`

---

### Step 3: Length = 3

* `dp[0][2]` ("bbb") → match → `2 + dp[1][1] = 3`
* `dp[1][3]` ("bba") → no match → `max(1, 2) = 2`
* `dp[2][4]` ("bab") → match → `2 + dp[3][3] = 3`

---

### Step 4: Length = 4

* `dp[0][3]` ("bbba") → no match → `max(2, 3) = 3`
* `dp[1][4]` ("bbab") → match → `2 + dp[2][3] = 3`

---

### Step 5: Length = 5 (full string)

* `dp[0][4]` ("bbbab")
* `s[0] == s[4] == 'b'`

```
dp[0][4] = 2 + dp[1][3] = 2 + 2 = 4
```

---

### Final DP Table (conceptual)

```
    0 1 2 3 4
      b b b a b
    -------------
0 b | 1 2 3 3 4
1 b | 0 1 2 2 3
2 b | 0 0 1 1 3
3 a | 0 0 0 1 1
4 b | 0 0 0 0 1
```

---

## Final Answer

```
Longest Palindromic Subsequence length = dp[0][n-1] = 4
```

Example subsequence: `"bbbb"`

---

## Python 3 Implementation (Typed)

```python
from typing import List

class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp: List[List[int]] = [[0] * n for _ in range(n)]

        # Base case: single characters
        for i in range(n):
            dp[i][i] = 1

        # Fill table for substrings of length >= 2
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                if s[i] == s[j]:
                    dp[i][j] = 2 + dp[i + 1][j - 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

        return dp[0][n - 1]
```

---

## Complexity Analysis

* **Time Complexity:** `O(n²)`
* **Space Complexity:** `O(n²)`

---

If you want next:

* Space-optimized DP (`O(n)`)
* Mapping this problem to **LCS(s, reverse(s))**
* Or a **DP table visualization with arrows**

Tell me how deep you want to go.