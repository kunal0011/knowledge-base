---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 5: Longest Palindromic Substring"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 5: Longest Palindromic Substring

## LeetCode 5 — Longest Palindromic Substring

### Problem Statement

Given a string `s`, return the **longest substring** of `s` that is a **palindrome**.

A palindrome reads the same forward and backward.

---

## Dynamic Programming Approach

### Key Insight

A substring `s[l..r]` is a palindrome **if and only if**:

1. `s[l] == s[r]`
2. The inner substring `s[l+1..r-1]` is also a palindrome  
   (or the length is ≤ 2, which is trivially palindromic)

This makes the problem ideal for **interval DP**.

---

## 1. DP State Definition

Let:

```
dp[l][r] = True  if substring s[l..r] is a palindrome
           False otherwise
```

Where:

* `l` = starting index
* `r` = ending index
* `0 ≤ l ≤ r < n`

---

## 2. Base Cases

1. **Single character substrings**

```
dp[i][i] = True
```

Every single character is a palindrome.

2. **Two-character substrings**

```
dp[i][i+1] = (s[i] == s[i+1])
```

---

## 3. State Transition

For substrings of length ≥ 3:

```
dp[l][r] = (s[l] == s[r]) AND dp[l+1][r-1]
```

---

## 4. Traversal Order (Very Important)

Because `dp[l][r]` depends on `dp[l+1][r-1]`,  
we must **increase substring length gradually**.

Traversal strategy:

```
for length from 3 to n:
    for l from 0 to n - length:
        r = l + length - 1
```

This guarantees `dp[l+1][r-1]` is already computed.

---

## 5. Tracking the Answer

Maintain:

* `start`: starting index of longest palindrome
* `max_len`: length of longest palindrome found so far

Update when:

```
if dp[l][r] == True and (r - l + 1) > max_len
```

---

## 6. Python 3 DP Implementation (with typing)

```python
from typing import List

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n == 0:
            return ""

        dp: List[List[bool]] = [[False] * n for _ in range(n)]

        start = 0
        max_len = 1

        # Base case: single characters
        for i in range(n):
            dp[i][i] = True

        # Base case: two characters
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_len = 2

        # Length >= 3
        for length in range(3, n + 1):
            for l in range(0, n - length + 1):
                r = l + length - 1
                if s[l] == s[r] and dp[l + 1][r - 1]:
                    dp[l][r] = True
                    start = l
                    max_len = length

        return s[start:start + max_len]
```

---

## 7. Example Walkthrough

### Input

```
s = "babad"
```

Indices:

```
0 1 2 3 4
b a b a d
```

---

### Step 1: Initialize DP Table

Single characters:

```
dp[i][i] = True
```

```
b   a   b   a   d
b [ T   F   F   F   F ]
a [     T   F   F   F ]
b [         T   F   F ]
a [             T   F ]
d [                 T ]
```

---

### Step 2: Length = 2

Check adjacent characters:

* `"ba"` → False
* `"ab"` → False
* `"ba"` → False
* `"ad"` → False

No updates.

---

### Step 3: Length = 3

Check:

* `l=0, r=2` → `"bab"`

  * `s[0] == s[2]` ✔
  * `dp[1][1] == True`
  * `dp[0][2] = True`
* `l=1, r=3` → `"aba"`

  * `s[1] == s[3]` ✔
  * `dp[2][2] == True`
  * `dp[1][3] = True`

DP snapshot:

```
b   a   b   a   d
b [ T   F   T   F   F ]
a [     T   F   T   F ]
b [         T   F   F ]
a [             T   F ]
d [                 T ]
```

Longest so far: `"bab"` or `"aba"` (length = 3)

---

### Step 4: Length = 4

* `"baba"` → ends mismatch
* `"abad"` → ends mismatch

No updates.

---

### Step 5: Length = 5

* `"babad"` → ends mismatch

Stop.

---

## 8. Final Answer

```text
Output: "bab"
```

(`"aba"` is also valid; either is acceptable)

---

## 9. Complexity Analysis

* **Time Complexity:** `O(n²)`
* **Space Complexity:** `O(n²)`