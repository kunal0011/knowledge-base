---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 91: Decode Ways"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 91: Decode Ways

**LeetCode 91 – Decode Ways**, focusing on **state definition, transition, DP table construction**, and a **worked example**.

---

## LeetCode 91 – Decode Ways

### Problem Statement

You are given a string `s` containing only digits.  
Each digit or pair of digits can be mapped to letters as:

```python
'1' -> 'A'
'2' -> 'B'
...
'26' -> 'Z'
```

Return the **total number of ways** to decode the string.

**Constraints**

* `'0'` cannot be decoded alone.
* Two-digit numbers must be in the range `10` to `26`.

---

## Key Observation

At any position `i`, decoding depends only on:

1. **Single digit** ending at `i`
2. **Two digits** ending at `i`

This naturally leads to **1-D Dynamic Programming**.

---

## DP State Definition

### State

```
dp[i] = number of ways to decode the substring s[0 : i]
```

> `i` represents **length**, not index.

---

### Base Cases

```
dp[0] = 1        # empty string has 1 valid decoding
dp[1] = 1 if s[0] != '0' else 0
```

---

## State Transition

For `i >= 2`, consider two cases:

### 1️⃣ Single Digit Decode

If `s[i-1] != '0'`  
→ decode last digit alone

```
dp[i] += dp[i-1]
```

---

### 2️⃣ Two Digit Decode

If `10 <= int(s[i-2:i]) <= 26`  
→ decode last two digits together

```
dp[i] += dp[i-2]
```

---

### Final Transition Formula

```
dp[i] = 0
if s[i-1] != '0':
    dp[i] += dp[i-1]
if 10 <= int(s[i-2:i]) <= 26:
    dp[i] += dp[i-2]
```

---

## DP Table Construction (Example)

### Example Input

```
s = "226"
```

### Step-by-step DP Table

| i | substring | single digit valid? | two digit valid? | dp[i] |
| --- | --- | --- | --- | --- |
| 0 | "" | — | — | 1 |
| 1 | "2" | yes | — | 1 |
| 2 | "22" | yes | yes ("22") | 2 |
| 3 | "226" | yes | yes ("26") | 3 |

---

### Explanation

* `"2 | 2 | 6"`
* `"22 | 6"`
* `"2 | 26"`

Total = **3 ways**

---

## DP Array Visualization

```
Index (i):     0   1   2   3
String:        ""  2   22  226
dp[i]:         1   1   2   3
```

---

## Python 3 Implementation (with typing)

```python
from typing import List

class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0

        n = len(s)
        dp: List[int] = [0] * (n + 1)

        dp[0] = 1
        dp[1] = 1

        for i in range(2, n + 1):
            # Single digit decode
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]

            # Two digit decode
            two_digit = int(s[i - 2:i])
            if 10 <= two_digit <= 26:
                dp[i] += dp[i - 2]

        return dp[n]
```

---

## Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | `O(n)` |
| Space | `O(n)` (can be optimized to `O(1)`) |

---

## Common Edge Cases

| Input | Output | Reason |
| --- | --- | --- |
| `"0"` | 0 | Cannot decode |
| `"06"` | 0 | Leading zero |
| `"10"` | 1 | Only `"J"` |
| `"101"` | 1 | `"10 |

---

## Mental Model to Remember

> **Each index asks:**  
> “Can I decode **1 digit back**?”  
> “Can I decode **2 digits back**?”

If you want, I can also provide:

* O(1) space optimized version
* Backtracking + memo comparison
* Visualization with recursion tree
* Why greedy fails here