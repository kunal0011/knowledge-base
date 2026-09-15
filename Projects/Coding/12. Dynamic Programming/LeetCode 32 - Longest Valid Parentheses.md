---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 32: Longest Valid Parentheses"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 32: Longest Valid Parentheses

**LeetCode 32 – Longest Valid Parentheses**, covering **state definition, transition logic, DP table construction, and a worked example**.

---

## Problem Statement (LeetCode 32)

Given a string `s` consisting of `'('` and `')'`, return the length of the **longest valid (well-formed) parentheses substring**.

**Valid parentheses rules**

* Every `'('` must have a matching `')'`
* Parentheses must be correctly nested

---

## Why Dynamic Programming Works Here

A valid parentheses substring **must end at a `')'`**.  
Therefore, DP is ideal if we track:

> “What is the longest valid parentheses substring **ending at index i**?”

This allows us to **reuse previously computed valid segments**.

---

## DP State Definition

Let:

```
dp[i] = length of the longest valid parentheses substring ending at index i
```

### Key Properties

* `dp[i] = 0` if `s[i] == '('` (cannot end a valid substring)
* We only compute `dp[i]` when `s[i] == ')'`

---

## State Transition

There are **two valid structural cases** when `s[i] == ')'`.

---

### Case 1: Simple Pair "()"

```
... ()
     i
```

Condition:

```
s[i] == ')' and s[i - 1] == '('
```

Transition:

```
dp[i] = 2 + dp[i - 2]
```

Explanation:

* `()` contributes `2`
* Add any valid substring ending at `i - 2`

---

### Case 2: Nested or Extended Pattern "))"

```
... (  valid  )
            i
```

Condition:

```
s[i] == ')' and s[i - 1] == ')'
```

We check if there is a matching `'('` **before** the previous valid substring.

Index to check:

```
j = i - dp[i - 1] - 1
```

If:

```
j >= 0 and s[j] == '('
```

Then:

```
dp[i] = dp[i - 1] + 2 + dp[j - 1]
```

Explanation:

* `dp[i - 1]` → previous valid block
* `+2` → matching parentheses
* `dp[j - 1]` → valid substring before that block

---

## Full DP Transition Formula

```
if s[i] == ')':
    if s[i - 1] == '(':
        dp[i] = 2 + dp[i - 2]
    else if s[i - 1] == ')':
        j = i - dp[i - 1] - 1
        if j >= 0 and s[j] == '(':
            dp[i] = dp[i - 1] + 2 + dp[j - 1]
```

---

## DP Table Construction Example

### Input

```
s = ")()())"
```

Indices:

```
 i:  0 1 2 3 4 5
 s:  ) ( ) ( ) )
```

Initialize:

```
dp = [0, 0, 0, 0, 0, 0]
```

---

### Step-by-Step Table Fill

| i | s[i] | dp[i] | Explanation |
| --- | --- | --- | --- |
| 0 | `)` | 0 | Cannot form valid substring |
| 1 | `(` | 0 | Cannot end with '(' |
| 2 | `)` | 2 | "()" → dp[2] = 2 |
| 3 | `(` | 0 | Cannot end with '(' |
| 4 | `)` | 4 | "( )" + dp[2] → "()()" |
| 5 | `)` | 0 | No matching '(' |

Final DP table:

```
dp = [0, 0, 2, 0, 4, 0]
```

---

## Final Answer

```
max(dp) = 4
```

Longest valid parentheses substring is:

```
"()()"
```

---

## Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n: int = len(s)
        dp: List[int] = [0] * n
        max_len: int = 0

        for i in range(1, n):
            if s[i] == ')':
                # Case 1: "()"
                if s[i - 1] == '(':
                    dp[i] = 2
                    if i >= 2:
                        dp[i] += dp[i - 2]

                # Case 2: "))"
                else:
                    j = i - dp[i - 1] - 1
                    if j >= 0 and s[j] == '(':
                        dp[i] = dp[i - 1] + 2
                        if j >= 1:
                            dp[i] += dp[j - 1]

                max_len = max(max_len, dp[i])

        return max_len
```

---

## Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(n)** |

---

## Key DP Insight (Important for Interviews)

> A valid parentheses substring **must end at `')'`**, and every valid segment can be built by **extending a smaller valid segment**.

---

If you want next:

* **DP vs Stack comparison**
* **Visual DP tree / interval expansion**
* **Optimized O(1) space solution**
* **Why DP handles nested cases better than brute force**

Tell me how deep you want to go.