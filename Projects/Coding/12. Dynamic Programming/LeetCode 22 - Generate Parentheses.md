---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 22: Generate Parentheses"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 22: Generate Parentheses

**LeetCode 22 – Generate Parentheses**, with **explicit state definition, transition logic, DP table construction, and a worked example**.  
This is intentionally presented **without backtracking**, to highlight the DP perspective.

---

## Problem Statement (Recap)

Given an integer `n`, generate all combinations of `n` pairs of **well-formed parentheses**.

**Input:** `n = 3`  
**Output:**

```
["((()))","(()())","(())()","()(())","()()()"]
```

---

## Key DP Observation

A valid parentheses string of `n` pairs can be **constructed by splitting it into two valid parts**:

```
("(" + left_substring + ")" + right_substring)
```

Where:

* `left_substring` uses `k` pairs
* `right_substring` uses `n - 1 - k` pairs
* `k` ranges from `0` to `n-1`

This is structurally identical to **Catalan-number-style decomposition**.

---

## DP State Definition

### State

```
dp[i] = list of all valid parentheses strings using exactly i pairs
```

### Base Case

```
dp[0] = [""]
```

(Empty string is a valid representation of 0 pairs)

---

## State Transition

For `i >= 1`:

```
dp[i] = []
for k in range(0, i):
    for left in dp[k]:
        for right in dp[i - 1 - k]:
            dp[i].append("(" + left + ")" + right)
```

### Why this works

* The **first opening parenthesis** must match some closing parenthesis.
* Everything inside that pair must be valid (`dp[k]`)
* Everything after must also be valid (`dp[i - 1 - k]`)

This ensures **correct nesting and ordering**.

---

## DP Table Construction (Step-by-Step Example)

### Step 0

```
dp[0] = [""]
```

---

### Step 1 (i = 1)

Possible splits:

```
k = 0
("(" + dp[0] + ")" + dp[0])
```

Result:

```
dp[1] = ["()"]
```

---

### Step 2 (i = 2)

#### k = 0

```
("(" + "" + ")" + "()") → "()()"
```

#### k = 1

```
("(" + "()" + ")" + "") → "(())"
```

Result:

```
dp[2] = ["()()", "(())"]
```

---

### Step 3 (i = 3)

| k | Left (dp[k]) | Right (dp[2-k]) | Generated |
| --- | --- | --- | --- |
| 0 | "" | dp[2] | "()()", "(())" |
| 1 | "()" | dp[1] | "(())()" |
| 2 | dp[2] | "" | "(()())", "((()))" |

Full construction:

```
k=0 → "(" + "" + ")" + "()()" → "()()()"
     → "(" + "" + ")" + "(())" → "()(())"

k=1 → "(" + "()" + ")" + "()" → "(())()"

k=2 → "(" + "()()" + ")" → "(()())"
     → "(" + "(())" + ")" → "((()))"
```

Final:

```
dp[3] = ["()()()", "()(())", "(())()", "(()())", "((()))"]
```

---

## DP Table Summary

| i | dp[i] |
| --- | --- |
| 0 | [""] |
| 1 | ["()"] |
| 2 | ["()()", "(())"] |
| 3 | ["()()()", "()(())", "(())()", "(()())", "((()))"] |

---

## Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        dp: List[List[str]] = [[] for _ in range(n + 1)]
        dp[0] = [""]

        for i in range(1, n + 1):
            for k in range(i):
                for left in dp[k]:
                    for right in dp[i - 1 - k]:
                        dp[i].append("(" + left + ")" + right)

        return dp[n]
```

---

## Time & Space Complexity

* **Time:**  
  `O(Cn)` where `Cn` is the `n`th Catalan number
* **Space:**  
  `O(Cn)` for storing all valid strings

---

## DP vs Backtracking (Conceptual Difference)

| Aspect | DP | Backtracking |
| --- | --- | --- |
| Idea | Build from smaller valid solutions | Explore character-by-character |
| State | dp[i] = all valid strings of size i | (open, close, string) |
| Overlaps | Explicitly reused | Implicit via recursion |
| Use Case | Structural enumeration | Constraint-driven generation |

---

If you want next:

* DP → **Catalan number mathematical derivation**
* Visual **DP construction tree**
* Comparison with **backtracking tree (state-space)**

Say the word.