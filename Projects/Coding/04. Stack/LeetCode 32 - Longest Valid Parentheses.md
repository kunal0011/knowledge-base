---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 32: Longest Valid Parentheses"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 32: Longest Valid Parentheses

Below is a **complete, structured explanation of LeetCode 32 – Longest Valid Parentheses**, aligned with how this problem is typically reasoned in interviews and competitive programming.

---

## 1. Problem Statement (LeetCode 32)

Given a string `s` consisting of only `'('` and `')'`, return the **length of the longest valid (well-formed) parentheses substring**.

A substring is valid if:

* Every opening `'('` has a corresponding closing `')'`
* Parentheses are properly nested

**Examples**

* Input: `"(()"` → Output: `2` (`"()"`)
* Input: `")()())"` → Output: `4` (`"()()"`)
* Input: `""` → Output: `0`

---

## 2. Key Observation

1. A valid parentheses substring must:

   * Never have more `')'` than `'('` at any prefix
   * End at a position where parentheses are balanced
2. The problem is **not** about counting pairs globally — it is about finding the **maximum length contiguous substring**that satisfies validity constraints.
3. Whenever we encounter an **invalid closing parenthesis `')'`**, all substrings extending across that index become invalid.

---

## 3. Stack-Based Key Insight (Core Idea)

### Why Stack?

* Parentheses matching is inherently **LIFO**
* Stack allows us to:

  * Track indices of unmatched `'('`
  * Compute substring lengths using index differences

### Critical Insight

Instead of pushing parentheses characters, **push indices**.

We also:

* Push `-1` initially as a **base index**
* This helps compute length when a valid substring starts from index `0`

---

## 4. Stack Algorithm (Step-by-Step Logic)

1. Initialize stack with `-1`
2. Iterate through the string with index `i`
3. If `s[i] == '('`:

   * Push index `i`
4. Else (`s[i] == ')'`):

   * Pop from stack
   * If stack becomes empty:

     * Push current index `i` (new base)
   * Else:

     * Valid substring length = `i - stack[-1]`
     * Update maximum length

---

## 5. Python 3 Solution (with `typing`)

```python
from typing import List

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack: List[int] = [-1]  # base index
        max_len = 0

        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            else:
                stack.pop()

                if not stack:
                    # invalid closing, reset base
                    stack.append(i)
                else:
                    max_len = max(max_len, i - stack[-1])

        return max_len
```

---

## 6. Worked-Out Example (Detailed Walkthrough)

### Input

```
s = ")()())"
```

### Initialization

```
stack = [-1]
max_len = 0
```

---

### Step-by-Step Execution

| Index | Char | Action | Stack | Current Length | max\_len |
| --- | --- | --- | --- | --- | --- |
| 0 | `)` | pop → empty → push 0 | [0] | — | 0 |
| 1 | `(` | push 1 | [0, 1] | — | 0 |
| 2 | `)` | pop → len = 2 - 0 = 2 | [0] | 2 | 2 |
| 3 | `(` | push 3 | [0, 3] | — | 2 |
| 4 | `)` | pop → len = 4 - 0 = 4 | [0] | 4 | 4 |
| 5 | `)` | pop → empty → push 5 | [5] | — | 4 |

---

### Final Answer

```
Longest valid parentheses length = 4
```

Substring:

```
"()()"
```

---

## 7. Why `-1` Is Important

Example: `"()"`

Without `-1`:

* Stack becomes empty after matching
* Length calculation breaks

With `-1`:

```
i - stack[-1] = 1 - (-1) = 2
```

This ensures correct length calculation from the beginning.

---

## 8. Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)` (stack)

---

## 9. When to Use This Pattern

* Matching parentheses/brackets
* Longest valid substring problems
* Problems requiring **index-based span calculation**

---

If you want, I can also:

* Show **DP solution** and compare with stack
* Draw a **stack evolution diagram**
* Explain why **two-pointer** fails here

Just tell me.