---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 1249: Minimum Remove to Make Valid Parentheses"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 1249: Minimum Remove to Make Valid Parentheses

Below is a complete, interview-ready explanation of **LeetCode 1249 – Minimum Remove to Make Valid Parentheses**, structured exactly as requested.

---

## LeetCode 1249: Minimum Remove to Make Valid Parentheses

### Problem Statement

You are given a string `s` consisting of lowercase English letters and parentheses `'('` and `')'`.

A string is **valid** if:

1. Every opening parenthesis `'('` has a corresponding closing parenthesis `')'`.
2. Parentheses are closed in the correct order.

Your task is to **remove the minimum number of parentheses** so that the resulting string is valid and return **any valid result**.

---

### Key Observation

* Letters are always valid and should **never be removed**.
* Invalidity comes from:

  1. A `')'` that appears **without a matching `'('` before it**.
  2. A `'('` that is **never closed** by a corresponding `')'`.

Hence, we only need to identify and remove **unmatched parentheses**.

---

### Stack – Key Insight

The stack helps track **indices of unmatched parentheses**.

Core idea:

* Traverse the string once.
* Use a stack to store **indices of `'('`**.
* When encountering `')'`:

  * If the stack is not empty → match it with a `'('` (pop stack).
  * If the stack is empty → this `')'` is invalid.
* After traversal:

  * Any remaining `'('` indices in the stack are invalid.

All invalid indices are removed.

This guarantees **minimum removals**, because we only delete parentheses that cannot be matched.

---

### Algorithm Steps

1. Initialize:

   * `stack` → stores indices of `'('`
   * `invalid` → stores indices to remove
2. Traverse the string:

   * `'('` → push index to stack
   * `')'`:

     * stack empty → mark index invalid
     * else → pop stack
3. Add all remaining stack indices (unmatched `'('`) to invalid
4. Build result string excluding invalid indices

---

### Python 3 Solution (with typing)

```python
from typing import List, Set

class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack: List[int] = []
        invalid: Set[int] = set()

        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            elif ch == ')':
                if stack:
                    stack.pop()
                else:
                    invalid.add(i)

        # Remaining '(' in stack are invalid
        invalid.update(stack)

        # Build valid string
        result = []
        for i, ch in enumerate(s):
            if i not in invalid:
                result.append(ch)

        return ''.join(result)
```

---

### Worked Example (Step-by-Step)

#### Input

```
s = "a)b(c)d"
```

#### Index Mapping

```
Index: 0 1 2 3 4 5 6
Char : a ) b ( c ) d
```

---

#### Traversal

| Index | Char | Stack | Invalid |
| --- | --- | --- | --- |
| 0 | a | [] | {} |
| 1 | ) | [] | {1} |
| 2 | b | [] | {1} |
| 3 | ( | [3] | {1} |
| 4 | c | [3] | {1} |
| 5 | ) | [] | {1} |
| 6 | d | [] | {1} |

* Index `1` is an invalid `')'`
* All parentheses matched afterward

---

#### Final Construction

Remove index `1`:

```
"a)b(c)d" → "ab(c)d"
```

---

### Final Output

```
"ab(c)d"
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)` (stack + invalid indices)

---

### Why Stack Is the Optimal Choice

* Naturally models nested parentheses
* Guarantees minimal deletions
* Works in a single pass
* Simple and robust for interview scenarios

---

If you want:

* Two-pass greedy approach (without stack)
* Visualization with stack state per character
* Variations of this problem and pattern classification

I can provide those next.