---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 44: Wildcard Matching"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 44: Wildcard Matching

**LeetCode 44 – Wildcard Matching**, covering the problem statement, key observations, the greedy strategy (why it works), a Python 3 implementation with typing, and a fully worked example.

---

## 1. Problem Statement (LeetCode 44 – Wildcard Matching)

You are given:

* A string `s`
* A pattern `p`

The pattern supports two special wildcard characters:

* `?` → matches **exactly one** character
* `*` → matches **any sequence of characters** (including the empty sequence)

### Objective

Return `True` if the pattern `p` matches the **entire** string `s`. Otherwise, return `False`.

### Constraints (important for solution choice)

* `1 ≤ len(s), len(p) ≤ 2000`
* Must match the **whole string**, not a substring

---

## 2. Key Observations

1. This is **not regex matching**  
   Only `?` and `*` are supported, with very specific semantics.
2. `*` is the only character that introduces ambiguity

   * It can match zero characters
   * Or one
   * Or many
3. A naive recursive or DP solution works but:

   * DP is `O(n × m)` time and space
   * Can be optimized, but still heavy
4. A **greedy two-pointer approach** works in linear time  
   This is non-trivial but relies on a crucial observation:

   * Only `*` can “absorb” mismatches
   * The **last seen `*`** can always be expanded if needed

---

## 3. Greedy Strategy – Core Insight

### Idea

Use two pointers:

* `i` → pointer for string `s`
* `j` → pointer for pattern `p`

Additionally:

* `star_idx` → index of the **last `*`** seen in `p`
* `match` → index in `s` where the last `*` started matching

### Matching Rules

1. **Exact match or `?`**

   * If `p[j] == s[i]` or `p[j] == '?'`
   * Move both pointers forward
2. **Encounter `*`**

   * Record:

     * `star_idx = j`
     * `match = i`
   * Move pattern pointer `j` forward
   * Initially assume `*` matches **empty**
3. **Mismatch**

   * If we have seen a `*` before:

     * Backtrack:

       * Let the `*` match **one more character**
       * Increment `match`
       * Set `i = match`
       * Set `j = star_idx + 1`
   * Else:

     * No way to fix mismatch → return `False`
4. **End of string**

   * Remaining pattern characters must all be `*`

---

## 4. Why the Greedy Works

* Only `*` can compensate for mismatches
* Always expanding the **most recent `*`** is sufficient
* Earlier `*` choices do not need reconsideration
* This avoids exponential backtracking

Time Complexity: **O(n + m)**  
Space Complexity: **O(1)**

---

## 5. Python 3 Solution (with Typing)

```python
from typing import *

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        i: int = 0  # pointer for s
        j: int = 0  # pointer for p
        star_idx: int = -1
        match: int = 0

        while i < len(s):
            # Case 1: exact match or '?'
            if j < len(p) and (p[j] == s[i] or p[j] == '?'):
                i += 1
                j += 1

            # Case 2: '*'
            elif j < len(p) and p[j] == '*':
                star_idx = j
                match = i
                j += 1

            # Case 3: mismatch but we have seen '*'
            elif star_idx != -1:
                j = star_idx + 1
                match += 1
                i = match

            # Case 4: mismatch and no '*'
            else:
                return False

        # Remaining pattern characters must be all '*'
        while j < len(p) and p[j] == '*':
            j += 1

        return j == len(p)
```

---

## 6. Complete Worked Example

### Example

```
s = "adceb"
p = "*a*b"
```

### Step-by-step Execution

| i (s) | j (p) | s[i] | p[j] | Action |
| --- | --- | --- | --- | --- |
| 0 | 0 | a | \* | star\_idx = 0, match = 0 |
| 0 | 1 | a | a | match → i=1, j=2 |
| 1 | 2 | d | \* | star\_idx = 2, match = 1 |
| 1 | 3 | d | b | mismatch → expand `*` |
| 2 | 3 | c | b | mismatch → expand `*` |
| 3 | 3 | e | b | mismatch → expand `*` |
| 4 | 3 | b | b | match → i=5, j=4 |

Now:

* `i == len(s)`
* `j == len(p)`

### Result

```
True
```

The second `*` successfully absorbed `"dce"`.

---

## 7. Common Pitfalls

1. Forgetting to skip remaining `*` at the end of pattern
2. Trying to backtrack multiple `*` levels (unnecessary)
3. Assuming greedy fails — this specific problem is proven greedy-solvable

---

## 8. When to Prefer DP Instead

* If wildcard rules change
* If partial matching is allowed
* If constraints include additional operators

For the given problem, **greedy is optimal and preferred**.

---

If you want, I can also:

* Provide the **DP solution with table visualization**
* Compare **DP vs Greedy**
* Explain **why greedy fails for regex but works here**
* Add **dry-run diagrams for multiple cases**