---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 301: Remove Invalid Parentheses"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 301: Remove Invalid Parentheses

## LeetCode 301 — Remove Invalid Parentheses

---

## Problem Statement

Given a string `s` that contains parentheses `'('`, `')'` and lowercase English letters, remove the **minimum number of invalid parentheses** to make the input string valid.

Return **all possible results**.  
The order of results does not matter.

**Validity Rule**

* Every `'('` must have a corresponding `')'`
* At no point should `')'` exceed `'('` when scanning left to right

**Example**

```text
Input:  s = "()())()"
Output: ["()()()", "(())()"]
```

---

## Key Observations (Critical for LC 301)

1. This is **NOT** a standard generate-all-and-filter problem  
   → Search space is exponential, so **pruning is mandatory**.
2. We must:

   * Remove the **minimum number** of parentheses
   * Generate **all distinct valid strings** with that minimum removal
3. First compute:

   * `left_rem`: number of extra `'('` to remove
   * `right_rem`: number of extra `')'` to remove
4. Backtracking decisions:

   * For each parenthesis, we can **either keep it or remove it**
   * Removal is only allowed if we still have removals left
5. Validity must be enforced **during construction**:

   * At any time: `close_count <= open_count`

---

## High-Level Strategy

### Two-Phase Solution

1. **Preprocessing pass**

   * Determine minimum removals needed
2. **Backtracking DFS**

   * Generate only valid strings using exactly those removals

This guarantees:

* Minimal removals
* No duplicate answers
* Early pruning of invalid paths

---

## Python 3 Solution (with Typing)

```python
from typing import List, Set

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        # Step 1: calculate minimum removals
        left_rem = right_rem = 0
        for ch in s:
            if ch == '(':
                left_rem += 1
            elif ch == ')':
                if left_rem > 0:
                    left_rem -= 1
                else:
                    right_rem += 1

        result: Set[str] = set()

        def backtrack(
            index: int,
            open_count: int,
            close_count: int,
            left_rem: int,
            right_rem: int,
            path: List[str]
        ) -> None:
            # End of string
            if index == len(s):
                if left_rem == 0 and right_rem == 0:
                    result.add("".join(path))
                return

            ch = s[index]

            # Option 1: Remove current parenthesis (if possible)
            if ch == '(' and left_rem > 0:
                backtrack(index + 1, open_count, close_count, left_rem - 1, right_rem, path)
            if ch == ')' and right_rem > 0:
                backtrack(index + 1, open_count, close_count, left_rem, right_rem - 1, path)

            # Option 2: Keep current character
            path.append(ch)

            if ch not in "()":
                backtrack(index + 1, open_count, close_count, left_rem, right_rem, path)
            elif ch == '(':
                backtrack(index + 1, open_count + 1, close_count, left_rem, right_rem, path)
            elif ch == ')' and close_count < open_count:
                backtrack(index + 1, open_count, close_count + 1, left_rem, right_rem, path)

            path.pop()

        backtrack(0, 0, 0, left_rem, right_rem, [])
        return list(result)
```

---

## Example Walkthrough

### Input

```
s = "()())()"
```

### Step 1: Compute removals

```
Extra ')' = 1
Extra '(' = 0
```

So:

```
left_rem = 0
right_rem = 1
```

---

### Step 2: DFS with pruning

We explore:

* Remove **exactly one `')'`**
* Maintain validity (`close_count ≤ open_count`)
* Deduplicate using a set

Valid results:

```
()()()
(())()
```

---

## Backtracking Tree — Conceptual (Complete View)

> Below is a **conceptual navigation tree**, not index-exact, showing how choices branch and prune.

![https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AybdSPpr2CUOoWgAwynrxrw.png?utm_source=chatgpt.com](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AybdSPpr2CUOoWgAwynrxrw.png?utm_source=chatgpt.com)

![https://zxi.mytechroad.com/blog/wp-content/uploads/2017/12/301-ep139-1-1.png?utm_source=chatgpt.com](https://zxi.mytechroad.com/blog/wp-content/uploads/2017/12/301-ep139-1-1.png?utm_source=chatgpt.com)

![https://i.sstatic.net/ufjSk.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/1zGaGnicu5mVUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw7O9M7wtog3z4jKcHV3M8gpcCvP8CjOCasKNwsvSg9zMXEuLMkyTwoJNq1I0XXN0Y2sUCsGAIQ3Jn0?utm_source=chatgpt.com)

### Conceptual Tree (Simplified)

```
"()())()"
                         / remove )           \ keep )
                     "()()()"                "()())()"
                        ✓                   /        \
                                   remove )          keep )
                                "(())()" ✓        ❌ invalid
```

### What Happens in the Tree

* **Remove branches** consume `left_rem` / `right_rem`
* **Keep branches** must maintain validity
* Any path where:

  * removals exceed allowed
  * or `close_count > open_count`  
    → **immediately pruned**

---

## Why This Works (Key Insight)

| Constraint | Enforcement |
| --- | --- |
| Minimum removals | Precomputed `left_rem`, `right_rem` |
| Valid parentheses | `close_count ≤ open_count` |
| No duplicates | Result stored in `set` |
| Performance | Early pruning |

This converts a brute-force `2^n` problem into a **highly pruned DFS**.

---

## Complexity Analysis

* **Time Complexity:** Exponential in worst case, but heavily pruned
* **Space Complexity:** `O(n)` recursion depth

---

## Pattern Recognition

LeetCode 301 is a **Hard backtracking + pruning** problem that combines:

* Constraint tracking
* Duplicate avoidance
* Minimal-edit guarantees

It is a canonical example of:

> **“Search with bounded removals and validity constraints”**

---

### Interview-Ready One-Liner

> “We first compute the minimum number of invalid parentheses, then use DFS to remove exactly those many while maintaining validity and pruning invalid branches.”

If you want next, I can:

* Compare **DFS vs BFS** solutions for LC 301
* Show **what breaks if pruning is removed**
* Provide a **generic template** for minimal-removal backtracking problems