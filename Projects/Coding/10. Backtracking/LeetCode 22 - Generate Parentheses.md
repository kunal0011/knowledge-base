---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 22: Generate Parentheses"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 22: Generate Parentheses

## LeetCode 22 — Generate Parentheses

### Problem Statement

Given an integer `n`, generate **all combinations of well-formed parentheses** consisting of `n` pairs of parentheses.

**Constraints**

* `1 <= n <= 8`

**Example**

* Input: `n = 3`
* Output:

```
["((()))","(()())","(())()","()(())","()()()"]
```

---

### Key Observations

1. A valid parentheses string must satisfy **two invariants** at every prefix:

   * The number of `'('` used so far must be **≤ n**.
   * The number of `')'` used so far must be **≤ the number of '('** used so far.
2. Any prefix that violates these rules **can never lead** to a valid solution and should be pruned immediately.
3. This structure naturally maps to **backtracking / DFS**, where:

   * Each decision adds either `'('` or `')'`.
   * Invalid branches are cut early (pruning).
4. Total valid outputs correspond to the **Catalan number** `Cₙ`, which grows much slower than `2^(2n)` due to pruning.

---

### Approach (Backtracking)

We build the string step by step:

* Start with an empty string.
* Track:

  * `open_count`: number of `'('` used
  * `close_count`: number of `')'` used
* Recursive choices:

  * Add `'('` if `open_count < n`
  * Add `')'` if `close_count < open_count`
* Stop when the string length becomes `2 * n`.

This ensures:

* Only valid prefixes are explored.
* Time complexity is proportional to the number of valid results.

---

### Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result: List[str] = []

        def backtrack(current: str, open_count: int, close_count: int) -> None:
            # Base case: valid combination completed
            if len(current) == 2 * n:
                result.append(current)
                return

            # Try adding '(' if possible
            if open_count < n:
                backtrack(current + "(", open_count + 1, close_count)

            # Try adding ')' if it keeps the string valid
            if close_count < open_count:
                backtrack(current + ")", open_count, close_count + 1)

        backtrack("", 0, 0)
        return result
```

---

### Example Walkthrough (`n = 2`)

We want all valid strings of length `4`.

1. Start: `""`
2. Add `'('` → `"("`
3. Add `'('` → `"(("`
4. Can’t add `'('` (limit reached), add `')'` → `"(()"`
5. Add `')'` → `"(())"` ✓ valid
6. Backtrack to `"("`
7. Add `')'` → `"()"`
8. Add `'('` → `"()("`
9. Add `')'` → `"()()"` ✓ valid

**Final Output**

```
["(())", "()()"]
```

---

### Backtracking Tree Structure (for `n = 3`)

![https://i.sstatic.net/heAM3.png?utm_source=chatgpt.com](https://images.openai.com/static-rsc-1/SZa4OSUwo3JyAdp6_PLSjB4nV6VTrDvQ7S0l2MZX7-7I-KuO0vfG0N2CVDlNVFTokQ8BXV8ZgPXJupe2K80tSgo1HalTWO634WMTCQLxN73F0Bz_uWGaLgZF7RPn7XWmhqrfG0ZZQYAReUEn1s7z-A?utm_source=chatgpt.com)

![https://i.sstatic.net/ufjSk.png?utm_source=chatgpt.com](https://images.openai.com/static-rsc-1/4rqNUH8-1MngmZfGIFf9qjrZec9d1rzijWwbeLBTMUkX325lIutx8pmTaGTcncaFt23kyIzyubFMI3aE29o_YykyCwiojPUa3TnBjzDPnkVhC02XmLnW-ragOAlumPtgFuv5eQ-HNH_gJ9JsAMnlYA?utm_source=chatgpt.com)

![https://miro.medium.com/1%2Aj4s0A8Sve6Cyy3z3iWnyGQ.jpeg?utm_source=chatgpt.com](https://miro.medium.com/1%2Aj4s0A8Sve6Cyy3z3iWnyGQ.jpeg?utm_source=chatgpt.com)

**Tree Interpretation**

* Each node represents a prefix string.
* Left edge → add `'('`
* Right edge → add `')'`
* Branches are **pruned immediately** when `')' > '('`.
* Leaf nodes at depth `2n` are valid answers.

Example partial tree:

```
""
                  |
                 "("
               /      \
            "(("        "()"
           /    \          \
      "((("    "(()"       "()("
         |       |            |
     "((()"   "(())"        "()()"
         |       |
     "((())"   "(())()"
         |
     "((()))"
```

---

### Complexity Analysis

* **Time Complexity:** `O(Cₙ)` where `Cₙ` is the nth Catalan number.
* **Space Complexity:** `O(n)` recursion depth (excluding output storage).

---

### Why This Is a Canonical Backtracking Problem

* Clear **constraints-based pruning**
* Builds solutions incrementally
* Guarantees correctness by construction
* Forms the foundation for many other problems:

  * Valid combinations
  * Balanced strings
  * Constraint-satisfaction DFS problems