---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1221: Split a String in Balanced Strings"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1221: Split a String in Balanced Strings

Below is a structured, end-to-end explanation of **LeetCode 1221 – Split a String in Balanced Strings**, aligned to your requested format and level of rigor.

---

## LeetCode 1221 — Split a String in Balanced Strings

---

### Problem Statement

You are given a string `s` consisting only of the characters `'L'` and `'R'`.

A **balanced string** is defined as a string that contains an equal number of `'L'` and `'R'`.

Your task is to split the string into the **maximum number of balanced substrings**, such that:

* Each substring is balanced.
* Substrings are contiguous.
* Every character of `s` belongs to exactly one substring.

Return the **maximum number of balanced substrings** that can be obtained.

---

### Key Observation

This problem does **not** require backtracking, DP, or prefix arrays.

The critical insight is:

> As soon as the count of `'L'` and `'R'` becomes equal while scanning left to right, we can safely split the string at that position.

Why this works:

* The earliest possible split always leaves the remaining suffix as large as possible.
* Delaying a split can never increase the total number of balanced substrings.
* Therefore, a **greedy earliest-split strategy** is optimal.

---

### Greedy Solution – Core Idea

Maintain a running **balance counter**:

* Increment balance for `'L'`
* Decrement balance for `'R'`

Whenever:

```
balance == 0
```

it means:

* We have seen equal `'L'` and `'R'`
* A balanced substring ends here
* We increment our answer counter

Then we continue scanning the rest of the string.

---

### Greedy “Tricks” Used

1. **Balance Instead of Two Counters**

   * Using one integer is enough.
   * `balance = (#L so far) − (#R so far)`
2. **Earliest Valid Cut**

   * As soon as balance becomes zero, split immediately.
   * This maximizes the number of substrings.
3. **Single Pass**

   * Time Complexity: **O(n)**
   * Space Complexity: **O(1)**

---

### Python 3 Solution (With Typing)

```python
from typing import *

class Solution:
    def balancedStringSplit(self, s: str) -> int:
        balance: int = 0
        result: int = 0

        for ch in s:
            if ch == 'L':
                balance += 1
            else:  # ch == 'R'
                balance -= 1

            if balance == 0:
                result += 1

        return result
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```
s = "RLRRLLRLRL"
```

---

### Step-by-Step Processing

| Index | Character | Balance Change | Balance | Balanced Substring? | Count |
| --- | --- | --- | --- | --- | --- |
| 0 | R | -1 | -1 | No | 0 |
| 1 | L | +1 | 0 | Yes → split | 1 |
| 2 | R | -1 | -1 | No | 1 |
| 3 | R | -1 | -2 | No | 1 |
| 4 | L | +1 | -1 | No | 1 |
| 5 | L | +1 | 0 | Yes → split | 2 |
| 6 | R | -1 | -1 | No | 2 |
| 7 | L | +1 | 0 | Yes → split | 3 |
| 8 | R | -1 | -1 | No | 3 |
| 9 | L | +1 | 0 | Yes → split | 4 |

---

### Visual Split Representation

```
R L | R R L L | R L | R L
```

Each segment has equal `'L'` and `'R'`.

---

### Final Answer

```
Output = 4
```

---

### Why Greedy Is Correct (Intuition)

* Any balanced substring must end at a position where cumulative balance becomes zero.
* Taking the **earliest such position** guarantees:

  * One valid substring
  * Maximum remaining characters to form more substrings
* There is no advantage in merging two balanced substrings into one.

---

### Summary

| Aspect | Value |
| --- | --- |
| Technique | Greedy |
| Traversal | Single pass |
| Time Complexity | O(n) |
| Space Complexity | O(1) |
| Core Insight | Split immediately when balance reaches zero |

---

If you want, I can also:

* Show **counter-examples where delaying the split fails**
* Compare this with **prefix sum interpretation**
* Map this to a **stack-based mental model**

Just let me know.