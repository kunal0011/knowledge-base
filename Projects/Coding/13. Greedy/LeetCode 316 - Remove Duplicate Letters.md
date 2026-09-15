---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 316: Remove Duplicate Letters"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 316: Remove Duplicate Letters

**LeetCode 316 – Remove Duplicate Letters**, structured exactly as requested.

---

## LeetCode 316: Remove Duplicate Letters

### Problem Statement

Given a string `s`, remove duplicate letters so that **every letter appears exactly once** and the resulting string is **the smallest in lexicographical order** among all possible results.

**Constraints**

* `1 <= s.length <= 10^4`
* `s` consists of lowercase English letters.

**Example**

```text
Input:  s = "cbacdcbc"
Output: "acdb"
```

---

## Key Observations

1. **Each character must appear exactly once**

   * This is a **set constraint**.
2. **Lexicographically smallest result**

   * Among all valid permutations, we want the smallest string in dictionary order.
3. **Relative order matters**

   * We cannot arbitrarily sort characters.
   * We must **preserve relative ordering** of characters from the original string.
4. **Greedy choice**

   * At each step, we should:

     * Prefer a **smaller character**
     * But only if the removed character **appears again later**

---

## Greedy Strategy (Core Insight)

We build the result **incrementally** using a **monotonic stack**.

### When processing a character `c`:

We can safely remove the top character `x` from the stack **if all three conditions hold**:

1. `x > c`  
   → Removing `x` improves lexicographical order
2. `x` appears again later  
   → We can reinsert it later
3. `c` is not already in the result  
   → Ensures uniqueness

This guarantees:

* Lexicographically smallest sequence
* Each character appears once
* Order constraints are preserved

---

## Greedy Solution Tricks (Interview Gold)

### Trick 1: Last Occurrence Map

Store the **last index** of every character to know if it appears again.

```
last = {char: index}
```

### Trick 2: Stack for Building Answer

Use a stack to maintain characters in increasing lexicographical order.

### Trick 3: Visited Set

Ensure each character is added **only once**.

---

## Python 3 Solution (With Typing)

```python
from typing import Dict, List

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        last_occurrence: Dict[str, int] = {}
        for i, ch in enumerate(s):
            last_occurrence[ch] = i

        stack: List[str] = []
        in_stack = set()

        for i, ch in enumerate(s):
            # Skip if already included
            if ch in in_stack:
                continue

            # Greedy removal
            while (
                stack and
                stack[-1] > ch and
                last_occurrence[stack[-1]] > i
            ):
                removed = stack.pop()
                in_stack.remove(removed)

            stack.append(ch)
            in_stack.add(ch)

        return "".join(stack)
```

---

## Complete Worked Example (Step-by-Step)

### Input

```
s = "cbacdcbc"
```

### Step 1: Last Occurrence Index

```
c → 7
b → 6
a → 2
d → 4
```

---

### Step 2: Process Each Character

| Index | Char | Stack Before | Action | Stack After |
| --- | --- | --- | --- | --- |
| 0 | c | [] | push c | [c] |
| 1 | b | [c] | pop c (c > b and appears later) | [b] |
| 2 | a | [b] | pop b (b > a and appears later) | [a] |
| 3 | c | [a] | push c | [a, c] |
| 4 | d | [a, c] | push d | [a, c, d] |
| 5 | c | [a, c, d] | skip (already in stack) | [a, c, d] |
| 6 | b | [a, c, d] | cannot pop d (no future d) | [a, c, d, b] |
| 7 | c | [a, c, d, b] | skip | [a, c, d, b] |

---

### Final Stack

```
[a, c, d, b]
```

### Output

```
"acdb"
```

---

## Why This Greedy Works

* We **only remove characters** when:

  * A better (smaller) character is available
  * The removed character can be reinserted later
* This guarantees:

  * Global lexicographical optimality
  * Uniqueness
  * Order preservation

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(1)** (fixed alphabet of 26 letters) |

---

## Related Problems (Same Pattern)

* **LeetCode 1081** – Smallest Subsequence of Distinct Characters
* **LeetCode 402** – Remove K Digits
* **Monotonic Stack problems**

---

If you want, I can also:

* Draw the **greedy decision tree**
* Compare this with a **DP approach (why DP is worse here)**
* Provide a **dry-run template** you can use in interviews