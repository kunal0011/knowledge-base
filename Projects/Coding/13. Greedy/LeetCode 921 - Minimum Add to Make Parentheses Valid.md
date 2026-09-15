---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 921: Minimum Add to Make Parentheses Valid"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 921: Minimum Add to Make Parentheses Valid

Below is a complete, structured explanation of **LeetCode 921 – Minimum Add to Make Parentheses Valid**, aligned with how this problem is typically reasoned in interviews and competitive programming.

---

## 1. Problem Statement

**LeetCode 921 — Minimum Add to Make Parentheses Valid**

You are given a string `s` consisting only of the characters `'('` and `')'`.

A parentheses string is **valid** if:

1. Every opening parenthesis `'('` has a corresponding closing parenthesis `')'`.
2. Parentheses are closed in the correct order.

You may add parentheses at **any position** in the string.

**Return the minimum number of parentheses you must add to make the string valid.**

---

### Example

```text
Input:  s = "()))(("
Output: 4
```

---

## 2. Key Observation

A valid parentheses string satisfies **two invariants**:

1. At no point should closing parentheses exceed opening ones  
   → prefix condition
2. Total number of `'('` must equal total number of `')'`  
   → global balance condition

This naturally suggests **tracking balance while scanning the string once**.

---

## 3. Greedy Insight (Core Trick)

We process the string from left to right and apply the following greedy logic:

### State Variables

* `balance`: number of unmatched `'('` seen so far
* `additions`: number of parentheses we must insert

### Greedy Rules

1. If we see `'('`  
   → increment `balance`
2. If we see `')'`:

   * If `balance > 0`  
     → match it with an existing `'('` → decrement `balance`
   * Else (`balance == 0`)  
     → this `')'` has no matching `'('`  
     → **we must add one `'('`** before it  
     → increment `additions`

### After full traversal

* Any remaining `balance` represents unmatched `'('`
* Each requires one `')'` to close

```
Total additions = additions + balance
```

---

## 4. Why Greedy Works

* Every unmatched `')'` **must** be fixed immediately (you cannot match it later).
* Delaying fixes only increases future work.
* Matching whenever possible minimizes insertions.
* Remaining `'('` can only be closed by adding `')'`.

This is a classic **local-optimal ⇒ global-optimal** greedy structure.

---

## 5. Python 3 Solution (with Typing)

```python
from typing import *

class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        balance: int = 0      # unmatched '('
        additions: int = 0    # required insertions
        
        for ch in s:
            if ch == '(':
                balance += 1
            else:  # ch == ')'
                if balance > 0:
                    balance -= 1
                else:
                    additions += 1  # need to add '(' before this ')'
        
        # remaining '(' need ')'
        return additions + balance
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
s = "()))(("
```

### Initialization

```
balance = 0
additions = 0
```

---

### Step-by-step Processing

| Index | Char | Action | balance | additions | Explanation |
| --- | --- | --- | --- | --- | --- |
| 0 | '(' | balance += 1 | 1 | 0 | New open |
| 1 | ')' | balance -= 1 | 0 | 0 | Matched |
| 2 | ')' | additions += 1 | 0 | 1 | No '(' to match |
| 3 | ')' | additions += 1 | 0 | 2 | No '(' to match |
| 4 | '(' | balance += 1 | 1 | 2 | Open |
| 5 | '(' | balance += 1 | 2 | 2 | Open |

---

### After Traversal

```
balance = 2   # two unmatched '('
additions = 2 # two unmatched ')'
```

### Final Calculation

```
Total = additions + balance
      = 2 + 2
      = 4
```

---

## 7. Visual Interpretation

Original:

```
()))((
```

Fixes needed:

```
( ( ) ) ) ( ( )
^   ^           ^
add '('     add ')'
```

---

## 8. Time & Space Complexity

* **Time:** `O(n)` — single pass
* **Space:** `O(1)` — constant extra space

---

## 9. Interview Takeaways

* This problem is **not a stack problem**, despite appearances.
* Balance tracking is sufficient.
* Greedy works because invalid `')'` cannot be fixed later.
* Very common question to test prefix constraints and greedy reasoning.

---

If you want, I can also:

* Show a **stack-based solution and why it’s inferior**
* Convert this into a **formal invariant-based proof**
* Compare with **LeetCode 1541 / 1249 variations**

Just tell me how deep you want to go.