---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 678: Valid Parenthesis String"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 678: Valid Parenthesis String

Below is a **complete, structured explanation of LeetCode 678 – *Valid Parenthesis String***, aligned with your usual learning pattern (problem → observation → greedy trick → code → step-by-step execution).

---

## LeetCode 678 – Valid Parenthesis String

### Problem Statement

You are given a string `s` containing only three characters:

* `'('` : opening parenthesis
* `')'` : closing parenthesis
* `'*'` : wildcard character

The wildcard `'*'` can represent **any one of the following**:

* `'('`
* `')'`
* an empty string `""`

Return **true** if the string can be interpreted as a **valid parentheses string**, otherwise return **false**.

#### Definition of a Valid Parentheses String

1. Every opening parenthesis `'('` must have a corresponding closing parenthesis `')'`.
2. Parentheses must be closed in the correct order.
3. The empty string is considered valid.

---

## Key Observations

1. **Brute force is infeasible**

   * Each `'*'` has 3 possibilities.
   * Worst case: `3^n` combinations.
2. **Stack-based solutions become complex**

   * Because `'*'` can behave differently depending on context.
3. **We do NOT need exact positions**

   * We only need to know **whether a valid interpretation exists**, not what it is.
4. **Key Insight**

   * At any point, the number of open parentheses can lie within a **range**:

     * **Minimum possible open count**
     * **Maximum possible open count**

---

## Greedy Solution Trick (Core Idea)

Instead of tracking exact matches, track a **range of possible open parentheses**.

### Two Variables

* `low` → minimum possible number of unmatched `'('`
* `high` → maximum possible number of unmatched `'('`

---

### How Each Character Affects the Range

| Character | Effect on `low` | Effect on `high` |
| --- | --- | --- |
| `'('` | `low += 1` | `high += 1` |
| `')'` | `low -= 1` | `high -= 1` |
| `'*'` | `low -= 1` | `high += 1` |

Explanation for `'*'`:

* If `'*'` → `')'` → open count decreases
* If `'*'` → `'('` → open count increases
* If `'*'` → `""` → open count unchanged  
  So it expands the range.

---

### Critical Greedy Constraints

1. **`high` must never be negative**

   * If `high < 0`, even the best case has more `')'` than `'('` → invalid immediately.
2. **Clamp `low` to 0**

   * Open parentheses cannot be negative.
   * `low = max(low, 0)`
3. **Final Condition**

   * At the end, `low == 0`
   * Means there exists an interpretation with all parentheses closed.

---

## Python 3 Solution (Greedy, O(n) Time, O(1) Space)

```python
from typing import *

class Solution:
    def checkValidString(self, s: str) -> bool:
        low: int = 0   # minimum open parentheses
        high: int = 0  # maximum open parentheses

        for ch in s:
            if ch == '(':
                low += 1
                high += 1
            elif ch == ')':
                low -= 1
                high -= 1
            else:  # '*'
                low -= 1      # treat '*' as ')'
                high += 1     # treat '*' as '('

            # Too many closing brackets
            if high < 0:
                return False

            # Clamp low to 0
            if low < 0:
                low = 0

        return low == 0
```

---

## Complete Worked Example (Step-by-Step)

### Example

```
s = "(*))"
```

---

### Initialization

```
low = 0
high = 0
```

---

### Step 1: Character `'('`

```
low += 1  → 1
high += 1 → 1
```

Range:

```
[1, 1]
```

---

### Step 2: Character `'*'`

```
low -= 1  → 0   (treat '*' as ')')
high += 1 → 2   (treat '*' as '(')
```

Range:

```
[0, 2]
```

Interpretations:

* `'*' = ')'` → "()"
* `'*' = '('` → "(("
* `'*' = ""` → "("

All are still valid possibilities.

---

### Step 3: Character `')'`

```
low -= 1  → -1
high -= 1 → 1
```

Clamp:

```
low = max(0, -1) → 0
```

Range:

```
[0, 1]
```

---

### Step 4: Character `')'`

```
low -= 1  → -1
high -= 1 → 0
```

Clamp:

```
low = 0
```

Range:

```
[0, 0]
```

---

### Final Check

```
low == 0 → True
```

There exists a valid interpretation:

```
"(*))" → "(())"
```

---

## Why This Greedy Works

* We **never commit early** to what `'*'` means.
* We maintain **all feasible states simultaneously** via a range.
* If **any valid path exists**, it survives within `[low, high]`.
* If all paths fail, the range collapses.

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | O(n) |
| Space | O(1) |
| Technique | Greedy + Range Tracking |

---

## Pattern Classification (Useful for Interviews)

* **Greedy with State Range**
* Similar to:

  * Interval DP optimization
  * Feasibility tracking problems
  * “Exists a valid configuration” problems

---

If you want, next we can:

* Compare this with **stack-based approaches** and explain why they are inferior here
* Convert this into a **proof-style explanation**
* Show **why two-pass greedy also works** for this problem

Tell me how deep you want to go.