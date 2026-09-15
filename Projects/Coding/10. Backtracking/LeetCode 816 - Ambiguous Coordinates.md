---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 816: Ambiguous Coordinates"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 816: Ambiguous Coordinates

## LeetCode 816 — Ambiguous Coordinates

---

### Problem Statement

We are given a string `s` representing a coordinate in the form `"(123)"` (parentheses included, no spaces).  
Return **all possible valid coordinate representations** by inserting:

* **one comma** to separate x and y
* **optional decimal points** in x and y

**Rules for validity**

1. No extra leading zeros:

   * `"0"` is valid
   * `"0.x"` is valid
   * `"00"`, `"01"`, `"01.2"` are **invalid**
2. No trailing zeros in decimals:

   * `"1.0"` is invalid
   * `"1.20"` is invalid
3. At least one digit before and after a decimal point.

**Example**

```text
Input: s = "(123)"
Output: ["(1, 23)", "(1, 2.3)", "(12, 3)", "(1.2, 3)"]
```

---

## Key Observations

1. The problem has **two independent decisions**:

   * Where to **split** the string into x and y
   * How to **place decimal points** in each part
2. Each substring can generate **multiple valid numeric forms**
3. This is not classical backtracking on characters — it is:

   * **Split enumeration**
   * Followed by **local generation + validation**
4. The “tree” here is **conceptual**, not recursive over every character.

---

## High-Level Approach

1. Remove parentheses.
2. Try every possible split:

   ```
   left = s[0:i], right = s[i:]
   ```
3. For each side:

   * Generate all valid numeric representations.
4. Combine left × right.

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        digits = s[1:-1]  # remove '(' and ')'
        n = len(digits)
        result: List[str] = []

        def generate(part: str) -> List[str]:
            res: List[str] = []

            # Whole number (no decimal)
            if part == "0" or not part.startswith("0"):
                res.append(part)

            # Decimal placements
            for i in range(1, len(part)):
                left, right = part[:i], part[i:]

                # Leading zero rule
                if left.startswith("0") and left != "0":
                    continue
                # Trailing zero rule
                if right.endswith("0"):
                    continue

                res.append(left + "." + right)

            return res

        for i in range(1, n):
            left_part = digits[:i]
            right_part = digits[i:]

            left_nums = generate(left_part)
            right_nums = generate(right_part)

            for x in left_nums:
                for y in right_nums:
                    result.append(f"({x}, {y})")

        return result
```

---

## Example Explanation (`s = "(123)"`)

Digits extracted:

```
"123"
```

### Possible splits

1. `"1" | "23"`
2. `"12" | "3"`

---

### Split 1: `"1" | "23"`

* `"1"` → `["1"]`
* `"23"` → `["23", "2.3"]`

Produces:

```
(1, 23)
(1, 2.3)
```

---

### Split 2: `"12" | "3"`

* `"12"` → `["12", "1.2"]`
* `"3"` → `["3"]`

Produces:

```
(12, 3)
(1.2, 3)
```

---

## Backtracking Tree Structure (Complete Conceptual Tree)

> This tree represents **logical navigation**, not function calls.

```
Start: "(123)"
        |
        |── Remove parentheses
        ↓
      "123"
        |
        |── Split at i = 1
        |     |
        |     |── Left = "1"
        |     |     └── ["1"]
        |     |
        |     |── Right = "23"
        |           ├── "23"
        |           └── "2.3"
        |
        |     → (1, 23)
        |     → (1, 2.3)
        |
        |── Split at i = 2
              |
              |── Left = "12"
              |     ├── "12"
              |     └── "1.2"
              |
              |── Right = "3"
                    └── ["3"]

              → (12, 3)
              → (1.2, 3)
```

---

## Why This Is Still Backtracking (Conceptually)

* We explore **all structural choices**
* Invalid numeric forms are **pruned early**
* Cartesian product at each split resembles DFS branching
* The tree is shallow but wide

---

## Complexity Analysis

* Let `n = len(digits)`
* Splits: `O(n)`
* Decimal placements per part: `O(n)`
* **Time Complexity:** `O(n³)` worst case
* **Space Complexity:** `O(n²)` for generated candidates

---

## Pattern Recognition

This problem exemplifies the **“Split + Validate + Combine”** backtracking pattern:

* Enumerate structural splits
* Generate local valid states
* Combine independent branches

---

### One-Line Interview Summary

> “We try every possible split of the digits into x and y, generate all valid decimal representations for each side under strict formatting rules, and combine them.”