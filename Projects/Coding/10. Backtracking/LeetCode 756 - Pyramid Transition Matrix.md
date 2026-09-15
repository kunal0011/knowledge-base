---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 756: Pyramid Transition Matrix"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 756: Pyramid Transition Matrix

## LeetCode 756 — Pyramid Transition Matrix

---

### Problem Statement

You are given a string `bottom` and a list of strings `allowed`.  
Each string in `allowed` represents a **valid transition** of the form:

```python
XY -> Z
```

Meaning: if two adjacent blocks `X` and `Y` are next to each other in the current row, you can place block `Z` above them in the next row.

Your task is to determine **whether it is possible to build a pyramid** such that:

* The bottom row is exactly `bottom`
* Each upper row is built using valid transitions
* The pyramid ends with **a single block at the top**

Return `True` if possible, otherwise `False`.

---

### Key Observations

1. Each level of the pyramid is **one character shorter** than the level below.
2. The choice of blocks for the next row depends on **adjacent pairs** in the current row.
3. Multiple transitions may exist for the same pair → **branching choices**.
4. This is a **backtracking / DFS** problem with:

   * Depth = height of the pyramid (`len(bottom)`)
   * Branching = number of allowed transitions per pair
5. If **any path** leads to a valid single-block top, return `True`.

---

## Core Idea (High-Level)

1. Preprocess `allowed` into a mapping:

   ```
   (X, Y) → [Z1, Z2, ...]
   ```
2. Recursively:

   * Build the **next row** from left to right
   * Once the next row is complete, recurse on that row
3. If at any level no valid transition exists, **backtrack**

---

## Python 3 Solution (with Typing)

```python
from typing import List, Dict
from collections import defaultdict

class Solution:
    def pyramidTransition(self, bottom: str, allowed: List[str]) -> bool:
        transitions: Dict[str, List[str]] = defaultdict(list)

        # Build transition map
        for rule in allowed:
            transitions[rule[:2]].append(rule[2])

        def can_build(row: str) -> bool:
            # Base case: pyramid completed
            if len(row) == 1:
                return True

            # Generate all possible next rows via backtracking
            def backtrack(index: int, next_row: List[str]) -> bool:
                if index == len(row) - 1:
                    return can_build("".join(next_row))

                pair = row[index:index + 2]
                if pair not in transitions:
                    return False

                for ch in transitions[pair]:
                    next_row.append(ch)
                    if backtrack(index + 1, next_row):
                        return True
                    next_row.pop()

                return False

            return backtrack(0, [])

        return can_build(bottom)
```

---

## Example Explanation

### Input

```
bottom = "BCD"
allowed = ["BCG", "CDE", "GEA", "FFF"]
```

### Transitions

```
BC → G
CD → E
GE → A
```

### Construction

```
Bottom:     B   C   D
Next row:     G   E
Top row:        A
```

Since we reach a single block at the top → **return True**

---

## Backtracking Tree Structure (Complete Conceptual Tree)

### Example

```
bottom = "ABC"
allowed = ["ABD", "BCE", "BCF", "DEF", "EFG"]
```

### Transition Map

```
AB → D
BC → E, F
DE → F
EF → G
```

---

### Conceptual Backtracking Tree

```
Level 0 (bottom)
--------------------------------
ABC

Level 1 (building next row from pairs AB, BC)
--------------------------------
             []
              |
        -----------------
        |               |
      D E               D F
      (from BC→E)       (from BC→F)
        |               |
      "DE"            "DF"

Level 2 (building above)
--------------------------------
     DE                DF
      |                |
     F                ❌
 (DE→F)        (no DF transition)

Level 3 (top)
--------------------------------
       F
       |
     (len == 1) ✓ SUCCESS
```

---

## How Navigation Happens

1. Fix the **row**
2. Build the **next row left-to-right**
3. For each adjacent pair:

   * Try **all possible transitions**
4. If a path fails at any level → backtrack
5. Stop immediately when a valid top is found

---

## Why This Is Pure Backtracking

* Choices depend on **previous choices**
* Tree is **irregular** (branching varies per pair)
* Pruning occurs naturally when no transition exists

---

## Complexity Analysis

* **Time Complexity:** Exponential in pyramid height (worst-case), but heavily pruned
* **Space Complexity:** `O(n²)` recursion + intermediate rows

---

## One-Line Interview Explanation

> “We recursively build the pyramid row by row using backtracking, trying all valid transitions for adjacent pairs and stopping as soon as a single-block top is achievable.”