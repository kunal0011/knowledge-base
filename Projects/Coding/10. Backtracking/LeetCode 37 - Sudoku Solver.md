---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 37: Sudoku Solver"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 37: Sudoku Solver

## LeetCode 37 — Sudoku Solver

---

## Problem Statement

Write a program to solve a **9×9 Sudoku puzzle** by filling the empty cells.

**Rules**

* Each row must contain digits `1–9` **exactly once**
* Each column must contain digits `1–9` **exactly once**
* Each 3×3 sub-box must contain digits `1–9` **exactly once**
* Empty cells are denoted by `'.'`
* There is **exactly one valid solution**

You must modify the board **in-place**.

---

## Key Observations (Critical)

1. This is a **constraint satisfaction problem (CSP)**.
2. Each empty cell is a **decision point** with up to 9 choices.
3. Most choices are invalid → **heavy pruning** is essential.
4. Backtracking works because:

   * Partial assignments can be validated early
   * Invalid branches are abandoned immediately
5. The problem is **not about brute force**; it is about **constraint propagation + DFS**.

---

## Constraints Used for Pruning

For a cell `(r, c)`:

* Digit must not appear in:

  * `row r`
  * `column c`
  * `3×3 box (r//3, c//3)`

These constraints reduce branching dramatically.

---

## Approach (Backtracking)

### State

* Board state (mutated in-place)

### Choice

* Pick an empty cell
* Try digits `'1'` to `'9'`

### Validation

* Check row, column, and sub-box constraints

### Termination

* If all cells are filled → solution found

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        def is_valid(r: int, c: int, ch: str) -> bool:
            for i in range(9):
                if board[r][i] == ch:
                    return False
                if board[i][c] == ch:
                    return False
                if board[(r // 3) * 3 + i // 3][(c // 3) * 3 + i % 3] == ch:
                    return False
            return True

        def backtrack() -> bool:
            for r in range(9):
                for c in range(9):
                    if board[r][c] == ".":
                        for ch in "123456789":
                            if is_valid(r, c, ch):
                                board[r][c] = ch
                                if backtrack():
                                    return True
                                board[r][c] = "."  # undo
                        return False
            return True  # solved

        backtrack()
```

---

## Example Explanation (High Level)

Given a partially filled board:

1. Find the **first empty cell**
2. Try digits `1–9`
3. Place the first valid digit
4. Move to the next empty cell
5. If later a cell has **no valid digit**, backtrack
6. Continue until all cells are filled

---

## Backtracking Tree Structure

(**Conceptual Complete Tree — Not Expanded Values**)

> Important: The actual tree is enormous. Below is the **conceptual navigation structure**, which is what interviewers expect you to understand.

---

### Conceptual Backtracking Tree

```
Start
 |
 |-- Fill Cell #1
 |     |
 |     |-- Try 1 ❌ (violates constraint)
 |     |-- Try 2 ❌
 |     |-- Try 3 ✅
 |           |
 |           |-- Fill Cell #2
 |           |     |
 |           |     |-- Try 1 ❌
 |           |     |-- Try 2 ✅
 |           |           |
 |           |           |-- Fill Cell #3
 |           |           |     |
 |           |           |     |-- Try 1 ❌
 |           |           |     |-- Try 2 ❌
 |           |           |     |-- Try 3 ❌
 |           |           |     |-- Try 4 ✅
 |           |           |           |
 |           |           |           |-- Fill Cell #4
 |           |           |                 |
 |           |           |                 |-- ...
 |           |           |
 |           |           |-- (No valid digit)
 |           |                 |
 |           |                 ← BACKTRACK
 |           |
 |           |-- Try next digit for Cell #2
 |                 |
 |                 |-- ...
 |
 |-- Continue until all cells filled
 |
Solution Found ✔
```

---

### How to Read This Tree

* **Each level** = filling one empty cell
* **Each branch** = trying a digit `1–9`
* ❌ = constraint violation → prune immediately
* ✅ = valid placement → go deeper
* Backtrack occurs when **no valid digit exists** at a level

---

## Why This Tree Works

* The tree is **deep (up to 81 levels)** but **narrow** due to pruning
* Most branches die early
* Only one path reaches a valid leaf

---

## Complexity (Theoretical)

* **Worst-case:** `O(9^81)` (theoretical upper bound)
* **Practical:** Much smaller due to aggressive pruning
* Sudoku is NP-complete, but constraints make it tractable

---

## Pattern Recognition

LeetCode 37 is the **canonical example** of:

* Constraint Satisfaction
* DFS + Backtracking
* Early pruning
* In-place state mutation

This pattern appears in:

* N-Queens (LC 51)
* Crossword filling
* Timetabling / scheduling systems

---

### One-Line Interview Explanation

> “We use DFS backtracking, try digits in empty cells, validate against Sudoku constraints, and backtrack immediately when a placement leads to no valid continuation.”

If you want next, I can:

* Optimize this using **bitmasks**
* Show **MRV heuristic** (choose cell with fewest options)
* Compare **Sudoku vs N-Queens** trees structurally