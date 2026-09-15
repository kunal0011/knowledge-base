---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 36: Valid Sudoku"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 36: Valid Sudoku

---

### Problem Statement

Determine if a `9 x 9` Sudoku board is valid. Only the filled cells need to be validated according to standard Sudoku rules.

---

### Key Observation

* Each row, column, and 3x3 sub-grid can contain each digit 1-9 at most once.
* A 3x3 sub-grid at cell `(r, c)` is uniquely indexed by `(r // 3, c // 3)`.

---

### Core Technique: Hash Sets for Rows, Cols, and 3x3 Boxes

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = defaultdict(set)
        cols = defaultdict(set)
        boxes = defaultdict(set)
        
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val == '.':
                    continue
                    
                box_idx = (r // 3, c // 3)
                if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                    return False
                    
                rows[r].add(val)
                cols[c].add(val)
                boxes[box_idx].add(val)
                
        return True
```

---

### Worked-Out Example

```python
Cell (0, 0) = '5':
  Row 0 -> add '5'
  Col 0 -> add '5'
  Box (0, 0) -> add '5'
If duplicate '5' found later in row 0, col 0, or box (0,0) -> return False
```

---

### Complexity Analysis

* **Time Complexity:** `O(1) (fixed 81 cells)`
* **Space Complexity:** `O(1) (fixed hash tables)`

---

### Takeaway Pattern

Map 2D coordinates to 3x3 block coordinates using integer division `(r // 3, c // 3)`.