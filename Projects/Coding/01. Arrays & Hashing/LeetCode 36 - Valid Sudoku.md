---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 36: Valid Sudoku"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
---

# LeetCode 36: Valid Sudoku

**Target Companies:** Amazon (Top Classic), Google, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Hash Set / 2D Grid Coordinate Hashing / Bitmasking

---

### Problem Statement

Determine if a $9 \times 9$ Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
1. Each row must contain the digits `1-9` without repetition.
2. Each column must contain the digits `1-9` without repetition.
3. Each of the nine $3 \times 3$ sub-boxes of the grid must contain the digits `1-9` without repetition.

**Note:**
- A Sudoku board (partially filled) could be valid but is not necessarily solvable.
- Only the filled cells need to be validated.

---

### Input & Output Formats & Constraints

- **Input:** `board: List[List[str]]` ($9 \times 9$ matrix of characters `'1'-'9'` or `'.'`)
- **Output:** `bool`
- **Constraints:**
  - `board.length == 9`
  - `board[i].length == 9`
  - `board[i][j]` is a digit `'1'-'9'` or `'.'`.

---

### Key Idea & Intuition

The problem requires checking that no digit from `'1'` to `'9'` appears more than once in any row, column, or $3 \times 3$ sub-box.

#### Grid Coordinate Mapping for $3 \times 3$ Boxes:
A cell at coordinate `(r, c)` belongs to:
- Row `r` ($0 \le r < 9$)
- Column `c` ($0 \le c < 9$)
- Box index:
  $$\text{box\_index} = \left(\lfloor r / 3 \rfloor, \lfloor c / 3 \rfloor\right) \quad \text{or as 1D index: } 3 \cdot \lfloor r / 3 \rfloor + \lfloor c / 3 \rfloor \in [0, 8]$$

#### Single-Pass Validation:
Instead of making 3 separate passes (one for rows, one for columns, one for boxes), we can traverse the 81 cells once in a single nested loop:
- Maintain 3 lookup tables / hash sets / bitmasks:
  - `rows[r]`: digits seen in row $r$
  - `cols[c]`: digits seen in column $c$
  - `boxes[b]`: digits seen in box $b$
- For each non-empty cell `val != '.'`:
  - If `val` has already been recorded in `rows[r]`, `cols[c]`, or `boxes[b]`, return `False` immediately.
  - Otherwise, register `val` in all three tracking structures.
- If all 81 cells pass without collisions, return `True`.

---

### Solution Approach (Step-by-Step)

1. **Initialize Tracking Structures:**
   - Allocate three $9 \times 9$ boolean arrays (or hash sets) for `rows`, `cols`, and `boxes`.
2. **Scan Grid Cells:**
   - Loop `r` from $0$ to $8$:
     - Loop `c` from $0$ to $8$:
       - If `board[r][c] == '.'`, continue.
       - Map character digit to $0$-based index: `d = board[r][c] - '1'`.
       - Compute box index: `b = (r // 3) * 3 + (c // 3)`.
       - Check if `rows[r][d]`, `cols[c][d]`, or `boxes[b][d]` is already true.
       - If any is true, return `False` (duplicate detected).
       - Mark `rows[r][d] = cols[c][d] = boxes[b][d] = true`.
3. **Return True:**
   - If no conflicts are detected after inspecting all cells, the board is valid.

---

### Visual Algorithm Walkthrough

```
Sub-box Indexing:
    c = 0 1 2   3 4 5   6 7 8
r=0-2 [ Box 0 ] [ Box 1 ] [ Box 2 ]
r=3-5 [ Box 3 ] [ Box 4 ] [ Box 5 ]
r=6-8 [ Box 6 ] [ Box 7 ] [ Box 8 ]

Formula: box_index = (r // 3) * 3 + (c // 3)

Cell (r=4, c=7):
  r // 3 = 1
  c // 3 = 2
  box_index = 1 * 3 + 2 = 5 -> Box 5.
```

---

### Solved Examples with Multiple Inputs

| Scenario | Board State | Validation Result | Reason |
| :--- | :--- | :--- | :--- |
| Standard Valid Partial Board | Digits 1-9 placed without overlap in rows, cols, boxes | `True` | All constraints satisfied |
| Row Conflict | Two `'8'`s in row 0 at `board[0][0]` and `board[0][4]` | `False` | `rows[0]['8']` collision |
| Column Conflict | Two `'3'`s in column 2 at `board[1][2]` and `board[7][2]` | `False` | `cols[2]['3']` collision |
| Sub-box Conflict | Two `'5'`s at `board[0][0]` and `board[2][2]` | `False` | Both belong to Box 0: `boxes[0]['5']` collision |
| Completely Empty Board | All 81 cells are `'.'` | `True` | No duplicate numbers |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val == '.':
                    continue
                    
                b = (r // 3) * 3 + (c // 3)
                if val in rows[r] or val in cols[c] or val in boxes[b]:
                    return False
                    
                rows[r].add(val)
                cols[c].add(val)
                boxes[b].add(val)
                
        return True
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    bool isValidSudoku(std::vector<std::vector<char>>& board) {
        int rows[9][9] = {0};
        int cols[9][9] = {0};
        int boxes[9][9] = {0};

        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                if (board[r][c] == '.') continue;
                int digit = board[r][c] - '1';
                int b = (r / 3) * 3 + (c / 3);

                if (rows[r][digit] || cols[c][digit] || boxes[b][digit]) {
                    return false;
                }

                rows[r][digit] = cols[c][digit] = boxes[b][digit] = 1;
            }
        }
        return true;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        int[][] rows = new int[9][9];
        int[][] cols = new int[9][9];
        int[][] boxes = new int[9][9];

        for (int r = 0; r < 9; r++) {
            for (int c = 0; c < 9; c++) {
                if (board[r][c] == '.') continue;
                int digit = board[r][c] - '1';
                int b = (r / 3) * 3 + (c / 3);

                if (rows[r][digit] == 1 || cols[c][digit] == 1 || boxes[b][digit] == 1) {
                    return false;
                }

                rows[r][digit] = cols[c][digit] = boxes[b][digit] = 1;
            }
        }
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(1)$ — Strictly 81 cells are inspected. For each cell, array lookups and bitwise/hash operations run in constant $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space — Three fixed-size matrices of $9 \times 9 = 81$ entries.

---

### Takeaway Pattern & Interview Traps

1. **Sub-box Index Formula:**
   - Remember the flattening equation `(r / 3) * 3 + (c / 3)` which maps row and column coordinates into a single integer in $[0, 8]$.
2. **Validity vs. Solvability:**
   - The problem does NOT ask whether the board has a valid full solution (which requires exponential backtracking). It only checks consistency of the existing digits on the board.
3. **Bitmasking Optimization:**
   - In place of arrays or hash sets, an integer bitmask (`1 << digit`) can be used for each row, col, and box, reducing auxiliary memory to just three arrays of 9 integers.
