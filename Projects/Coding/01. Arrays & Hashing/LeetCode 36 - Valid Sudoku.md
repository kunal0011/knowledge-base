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
**Topic:** Hash Set / 2D Grid Coordinate Hashing

---

### Problem Statement

Determine if a $9 \times 9$ Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
1. Each row must contain the digits `1-9` without repetition.
2. Each column must contain the digits `1-9` without repetition.
3. Each of the nine $3 \times 3$ sub-boxes of the grid must contain the digits `1-9` without repetition.

Note: A Sudoku board (partially filled) could be valid but is not necessarily solvable. Only the filled cells need to be validated.

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

- **Grid Coordinate Mapping for $3 \times 3$ Boxes:**
  - A cell at `(r, c)` belongs to $3 \times 3$ box index `(r // 3, c // 3)`.
- **Three Tracking Sets:**
  - Maintain sets or bitmasks for:
    - `rows[r]`: Set of digits seen in row $r$.
    - `cols[c]`: Set of digits seen in column $c$.
    - `boxes[(r // 3, c // 3)]`: Set of digits seen in box $(r // 3, c // 3)$.
  - Single pass through all 81 cells verifies the board in constant $O(1)$ operations.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
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
                    
                b_idx = (r // 3, c // 3)
                if val in rows[r] or val in cols[c] or val in boxes[b_idx]:
                    return False
                    
                rows[r].add(val)
                cols[c].add(val)
                boxes[b_idx].add(val)
                
        return True
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_set>

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

- **Time Complexity:** $O(1)$ — Strictly 81 cells checked.
- **Space Complexity:** $O(1)$ — Arrays of fixed size $9 \times 9$.
