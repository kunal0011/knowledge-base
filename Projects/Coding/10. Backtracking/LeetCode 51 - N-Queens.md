---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 51: N-Queens"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 51: N-Queens

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, Uber, Bloomberg  
**Difficulty:** Hard  
**Topic:** Backtracking / Constraint Satisfaction / Diagonal Invariants / Bitmasking  

---

### Problem Statement

The **n-queens** puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.

Given an integer `n`, return *all distinct solutions to the **n-queens puzzle***. You may return the answer in **any order**.

Each solution contains a distinct board configuration of the n-queens' placement, where `'Q'` and `'.'` both indicate a queen and an empty space, respectively.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `List[List[str]]` containing all valid board layouts.
- **Constraints:**
  - $1 \le n \le 9$

---

### Key Idea & Intuition

- **Row-by-Row Formulation:**
  - On an $n \times n$ board, exactly $n$ queens must be placed. Since two queens cannot share a row, **each row must contain exactly one queen**.
  - Placing queens row by row ($r = 0, 1, \dots, n-1$) naturally eliminates all horizontal conflicts by design.
- **$\mathcal{O}(1)$ Conflict Checks via Mathematical Invariants:**
  - A queen placed at $(r, c)$ threatens:
    1. **Column $c$:** No other queen can have column index $c$.
    2. **Main Diagonal ($\backslash$):** Along any top-left to bottom-right diagonal, the difference $r - c$ is strictly invariant. To ensure non-negative indexing, shift by $n - 1$:
       $$\text{diag}_1 = r - c + n - 1$$
    3. **Anti-Diagonal ($/$):** Along any top-right to bottom-left diagonal, the sum $r + c$ is strictly invariant:
       $$\text{diag}_2 = r + c$$
- **Bitmask Optimization:**
  - For $n \le 9$, column, main diagonal, and anti-diagonal occupancy can be tracked using single integer bitmasks (`cols`, `diags1`, `diags2`).
  - Bitwise checking `(cols & (1 << c))` takes 1 CPU cycle.

---

### Solution Approach (Step-by-Step)

1. Initialize `results = []` and `queens = [-1] * n` (where `queens[r]` stores the column index of the queen in row $r$).
2. Maintain boolean arrays (or bitmasks) for:
   - `cols` of size $n$
   - `diag1` of size $2n$ ($r - c + n$)
   - `diag2` of size $2n$ ($r + c$)
3. Define `backtrack(r)`:
   - **Base Case:** If `r == n`:
     - Construct the $n \times n$ board: each row $i$ has `'.'*c + 'Q' + '.'*(n-1-c)` where $c = \text{queens}[i]$.
     - Append to `results`.
     - Return.
   - For `c` from $0$ to $n - 1$:
     - `d1 = r - c + n`, `d2 = r + c`.
     - If not `cols[c]` and not `diag1[d1]` and not `diag2[d2]`:
       - **Place Queen:**
         - `queens[r] = c`
         - `cols[c] = diag1[d1] = diag2[d2] = True`
       - **Explore:**
         - `backtrack(r + 1)`
       - **Backtrack:**
         - `cols[c] = diag1[d1] = diag2[d2] = False`
4. Call `backtrack(0)` and return `results`.

---

### Visual Algorithm Walkthrough

For $n = 4$, diagonal invariants:

```
Board (r, c):
(0,0) (0,1) (0,2) (0,3)
(1,0) (1,1) (1,2) (1,3)
(2,0) (2,1) (2,2) (2,3)
(3,0) (3,1) (3,2) (3,3)

Anti-diagonal (r + c):
 0   1   2   3
 1   2   3   4
 2   3   4   5
 3   4   5   6

Main diagonal (r - c):
 0  -1  -2  -3
 1   0  -1  -2
 2   1   0  -1
 3   2   1   0
```

Placement trace for 1st valid solution ($n = 4$):
1. Row 0: Place at Col 1 $\implies$ `[ . Q . . ]`
2. Row 1: Col 0 (attacked d1), Col 1 (attacked col), Col 2 (attacked d2). Place at Col 3 $\implies$ `[ . . . Q ]`
3. Row 2: Place at Col 0 $\implies$ `[ Q . . . ]`
4. Row 3: Place at Col 2 $\implies$ `[ . . Q . ]`
Result: `[".Q..", "...Q", "Q...", "..Q."]` (Valid!)

---

### Solved Examples with Multiple Inputs

| Test Case | `n` | Total Solutions | Solutions |
| :--- | :--- | :--- | :--- |
| **Example 1** | `4` | `2` | `[[".Q..","...Q","Q...","..Q."], ["..Q.","Q...","...Q",".Q.."]]` |
| **Single Queen** | `1` | `1` | `[["Q"]]` |
| **No Solution** | `2` | `0` | `[]` |
| **No Solution** | `3` | `0` | `[]` |
| **Classic Eight Queens** | `8` | `92` | 92 distinct valid boards |
| **Max Constraint** | `9` | `352` | 352 distinct valid boards |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        """
        Solves the N-Queens puzzle using backtracking with O(1) diagonal checks.
        """
        results: List[List[str]] = []
        queens = [-1] * n  # queens[r] is the column position of queen in row r

        cols = [False] * n
        diag1 = [False] * (2 * n)  # r - c + n
        diag2 = [False] * (2 * n)  # r + c

        def backtrack(r: int) -> None:
            if r == n:
                # Build board representation
                board = []
                for c in queens:
                    row = ['.'] * n
                    row[c] = 'Q'
                    board.append("".join(row))
                results.append(board)
                return

            for c in range(n):
                d1 = r - c + n
                d2 = r + c
                if not cols[c] and not diag1[d1] and not diag2[d2]:
                    # Choose
                    queens[r] = c
                    cols[c] = diag1[d1] = diag2[d2] = True

                    # Recurse
                    backtrack(r + 1)

                    # Backtrack
                    cols[c] = diag1[d1] = diag2[d2] = False

        backtrack(0)
        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::vector<std::vector<std::string>> solveNQueens(int n) {
        std::vector<std::vector<std::string>> results;
        std::vector<int> queens(n, -1);
        std::vector<bool> cols(n, false);
        std::vector<bool> diag1(2 * n, false);
        std::vector<bool> diag2(2 * n, false);

        backtrack(0, n, queens, cols, diag1, diag2, results);
        return results;
    }

private:
    void backtrack(int r, int n, std::vector<int>& queens,
                   std::vector<bool>& cols, std::vector<bool>& diag1, std::vector<bool>& diag2,
                   std::vector<std::vector<std::string>>& results) {
        if (r == n) {
            std::vector<std::string> board(n, std::string(n, '.'));
            for (int i = 0; i < n; ++i) {
                board[i][queens[i]] = 'Q';
            }
            results.push_back(board);
            return;
        }

        for (int c = 0; c < n; ++c) {
            int d1 = r - c + n;
            int d2 = r + c;
            if (!cols[c] && !diag1[d1] && !diag2[d2]) {
                queens[r] = c;
                cols[c] = diag1[d1] = diag2[d2] = true;

                backtrack(r + 1, n, queens, cols, diag1, diag2, results);

                cols[c] = diag1[d1] = diag2[d2] = false; // Backtrack
            }
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> results = new ArrayList<>();
        int[] queens = new int[n];
        boolean[] cols = new boolean[n];
        boolean[] diag1 = new boolean[2 * n];
        boolean[] diag2 = new boolean[2 * n];

        backtrack(0, n, queens, cols, diag1, diag2, results);
        return results;
    }

    private void backtrack(int r, int n, int[] queens,
                          boolean[] cols, boolean[] diag1, boolean[] diag2,
                          List<List<String>> results) {
        if (r == n) {
            List<String> board = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                char[] row = new char[n];
                Arrays.fill(row, '.');
                row[queens[i]] = 'Q';
                board.add(new String(row));
            }
            results.add(board);
            return;
        }

        for (int c = 0; c < n; c++) {
            int d1 = r - c + n;
            int d2 = r + c;
            if (!cols[c] && !diag1[d1] && !diag2[d2]) {
                queens[r] = c;
                cols[c] = diag1[d1] = diag2[d2] = true;

                backtrack(r + 1, n, queens, cols, diag1, diag2, results);

                cols[c] = diag1[d1] = diag2[d2] = false; // Backtrack
            }
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N!)$.
  - The first queen has $N$ choices, the second has at most $N - 2$, the third has at most $N - 4$, bounding total states strictly by $\mathcal{O}(N!)$.
  - Generating and copying the board at each leaf takes $\mathcal{O}(N^2)$.
  - For $N = 9$, $9! = 362,880$, running in $< 8 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack, `queens` column array, and boolean occupancy tracking arrays.

---

### Takeaway Pattern & Interview Traps

- **Mathematical Invariant Identification:** Remembering $r - c + n$ and $r + c$ for diagonal lines eliminates the need to traverse the board diagonally in $\mathcal{O}(N)$ time.
- **Board Representation:** Do not pass an $N \times N$ 2D character matrix around the recursion stack. A 1D integer array `queens[r] = c` represents the board compactly and only builds the formatted strings upon reaching $r = n$.