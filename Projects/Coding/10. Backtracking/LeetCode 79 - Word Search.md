---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 79: Word Search"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 79: Word Search

## LeetCode 79 — Word Search

---

### Problem Statement

Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid.

**Rules**

* The word must be constructed from **sequentially adjacent cells**.
* Adjacent cells are horizontally or vertically neighboring.
* The **same cell cannot be used more than once** in a word.

**Constraints**

* `1 ≤ m, n ≤ 6`
* `1 ≤ word.length ≤ 15`
* `board[i][j]` and `word[k]` are lowercase English letters.

**Example**

```text
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

word = "ABCCED"

Output: true
```

---

## Key Observations

1. This is a **path existence** problem, not a path enumeration problem.
2. From each cell, we can move in **4 directions**: up, down, left, right.
3. A cell **cannot be reused** in the same path → requires visited marking.
4. This is classic **DFS + backtracking on a grid**.
5. As soon as the word is found, we can **short-circuit and return true**.
6. Backtracking is necessary to:

   * Mark a cell as visited
   * Explore
   * Restore the cell for other paths

---

## Approach (DFS Backtracking)

### Strategy

1. Iterate over every cell in the grid.
2. If the cell matches `word[0]`, start DFS from there.
3. At DFS step `index`:

   * If `index == len(word)` → word found.
   * If out of bounds or mismatch → stop.
4. Temporarily mark the current cell as visited.
5. Explore all 4 directions.
6. Restore the cell after recursion (backtrack).

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        rows, cols = len(board), len(board[0])

        def dfs(r: int, c: int, index: int) -> bool:
            if index == len(word):
                return True

            if (
                r < 0 or r >= rows or
                c < 0 or c >= cols or
                board[r][c] != word[index]
            ):
                return False

            # mark as visited
            temp = board[r][c]
            board[r][c] = "#"

            found = (
                dfs(r + 1, c, index + 1) or
                dfs(r - 1, c, index + 1) or
                dfs(r, c + 1, index + 1) or
                dfs(r, c - 1, index + 1)
            )

            # backtrack
            board[r][c] = temp
            return found

        for i in range(rows):
            for j in range(cols):
                if dfs(i, j, 0):
                    return True

        return False
```

---

## Example Walkthrough

### Input

```
word = "ABCCED"
```

### Path Taken

```
A → B → C → C → E → D
```

Coordinates:

```
(0,0) → (0,1) → (0,2) → (1,2) → (2,2) → (2,1)
```

Each step:

* Matches the next character
* Moves to an adjacent cell
* Marks the previous cell as visited

---

## Backtracking Tree Structure (Navigation View)

![https://miro.medium.com/v2/resize%3Afit%3A1400/1%2A7StmdAvxN8pqHKZo5j9J0w.jpeg?utm_source=chatgpt.com](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2A7StmdAvxN8pqHKZo5j9J0w.jpeg?utm_source=chatgpt.com)

![https://media.geeksforgeeks.org/wp-content/uploads/20240502130438/dfs.webp?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/uploads/20240502130438/dfs.webp?utm_source=chatgpt.com)

![https://media.geeksforgeeks.org/wp-content/uploads/geekfofgeek_dfs_graph.png?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/uploads/geekfofgeek_dfs_graph.png?utm_source=chatgpt.com)

### Conceptual Tree (Starting at 'A' at (0,0))

```
A(0,0)
               --------------------------------
               |              |               |
            B(0,1)          ❌              ❌
          ---------
          |       |
       C(0,2)    ❌
          |
       C(1,2)
          |
       E(2,2)
          |
       D(2,1)  ✓ FOUND
```

**Tree Characteristics**

* Depth = `len(word)`
* Branching factor ≤ 4
* Many branches terminate early due to mismatch
* Backtracking restores the board after each failed path

---

## Why Backtracking Is Required

Without backtracking:

* Cells would remain marked as visited
* Valid alternative paths would be blocked

Backtracking ensures:

```
choose → explore → un-choose
```

---

## Complexity Analysis

* **Time Complexity:** `O(m * n * 4^L)`

  * `L` = length of word
* **Space Complexity:** `O(L)` recursion stack

---

## Pattern Recognition

This problem exemplifies the **Grid DFS Backtracking** pattern:

* Used in maze solving, island problems, word puzzles
* Requires visited-state management
* Early termination when goal is reached

---

### One-Line Interview Explanation

> “We run DFS from each cell that matches the first character, mark cells as visited during exploration, and backtrack after each path to safely explore alternatives.”

If you want next, I can:

* Optimize using **character frequency pruning**
* Compare with **Word Search II (Trie-based)**
* Convert this into a **generic grid DFS templat**