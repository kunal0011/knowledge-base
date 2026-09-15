---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 79: Word Search"
tags:
  - leetcode
  - coding
  - backtracking
  - matrix
  - array
  - string
  - amazon
  - google
---

# LeetCode 79: Word Search

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Grid DFS / In-Place State Marking  

---

### Problem Statement

Given an `m x n` grid of characters `board` and a string `word`, return `true` *if `word` exists in the grid*.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

---

### Input & Output Formats & Constraints

- **Input:** `board: List[List[str]]`, `word: str`
- **Output:** `bool`
- **Constraints:**
  - $m == \text{board.length}$
  - $n == \text{board}[i]\text{.length}$
  - $1 \le m, n \le 6$
  - $1 \le \text{word.length} \le 15$
  - `board` and `word` consist of only lowercase and uppercase English letters.

---

### Key Idea & Intuition

- **Grid DFS Backtracking:**
  - Start a DFS traversal from every cell `(r, c)` where `board[r][c] == word[0]`.
  - At each step, if `board[r][c] == word[idx]`:
    - If `idx == len(word) - 1`, the complete word has been matched $\implies$ return `True`.
    - Temporarily mark cell `(r, c)` as visited (by replacing with `'#'`) to prevent reusing the same cell within the current path.
    - Recurse in 4 directions: $(r+1, c), (r-1, c), (r, c+1), (r, c-1)$.
    - Restore the original character (`board[r][c] = orig_char`) when backtracking.
- **Critical Production-Grade Pruning Heuristics:**
  1. **Character Count Feasibility:**
     - Count character frequencies across `board`. If `board` contains fewer instances of any letter than `word` requires, return `False` immediately.
  2. **Frequency-Based Direction Reversal:**
     - If the starting character `word[0]` is more common on `board` than the ending character `word[-1]`, reverse `word` (`word = word[::-1]`)!
     - Searching from the rarer character dramatically reduces the number of starting cells and initial branching factor (e.g., searching for `"aaaaab"` on a grid filled with `'a'`s).

---

### Solution Approach (Step-by-Step)

1. Check if `len(word) > m * n`: return `False`.
2. Count frequencies of letters in `board` and `word`:
   - If any character in `word` appears more times than in `board`, return `False`.
   - If `count(word[0]) > count(word[-1])`, set `word = word[::-1]`.
3. Define `dfs(r, c, idx)`:
   - If `board[r][c] != word[idx]`: return `False`.
   - If `idx == len(word) - 1`: return `True`.
   - Save character: `temp = board[r][c]`.
   - Mark visited: `board[r][c] = '#'`.
   - For `(dr, dc)` in `[(1,0), (-1,0), (0,1), (0,-1)]`:
     - `nr, nc = r + dr, c + dc`
     - If $0 \le nr < m$ and $0 \le nc < n$ and `board[nr][nc] != '#'`:
       - If `dfs(nr, nc, idx + 1)`:
         - `board[r][c] = temp` (restore)
         - Return `True`
   - Backtrack: `board[r][c] = temp`.
   - Return `False`.
4. Loop through each cell $(r, c)$:
   - If `board[r][c] == word[0]` and `dfs(r, c, 0)`:
     - Return `True`.
5. Return `False`.

---

### Visual Algorithm Walkthrough

Let `board` =
```
[
  ['A', 'B', 'C', 'E'],
  ['S', 'F', 'C', 'S'],
  ['A', 'D', 'E', 'E']
]
word = "ABCCED"
```

Execution steps:
```
1. (0,0) matches 'A' -> mark '#'
   - Neighbor (0,1) matches 'B' -> mark '#'
     - Neighbor (0,2) matches 'C' -> mark '#'
       - Neighbor (1,2) matches 'C' -> mark '#'
         - Neighbor (2,2) matches 'E' -> mark '#'
           - Neighbor (2,1) matches 'D' -> idx == 5 == len(word)-1!
             MATCH FOUND! Returns True up call stack!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `board` | `word` | Path Found | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard True** | `[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]` | `"ABCCED"` | `(0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,1)` | `true` |
| **Standard False** | Same board | `"ABCB"` | Cannot reuse cell `(0,1)` `'B'` | `false` |
| **Single Character** | `[["a"]]` | `"a"` | `(0,0)` matches | `true` |
| **Frequency Prune** | Grid of all `'a'` | `"aaaaab"` | No `'b'` on board $\implies$ pruned in $\mathcal{O}(1)$ | `false` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List
from collections import Counter

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        """
        Searches for word in 2D grid using backtracking DFS.
        Applies character count check and directional reversal heuristics.
        """
        rows, cols = len(board), len(board[0])
        if len(word) > rows * cols:
            return False

        # Frequency check
        board_counts = Counter(ch for row in board for ch in row)
        word_counts = Counter(word)
        for ch, count in word_counts.items():
            if board_counts[ch] < count:
                return False

        # Reverse heuristic: start from rarer endpoint character
        if board_counts[word[0]] > board_counts[word[-1]]:
            word = word[::-1]

        def dfs(r: int, c: int, idx: int) -> bool:
            if board[r][c] != word[idx]:
                return False

            if idx == len(word) - 1:
                return True

            # Mark cell visited in-place
            temp = board[r][c]
            board[r][c] = '#'

            # Explore 4 directions
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                    if dfs(nr, nc, idx + 1):
                        board[r][c] = temp  # Restore state
                        return True

            # Backtrack
            board[r][c] = temp
            return False

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == word[0] and dfs(r, c, 0):
                    return True

        return False
```

#### C++17
```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    bool exist(std::vector<std::vector<char>>& board, std::string word) {
        int m = static_cast<int>(board.size());
        int n = static_cast<int>(board[0].size());
        if (word.size() > static_cast<size_t>(m * n)) return false;

        std::unordered_map<char, int> board_count;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                board_count[board[r][c]]++;
            }
        }

        std::unordered_map<char, int> word_count;
        for (char c : word) {
            if (++word_count[c] > board_count[c]) {
                return false;
            }
        }

        // Search in reverse if the last character is rarer
        if (board_count[word.front()] > board_count[word.back()]) {
            std::reverse(word.begin(), word.end());
        }

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (board[r][c] == word[0] && dfs(r, c, 0, m, n, board, word)) {
                    return true;
                }
            }
        }

        return false;
    }

private:
    bool dfs(int r, int c, size_t idx, int m, int n,
             std::vector<std::vector<char>>& board, const std::string& word) {
        if (board[r][c] != word[idx]) return false;
        if (idx == word.size() - 1) return true;

        char temp = board[r][c];
        board[r][c] = '#'; // Mark visited

        static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (const auto& d : dirs) {
            int nr = r + d[0];
            int nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && board[nr][nc] != '#') {
                if (dfs(nr, nc, idx + 1, m, n, board, word)) {
                    board[r][c] = temp;
                    return true;
                }
            }
        }

        board[r][c] = temp; // Backtrack
        return false;
    }
};
```

#### Java
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        if (word.length() > m * n) {
            return false;
        }

        Map<Character, Integer> boardCount = new HashMap<>();
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                boardCount.put(board[r][c], boardCount.getOrDefault(board[r][c], 0) + 1);
            }
        }

        Map<Character, Integer> wordCount = new HashMap<>();
        for (char c : word.toCharArray()) {
            int count = wordCount.getOrDefault(c, 0) + 1;
            wordCount.put(c, count);
            if (count > boardCount.getOrDefault(c, 0)) {
                return false;
            }
        }

        // Reverse word if last character has lower frequency
        if (boardCount.getOrDefault(word.charAt(0), 0) > boardCount.getOrDefault(word.charAt(word.length() - 1), 0)) {
            word = new StringBuilder(word).reverse().toString();
        }

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (board[r][c] == word.charAt(0) && dfs(r, c, 0, m, n, board, word)) {
                    return true;
                }
            }
        }

        return false;
    }

    private boolean dfs(int r, int c, int idx, int m, int n, char[][] board, String word) {
        if (board[r][c] != word.charAt(idx)) {
            return false;
        }
        if (idx == word.length() - 1) {
            return true;
        }

        char temp = board[r][c];
        board[r][c] = '#'; // Mark visited

        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] d : dirs) {
            int nr = r + d[0];
            int nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && board[nr][nc] != '#') {
                if (dfs(nr, nc, idx + 1, m, n, board, word)) {
                    board[r][c] = temp;
                    return true;
                }
            }
        }

        board[r][c] = temp; // Backtrack
        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M \cdot N \cdot 3^L)$ where $L = \text{word.length} \le 15$.
  - There are $M \times N$ starting cells.
  - From each cell, the first step has 4 neighbors; each subsequent step cannot revisit the immediate parent cell, leaving at most 3 directions.
  - With frequency checking and reverse word pruning, bad test cases like grids filled with `'a'` terminate in $\mathcal{O}(1)$ or $< 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(L)$ auxiliary space for the recursion call stack depth ($L \le 15$). Marking the board in-place uses $\mathcal{O}(1)$ extra memory.

---

### Takeaway Pattern & Interview Traps

- **In-Place Marking (`#`):** Reusing the board array to mark visited cells avoids allocating an $M \times N$ boolean array or maintaining a hash set of coordinate tuples.
- **The Reverse Word Optimization:** This single trick cuts execution time on pathological inputs (like $1000$ `'a'`s followed by `'b'`) from TLE to $0 \text{ ms}$.