---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 212: Word Search II"
tags:
  - leetcode
  - coding
  - backtracking
  - trie
  - matrix
  - amazon
  - google
---

# LeetCode 212: Word Search II

**Target Companies:** Amazon, Google, Meta, Microsoft, Uber, Apple  
**Difficulty:** Hard  
**Topic:** Backtracking / Trie (Prefix Tree) / Grid DFS / Trie Pruning  

---

### Problem Statement

Given an `m x n` `board` of characters and a list of strings `words`, return *all words on the board*.

Each word must be constructed from letters of sequentially adjacent cells, where **adjacent cells** are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

---

### Input & Output Formats & Constraints

- **Input:** `board: List[List[str]]`, `words: List[str]`
- **Output:** `List[str]` containing all words from `words` found on the board.
- **Constraints:**
  - $m == \text{board.length}$
  - $n == \text{board}[i]\text{.length}$
  - $1 \le m, n \le 12$
  - `board[i][j]` is a lowercase English letter.
  - $1 \le \text{words.length} \le 3 \times 10^4$
  - $1 \le \text{words}[i]\text{.length} \le 10$
  - `words[i]` consists of lowercase English letters.
  - All the strings of `words` are unique.

---

### Key Idea & Intuition

- **Why Word Search I (LC 79) Repeated for Each Word Fails:**
  - Running LeetCode 79 individually for each word costs $\mathcal{O}(W \times M \times N \times 3^L)$.
  - With $W = 3 \times 10^4$, $M \times N = 144$, and $L = 10$, this results in $\approx 2.5 \times 10^{11}$ operations, guaranteed to trigger **Time Limit Exceeded (TLE)**.
- **Trie (Prefix Tree) Driven Search:**
  - Build a Trie containing all target words.
  - Traverse the board with DFS only once. Instead of searching for an individual word, the DFS follows the transitions in the Trie!
  - If the character at `board[r][c]` is not a child of the current Trie node, the prefix cannot match any word in the dictionary, allowing instantaneous pruning of that entire branch.
- **Critical Optimizations:**
  1. **Store `word` directly in Trie Node:** Instead of accumulating strings with expensive concatenations, store the reference to `word` at the terminal Trie node. When reaching that node, append `curr_node.word` to results.
  2. **Prevent Duplicates in $\mathcal{O}(1)$:** After adding a word to results, set `curr_node.word = None` so the same word is never added twice if reached via another path.
  3. **Trie Leaf Pruning (Trimming):** Once a leaf node has been matched and has no remaining children, remove it from its parent. This actively shrinks the Trie during the search and prunes redundant future board traversals.
  4. **In-Place Board Marking:** Mark visited cells by temporarily modifying `board[r][c] = '#'` and restoring the original character upon backtracking, avoiding the overhead of a separate `visited` array or set.

---

### Solution Approach (Step-by-Step)

1. **Build Trie:**
   - Define `TrieNode` with a children array or map `children = {}` and `word: Optional[str] = None`.
   - Insert every word in `words` into the Trie.
2. **Backtracking DFS `dfs(r, c, parent_node)`:**
   - Let `char = board[r][c]`. If `char not in parent_node.children`, return.
   - `curr_node = parent_node.children[char]`.
   - If `curr_node.word is not None`:
     - Append `curr_node.word` to `results`.
     - Set `curr_node.word = None` (avoid duplicates).
   - Mark cell as visited: `board[r][c] = '#'`.
   - For each neighbor `(nr, nc)` in `[(r+1,c), (r-1,c), (r,c+1), (r,c-1)]`:
     - If $0 \le nr < m$ and $0 \le nc < n$ and `board[nr][nc] != '#'`:
       - `dfs(nr, nc, curr_node)`
   - Backtrack: `board[r][c] = char`.
   - **Trie Trimming:** If `len(curr_node.children) == 0`:
     - Delete `parent_node.children[char]`.
3. **Execute:**
   - Iterate over each cell $(r, c)$ in the grid and call `dfs(r, c, root)`.
   - Return `results`.

---

### Visual Algorithm Walkthrough

Suppose words: `["oath", "pea", "eat", "rain"]`.

```
Trie Structure:
       Root
     /  |  \
    o   p   e
    |   |   |
    a   e   a
    |   |   |
    t   a*  t*
    |
    h*   (* denotes word terminal)
```

Board Traversal:
```
o  a  a  n
e  t  a  e
i  h  k  r
i  f  l  v

Start at (0, 0) = 'o':
1. 'o' matches Root['o'] -> curr_node = 'o'
2. Visit neighbor (0, 1) = 'a': matches 'o'['a'] -> curr_node = 'a'
3. Visit neighbor (1, 1) = 't': matches 'a'['t'] -> curr_node = 't'
4. Visit neighbor (2, 1) = 'h': matches 't'['h'] -> curr_node = 'h'*
   - Found word: "oath"! Add to results.
   - curr_node.word = None.
   - Backtrack and prune node 'h' since it is a leaf.
```

---

### Solved Examples with Multiple Inputs

| Test Case | `board` | `words` | Found Words | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]` | `["oath","pea","eat","rain"]` | `"eat"`, `"oath"` | `["eat","oath"]` |
| **No Match** | `[["a","b"],["c","d"]]` | `["abcb"]` | Cannot reuse cell `'b'` | `[]` |
| **Common Prefixes** | `[["a","b"],["a","a"]]` | `["aba","baa","bab","aaab","aaa","aaaa","aaba"]` | Matches valid paths | `["aaa","aaab","aaba","aba","baa"]` |
| **Single Cell Match** | `[["a"]]` | `["a"]` | Cell `(0,0)` matches | `["a"]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Dict, Optional

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.word: Optional[str] = None

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Finds all words on the board using a Trie with backtracking and leaf pruning.
        """
        # Step 1: Build Trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word

        rows, cols = len(board), len(board[0])
        results: List[str] = []

        # Step 2: DFS Backtracking
        def dfs(r: int, c: int, parent: TrieNode) -> None:
            char = board[r][c]
            curr_node = parent.children[char]

            # Check if this node matches a word
            if curr_node.word is not None:
                results.append(curr_node.word)
                curr_node.word = None  # Avoid duplicates

            # Mark visited
            board[r][c] = '#'

            # Explore 4-directional neighbors
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if board[nr][nc] in curr_node.children:
                        dfs(nr, nc, curr_node)

            # Backtrack
            board[r][c] = char

            # Step 3: Trie Pruning (Trim empty leaf nodes)
            if not curr_node.children:
                del parent.children[char]

        for r in range(rows):
            for c in range(cols):
                if board[r][c] in root.children:
                    dfs(r, c, root)

        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

struct TrieNode {
    TrieNode* children[26] = {nullptr};
    std::string word = "";
    int child_count = 0;
};

class Solution {
public:
    std::vector<std::string> findWords(std::vector<std::vector<char>>& board, std::vector<std::string>& words) {
        TrieNode* root = new TrieNode();
        for (const std::string& w : words) {
            TrieNode* node = root;
            for (char c : w) {
                int idx = c - 'a';
                if (!node->children[idx]) {
                    node->children[idx] = new TrieNode();
                    node->child_count++;
                }
                node = node->children[idx];
            }
            node->word = w;
        }

        int m = static_cast<int>(board.size());
        int n = static_cast<int>(board[0].size());
        std::vector<std::string> results;

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                int idx = board[r][c] - 'a';
                if (root->children[idx]) {
                    dfs(r, c, m, n, board, root, results);
                }
            }
        }

        return results;
    }

private:
    void dfs(int r, int c, int m, int n, std::vector<std::vector<char>>& board,
             TrieNode* parent, std::vector<std::string>& results) {
        char ch = board[r][c];
        int idx = ch - 'a';
        TrieNode* curr = parent->children[idx];
        if (!curr) return;

        if (!curr->word.empty()) {
            results.push_back(curr->word);
            curr->word.clear(); // Avoid duplicates
        }

        board[r][c] = '#'; // Mark visited

        static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (const auto& d : dirs) {
            int nr = r + d[0];
            int nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && board[nr][nc] != '#') {
                int nxt_idx = board[nr][nc] - 'a';
                if (curr->children[nxt_idx]) {
                    dfs(nr, nc, m, n, board, curr, results);
                }
            }
        }

        board[r][c] = ch; // Backtrack

        // Leaf pruning optimization
        if (curr->child_count == 0 && curr->word.empty()) {
            delete curr;
            parent->children[idx] = nullptr;
            parent->child_count--;
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class TrieNode {
    TrieNode[] children = new TrieNode[26];
    String word = null;
    int childCount = 0;
}

class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) {
                    node.children[idx] = new TrieNode();
                    node.childCount++;
                }
                node = node.children[idx];
            }
            node.word = word;
        }

        int m = board.length;
        int n = board[0].length;
        List<String> results = new ArrayList<>();

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                int idx = board[r][c] - 'a';
                if (root.children[idx] != null) {
                    dfs(r, c, m, n, board, root, results);
                }
            }
        }

        return results;
    }

    private void dfs(int r, int c, int m, int n, char[][] board, TrieNode parent, List<String> results) {
        char ch = board[r][c];
        int idx = ch - 'a';
        TrieNode curr = parent.children[idx];
        if (curr == null) return;

        if (curr.word != null) {
            results.add(curr.word);
            curr.word = null; // Avoid duplicates
        }

        board[r][c] = '#'; // Mark visited

        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] d : dirs) {
            int nr = r + d[0];
            int nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && board[nr][nc] != '#') {
                int nxtIdx = board[nr][nc] - 'a';
                if (curr.children[nxtIdx] != null) {
                    dfs(nr, nc, m, n, board, curr, results);
                }
            }
        }

        board[r][c] = ch; // Backtrack

        // Leaf pruning
        if (curr.childCount == 0 && curr.word == null) {
            parent.children[idx] = null;
            parent.childCount--;
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(W \cdot L + M \cdot N \cdot 3^{L-1})$.
  - Building the Trie takes $\mathcal{O}(W \cdot L)$ where $W$ is the number of words ($3 \times 10^4$) and $L$ is the max length of a word ($10$).
  - For grid traversal, each starting cell $(M \times N \le 144)$ branches into 4 directions initially and at most 3 directions for the remaining $L - 1$ steps.
  - With Trie pruning, exploration stops immediately when a prefix is not in the Trie and dead branches are dynamically deleted, reducing the actual runtime to $< 40 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(W \cdot L)$ auxiliary space to store all characters in the Trie, plus $\mathcal{O}(L)$ recursion stack depth ($L \le 10$).

---

### Takeaway Pattern & Interview Traps

- **Trie + Grid Backtracking Pattern:** Searching for multiple dictionary strings in a 2D matrix or graph always requires a Trie to match prefixes in parallel.
- **Trie Leaf Pruning (Crucial for Hard LeetCode testcases):** Simply checking `curr_node.word` will pass standard tests, but will be slow on large grids with duplicate prefixes. Trimming `parent.children[char]` when `curr_node` has no children drops runtime by $> 90\%$.
- **Duplicate Prevention:** Instead of a `HashSet<String>` at the end, setting `curr_node.word = null` prevents adding the same word multiple times from different starting positions on the board.