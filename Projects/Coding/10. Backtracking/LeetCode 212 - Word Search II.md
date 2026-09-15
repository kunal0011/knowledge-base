---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 212: Word Search II"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 212: Word Search II

## LeetCode 212 — Word Search II

---

## Problem Statement

Given an `m x n` board of characters and a list of strings `words`, return **all words on the board**.

**Rules**

* Words must be constructed from **adjacent cells** (horizontal or vertical).
* The **same cell cannot be used more than once** in a single word.
* Return all words that can be found on the board.

**Constraints**

* `1 ≤ m, n ≤ 12`
* `1 ≤ words.length ≤ 3 * 10⁴`
* `1 ≤ words[i].length ≤ 10`
* All words consist of lowercase English letters.

---

## Key Observations (Critical)

1. A naive DFS **for each word** is too slow  
   → `O(words × board × 4^L)` (TLE).
2. Many words share **common prefixes**.
3. We must:

   * Search **all words simultaneously**
   * Stop early when the current prefix is invalid
4. The correct data structure is a **Trie (Prefix Tree)** combined with **backtracking**.
5. The board traversal is DFS, but the **search space is pruned by the Trie**.

---

## High-Level Approach

1. Build a **Trie** from all words.
2. Start DFS from **every cell** in the board.
3. While exploring neighbors:

   * Move forward in the Trie
   * Stop immediately if the prefix does not exist
4. When a Trie node marks a complete word → add to result.
5. Mark cells as visited during DFS and restore after backtracking.

---

## Python 3 Solution (with Typing)

```python
from typing import List, Dict, Optional

class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.word: Optional[str] = None  # stores complete word at terminal node

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()

        # Build Trie
        for word in words:
            node = root
            for ch in word:
                node = node.children.setdefault(ch, TrieNode())
            node.word = word

        rows, cols = len(board), len(board[0])
        result: List[str] = []

        def backtrack(r: int, c: int, node: TrieNode) -> None:
            char = board[r][c]
            if char not in node.children:
                return

            next_node = node.children[char]

            # Found a word
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None  # avoid duplicates

            board[r][c] = "#"  # mark visited

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                    backtrack(nr, nc, next_node)

            board[r][c] = char  # restore

        for i in range(rows):
            for j in range(cols):
                backtrack(i, j, root)

        return result
```

---

## Example Explanation

### Input

```
board =
[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

words = ["oath","pea","eat","rain"]
```

### Output

```
["oath","eat"]
```

---

## Complete Conceptual Backtracking Tree (Key Requirement)

### Trie (Prefix Structure)

```
(root)
 ├── o
 │    └── a
 │         └── t
 │              └── h  (word = "oath")
 ├── e
 │    └── a
 │         └── t  (word = "eat")
 ├── p → e → a
 └── r → a → i → n
```

---

### Board-Based DFS + Trie Navigation (Conceptual Tree)

Starting at cell `'o'` (0,0):

```
DFS from board[0][0] = 'o'
Trie: root → o
|
└── 'o'
    |
    └── 'a' (0,1)
        |
        └── 't' (1,1)
            |
            └── 'h' (2,1)
                |
                └── FOUND "oath"
```

---

Starting at cell `'e'` (1,3):

```
DFS from board[1][3] = 'e'
Trie: root → e
|
└── 'e'
    |
    └── 'a' (1,2)
        |
        └── 't' (1,1)
            |
            └── FOUND "eat"
```

---

### Pruned Branch Example (Early Termination)

```
DFS from 'o'
 |
 └── 'o' → 'a'
        |
        └── 'a' ❌   (No Trie child)
            PRUNE ENTIRE SUBTREE
```

---

## Key Insight About the Tree

* **Depth = max word length**
* **Branching = at most 4 directions**
* Trie ensures:

  * Valid prefixes continue
  * Invalid prefixes terminate immediately
* The DFS tree is **guided**, not brute-force

---

## Why This Works Efficiently

| Technique | Benefit |
| --- | --- |
| Trie | Prefix pruning |
| DFS | Board traversal |
| In-place marking | No extra visited matrix |
| Word removal | Avoid duplicate results |

---

## Complexity Analysis

* **Time Complexity:**  
  `O(m × n × 4^L)` in the worst case, but **heavily pruned** by Trie.
* **Space Complexity:**  
  `O(total letters in Trie + recursion depth)`

---

## One-Line Interview Explanation

> “We build a Trie from all words and perform DFS from each board cell, pruning paths that don’t match Trie prefixes and collecting words when reaching terminal nodes.”

---

If you want next:

* Visual comparison of **LC 79 vs LC 212**
* Explanation of **why removing `node.word` avoids duplicates**
* Memory-optimized Trie variants