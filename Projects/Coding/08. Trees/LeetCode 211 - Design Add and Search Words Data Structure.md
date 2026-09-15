---
date: "2026-08-29"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 211: Design Add and Search Words Data Structure"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 211: Design Add and Search Words Data Structure

**Target Companies:** Google (Wildcard Pattern Matching Trie)

---

### Problem Statement

Design a data structure that supports adding words and finding if a string matches any previously added string, where `'.'` matches any character.

---

### Key Observation

* Store words in a Trie.
* When searching a exact character, traverse standard child node.
* When encountering wildcard `'.'`, explore ALL child branches using DFS/backtracking.

---

### Core Technique: Trie with DFS Wildcard Branching

---

### Python 3 Solution (with typing)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_end = True

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, idx: int) -> bool:
            if idx == len(word):
                return node.is_end
            c = word[idx]
            if c == '.':
                for child in node.children.values():
                    if dfs(child, idx + 1):
                        return True
                return False
            else:
                if c not in node.children:
                    return False
                return dfs(node.children[c], idx + 1)
                
        return dfs(self.root, 0)
```

---

### Worked-Out Example

```python
addWord("bad"), addWord("dad"), addWord("mad")
search("pad") -> False
search("bad") -> True
search(".ad") -> matches 'b', 'd', 'm' -> True
search("b..") -> matches 'a' -> 'd' -> True
```

---

### Complexity Analysis

* **Time Complexity:** `O(L) for addWord, O(26^L) worst case for search with all wildcards`
* **Space Complexity:** `O(total characters)`

---

### Takeaway Pattern

Branch recursive DFS across all `node.children.values()` when matching regex `.` wildcards in a Trie.