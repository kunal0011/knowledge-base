---
date: "2026-08-29"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 208: Implement Trie (Prefix Tree)"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 208: Implement Trie (Prefix Tree)

**Target Companies:** Google (Search Autocomplete), Amazon

---

### Problem Statement

Implement a Trie (Prefix Tree) with `insert`, `search`, and `startsWith` methods.

---

### Key Observation

* Each TrieNode holds a dictionary/array of child characters and an `is_end_of_word` boolean flag.
* `insert(word)`: Traverse down, creating nodes for characters that do not exist yet.
* `startsWith(prefix)`: Checks if path exists. `search(word)` additionally checks `is_end == True`.

---

### Core Technique: Trie (Prefix Tree) Data Structure Implementation

---

### Python 3 Solution (with typing)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_end = True

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.is_end

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True
```

---

### Worked-Out Example

```python
insert("apple") -> root -> 'a' -> 'p' -> 'p' -> 'l' -> 'e' (is_end=True)
search("apple") -> True
search("app") -> False (not end of word)
startsWith("app") -> True
```

---

### Complexity Analysis

* **Time Complexity:** `O(L) for insert, search, startsWith where L is word length`
* **Space Complexity:** `O(total characters inserted)`

---

### Takeaway Pattern

The fundamental data structure for prefix searching, autocomplete, and spell checkers.