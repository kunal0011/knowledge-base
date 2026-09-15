---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 208: Implement Trie (Prefix Tree)"
tags:
  - leetcode
  - coding
  - trees
  - trie
  - design
  - amazon
  - google
---

# LeetCode 208: Implement Trie (Prefix Tree)

**Target Companies:** Google (Search Autocomplete & Spell Checking), Amazon, Microsoft, Twitter, Uber  
**Difficulty:** Medium  
**Topic:** Trie (Prefix Tree) Data Structure / Tree Design / Character Traversal

---

### Problem Statement

A **trie** (pronounced as "try") or **prefix tree** is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the `Trie` class:
- `Trie()` Initializes the trie object.
- `void insert(String word)` Inserts the string `word` into the trie.
- `boolean search(String word)` Returns `true` if the string `word` is in the trie (i.e., was inserted before), and `false` otherwise.
- `boolean startsWith(String prefix)` Returns `true` if there is a previously inserted string `word` that has the prefix `prefix`, and `false` otherwise.

---

### Input & Output Formats & Constraints

- **Operations:**
  - `insert(word: str) -> None`
  - `search(word: str) -> bool`
  - `startsWith(prefix: str) -> bool`
- **Constraints:**
  - $1 \le \text{word.length}, \text{prefix.length} \le 2000$
  - `word` and `prefix` consist only of lowercase English letters.
  - At most $3 \times 10^4$ calls in total will be made to `insert`, `search`, and `startsWith`.

---

### Key Idea & Intuition

- **Trie Invariant:**
  - Unlike a binary tree where nodes store keys, positions in a Trie determine the key associated with that node.
  - All descendants of a node share a common prefix associated with that node.
  - Each `TrieNode` consists of:
    1. A collection of child pointers (either an array of size 26 for `'a'` through `'z'`, or a hash map).
    2. A boolean flag `is_end` (or `is_terminal`), signifying that a complete word terminates at this node.
- **Differentiating `search` vs `startsWith`:**
  - Both methods traverse down character by character:
    - If a character child pointer does not exist, both return `False`.
    - At the end of the query string:
      - `startsWith` returns `True` (a valid prefix path exists).
      - `search` returns `curr.is_end` (the path represents a complete word, not just an incomplete prefix).

---

### Solution Approach (Step-by-Step)

1. Define `TrieNode`:
   - `children`: Array of size 26 or dictionary.
   - `is_end`: Boolean flag initialized to `False`.
2. In `Trie.__init__()`:
   - Initialize `root = TrieNode()`.
3. In `insert(word)`:
   - Start at `curr = root`.
   - For each character $c$ in `word`:
     - If child for $c$ doesn't exist, instantiate a new `TrieNode`.
     - Advance `curr = curr.children[c]`.
   - Mark `curr.is_end = True`.
4. In `search(word)`:
   - Start at `curr = root`.
   - For each character $c$ in `word`:
     - If child $c$ is missing, return `False`.
     - Advance `curr = curr.children[c]`.
   - Return `curr.is_end`.
5. In `startsWith(prefix)`:
   - Start at `curr = root`.
   - For each character $c$ in `prefix`:
     - If child $c$ is missing, return `False`.
     - Advance `curr = curr.children[c]`.
   - Return `True`.

---

### Visual Algorithm Walkthrough

```
Inserting: "apple", "app", "apt"

         (root)
           |
          'a'
           |
          'p'
           |
          'p' (is_end = True)  <- Word: "app"
         /   \
       'l'   't' (is_end = True) <- Word: "apt"
        |
       'e' (is_end = True)     <- Word: "apple"

Queries:
- startsWith("ap") -> Traverse 'a' -> 'p' -> Found node! Return True.
- search("app")     -> Traverse 'a' -> 'p' -> 'p' -> is_end is True! Return True.
- search("appl")    -> Traverse 'a' -> 'p' -> 'p' -> 'l' -> is_end is False. Return False.
- startsWith("apt") -> Traverse 'a' -> 'p' -> 't' -> Found node! Return True.
- search("bat")     -> 'b' not in root.children -> Return False immediately.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Lifecycle
- **Calls:**
  ```python
  trie = Trie()
  trie.insert("apple")
  trie.search("apple")   # return True
  trie.search("app")     # return False
  trie.startsWith("app") # return True
  trie.insert("app")
  trie.search("app")     # return True
  ```
- **Step Trace:**
  | Operation | Argument | Path Traversed | End Node `is_end` | Return Value |
  | :--- | :--- | :--- | :--- | :--- |
  | `insert` | `"apple"` | Root $\to$ a $\to$ p $\to$ p $\to$ l $\to$ e | Marked `True` | `None` |
  | `search` | `"apple"` | Root $\to$ a $\to$ p $\to$ p $\to$ l $\to$ e | `True` | `True` |
  | `search` | `"app"` | Root $\to$ a $\to$ p $\to$ p | `False` | `False` |
  | `startsWith` | `"app"` | Root $\to$ a $\to$ p $\to$ p | Exists | `True` |
  | `insert` | `"app"` | Root $\to$ a $\to$ p $\to$ p | Marked `True` | `None` |
  | `search` | `"app"` | Root $\to$ a $\to$ p $\to$ p | `True` | `True` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
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
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_end = True

    def search(self, word: str) -> bool:
        curr = self.root
        for char in word:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return curr.is_end

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return True
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class TrieNode {
public:
    std::vector<TrieNode*> children;
    bool isEnd;

    TrieNode() : children(26, nullptr), isEnd(false) {}
    ~TrieNode() {
        for (auto child : children) {
            delete child;
        }
    }
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    ~Trie() {
        delete root;
    }

    void insert(const std::string& word) {
        TrieNode* curr = root;
        for (char c : word) {
            int idx = c - 'a';
            if (curr->children[idx] == nullptr) {
                curr->children[idx] = new TrieNode();
            }
            curr = curr->children[idx];
        }
        curr->isEnd = true;
    }

    bool search(const std::string& word) {
        TrieNode* curr = root;
        for (char c : word) {
            int idx = c - 'a';
            if (curr->children[idx] == nullptr) {
                return false;
            }
            curr = curr->children[idx];
        }
        return curr->isEnd;
    }

    bool startsWith(const std::string& prefix) {
        TrieNode* curr = root;
        for (char c : prefix) {
            int idx = c - 'a';
            if (curr->children[idx] == nullptr) {
                return false;
            }
            curr = curr->children[idx];
        }
        return true;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class TrieNode {
    TrieNode[] children;
    boolean isEnd;

    public TrieNode() {
        children = new TrieNode[26];
        isEnd = false;
    }
}

class Trie {
    private final TrieNode root;

    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word) {
        TrieNode curr = root;
        for (int i = 0; i < word.length(); i++) {
            int idx = word.charAt(i) - 'a';
            if (curr.children[idx] == null) {
                curr.children[idx] = new TrieNode();
            }
            curr = curr.children[idx];
        }
        curr.isEnd = true;
    }

    public boolean search(String word) {
        TrieNode curr = root;
        for (int i = 0; i < word.length(); i++) {
            int idx = word.charAt(i) - 'a';
            if (curr.children[idx] == null) {
                return false;
            }
            curr = curr.children[idx];
        }
        return curr.isEnd;
    }

    public boolean startsWith(String prefix) {
        TrieNode curr = root;
        for (int i = 0; i < prefix.length(); i++) {
            int idx = prefix.charAt(i) - 'a';
            if (curr.children[idx] == null) {
                return false;
            }
            curr = curr.children[idx];
        }
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `insert(word)`: $\mathcal{O}(L)$ where $L$ is the length of `word`.
  - `search(word)`: $\mathcal{O}(L)$ where $L$ is the length of `word`.
  - `startsWith(prefix)`: $\mathcal{O}(P)$ where $P$ is the length of `prefix`.
  - Each step is a direct $\mathcal{O}(1)$ pointer hop or hash map lookup.
- **Space Complexity:** $\mathcal{O}(|\Sigma| \cdot L \cdot N)$ where $|\Sigma| = 26$ is alphabet size, $L$ is average word length, and $N$ is number of inserted words. In the worst case with no shared prefixes, every character allocates a new node.

---

### Takeaway Pattern & Interview Traps

1. **Fixed Array vs. Hash Map Trade-off:**
   - Fixed Array `[26]`: Constant time array index `char - 'a'`, highly cache-friendly, but allocates 26 references per node regardless of fan-out.
   - Hash Map `dict`: Dynamically allocates only existing children, saving memory when branching is sparse or alphabet includes unicode characters.
2. **Memory Leaks in C++:** In C++, make sure destructors recursively clean up allocated child nodes to prevent memory leaks in production systems.
3. **Prefix vs Word Termination:** Always verify `curr.is_end` in `search(word)` to distinguish between an actual word and a prefix of a longer word.