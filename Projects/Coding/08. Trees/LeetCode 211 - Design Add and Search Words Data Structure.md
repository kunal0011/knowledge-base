---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 211: Design Add and Search Words Data Structure"
tags:
  - leetcode
  - coding
  - trees
  - trie
  - dfs
  - backtracking
  - amazon
  - google
---

# LeetCode 211: Design Add and Search Words Data Structure

**Target Companies:** Google (Wildcard Autocomplete & Pattern Matching), Meta, Amazon, Microsoft  
**Difficulty:** Medium  
**Topic:** Trie / DFS Backtracking / Wildcard Pattern Matching (`'.'`)

---

### Problem Statement

Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the `WordDictionary` class:
- `WordDictionary()` Initializes the object.
- `void addWord(word)` Adds `word` to the data structure, it can be matched later.
- `bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.

---

### Input & Output Formats & Constraints

- **Operations:**
  - `addWord(word: str) -> None`
  - `search(word: str) -> bool`
- **Constraints:**
  - $1 \le \text{word.length} \le 25$
  - `word` in `addWord` consists of lowercase English letters.
  - `word` in `search` consists of `'.'` or lowercase English letters.
  - There will be at most $2$ dots in `word` for search queries.
  - At most $10^4$ calls in total will be made to `addWord` and `search`.

---

### Key Idea & Intuition

- **Trie Representation:**
  - Store all words in a standard prefix tree (Trie). Each node contains a map/array of `children` and a boolean `is_end`.
- **Handling Wildcard `'.'`: Backtracking DFS:**
  - When querying character $c$ at index $i$:
    1. **Exact Character ($c \ne '.'$):** Directly check if child exists in `node.children`. If so, advance to child and index $i + 1$. If not, this branch fails (return `False`).
    2. **Wildcard ($c == '.'$):** The dot can match *any* character. Therefore, we must branch into **every existing child node** of the current node using recursive DFS. If any recursive search returns `True`, immediately propagate `True`.
  - At index $i == \text{len}(word)$, return `node.is_end`.

---

### Solution Approach (Step-by-Step)

1. `addWord(word)`:
   - Identical to standard Trie insertion. Traverse character by character, allocating nodes as needed, and mark `curr.is_end = True`.
2. `search(word)`:
   - Define recursive helper `dfs(node, idx)`:
     - Base Case: If `idx == len(word)`, return `node.is_end`.
     - Let `char = word[idx]`.
     - If `char == '.'`:
       - For each child node in `node.children`:
         - If `dfs(child, idx + 1)` is `True`: return `True`.
       - If no child branch returns `True`, return `False`.
     - Else (`char` is an explicit letter):
       - If `char` not in `node.children`, return `False`.
       - Return `dfs(node.children[char], idx + 1)`.
   - Call `dfs(root, 0)`.

---

### Visual Algorithm Walkthrough

```
Words Added: "bad", "dad", "mad"

         (root)
       /   |   \
     'b'  'd'  'm'
      |    |    |
     'a'  'a'  'a'
      |    |    |
     'd'  'd'  'd'
    (end)(end)(end)

Search ".ad":
1. Root: word[0] = '.' -> Try all children:
   - Branch 1: child 'b' -> dfs('b', idx=1)
     - word[1] = 'a' -> child 'a' exists -> dfs('a', idx=2)
       - word[2] = 'd' -> child 'd' exists -> dfs('d', idx=3)
         - idx == 3 == len(word) -> return node.is_end (True!)
   - Early return True! Match found.

Search "b..":
1. Root: word[0] = 'b' -> child 'b' exists.
2. Node 'b': word[1] = '.' -> explore children:
   - child 'a' exists -> dfs('a', idx=2)
     - word[2] = '.' -> explore children:
       - child 'd' exists -> dfs('d', idx=3) -> is_end is True!
   - Early return True!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Lifecycle with Wildcards
- **Calls:**
  ```python
  wd = WordDictionary()
  wd.addWord("bad")
  wd.addWord("dad")
  wd.addWord("mad")
  wd.search("pad") # False
  wd.search("bad") # True
  wd.search(".ad") # True
  wd.search("b..") # True
  ```
- **Step Trace:**
  | Query | Breakdown | Path Explored | Result |
  | :--- | :--- | :--- | :--- |
  | `"pad"` | `'p'` not in root | Root fails | `False` |
  | `"bad"` | `'b' -> 'a' -> 'd'` | Exact path matches and `is_end == True` | `True` |
  | `".ad"` | `'.'` branches into `b, d, m` | First branch `'b' -> 'a' -> 'd'` valid | `True` |
  | `"b.."` | `'b'` $\to$ `'.'` (branches to `a`) $\to$ `'.'` (branches to `d`) | Valid word reached | `True` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
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
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_end = True

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, idx: int) -> bool:
            if idx == len(word):
                return node.is_end
            
            char = word[idx]
            if char == '.':
                # Wildcard: explore all non-null children
                for child in node.children.values():
                    if dfs(child, idx + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return dfs(node.children[char], idx + 1)

        return dfs(self.root, 0)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class WordDictionary {
private:
    struct TrieNode {
        std::vector<TrieNode*> children;
        bool isEnd;
        TrieNode() : children(26, nullptr), isEnd(false) {}
        ~TrieNode() {
            for (auto child : children) {
                delete child;
            }
        }
    };

    TrieNode* root;

    bool dfs(TrieNode* node, const std::string& word, int idx) {
        if (!node) return false;
        if (idx == word.length()) {
            return node->isEnd;
        }

        char c = word[idx];
        if (c == '.') {
            for (int i = 0; i < 26; ++i) {
                if (node->children[i] && dfs(node->children[i], word, idx + 1)) {
                    return true;
                }
            }
            return false;
        } else {
            int childIdx = c - 'a';
            return dfs(node->children[childIdx], word, idx + 1);
        }
    }

public:
    WordDictionary() {
        root = new TrieNode();
    }

    ~WordDictionary() {
        delete root;
    }

    void addWord(const std::string& word) {
        TrieNode* curr = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!curr->children[idx]) {
                curr->children[idx] = new TrieNode();
            }
            curr = curr->children[idx];
        }
        curr->isEnd = true;
    }

    bool search(const std::string& word) {
        return dfs(root, word, 0);
    }
};
```

#### 3. Java (Modern, Typed)
```java
class WordDictionary {
    private static class TrieNode {
        TrieNode[] children = new TrieNode[26];
        boolean isEnd = false;
    }

    private final TrieNode root;

    public WordDictionary() {
        root = new TrieNode();
    }

    public void addWord(String word) {
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
        return dfs(root, word, 0);
    }

    private boolean dfs(TrieNode node, String word, int idx) {
        if (node == null) {
            return false;
        }
        if (idx == word.length()) {
            return node.isEnd;
        }

        char c = word.charAt(idx);
        if (c == '.') {
            for (TrieNode child : node.children) {
                if (child != null && dfs(child, word, idx + 1)) {
                    return true;
                }
            }
            return false;
        } else {
            int childIdx = c - 'a';
            return dfs(node.children[childIdx], word, idx + 1);
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `addWord(word)`: $\mathcal{O}(L)$ where $L$ is word length ($L \le 25$).
  - `search(word)`:
    - Without wildcards: $\mathcal{O}(L)$.
    - With wildcards: In the worst-case scenario where query is all dots `...`, search inspects all nodes up to depth $L$, giving $\mathcal{O}(26^L)$ worst-case theoretical bound. Given constraints (at most 2 dots), branching is bounded by $\mathcal{O}(26^2 \cdot L) \approx \mathcal{O}(L)$.
- **Space Complexity:** $\mathcal{O}(|\Sigma| \cdot L \cdot N)$ where $|\Sigma| = 26$ is alphabet size, $L$ is word length, and $N$ is number of distinct words. Search recursion call stack takes $\mathcal{O}(L)$ frames.

---

### Takeaway Pattern & Interview Traps

1. **Short-Circuit on First Match:** In the wildcard loop `for child in node.children`, immediately `return True` upon finding the first valid path. Never continue searching remaining children once a valid path is found.
2. **End-of-Word Validation:** When reaching the end of the query string (`idx == len(word)`), return `node.is_end`, NOT `True`. A prefix match is not a valid word match.
3. **Dictionary Branch Filtering:** In Python, iterating over `node.children.values()` automatically skips nonexistent character branches, saving iteration overhead compared to checking 26 possible indices.