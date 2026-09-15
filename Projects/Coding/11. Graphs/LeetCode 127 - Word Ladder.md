---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 127: Word Ladder"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 127: Word Ladder

---

### Problem Statement

Given two words `beginWord` and `endWord`, and a dictionary `wordList`, return the number of words in the shortest transformation sequence from `beginWord` to `endWord`.

---

### Key Observation

* Each word is a node in a graph. An edge exists if words differ by exactly 1 character.
* Shortest path on unweighted graph $
  ightarrow$ **Breadth-First Search (BFS)**.
* Preprocess wildcard patterns (e.g. `*ot -> [hot, dot, lot]`) in a Hash Map to find valid next words in `O(26 * L)` instead of `O(N * L)`.

---

### Core Technique: BFS with Pattern Wildcard Graph Modeling

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque, defaultdict

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
            
        patterns = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                pattern = word[:i] + "*" + word[i+1:]
                patterns[pattern].append(word)
                
        queue = deque([(beginWord, 1)])
        visited = {beginWord}
        
        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length
                
            for i in range(len(word)):
                pattern = word[:i] + "*" + word[i+1:]
                for neighbor in patterns[pattern]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, length + 1))
                        
        return 0
```

---

### Worked-Out Example

```python
beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
hit (len 1) -> hot (len 2) -> dot/lot (len 3) -> dog/log (len 4) -> cog (len 5)
Result = 5
```

---

### Complexity Analysis

* **Time Complexity:** `O(M^2 * N) where M is word length, N is number of words`
* **Space Complexity:** `O(M^2 * N)`

---

### Takeaway Pattern

Use wildcard pattern indexing `word[:i] + '\*' + word[i+1:]` for instant neighbor discovery in string graphs.