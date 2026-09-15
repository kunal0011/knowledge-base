---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 269: Alien Dictionary"
tags:
  - leetcode
  - coding
  - graphs
  - topological-sort
  - bfs
  - amazon
  - google
---

# LeetCode 269: Alien Dictionary

**Target Companies:** Meta (All-Time #1 Top Topological Sort Classic), Google, Amazon, Microsoft, Airbnb  
**Difficulty:** Hard  
**Topic:** Topological Sort / Directed Graph Cycle Detection / Lexicographical Ordering

---

### Problem Statement

There is a new alien language that uses the English alphabet. However, the order of the letters is unknown to you.

You are given a list of strings `words` from the alien language's dictionary, where the strings in `words` are **sorted lexicographically** by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in **lexicographically increasing order** by the new language's rules. If there is no possible letter order, return `""`. If there are multiple valid letter orders, return **any of them**.

---

### Input & Output Formats & Constraints

- **Input:** `words: List[str]`
- **Output:** `str` (String of unique characters in valid topological order, or `""` if invalid/cycle detected).
- **Constraints:**
  - $1 \le \text{words.length} \le 100$
  - $1 \le \text{words}[i].\text{length} \le 100$
  - `words[i]` consists of only lowercase English letters.

---

### Key Idea & Intuition

- **Graph Formulation from Sorted Word List:**
  - In a lexicographically sorted dictionary, comparing two adjacent words $w_1$ and $w_2$ gives the relative ordering of their first differing characters:
    - If $w_1 = \text{"wrt"}$ and $w_2 = \text{"wrf"}$, their first difference is at index 2: `'t' != 'f'`.
    - This establishes a directed dependency: `'t'` must come before `'f'` ($t \to f$).
- **The Critical Prefix Invalidation Trap:**
  - If $w_2$ is a proper prefix of $w_1$ (for example: $w_1 = \text{"apple"}$, $w_2 = \text{"app"}$), the order is inherently invalid because a shorter prefix must ALWAYS appear before a longer word.
  - In this case, return `""` immediately!
- **Topological Sorting (Kahn's Algorithm):**
  1. Populate an adjacency list and in-degree map containing every unique character present in all words.
  2. For every adjacent pair $(w_1, w_2)$: find the first differing character, add directed edge $w_1[j] \to w_2[j]$, and increment `indegree[w2[j]]`. Stop checking further characters for this pair!
  3. Push all characters with `indegree == 0` into a BFS queue.
  4. Perform standard Kahn's BFS. If the output string contains all unique characters, return it. If fewer characters were processed, a cycle exists ($a \to b \to a$) $\implies$ return `""`.

---

### Solution Approach (Step-by-Step)

1. Collect all unique characters in `words`. Initialize `adj = {c: set() for c in chars}` and `indegree = {c: 0 for c in chars}`.
2. For $i$ from $0$ to $\text{len}(words) - 2$:
   - Let $w_1 = words[i], w_2 = words[i+1]$.
   - If $\text{len}(w_1) > \text{len}(w_2)$ and $w_1.\text{startswith}(w_2)$:
     - Return `""` (invalid prefix order).
   - Find first index $j$ where $w_1[j] \ne w_2[j]$:
     - If $w_2[j]$ not in `adj[w1[j]]`:
       - `adj[w1[j]].add(w2[j])`
       - `indegree[w2[j]] += 1`
     - Break (only the first differing character matters).
3. Initialize `queue = deque([c for c in indegree if indegree[c] == 0])`.
4. Initialize `order = []`.
5. While `queue` is not empty:
   - Pop character `curr`.
   - `order.append(curr)`.
   - For `neighbor` in `adj[curr]`:
     - `indegree[neighbor] -= 1`
     - If `indegree[neighbor] == 0`:
       - `queue.append(neighbor)`
6. Return `"".join(order)` if `len(order) == len(indegree)` else `""`.

---

### Visual Algorithm Walkthrough

```
Input: words = ["wrt","wrf","er","ett","rftt"]

Unique Characters: {'w', 'r', 't', 'f', 'e'} (5 total)

Pairwise Comparisons:
1. "wrt" vs "wrf": first diff at index 2 ('t' != 'f') -> Edge: t -> f
2. "wrf" vs "er":  first diff at index 0 ('w' != 'e') -> Edge: w -> e
3. "er"  vs "ett": first diff at index 1 ('r' != 't') -> Edge: r -> t
4. "ett" vs "rftt":first diff at index 0 ('e' != 'r') -> Edge: e -> r

Directed Graph Edges:
  w -> e -> r -> t -> f

In-Degrees:
w: 0
e: 1
r: 1
t: 1
f: 1

Kahn's BFS Order:
1. Queue = ['w']
2. Pop 'w' -> order = ['w'] -> decrement 'e' (in-degree becomes 0 -> enqueue 'e')
3. Pop 'e' -> order = ['w', 'e'] -> decrement 'r' (enqueue 'r')
4. Pop 'r' -> order = ['w', 'e', 'r'] -> decrement 't' (enqueue 't')
5. Pop 't' -> order = ['w', 'e', 'r', 't'] -> decrement 'f' (enqueue 'f')
6. Pop 'f' -> order = ['w', 'e', 'r', 't', 'f']

Output: "wertf" (5 unique characters ordered, valid DAG)
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Valid Dictionary
- **Input:** `words = ["wrt","wrf","er","ett","rftt"]`
- **Output:** `"wertf"`

#### Example 2: Invalid Prefix Ordering
- **Input:** `words = ["abc","ab"]`
- **Trace:** `"ab"` is a prefix of `"abc"`, but `"abc"` appears first.
- **Output:** `""`

#### Example 3: Cycle Detected
- **Input:** `words = ["z","x","z"]`
- **Edges:**
  - `"z"` vs `"x"` $\implies z \to x$
  - `"x"` vs `"z"` $\implies x \to z$
- **Cycle:** $z \leftrightarrow x$. Neither character has in-degree 0.
- **Output:** `""`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict, deque

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        # 1. Initialize adjacency list and in-degree map for all unique chars
        adj = {c: set() for w in words for c in w}
        indegree = {c: 0 for c in adj}
        
        # 2. Build graph by comparing adjacent words
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            min_len = min(len(w1), len(w2))
            
            # Check for invalid prefix condition (e.g. "abc" before "ab")
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                return ""
                
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in adj[w1[j]]:
                        adj[w1[j]].add(w2[j])
                        indegree[w2[j]] += 1
                    break  # Only the first differing character establishes order
                    
        # 3. Kahn's algorithm BFS
        queue = deque([c for c in indegree if indegree[c] == 0])
        order = []
        
        while queue:
            curr = queue.popleft()
            order.append(curr)
            
            for neighbor in adj[curr]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    
        # 4. Check for cycles
        if len(order) == len(indegree):
            return "".join(order)
        return ""
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>

class Solution {
public:
    std::string alienOrder(std::vector<std::string>& words) {
        std::unordered_map<char, std::unordered_set<char>> adj;
        std::unordered_map<char, int> indegree;

        // Initialize all unique characters
        for (const auto& w : words) {
            for (char c : w) {
                if (indegree.find(c) == indegree.end()) {
                    indegree[c] = 0;
                }
            }
        }

        // Build edges
        for (size_t i = 0; i + 1 < words.size(); ++i) {
            const std::string& w1 = words[i];
            const std::string& w2 = words[i + 1];
            size_t minLen = std::min(w1.length(), w2.length());

            if (w1.length() > w2.length() && w1.substr(0, minLen) == w2) {
                return "";
            }

            for (size_t j = 0; j < minLen; ++j) {
                if (w1[j] != w2[j]) {
                    if (adj[w1[j]].find(w2[j]) == adj[w1[j]].end()) {
                        adj[w1[j]].insert(w2[j]);
                        indegree[w2[j]]++;
                    }
                    break;
                }
            }
        }

        // BFS Kahn's algorithm
        std::queue<char> q;
        for (const auto& [c, deg] : indegree) {
            if (deg == 0) q.push(c);
        }

        std::string order = "";
        while (!q.empty()) {
            char curr = q.front();
            q.pop();
            order += curr;

            for (char next : adj[curr]) {
                if (--indegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        return order.length() == indegree.size() ? order : "";
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> adj = new HashMap<>();
        Map<Character, Integer> indegree = new HashMap<>();

        // Initialize map for all characters present in words
        for (String word : words) {
            for (char c : word.toCharArray()) {
                adj.putIfAbsent(c, new HashSet<>());
                indegree.putIfAbsent(c, 0);
            }
        }

        // Build edges between adjacent words
        for (int i = 0; i < words.length - 1; i++) {
            String w1 = words[i];
            String w2 = words[i + 1];
            int minLen = Math.min(w1.length(), w2.length());

            // Edge case: prefix check
            if (w1.length() > w2.length() && w1.startsWith(w2)) {
                return "";
            }

            for (int j = 0; j < minLen; j++) {
                char c1 = w1.charAt(j);
                char c2 = w2.charAt(j);
                if (c1 != c2) {
                    if (!adj.get(c1).contains(c2)) {
                        adj.get(c1).add(c2);
                        indegree.put(c2, indegree.get(c2) + 1);
                    }
                    break;
                }
            }
        }

        // Queue all nodes with indegree 0
        Queue<Character> queue = new ArrayDeque<>();
        for (char c : indegree.keySet()) {
            if (indegree.get(c) == 0) {
                queue.offer(c);
            }
        }

        StringBuilder order = new StringBuilder();
        while (!queue.isEmpty()) {
            char curr = queue.poll();
            order.append(curr);

            for (char neighbor : adj.get(curr)) {
                indegree.put(neighbor, indegree.get(neighbor) - 1);
                if (indegree.get(neighbor) == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        return order.length() == indegree.size() ? order.toString() : "";
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(C)$ where $C$ is the total sum of lengths of all strings in `words`. Initializing the character map takes $\mathcal{O}(C)$. Comparing adjacent words takes at most $\min(\text{len}(w_1), \text{len}(w_2))$ comparisons, bounded by $C$. The graph contains $|V| \le 26$ vertices and $|E| \le 26^2$ edges, so Kahn's BFS takes $\mathcal{O}(|V| + |E|) = \mathcal{O}(1)$ time. Overall time is strictly linear in input size $\mathcal{O}(C)$.
- **Space Complexity:** $\mathcal{O}(1)$ or $\mathcal{O}(|\Sigma| + |\Sigma|^2)$ auxiliary space where $|\Sigma| = 26$ is the English alphabet size.

---

### Takeaway Pattern & Interview Traps

1. **The Proper Prefix Trap:** If `"apple"` appears before `"app"`, no valid alphabet order exists because prefixes MUST be shorter and appear first in any lexicographical system. Checking `len(w1) > len(w2) and w1[:min_len] == w2` is an absolute requirement that causes most candidate submissions to fail hidden test cases.
2. **First Difference Only:** Once a pair of differing characters $w_1[j] \ne w_2[j]$ is found, you MUST `break`. Subsequent characters in those two words convey no ordering information!
3. **Include All Isolated Characters:** Characters that have no incoming or outgoing edges (such as a single letter word `["z"]`) must still appear in the output. Initializing all characters from every word ensures complete output.