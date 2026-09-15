---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 127: Word Ladder"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - hash-table
  - amazon
  - google
---

# LeetCode 127: Word Ladder

**Target Companies:** Amazon (All-Time Signature BFS Classic), Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Hard  
**Topic:** Unweighted Shortest Path / Breadth-First Search (BFS) / Word Transformation Graph

---

### Problem Statement

A **transformation sequence** from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s_1 -> s_2 -> ... -> s_k` such that:
- Every adjacent pair of words differs by a single letter.
- Every $s_i$ for $1 \le i \le k$ is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
- $s_k == \text{endWord}$

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return the **number of words** in the **shortest transformation sequence** from `beginWord` to `endWord`, or `0` if no such sequence exists.

---

### Input & Output Formats & Constraints

- **Input:** `beginWord: str`, `endWord: str`, `wordList: List[str]`
- **Output:** `int` — Length of shortest transformation sequence (number of words including `beginWord` and `endWord`), or `0`.
- **Constraints:**
  - $1 \le \text{beginWord.length} \le 10$
  - $\text{endWord.length} == \text{beginWord.length}$
  - $1 \le \text{wordList.length} \le 5000$
  - $\text{wordList}[i].\text{length} == \text{beginWord.length}$
  - `beginWord`, `endWord`, and `wordList[i]` consist of lowercase English letters.
  - `beginWord != endWord`
  - All strings in `wordList` are **unique**.

---

### Key Idea & Intuition

- **Graph Abstraction:**
  - Model each word as a vertex in an unweighted, undirected graph.
  - An edge exists between two words if and only if they differ by exactly one character.
  - Finding the shortest sequence from `beginWord` to `endWord` is equivalent to finding the **shortest path in an unweighted graph**, for which **Breadth-First Search (BFS)** is guaranteed optimal.
- **Efficient Neighbor Generation ($\mathcal{O}(26 \cdot L)$ vs $\mathcal{O}(N \cdot L)$):**
  - Naive pair comparison compares current word with all $N$ words in `wordList`, costing $\mathcal{O}(N \cdot L)$ per step.
  - Instead, convert `wordList` into a hash set `word_set`.
  - For a word of length $L$, try substituting each of its $L$ character positions with all 26 lowercase English letters `'a'` through `'z'`.
  - Check if each generated word exists in `word_set`.
  - Number of lookups is $26 \times L \le 260$. Since $260 \ll N = 5000$, this reduces neighbor generation time dramatically!
  - When a valid word is found, remove it from `word_set` immediately (acting as both visited set and neighbor filter).

---

### Solution Approach (Step-by-Step)

1. Convert `wordList` to a set: `word_set = set(wordList)`.
2. Edge case: If `endWord not in word_set`: return `0`.
3. Initialize `queue = deque([(beginWord, 1)])`.
4. While `queue` is not empty:
   - Pop `(curr_word, steps)`.
   - If `curr_word == endWord`: return `steps`.
   - For index $i$ from $0$ to $L - 1$:
     - For char $c$ in `'abcdefghijklmnopqrstuvwxyz'`:
       - If $c == curr\_word[i]$: continue.
       - Construct candidate: `next_word = curr_word[:i] + c + curr_word[i+1:]`.
       - If `next_word in word_set`:
         - Remove `next_word` from `word_set`.
         - Append `(next_word, steps + 1)` to `queue`.
5. If queue empties without reaching `endWord`, return `0`.

---

### Visual Algorithm Walkthrough

```
beginWord = "hit", endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]

BFS Level 1 (steps = 1):
Queue: [("hit", 1)]

Level 2 (steps = 2):
Pop ("hit", 1) -> mutate 'h'->'c' (cit? No), ..., 'i'->'o' ("hot"? Yes!)
Enqueue ("hot", 2)
Queue: [("hot", 2)]
Remove "hot" from word_set.

Level 3 (steps = 3):
Pop ("hot", 2) -> mutate "hot":
  - 'h'->'d' -> "dot" (Yes!)
  - 'h'->'l' -> "lot" (Yes!)
Enqueue ("dot", 3), ("lot", 3)
Queue: [("dot", 3), ("lot", 3)]

Level 4 (steps = 4):
Pop ("dot", 3) -> mutates to "dog" (Yes!)
Pop ("lot", 3) -> mutates to "log" (Yes!)
Queue: [("dog", 4), ("log", 4)]

Level 5 (steps = 5):
Pop ("dog", 4) -> mutates to "cog" (Matches endWord!)
Return steps = 5!

Shortest Sequence: "hit" -> "hot" -> "dot" -> "dog" -> "cog" (Length 5)
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Valid Transformation
- **Input:** `beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]`
- **Tracing Table:**
  | Step / Level | Word Popped | New Valid Neighbors Enqueued | Remaining in `word_set` |
  | :--- | :--- | :--- | :--- |
  | 1 | `"hit"` | `[("hot", 2)]` | `dot, dog, lot, log, cog` |
  | 2 | `"hot"` | `[("dot", 3), ("lot", 3)]` | `dog, log, cog` |
  | 3 | `"dot"` | `[("dog", 4)]` | `log, cog` |
  | 3 | `"lot"` | `[("log", 4)]` | `cog` |
  | 4 | `"dog"` | `[("cog", 5)]` (matches `endWord`) | - |
- **Output:** `5`

#### Example 2: Target Word Missing from Dictionary
- **Input:** `beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]`
- **Check:** `"cog"` not in `wordList` $\implies$ return `0` immediately.
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if endWord not in word_set:
            return 0
            
        queue = deque([(beginWord, 1)])
        # beginWord does not need to be in wordList
        if beginWord in word_set:
            word_set.remove(beginWord)
            
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
        while queue:
            word, steps = queue.popleft()
            if word == endWord:
                return steps
                
            for i in range(len(word)):
                prefix = word[:i]
                suffix = word[i+1:]
                for c in alphabet:
                    if c == word[i]:
                        continue
                    next_word = prefix + c + suffix
                    if next_word in word_set:
                        word_set.remove(next_word)
                        queue.append((next_word, steps + 1))
                        
        return 0
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>
#include <unordered_set>
#include <queue>

class Solution {
public:
    int ladderLength(std::string beginWord, std::string endWord, std::vector<std::string>& wordList) {
        std::unordered_set<std::string> wordSet(wordList.begin(), wordList.end());
        if (wordSet.find(endWord) == wordSet.end()) {
            return 0;
        }

        std::queue<std::pair<std::string, int>> q;
        q.push({beginWord, 1});
        wordSet.erase(beginWord);

        while (!q.empty()) {
            auto [word, steps] = q.front();
            q.pop();

            if (word == endWord) {
                return steps;
            }

            for (int i = 0; i < word.length(); ++i) {
                char originalChar = word[i];
                for (char c = 'a'; c <= 'z'; ++c) {
                    if (c == originalChar) continue;
                    word[i] = c;

                    if (wordSet.find(word) != wordSet.end()) {
                        wordSet.erase(word);
                        q.push({word, steps + 1});
                    }
                }
                word[i] = originalChar; // backtrack character
            }
        }

        return 0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> wordSet = new HashSet<>(wordList);
        if (!wordSet.contains(endWord)) {
            return 0;
        }

        Queue<String> queue = new ArrayDeque<>();
        queue.offer(beginWord);
        wordSet.remove(beginWord);
        int steps = 1;

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            for (int k = 0; k < levelSize; k++) {
                String word = queue.poll();
                if (word.equals(endWord)) {
                    return steps;
                }

                char[] chars = word.toCharArray();
                for (int i = 0; i < chars.length; i++) {
                    char original = chars[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        if (c == original) continue;
                        chars[i] = c;
                        String nextWord = new String(chars);

                        if (wordSet.contains(nextWord)) {
                            wordSet.remove(nextWord);
                            queue.offer(nextWord);
                        }
                    }
                    chars[i] = original;
                }
            }
            steps++;
        }

        return 0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M^2 \times N)$ — Where $M$ is the length of each word and $N$ is the total number of words in `wordList`. For each word popped from the queue, we iterate $M$ times, trying $26$ letters. Creating the candidate word of length $M$ takes $\mathcal{O}(M)$. Set lookups take $\mathcal{O}(M)$ time (string hashing). Overall runtime is $\mathcal{O}(26 \times M^2 \times N) = \mathcal{O}(M^2 \times N)$.
- **Space Complexity:** $\mathcal{O}(M \times N)$ — `wordSet` holds $N$ words of length $M$, and the BFS queue holds up to $N$ words.

---

### Takeaway Pattern & Interview Traps

1. **Length Count vs Transformation Count:** The problem asks for the *number of words* in the sequence, not the number of edges/transformations. For `hit -> hot`, the word count is $2$ (not $1$). Initializing `steps = 1` cleanly aligns with this requirement.
2. **Deleting from Set as Visited Marking:** Removing `next_word` from `word_set` as soon as it is enqueued eliminates the need for a separate `visited` set and prevents multiple paths from enqueuing the same word.
3. **Bidirectional BFS Optimization:** For large test cases, running BFS simultaneously from `beginWord` and `endWord` meeting in the middle reduces the search space from $\mathcal{O}(b^d)$ to $\mathcal{O}(2 \times b^{d/2})$, which is an impressive optimization to propose in Google/Amazon interviews.