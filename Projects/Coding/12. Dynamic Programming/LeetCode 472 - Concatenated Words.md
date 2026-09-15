---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 472: Concatenated Words"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - hash-table
  - trie
  - amazon
  - google
---

# LeetCode 472: Concatenated Words

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Word Break Extension / Hash Set / Sorting  

---

### Problem Statement

Given an array of strings `words` (**without duplicates**), return all the **concatenated words** in the given list of words.

A **concatenated word** is defined as a string that can be comprised entirely of at least two shorter words (not necessarily distinct) in the given array.

---

### Input & Output Formats & Constraints

- **Input:** `words: List[str]` — Array of distinct non-empty strings.
- **Output:** `List[str]` — All words formed by concatenating $\ge 2$ shorter words.
- **Constraints:**
  - $1 \le \text{words.length} \le 10^4$
  - $1 \le \text{words}[i].\text{length} \le 30$
  - The total number of characters in all words will not exceed $10^5$.
  - All the strings of `words` are unique.
  - All input strings consist of lowercase English letters only.

---

### Key Idea & Intuition

1. **Reduction to Word Break (LeetCode 139):**
   - For a single word $W$, determining if it can be partitioned into dictionary words is the classic **Word Break** problem:
     $$\text{dp}[i] = \exists j < i \text{ s.t. } \text{dp}[j] \land W[j \dots i-1] \in \text{dictionary}$$
   - But we must enforce the condition: **at least two component words must be used** (a word cannot simply match itself).

2. **The Sorting by Length Strategy:**
   - If we sort the word list by **length ascending**:
     - Any word $W$ can only be formed by concatenating words that are **strictly shorter** than $W$ (or of equal length, but since components must be shorter, equal length components cannot form a word without empty strings).
     - We can maintain an active `word_set` dictionary.
     - When inspecting word $W$, the dictionary contains **only words processed before $W$** (i.e. shorter words).
     - If Word Break DP returns `true` on $W$ against this dictionary, $W$ is guaranteed to be a valid concatenated word!
     - Afterwards, insert $W$ into `word_set` so that subsequent, longer words can use $W$ as a component.

3. **Word Break DP Formulation:**
   - For a word $W$ of length $L$:
   - $\text{dp}[0] = \text{true}$ (empty prefix is trivially matched).
   - For $i$ from $1$ to $L$:
     - For $j$ from $0$ to $i - 1$:
       - If $\text{dp}[j] == \text{true}$ and $W[j \dots i - 1] \in \text{word\_set}$:
         - $\text{dp}[i] = \text{true}$, break.
   - If $\text{dp}[L] == \text{true}$, add $W$ to result.

---

### Solution Approach (Step-by-Step)

1. **Sort Words:**
   - Sort `words` by length in ascending order.
2. **Initialize Dictionary & Result:**
   - `word_set = set()`
   - `result = []`
3. **Process Each Word:**
   - Define helper `can_form(word)`:
     - $L = \text{len}(word)$.
     - If $L == 0$: return `False`.
     - $\text{dp} = [\text{False}] \times (L + 1)$, $\text{dp}[0] = \text{True}$.
     - For $i$ from $1$ to $L$:
       - For $j$ from $0$ to $i - 1$:
         - If $\text{dp}[j]$ and $\text{word}[j:i] \in \text{word\_set}$:
           - $\text{dp}[i] = \text{True}$
           - break
     - Return $\text{dp}[L]$.
   - For each `w` in sorted `words`:
     - If `can_form(w)`: append `w` to `result`.
     - Add `w` to `word_set`.
4. **Return:**
   - Return `result`.

---

### Visual Algorithm Walkthrough

Suppose `words = ["cat", "cats", "dog", "catsdog"]`.

**Step 1: Sorted by length:**
`["cat", "dog", "cats", "catsdog"]`

```
1. Word: "cat" (len 3)
   word_set = {}
   can_form("cat") -> False
   Add "cat" to word_set: {"cat"}

2. Word: "dog" (len 3)
   word_set = {"cat"}
   can_form("dog") -> False
   Add "dog" to word_set: {"cat", "dog"}

3. Word: "cats" (len 4)
   word_set = {"cat", "dog"}
   j=0..3: "cat" in set (dp[3]=True), but "s" not in set
   can_form("cats") -> False
   Add "cats" to word_set: {"cat", "dog", "cats"}

4. Word: "catsdog" (len 7)
   word_set = {"cat", "dog", "cats"}
   dp[0] = True
   i = 3: j = 0 -> "cat" in set -> dp[3] = True
   i = 4: j = 0 -> "cats" in set -> dp[4] = True
   i = 7: j = 4 -> dp[4] is True and "catsdog"[4:7] == "dog" in set!
          -> dp[7] = True!
   can_form("catsdog") -> True!
   Append "catsdog" to result.

Final Result: ["catsdog"]
```

---

### Solved Examples with Multiple Inputs

| Case | `words` | Detected Concatenations | Result |
|---|---|---|---|
| **Standard** | `["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]` | `"catsdogcats"` (`cats`+`dog`+`cats`), `"dogcatsdog"`, `"ratcatdogcat"` | `["catsdogcats","dogcatsdog","ratcatdogcat"]` |
| **All Atomic** | `["a", "b", "c"]` | No word is composed of others | `[]` |
| **Chain Composition** | `["a", "aa", "aaa"]` | `"aa"` (`a`+`a`), `"aaa"` (`a`+`a`+`a` or `a`+`aa`) | `["aa", "aaa"]` |
| **Duplicate Fragments** | `["cat", "dog", "catdog"]` | `"catdog"` (`cat`+`dog`) | `["catdog"]` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Set

class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        # Sort by length ascending so shorter components are processed first
        words.sort(key=len)
        word_set: Set[str] = set()
        result: List[str] = []
        
        def can_form(word: str) -> bool:
            n = len(word)
            if n == 0:
                return False
            
            dp = [False] * (n + 1)
            dp[0] = True
            
            for i in range(1, n + 1):
                for j in range(i):
                    if dp[j] and word[j:i] in word_set:
                        dp[i] = True
                        break
                        
            return dp[n]

        for w in words:
            if can_form(w):
                result.append(w)
            word_set.add(w)
            
        return result
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>

class Solution {
public:
    std::vector<std::string> findAllConcatenatedWordsInADict(std::vector<std::string>& words) {
        // Sort by length ascending
        std::sort(words.begin(), words.end(), [](const std::string& a, const std::string& b) {
            return a.size() < b.size();
        });

        std::unordered_set<std::string> word_set;
        std::vector<std::string> result;

        auto can_form = [&](const std::string& word) -> bool {
            int n = word.size();
            if (n == 0) return false;

            std::vector<bool> dp(n + 1, false);
            dp[0] = true;

            for (int i = 1; i <= n; ++i) {
                for (int j = 0; j < i; ++j) {
                    if (dp[j] && word_set.find(word.substr(j, i - j)) != word_set.end()) {
                        dp[i] = true;
                        break;
                    }
                }
            }
            return dp[n];
        };

        for (const std::string& w : words) {
            if (can_form(w)) {
                result.push_back(w);
            }
            word_set.insert(w);
        }

        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

class Solution {
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        // Sort by string length ascending
        Arrays.sort(words, Comparator.comparingInt(String::length));

        Set<String> wordSet = new HashSet<>();
        List<String> result = new ArrayList<>();

        for (String word : words) {
            if (canForm(word, wordSet)) {
                result.add(word);
            }
            wordSet.add(word);
        }

        return result;
    }

    private boolean canForm(String word, Set<String> wordSet) {
        int n = word.length();
        if (n == 0) return false;

        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordSet.contains(word.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N + N \times L^2)$  
  Where $N \le 10^4$ is the number of words and $L \le 30$ is the maximum word length. Sorting takes $\mathcal{O}(N \log N \times L)$. For each word, the Word Break DP takes $\mathcal{O}(L^2)$ substring slices and hash lookups. Since $L \le 30$, $L^2 \approx 900$. Total operations $\approx 10^4 \times 900 \approx 9 \times 10^6$, which finishes in $\approx 80$ ms.
- **Space Complexity:** $\mathcal{O}(N \times L)$  
  The hash set stores up to $N$ strings of length at most $L$. The DP array for each word takes $\mathcal{O}(L)$ memory.

---

### Takeaway Pattern & Interview Traps

1. **Incremental Dictionary Prevents Self-Matching:**
   - If you populate the set with all words upfront, every word would trivially match itself via $j = 0$ in one step!
   - By sorting by length and only querying words shorter than the current word, self-matching is structurally impossible, ensuring every positive result consists of $\ge 2$ components.
2. **Handling Empty Strings:**
   - If the input contains `""`, an empty string would trivially satisfy `dp[0]` and loop indefinitely if not skipped. Ensure `n == 0` is pruned immediately.