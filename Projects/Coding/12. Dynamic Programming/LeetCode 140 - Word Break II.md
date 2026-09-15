---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 140: Word Break II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - backtracking
  - memoization
  - string
  - amazon
  - bloomberg
  - meta
  - google
---

# LeetCode 140: Word Break II

**Target Companies:** Amazon, Bloomberg, Meta, Google, Microsoft, Apple  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Backtracking / Memoization  

---

### Problem Statement

Given a string `s` and a dictionary of strings `wordDict`, add spaces in `s` to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in **any order**.

**Note** that the same word in the dictionary may be reused multiple times in the segmentation.

---

### Input & Output Formats & Constraints

- **Input:**
  - A string `s` ($1 \le |s| \le 20$).
  - A list of unique strings `wordDict` ($1 \le |wordDict| \le 1000$, $1 \le |wordDict[i]| \le 10$).
- **Output:** A list of strings `List[str]` representing all valid space-separated sentences.
- **Constraints:**
  - `1 <= s.length <= 20`
  - `1 <= wordDict.length <= 1000`
  - `1 <= wordDict[i].length <= 10`
  - `s` and `wordDict[i]` consist of only lowercase English letters.
  - All the strings of `wordDict` are **unique**.
  - Input is generated such that the total number of sentences does not exceed $10^5$.

---

### Key Idea & Intuition

#### Suffix Decomposition with Memoization
While LeetCode 139 only required a boolean answer (can we segment?), LeetCode 140 requires constructing **all valid sentences**.
Because different prefixes can share the exact same valid suffix segmentations, recomputing suffix sentences from scratch causes exponential explosion.
For example, in `s = "catsanddog"`, both prefixes `"cat" + "sand"` and `"cats" + "and"` arrive at suffix `"dog"`. Computing the sentences for `"dog"` once and memoizing the result prevents redundant work.

#### State Definition & Recurrence
Let `memo[start]` be the list of all sentences that can be formed from the suffix $s[start:]$:
- **Base Case:** When $start == |s|$, we have reached the end of the string. The only sentence for an empty suffix is the empty string `[""]`.
- **Transitions:**
  For each $end \in [start + 1, |s|]$:
  - Let $word = s[start..end-1]$.
  - If $word \in wordDict$:
    - Recursively fetch all suffix sentences for $s[end:]$: `sub_sentences = dfs(end)`.
    - For each `sub` in `sub_sentences`:
      - If `sub == ""`, form `word`.
      - Else, form `word + " " + sub`.
- Cache and return `memo[start]`.

---

### Solution Approach (Step-by-Step)

1. **Hash Set for Fast Lookups:**
   - Convert `wordDict` into a hash set `word_set`.
2. **Memoized DFS Helper:**
   - Define `dfs(start)` returning `List[str]`:
     - If $start \in memo$, return $memo[start]$.
     - If $start == |s|$, return `[""]`.
     - Initialize `sentences = []`.
     - For $end$ from $start + 1$ to $|s| + 1$:
       - $word = s[start:end]$.
       - If $word \in word\_set$:
         - For each $sub$ in $dfs(end)$:
           - If $sub == ""$:
             - $sentences.append(word)$.
           - Else:
             - $sentences.append(word + " " + sub)$.
     - Store $memo[start] = sentences$ and return.
3. **Execute:**
   - Return `dfs(0)`.

---

### Visual Algorithm Walkthrough

#### Trace for `s = "catsanddog"`, `wordDict = ["cat", "cats", "and", "sand", "dog"]`
```
Recursion Tree with Memoization:
dfs(0) [s = "catsanddog"]
 ├── word = "cat" (s[0:3]) -> calls dfs(3) [s = "sanddog"]
 │    └── word = "sand" (s[3:7]) -> calls dfs(7) [s = "dog"]
 │         └── word = "dog" (s[7:10]) -> calls dfs(10) -> returns [""]
 │              dfs(7) returns ["dog"]
 │         dfs(3) combines: "sand" + " " + "dog" -> returns ["sand dog"]
 │    From "cat", forms: "cat sand dog"
 │
 └── word = "cats" (s[0:4]) -> calls dfs(4) [s = "anddog"]
      └── word = "and" (s[4:7]) -> calls dfs(7) [s = "dog"]
           (CACHE HIT on dfs(7) -> immediately returns ["dog"])
      dfs(4) combines: "and" + " " + "dog" -> returns ["and dog"]
      From "cats", forms: "cats and dog"

Result from dfs(0):
[
  "cat sand dog",
  "cats and dog"
]
```

---

### Solved Examples with Multiple Inputs

| String $s$ | `wordDict` | Branching Sequences | Output Sentences |
|---|---|---|---|
| `"catsanddog"` | `["cat","cats","and","sand","dog"]` | `"cat" + "sand dog"`, `"cats" + "and dog"` | `["cat sand dog", "cats and dog"]` |
| `"pineapplepenapple"` | `["apple","pen","applepen","pine","pineapple"]` | Multiple overlaps | `["pine apple pen apple", "pineapple pen apple", "pine applepen apple"]` |
| `"catsandog"` | `["cats","dog","sand","and","cat"]` | No valid path to end | `[]` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> list[str]:
        word_set: set[str] = set(wordDict)
        memo: dict[int, list[str]] = {}
        n: int = len(s)
        
        def dfs(start: int) -> list[str]:
            if start in memo:
                return memo[start]
            if start == n:
                return [""]
                
            sentences: list[str] = []
            for end in range(start + 1, n + 1):
                word = s[start:end]
                if word in word_set:
                    sub_sentences = dfs(end)
                    for sub in sub_sentences:
                        if sub == "":
                            sentences.append(word)
                        else:
                            sentences.append(word + " " + sub)
                            
            memo[start] = sentences
            return sentences
            
        return dfs(0)
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class Solution {
private:
    std::unordered_map<int, std::vector<std::string>> memo;

    std::vector<std::string> dfs(int start, const std::string& s, const std::unordered_set<std::string>& word_set) {
        if (memo.count(start)) {
            return memo[start];
        }
        int n = static_cast<int>(s.size());
        if (start == n) {
            return {""};
        }

        std::vector<std::string> sentences;
        for (int end = start + 1; end <= n; ++end) {
            std::string word = s.substr(start, end - start);
            if (word_set.count(word)) {
                std::vector<std::string> sub_sentences = dfs(end, s, word_set);
                for (const std::string& sub : sub_sentences) {
                    if (sub.empty()) {
                        sentences.push_back(word);
                    } else {
                        sentences.push_back(word + " " + sub);
                    }
                }
            }
        }

        memo[start] = sentences;
        return sentences;
    }

public:
    std::vector<std::string> wordBreak(const std::string& s, const std::vector<std::string>& wordDict) {
        std::unordered_set<std::string> word_set(wordDict.begin(), wordDict.end());
        memo.clear();
        return dfs(0, s, word_set);
    }
};
```

#### Java 17
```java
import java.util.*;

class Solution {
    private Map<Integer, List<String>> memo = new HashMap<>();

    private List<String> dfs(int start, String s, Set<String> wordSet) {
        if (memo.containsKey(start)) {
            return memo.get(start);
        }
        int n = s.length();
        if (start == n) {
            return Collections.singletonList("");
        }

        List<String> sentences = new ArrayList<>();
        for (int end = start + 1; end <= n; end++) {
            String word = s.substring(start, end);
            if (wordSet.contains(word)) {
                List<String> subSentences = dfs(end, s, wordSet);
                for (String sub : subSentences) {
                    if (sub.isEmpty()) {
                        sentences.add(word);
                    } else {
                        sentences.add(word + " " + sub);
                    }
                }
            }
        }

        memo.put(start, sentences);
        return sentences;
    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        memo.clear();
        return dfs(0, s, wordSet);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(2^N + N^2)$, where $N = |s|$. In the worst case (e.g. `s = "aaa"`, `wordDict = ["a", "aa"]`), there can be $2^{N-1}$ valid sentence partitions. Memoization guarantees each suffix subproblem is computed once, so time is bounded by the total size of all output strings, which is $\le 10^5$ per constraints.
- **Space Complexity:** $\mathcal{O}(2^N + N^2)$ to store all memoized suffix lists, plus $\mathcal{O}(N)$ recursion depth for the call stack.

---

### Takeaway Pattern & Interview Traps

1. **Why Memoized DFS is Superior to Bottom-Up:** Bottom-up DP constructs full string lists for *all* indices, even if those indices cannot be reached from the prefix $s[0]$. Top-down memoized DFS only explores reachable prefixes.
2. **LC 139 Reachability Pruning:** If strings can be very long with many unmatchable suffixes, running LC 139 first to verify reachability before constructing strings avoids exploring dead ends.
3. **Empty Base Case String:** Returning `[""]` instead of `[]` for $start == n$ allows the loop over `sub_sentences` to execute once, creating the leaf words without special edge-case branching.