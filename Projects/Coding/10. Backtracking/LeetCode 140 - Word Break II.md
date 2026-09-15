---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 140: Word Break II"
tags:
  - leetcode
  - coding
  - backtracking
  - dynamic-programming
  - memoization
  - string
  - amazon
  - google
---

# LeetCode 140: Word Break II

**Target Companies:** Amazon, Google, Meta, Bloomberg, Microsoft, Uber  
**Difficulty:** Hard  
**Topic:** Backtracking / Top-Down Memoized DFS / String Partitioning  

---

### Problem Statement

Given a string `s` and a dictionary of strings `wordDict`, add spaces in `s` to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in **any order**.

**Note** that the same word in the dictionary may be reused multiple times in the segmentation.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`, `wordDict: List[str]`
- **Output:** `List[str]` containing all space-delimited valid sentences.
- **Constraints:**
  - $1 \le \text{s.length} \le 20$
  - $1 \le \text{wordDict.length} \le 1000$
  - $1 \le \text{wordDict}[i]\text{.length} \le 10$
  - `s` and `wordDict[i]` consist of only lowercase English letters.
  - All strings of `wordDict` are **unique**.
  - Input is generated such that the total number of sentences does not exceed $10^5$.

---

### Key Idea & Intuition

- **Differences from Word Break I (LeetCode 139):**
  - LeetCode 139 is a decision problem: *Can `s` be segmented?* This requires only a 1D boolean DP array.
  - LeetCode 140 requires enumerating **all valid sentences**.
- **The Exponential Trap & Why Memoization is Mandatory:**
  - Consider $s = \text{"aaaaaaaa"}$ and $\text{wordDict} = [\text{"a"}, \text{"aa"}, \text{"aaa"}]$. A naive backtracking search branches exponentially, recomputing sentences for suffix $\text{"aaaa"}$ many times.
  - Furthermore, consider a pathological input where $s = \text{"aaaaaaaaaab"}$ with words composed only of $\text{'a'}$. Naive backtracking will explore an exponential number of prefixes before failing at the final character $\text{'b'}$.
- **Memoized DFS (`memo: start_index -> List[str]`):**
  - Define `dfs(start)` which returns a list of all valid complete sentences that can be constructed from the suffix $s[\text{start}\dots]$.
  - For each word in `wordDict` (or by iterating prefix slices $s[\text{start}\dots\text{end}]$), if the prefix exists in `wordDict`:
    - Recursively call `dfs(end + 1)` to get all suffix sentences.
    - If the suffix is empty ($\text{end} + 1 == N$), the sentence is simply the prefix itself.
    - Otherwise, for each sub-sentence from the recursive call, form the sentence: `prefix + " " + sub_sentence`.
  - Cache the resulting list of sentences in `memo[start]`.

---

### Solution Approach (Step-by-Step)

1. **Convert Dictionary to Set:**
   - Store words in a hash set `word_set` for $\mathcal{O}(1)$ average-time prefix verification.
2. **Define Memoized DFS Function `dfs(start)`:**
   - If `start` is already in `memo`, return `memo[start]`.
   - If `start == len(s)`, return `[""]` (an empty string representing a valid base segmentation).
   - Initialize an empty list `sentences`.
   - For `end` from `start + 1` to `len(s) + 1`:
     - Extract `word = s[start : end]`.
     - If `word in word_set`:
       - Recursively call `sub_sentences = dfs(end)`.
       - For each `sub` in `sub_sentences`:
         - If `sub == ""`: append `word` to `sentences`.
         - Else: append `word + " " + sub` to `sentences`.
   - Store `memo[start] = sentences` and return `sentences`.
3. **Execute:**
   - Return `dfs(0)`.

---

### Visual Algorithm Walkthrough

Let `s = "catsanddog"`, `wordDict = ["cat", "cats", "and", "sand", "dog"]`.

```
dfs(0)  s[0..] = "catsanddog"
├── Match "cat" (len 3) -> dfs(3)  s[3..] = "sanddog"
│   └── Match "sand" (len 4) -> dfs(7)  s[7..] = "dog"
│       └── Match "dog" (len 3) -> dfs(10) Base case: [""]
│           Result for dfs(7): ["dog"]
│       Result for dfs(3): ["sand dog"]
│   Sentences from "cat": ["cat sand dog"]
│
└── Match "cats" (len 4) -> dfs(4)  s[4..] = "anddog"
    └── Match "and" (len 3) -> dfs(7)  [CACHED MEMO HIT!]
        Returns: ["dog"]
    Result for dfs(4): ["and dog"]
    Sentences from "cats": ["cats and dog"]

dfs(0) Returns: ["cat sand dog", "cats and dog"]
```

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | `wordDict` | Memoized DFS Hits | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `"catsanddog"` | `["cat","cats","and","sand","dog"]` | Index 7 reused by `"cats"` and `"cat"` branches | `["cat sand dog", "cats and dog"]` |
| **Multiple Options** | `"pineapplepenapple"` | `["apple","pen","applepen","pine","pineapple"]` | Multiple valid combinations of `"apple"`, `"pen"` | `["pine apple pen apple", "pineapple pen apple", "pine applepen apple"]` |
| **No Solution** | `"catsandog"` | `["cats","dog","sand","and","cat"]` | `"og"` cannot be matched | `[]` |
| **Single Word** | `"apple"` | `["apple"]` | Base match | `["apple"]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Dict

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        """
        Returns all valid segmentations of string s using words from wordDict.
        Utilizes top-down DFS with memoization to prune repeated suffix traversals.
        """
        word_set = set(wordDict)
        memo: Dict[int, List[str]] = {}

        def dfs(start: int) -> List[str]:
            if start in memo:
                return memo[start]

            if start == len(s):
                return [""]

            res: List[str] = []
            for end in range(start + 1, len(s) + 1):
                prefix = s[start:end]
                if prefix in word_set:
                    suffix_sentences = dfs(end)
                    for sub in suffix_sentences:
                        if sub == "":
                            res.append(prefix)
                        else:
                            res.append(f"{prefix} {sub}")

            memo[start] = res
            return res

        return dfs(0)
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class Solution {
public:
    std::vector<std::string> wordBreak(const std::string& s, const std::vector<std::string>& wordDict) {
        std::unordered_set<std::string> word_set(wordDict.begin(), wordDict.end());
        std::unordered_map<int, std::vector<std::string>> memo;
        return dfs(0, s, word_set, memo);
    }

private:
    std::vector<std::string> dfs(int start, const std::string& s,
                                const std::unordered_set<std::string>& word_set,
                                std::unordered_map<int, std::vector<std::string>>& memo) {
        if (memo.count(start)) {
            return memo[start];
        }

        if (start == static_cast<int>(s.size())) {
            return {""};
        }

        std::vector<std::string> res;
        for (int end = start + 1; end <= static_cast<int>(s.size()); ++end) {
            std::string prefix = s.substr(start, end - start);
            if (word_set.count(prefix)) {
                std::vector<std::string> sub_sentences = dfs(end, s, word_set, memo);
                for (const std::string& sub : sub_sentences) {
                    if (sub.empty()) {
                        res.push_back(prefix);
                    } else {
                        res.push_back(prefix + " " + sub);
                    }
                }
            }
        }

        memo[start] = res;
        return res;
    }
};
```

#### Java
```java
import java.util.*;

class Solution {
    public List<String> wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        Map<Integer, List<String>> memo = new HashMap<>();
        return dfs(0, s, wordSet, memo);
    }

    private List<String> dfs(int start, String s, Set<String> wordSet, Map<Integer, List<String>> memo) {
        if (memo.containsKey(start)) {
            return memo.get(start);
        }

        List<String> res = new ArrayList<>();
        if (start == s.length()) {
            res.add("");
            return res;
        }

        for (int end = start + 1; end <= s.length(); end++) {
            String prefix = s.substring(start, end);
            if (wordSet.contains(prefix)) {
                List<String> subSentences = dfs(end, s, wordSet, memo);
                for (String sub : subSentences) {
                    if (sub.isEmpty()) {
                        res.add(prefix);
                    } else {
                        res.add(prefix + " " + sub);
                    }
                }
            }
        }

        memo.put(start, res);
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(2^N + N^2)$ in the worst case where every partition of the string is valid (bounded by the number of output combinations $K \le 10^5$).
  - For a string of length $N \le 20$, there are $2^{N-1} \approx 5 \times 10^5$ possible ways to place spaces.
  - Substring extraction and hash set lookups take $\mathcal{O}(L)$ where $L \le 10$ is the max word length.
  - With memoization, each start index is processed once, and results are composed linearly with respect to the output size.
- **Space Complexity:** $\mathcal{O}(N \cdot 2^N)$ to store all output combinations in the memoization table and output list, with recursion stack depth $\mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

- **Top-Down Memoized DFS vs Pure Backtracking:** In problems where we must return all combinations (like Word Break II or Palindrome Partitioning), naive backtracking can explore identical suffix trees repeatedly. Caching the suffix results `memo[start]` converts exponential redundancy into a directed acyclic graph (DAG) traversal.
- **Pathological Dead Ends:** If the suffix contains a character not in any dictionary word, memoization ensures that `dfs(start)` immediately returns an empty list `[]` and subsequent calls hitting `start` instantly prune without re-evaluating.
- **Base Case Formulation:** Returning `[""]` for `start == len(s)` cleanly handles appending spaces: `if sub == "": prefix` else `prefix + " " + sub`.