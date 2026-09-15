---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 139: Word Break"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - hash-table
  - string
  - trie
  - amazon
  - meta
  - google
  - microsoft
---

# LeetCode 139: Word Break

**Target Companies:** Amazon, Meta, Google, Microsoft, Bloomberg, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Hash Table / String  

---

### Problem Statement

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

**Note** that the same word in the dictionary may be reused multiple times in the segmentation.

---

### Input & Output Formats & Constraints

- **Input:**
  - A string `s` ($1 \le |s| \le 300$) consisting of lowercase English letters.
  - A list of unique strings `wordDict` ($1 \le |wordDict| \le 1000$, $1 \le |wordDict[i]| \le 20$).
- **Output:** A boolean (`true` or `false`) indicating if `s` is segmentable.
- **Constraints:**
  - `1 <= s.length <= 300`
  - `1 <= wordDict.length <= 1000`
  - `1 <= wordDict[i].length <= 20`
  - All strings in `wordDict` are **unique**.

---

### Key Idea & Intuition

#### Prefix Decomposition Invariant
A string $s[0..i-1]$ of length $i$ can be segmented into valid words if and only if there exists some split point $j < i$ such that:
1. The prefix $s[0..j-1]$ of length $j$ can be validly segmented ($dp[j] == \text{true}$).
2. The remaining trailing substring $s[j..i-1]$ is a valid dictionary word ($s[j..i-1] \in wordDict$).

#### Recurrence Relation
Let $dp[i]$ be a boolean value indicating whether the prefix of length $i$ ($s[0..i-1]$) can be segmented:
- Base case: $dp[0] = \text{true}$ (the empty prefix requires 0 words and is vacuously valid).
- Transition for $i \in [1, n]$:
  $$dp[i] = \bigvee_{j = \max(0, i - L)}^{i - 1} \Big(dp[j] \land (s[j..i-1] \in wordDict)\Big)$$
  where $L = \max_{w \in wordDict} |w|$ is the maximum length of any word in `wordDict`.
- Final Answer: $dp[n]$.

#### The Max Word Length Pruning Optimization
In a standard unpruned loop, $j$ ranges from $0$ to $i - 1$, taking $\mathcal{O}(n^2)$ time. However, dictionary words have a maximum length $L \le 20$. Any substring $s[j..i-1]$ of length $> L$ can never belong to the dictionary. Limiting $j$ to the range $[\max(0, i - L), i - 1]$ drops the inner loop from $n$ iterations down to at most $L$ iterations, drastically speeding up execution.

---

### Solution Approach (Step-by-Step)

1. **Dictionary Set & Max Word Length:**
   - Store words in a hash set `word_set` for $\mathcal{O}(1)$ average-time lookups.
   - Compute `max_len = max(len(w) for w in wordDict)`.
2. **Initialize DP Array:**
   - Create boolean array `dp` of size $n + 1$ with `dp[0] = True` and all other entries `False`.
3. **Iterative Evaluation with Pruning:**
   - For $i$ from 1 to $n$:
     - For $j$ from $i - 1$ down to $\max(0, i - max\_len)$:
       - If $dp[j]$ and $s[j:i] \in word\_set$:
         - $dp[i] = \text{True}$.
         - Break early (one valid split suffices).
4. **Return:**
   - Return $dp[n]$.

---

### Visual Algorithm Walkthrough

#### Trace for `s = "leetcode"`, `wordDict = ["leet", "code"]`
```
n = 8, max_len = 4
dp = [True, False, False, False, False, False, False, False, False]
Index:  0      1      2      3      4      5      6      7      8

i = 1 ("l"):      j=0 -> "l" not in dict -> dp[1] = False
i = 2 ("le"):     j=1,0 -> not in dict -> dp[2] = False
i = 3 ("lee"):    j=2,1,0 -> not in dict -> dp[3] = False
i = 4 ("leet"):   j=0: dp[0] is True AND "leet" in dict -> dp[4] = True!
i = 5 ("leetc"):  j=4..1 -> no valid suffix
i = 6 ("leetco"): j=4..2 -> no valid suffix
i = 7 ("leetcod"):j=4..3 -> no valid suffix
i = 8 ("leetcode"):
  j = 4: dp[4] is True AND s[4:8] ("code") in dict -> dp[8] = True!

Result: dp[8] = True ("leet" + "code").
```

---

### Solved Examples with Multiple Inputs

| String $s$ | `wordDict` | Max Word Length $L$ | Valid Slices Identified | Output |
|---|---|---|---|---|
| `"leetcode"` | `["leet", "code"]` | 4 | `"leet"` $\to$ `"code"` | `true` |
| `"applepenapple"` | `["apple", "pen"]` | 5 | `"apple"` $\to$ `"pen"` $\to$ `"apple"` | `true` |
| `"catsandog"` | `["cats", "dog", "sand", "and", "cat"]` | 4 | `"cats"+"and"+"og"` (fail); `"cat"+"sand"+"og"` (fail) | `false` |
| `"a"` | `["b"]` | 1 | No matches | `false` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        word_set: set[str] = set(wordDict)
        max_len: int = max(len(w) for w in wordDict)
        n: int = len(s)
        
        # dp[i] indicates if s[0..i-1] can be segmented
        dp: list[bool] = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            # Only examine candidate words up to max_len
            for j in range(i - 1, max(-1, i - max_len - 1), -1):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
                    
        return dp[n]
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>

class Solution {
public:
    bool wordBreak(const std::string& s, const std::vector<std::string>& wordDict) {
        std::unordered_set<std::string> word_set(wordDict.begin(), wordDict.end());
        int max_len = 0;
        for (const auto& w : wordDict) {
            max_len = std::max(max_len, static_cast<int>(w.size()));
        }

        int n = static_cast<int>(s.size());
        std::vector<bool> dp(n + 1, false);
        dp[0] = true;

        for (int i = 1; i <= n; ++i) {
            for (int j = i - 1; j >= std::max(0, i - max_len); --j) {
                if (dp[j] && word_set.count(s.substr(j, i - j))) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n];
    }
};
```

#### Java 17
```java
import java.util.HashSet;
import java.util.List;
import java.util.Set;

class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordSet = new HashSet<>(wordDict);
        int maxLen = 0;
        for (String w : wordDict) {
            maxLen = Math.max(maxLen, w.length());
        }

        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        for (int i = 1; i <= n; i++) {
            for (int j = i - 1; j >= Math.max(0, i - maxLen); j--) {
                if (dp[j] && wordSet.contains(s.substring(j, i))) {
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

- **Time Complexity:** $\mathcal{O}(n \cdot \min(n, L) \cdot L)$, where $n = |s|$ and $L = \max_{w \in wordDict} |w|$. The outer loop runs $n$ times. The inner loop runs at most $L$ times due to max word length pruning. Substring extraction and hash matching take $\mathcal{O}(L)$ time. With $n \le 300$ and $L \le 20$, total operations are at most $300 \times 20 \times 20 \approx 1.2 \times 10^5$, executing in under $3$ ms.
- **Space Complexity:** $\mathcal{O}(n + D \cdot L)$ auxiliary space, where $D = |wordDict|$ for the hash set, and $\mathcal{O}(n)$ space for the 1D DP array.

---

### Takeaway Pattern & Interview Traps

1. **The Max-Length Pruning Difference:** Without bounding $j \ge i - L$, the inner loop does $n$ substring operations, resulting in $\mathcal{O}(n^3)$ worst-case runtime. Bounding by $L$ brings it to $\mathcal{O}(n \cdot L^2)$.
2. **Break on First True:** Once any valid prefix $j$ satisfies $dp[j] \land s[j..i-1] \in wordSet$, set $dp[i] = \text{true}$ and break immediately. Checking further $j$ values is redundant.
3. **Relation to BFS / Trie:** This problem can also be modeled as a shortest path / reachability graph search where nodes are indices $0 \dots n$ and directed edges exist between valid word boundaries.