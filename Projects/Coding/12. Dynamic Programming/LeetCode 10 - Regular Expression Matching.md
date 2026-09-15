---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 10: Regular Expression Matching"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - recursion
  - google
  - meta
  - amazon
  - microsoft
---

# LeetCode 10: Regular Expression Matching

**Target Companies:** Google, Meta, Amazon, Microsoft, Apple, Bloomberg, Uber  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / String / Recursion  

---

### Problem Statement

Given an input string `s` and a pattern `p`, implement regular expression matching with support for `'.'` and `'*'` where:
- `'.'` matches any single character.
- `'*'` matches zero or more of the preceding element.

The matching should cover the **entire** input string (not partial).

---

### Input & Output Formats & Constraints

- **Input:**
  - A string `s` ($1 \le |s| \le 20$) consisting of lowercase English letters.
  - A pattern `p` ($1 \le |p| \le 20$) consisting of lowercase English letters, `'.'`, and `'*'`.
- **Output:** A boolean (`true` or `false`) indicating whether the entire string `s` matches the pattern `p`.
- **Constraints:**
  - `1 <= s.length <= 20`
  - `1 <= p.length <= 20`
  - `s` contains only lowercase English letters.
  - `p` contains only lowercase English letters, `'.'`, and `'*'`.
  - It is guaranteed for each appearance of the character `'*'`, there will be a previous valid character to match.

---

### Key Idea & Intuition

#### Why Dynamic Programming?
Greedy matching fails because `'*'` introduces non-deterministic branching: it can match zero occurrences, one occurrence, or many occurrences of the preceding character. Trying to greedily consume characters can exhaust the string prematurely when fewer matches would allow the remainder of the pattern to succeed. Because substrings exhibit overlapping subproblems and optimal substructure, 2D Dynamic Programming solves this deterministically.

#### State Definition
Let $dp[i][j]$ be a boolean value indicating whether the prefix $s[0..i-1]$ of length $i$ matches the prefix $p[0..j-1]$ of length $j$:
- Dimensions: $(m + 1) \times (n + 1)$, where $m = |s|$ and $n = |p|$.
- Base case: $dp[0][0] = \text{true}$ (empty string matches empty pattern).
- First row ($i = 0$, empty string): A pattern can match an empty string if and only if it consists of zero or more pairs of `character + '*'`. For $j \ge 2$, if $p[j-1] == '*'$:
  $$dp[0][j] = dp[0][j-2]$$

#### State Transitions
For each $i \in [1, m]$ and $j \in [1, n]$:
1. **Case 1: Current pattern character is `'*'` ($p[j-1] == '*'$):**
   The preceding character in the pattern is $p[j-2]$. We have two fundamental choices:
   - **Zero Occurrences:** Discard the preceding character and `'*'` entirely:
     $$dp[i][j] = dp[i][j-2]$$
   - **One or More Occurrences:** If $p[j-2]$ matches $s[i-1]$ (i.e. $p[j-2] == s[i-1]$ or $p[j-2] == '.'$):
     We can consume one matching character from $s$ while keeping the pattern intact to allow future matches:
     $$dp[i][j] = dp[i][j] \lor dp[i-1][j]$$
2. **Case 2: Exact character match or `'.'` ($p[j-1] == s[i-1]$ or $p[j-1] == '.'$):**
   Both characters match directly, so the result depends on whether the preceding prefixes matched:
   $$dp[i][j] = dp[i-1][j-1]$$
3. **Case 3: Mismatch:**
   $dp[i][j] = \text{false}$.

---

### Solution Approach (Step-by-Step)

1. **Initialize DP Table:**
   - Create a boolean table `dp` of size $(m + 1) \times (n + 1)$ filled with `false`.
   - Set $dp[0][0] = \text{true}$.
2. **Initialize Row 0 (Empty String vs. Pattern):**
   - For $j$ from 2 to $n$:
     - If $p[j-1] == '*'$:
       - $dp[0][j] = dp[0][j-2]$.
3. **Fill DP Table:**
   - Iterate $i$ from 1 to $m$:
     - Iterate $j$ from 1 to $n$:
       - If $p[j-1] == '*'$;
         - $dp[i][j] = dp[i][j-2]$ (zero matches).
         - If $p[j-2] == s[i-1]$ or $p[j-2] == '.'$:
           - $dp[i][j] = dp[i][j] \lor dp[i-1][j]$.
       - Else if $p[j-1] == '.'$ or $p[j-1] == s[i-1]$:
         - $dp[i][j] = dp[i-1][j-1]$.
4. **Return Result:**
   - Return $dp[m][n]$.

---

### Visual Algorithm Walkthrough

#### Trace for $s = \text{"aab"}$, $p = \text{"c*a*b"}$
```
DP Table Dimensions: (3 + 1) rows x (5 + 1) cols
Empty row initialization:
dp[0][0] = T ("" matches "")
dp[0][1] ('c') = F
dp[0][2] ('*') = dp[0][0] = T ("" matches "c*")
dp[0][3] ('a') = F
dp[0][4] ('*') = dp[0][2] = T ("" matches "c*a*")
dp[0][5] ('b') = F

Filling row by row:
        ""   c   c*   a   a*   b
  "" [  T,  F,   T,  F,   T,  F ]
  a  [  F,  F,   F,  T,   T,  F ]
  a  [  F,  F,   F,  F,   T,  F ]
  b  [  F,  F,   F,  F,   F,  T ]

Cell Highlights:
- s = "a", p = "c*a":
  p[3]='a' matches s[0]='a', so dp[1][3] = dp[0][2] = T.
- s = "a", p = "c*a*":
  Zero copies: dp[1][4] |= dp[1][2] = F.
  One copy: p[2]='a' matches s[0]='a', so dp[1][4] |= dp[0][4] = T.
- s = "aab", p = "c*a*b":
  p[4]='b' matches s[2]='b', so dp[3][5] = dp[2][4] = T.

Final Answer: dp[3][5] = True.
```

---

### Solved Examples with Multiple Inputs

| String $s$ | Pattern $p$ | Empty Match Prefixes | Key Transition Step | Output |
|---|---|---|---|---|
| `"aa"` | `"a"` | None | Length mismatch ($m=2, n=1$) | `false` |
| `"aa"` | `"a*"` | `""` matches `""` | $s[1]$ matches $p[0]$ with $p[1]=='*'$ | `true` |
| `"ab"` | `".*"` | `""` matches `""` | `'.*'` matches any sequence | `true` |
| `"aab"` | `"c*a*b"` | `""` matches `"c*"`, `"c*a*"` | `"c*"` matches 0, `"a*"` matches 2 | `true` |
| `"mississippi"` | `"mis*is*p*."` | `""` matches `"mis*is*p*"`? No | $p[8]=='*'$ fails to match double 's' | `false` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m: int = len(s)
        n: int = len(p)
        
        # dp[i][j] indicates if s[0..i-1] matches p[0..j-1]
        dp: list[list[bool]] = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Base case: empty string matches empty pattern
        dp[0][0] = True
        
        # Deals with patterns like a*, a*b*, or a*b*c* matching empty string
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
                
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # Choice 1: match zero of preceding character
                    dp[i][j] = dp[i][j - 2]
                    # Choice 2: match one or more of preceding character
                    if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                        dp[i][j] = dp[i][j] or dp[i - 1][j]
                elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    
        return dp[m][n]
```

#### C++17
```cpp
#include <vector>
#include <string>

class Solution {
public:
    bool isMatch(const std::string& s, const std::string& p) {
        int m = static_cast<int>(s.size());
        int n = static_cast<int>(p.size());
        
        std::vector<std::vector<bool>> dp(m + 1, std::vector<bool>(n + 1, false));
        dp[0][0] = true;
        
        // Patterns like "a*", "a*b*" can match empty string
        for (int j = 2; j <= n; ++j) {
            if (p[j - 1] == '*') {
                dp[0][j] = dp[0][j - 2];
            }
        }
        
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    // Match 0 occurrences
                    dp[i][j] = dp[i][j - 2];
                    // Match 1 or more occurrences
                    if (p[j - 2] == s[i - 1] || p[j - 2] == '.') {
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else if (p[j - 1] == '.' || p[j - 1] == s[i - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        
        return dp[m][n];
    }
};
```

#### Java 17
```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        
        // Handle empty string matching pattern with '*'
        for (int j = 2; j <= n; j++) {
            if (p.charAt(j - 1) == '*') {
                dp[0][j] = dp[0][j - 2];
            }
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    // Zero occurrences of preceding character
                    dp[i][j] = dp[i][j - 2];
                    // One or more occurrences
                    char prev = p.charAt(j - 2);
                    if (prev == s.charAt(i - 1) || prev == '.') {
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else if (p.charAt(j - 1) == '.' || p.charAt(j - 1) == s.charAt(i - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        
        return dp[m][n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$, where $m = |s|$ and $n = |p|$. Every cell in the $(m+1) \times (n+1)$ matrix is computed in $\mathcal{O}(1)$ time. With $m, n \le 20$, the inner loop executes at most 400 times, taking $< 1$ millisecond.
- **Space Complexity:** $\mathcal{O}(m \times n)$ auxiliary space for the DP table. (Can be optimized to $\mathcal{O}(n)$ using two 1D rolling rows).

---

### Takeaway Pattern & Interview Traps

1. **Difference with Wildcard Matching (LeetCode 44):**
   - In LC 44, `'*'` stands alone and matches any sequence of characters directly.
   - In LC 10, `'*'` modifies the *preceding* element, which means it always looks 2 steps back in the pattern (`j - 2`).
2. **Lookahead vs. Lookbehind:** In top-down recursion, lookahead checking `p[j+1] == '*'` is customary. In bottom-up DP, lookbehind checking `p[j-1] == '*'` and referencing `p[j-2]` is much cleaner and eliminates out-of-bounds guards.
3. **Empty String Base Row:** Always initialize row 0 from index 2 onward. Forgetting $dp[0][j] = dp[0][j-2]$ causes patterns like `"a*"` or `".*"` to fail on empty or partially matched prefixes.