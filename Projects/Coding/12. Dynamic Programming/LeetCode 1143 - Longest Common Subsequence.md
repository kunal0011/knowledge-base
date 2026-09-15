---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 1143: Longest Common Subsequence"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - amazon
  - google
  - microsoft
  - meta
---

# LeetCode 1143: Longest Common Subsequence

**Target Companies:** Amazon, Google, Microsoft, Meta, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / String  

---

### Problem Statement

Given two strings `text1` and `text2`, return the length of their **longest common subsequence**. If there is no common subsequence, return `0`.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.
- For example, `"ace"` is a subsequence of `"abcde"`.

A **common subsequence** of two strings is a subsequence that is common to both strings.

---

### Input & Output Formats & Constraints

- **Input:** Two strings `text1` and `text2`.
- **Output:** An integer representing the length of the longest common subsequence.
- **Constraints:**
  - `1 <= text1.length, text2.length <= 1000`
  - `text1` and `text2` consist of only lowercase English characters.

---

### Key Idea & Intuition

#### Optimal Substructure & Overlapping Subproblems
Let $m = |text1|$ and $n = |text2|$. We consider prefixes $text1[0..i-1]$ of length $i$ and $text2[0..j-1]$ of length $j$.

When comparing the current trailing characters $text1[i-1]$ and $text2[j-1]$:
1. **Match Case ($text1[i-1] == text2[j-1]$):**
   - The matching character must belong to the optimal common subsequence.
   - We extend the LCS of the shorter prefixes by 1:
     $$dp[i][j] = 1 + dp[i-1][j-1]$$
2. **Mismatch Case ($text1[i-1] \ne text2[j-1]$):**
   - The two characters cannot both be the last character of a common subsequence.
   - Either $text1[i-1]$ is not used or $text2[j-1]$ is not used.
   - The answer is the maximum of ignoring one or the other:
     $$dp[i][j] = \max(dp[i-1][j], dp[i][j-1])$$

#### Space Optimization to $\mathcal{O}(\min(m, n))$
Because each state $dp[i][j]$ depends only on the current row and the previous row ($dp[i-1][j-1]$ and $dp[i-1][j]$), we can reduce the auxiliary space from a 2D table of size $(m+1) \times (n+1)$ to two rolling 1D arrays of size $\min(m, n) + 1$.

---

### Solution Approach (Step-by-Step)

1. **Table Sizing Optimization:**
   - Ensure `text2` is the shorter string (swap if $m < n$) to minimize space to $\mathcal{O}(\min(m, n))$.
2. **Initialize DP Arrays:**
   - Maintain a 1D array `prev` of length $n + 1$ initialized to 0.
   - `curr` of length $n + 1$ initialized to 0.
3. **Iterative Transitions:**
   - For $i$ from 1 to $m$:
     - For $j$ from 1 to $n$:
       - If $text1[i-1] == text2[j-1]$:
         - $curr[j] = 1 + prev[j-1]$
       - Else:
         - $curr[j] = \max(prev[j], curr[j-1])$
     - Swap `prev = curr` and reset `curr`.
4. **Return Result:**
   - Return `prev[n]`.

---

### Visual Algorithm Walkthrough

#### Trace for `text1 = "abcde"`, `text2 = "ace"`
```
text1 (m = 5), text2 (n = 3)

DP Table Construction:
        ""    a    c    e
  ""  [  0,   0,   0,   0 ]
  a   [  0,   1,   1,   1 ]  (text1[0]=='a' == text2[0]=='a' -> 1 + dp[0][0])
  b   [  0,   1,   1,   1 ]  (text1[1]=='b' != ace -> max(1, 0) = 1)
  c   [  0,   1,   2,   2 ]  (text1[2]=='c' == text2[1]=='c' -> 1 + dp[1][1] = 2)
  d   [  0,   1,   2,   2 ]  (mismatch)
  e   [  0,   1,   2,   3 ]  (text1[4]=='e' == text2[2]=='e' -> 1 + dp[3][2] = 3)

Final cell: dp[5][3] = 3.
Common Subsequence: "ace", length = 3.
```

---

### Solved Examples with Multiple Inputs

| `text1` | `text2` | Common Characters | DP Recurrence Match Points | Output Length |
|---|---|---|---|---|
| `"abcde"` | `"ace"` | `'a'`, `'c'`, `'e'` | $(1,1), (3,2), (5,3)$ | `3` |
| `"abc"` | `"abc"` | `'a'`, `'b'`, `'c'` | Diagonal matches throughout | `3` |
| `"abc"` | `"def"` | None | All cells evaluate to 0 | `0` |
| `"ezupkr"` | `"ubmrapg"` | `'u'`, `'p'`, `'r'` | Interspersed matches | `3` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # Space-optimize by ensuring text2 is shorter
        if len(text1) < len(text2):
            text1, text2 = text2, text1
            
        m, n = len(text1), len(text2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    curr[j] = 1 + prev[j - 1]
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
            
        return prev[n]
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int longestCommonSubsequence(std::string text1, std::string text2) {
        if (text1.size() < text2.size()) {
            std::swap(text1, text2);
        }
        
        int m = static_cast<int>(text1.size());
        int n = static_cast<int>(text2.size());
        
        std::vector<int> prev(n + 1, 0);
        std::vector<int> curr(n + 1, 0);
        
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (text1[i - 1] == text2[j - 1]) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    curr[j] = std::max(prev[j], curr[j - 1]);
                }
            }
            prev = curr;
        }
        
        return prev[n];
    }
};
```

#### Java 17
```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        if (text1.length() < text2.length()) {
            String temp = text1;
            text1 = text2;
            text2 = temp;
        }
        
        int m = text1.length();
        int n = text2.length();
        
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    curr[j] = 1 + prev[j - 1];
                } else {
                    curr[j] = Math.max(prev[j], curr[j - 1]);
                }
            }
            System.arraycopy(curr, 0, prev, 0, n + 1);
        }
        
        return prev[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$, where $m = |text1|$ and $n = |text2|$. We populate an $m \times n$ grid where each state calculation involves $\mathcal{O}(1)$ operations. With $m, n \le 1000$, this requires at most $10^6$ operations, which executes in under $15$ ms.
- **Space Complexity:** $\mathcal{O}(\min(m, n))$ auxiliary space. By swapping strings to ensure the inner loop iterates over the shorter string and using two rolling rows, memory usage drops from $1 \text{ MB}$ to $< 1 \text{ KB}$.

---

### Takeaway Pattern & Interview Traps

1. **Subsequence vs. Substring:** A *subsequence* does not require elements to be contiguous, allowing diagonal skipping ($dp[i-1][j]$ and $dp[i][j-1]$). In contrast, the *Longest Common Substring* requires consecutive characters, resetting $dp[i][j] = 0$ on mismatch.
2. **Reconstruction of the String:** While this problem only asks for length, interviewers often ask for the actual string. Storing the 2D table allows backtracking from $(m, n)$ back to $(0, 0)$ by following matching transitions.
3. **Core DP Archetype:** LCS forms the basis for Edit Distance (LC 72), Minimum ASCII Delete Sum (LC 712), Shortest Common Supersequence (LC 1092), and git diff algorithms (Myers Diff).