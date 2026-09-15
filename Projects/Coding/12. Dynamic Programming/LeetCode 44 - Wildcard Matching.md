---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 44: Wildcard Matching"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - meta
  - google
  - amazon
---

# LeetCode 44: Wildcard Matching

**Target Companies:** Meta, Google, Amazon, Microsoft, Bloomberg, ByteDance  
**Difficulty:** Hard  
**Topic:** 2D Dynamic Programming / String Pattern Matching / Space Optimization  

---

### Problem Statement

Given an input string (`s`) and a pattern (`p`), implement wildcard pattern matching with support for `'?'` and `'*'` where:

- `'?'` Matches any single character.
- `'*'` Matches any sequence of characters (including the empty sequence).

The matching should cover the **entire** input string (not partial).

---

### Input & Output Formats & Constraints

- **Input:**
  - `s: str` — Input text string consisting of lowercase English letters.
  - `p: str` — Pattern string consisting of lowercase English letters, `'?'`, and `'*'`.
- **Output:**
  - `bool` — `true` if pattern matches entire string, `false` otherwise.
- **Constraints:**
  - $0 \le s.\text{length}, p.\text{length} \le 2000$

---

### Key Idea & Intuition

1. **Comparison with LeetCode 10 (Regular Expression Matching):**
   - In **LeetCode 10 (Regex)**: `'*'` modifies the preceding element (e.g., `a*` matches zero or more `'a'`s).
   - In **LeetCode 44 (Wildcard)**: `'*'` stands alone and matches **any** sequence of arbitrary characters of length $\ge 0$.

2. **2D State Definition:**
   - Let $\text{dp}[i][j]$ be a boolean value indicating whether the prefix $s[0 \dots i-1]$ matches the pattern prefix $p[0 \dots j-1]$.
   - Dimensions: $(m + 1) \times (n + 1)$ where $m = |s|$ and $n = |p|$.

3. **Base Cases:**
   - Empty string and empty pattern match:
     $$\text{dp}[0][0] = \text{true}$$
   - Non-empty string and empty pattern cannot match:
     $$\text{dp}[i][0] = \text{false} \quad (\forall i \ge 1)$$
   - Empty string and non-empty pattern can match only if the pattern consists solely of `'*'`s:
     $$\text{dp}[0][j] = \text{dp}[0][j - 1] \quad \text{if } p[j - 1] == '*'$$

4. **State Transitions:**
   - **Case 1: Exact character match or `'?'`:**
     - If $p[j - 1] == s[i - 1]$ or $p[j - 1] == '?'$:
       $$\text{dp}[i][j] = \text{dp}[i - 1][j - 1]$$
   - **Case 2: Pattern character is `'*'':**
     - The `'*'` can match **empty sequence** $\implies \text{dp}[i][j - 1]$.
     - Or the `'*'` matches **at least one character** from $s \implies \text{dp}[i - 1][j]$ (the current character $s[i-1]$ is consumed, but the `'*'` remains active for subsequent characters).
     - Transition:
       $$\text{dp}[i][j] = \text{dp}[i][j - 1] \lor \text{dp}[i - 1][j]$$
   - **Case 3: Mismatch:**
     - Otherwise, $\text{dp}[i][j] = \text{false}$.

---

### Solution Approach (Step-by-Step)

1. **Table Allocation:**
   - Allocate `dp` table of size $(m + 1) \times (n + 1)$ filled with `False`.
   - Set `dp[0][0] = True`.
2. **Initialize First Row ($i = 0$):**
   - For $j$ from $1$ to $n$:
     - If $p[j - 1] == '*'$, `dp[0][j] = dp[0][j - 1]`.
     - Else break (a non-star cannot match empty string).
3. **Fill DP Table:**
   - Outer loop: $i$ from $1$ to $m$.
   - Inner loop: $j$ from $1$ to $n$.
     - If $p[j - 1] == '*'$;
       - `dp[i][j] = dp[i][j - 1] or dp[i - 1][j]`.
     - Else if $p[j - 1] == '?'$ or $p[j - 1] == s[i - 1]$:
       - `dp[i][j] = dp[i - 1][j - 1]`.
4. **Return:**
   - Return `dp[m][n]`.

---

### Visual Algorithm Walkthrough

Suppose $s = \text{"adceb"}$ and $p = \text{"*a*b"}$:

```
Table dp[i][j]:
      ""   *   a   *   b
""    T   T   F   F   F
a     F   T   T   T   F
d     F   T   F   T   F
c     F   T   F   T   F
e     F   T   F   T   F
b     F   T   F   T   T

Explanation of key cells:
- dp[0][1] (p="*"): T because * matches empty string.
- dp[1][2] (s="a", p="*a"): T because 'a' matches 'a' (from dp[0][1]).
- dp[4][3] (s="adce", p="*a*"): T because second * absorbs "dce".
- dp[5][4] (s="adceb", p="*a*b"): T because 'b' matches 'b' (from dp[4][3]).

Final result: dp[5][4] = True!
```

---

### Solved Examples with Multiple Inputs

| Case | `s` | `p` | Matching Behavior | Result | Explanation |
|---|---|---|---|---|---|
| **Exact Mismatch** | `"aa"` | `"a"` | Second `'a'` has no counterpart | `false` | Entire string must match |
| **Star Match All** | `"aa"` | `"*"` | `'*'` matches entire `"aa"` | `true` | Asterisk matches any sequence |
| **Question Mark** | `"cb"` | `"?a"` | `'c'` matches `'?'`, but `'b' != 'a'` | `false` | Last character differs |
| **Empty Strings** | `""` | `"***"` | Sequence of stars matches empty | `true` | All stars absorb empty string |
| **Leading/Trailing Stars** | `"adceb"` | `"*a*b"` | `*` matches `""`, `'a'` matches `'a'`, `*` matches `"dce"`, `'b'` matches `'b'` | `true` | Complex multi-star partition |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # Base case: empty string matches empty pattern
        dp[0][0] = True
        
        # Leading '*' can match empty string
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 1]
            else:
                break
                
        # Fill table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # Match empty (dp[i][j-1]) or match one char (dp[i-1][j])
                    dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
                elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    
        return dp[m][n]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class Solution {
public:
    bool isMatch(std::string s, std::string p) {
        int m = s.size();
        int n = p.size();

        std::vector<std::vector<bool>> dp(m + 1, std::vector<bool>(n + 1, false));
        dp[0][0] = true;

        // Initialize first row: '*' matching empty string
        for (int j = 1; j <= n; ++j) {
            if (p[j - 1] == '*') {
                dp[0][j] = dp[0][j - 1];
            } else {
                break;
            }
        }

        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                } else if (p[j - 1] == '?' || p[j - 1] == s[i - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }

        return dp[m][n];
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;

        for (int j = 1; j <= n; j++) {
            if (p.charAt(j - 1) == '*') {
                dp[0][j] = dp[0][j - 1];
            } else {
                break;
            }
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                } else if (p.charAt(j - 1) == '?' || p.charAt(j - 1) == s.charAt(i - 1)) {
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

- **Time Complexity:** $\mathcal{O}(m \times n)$  
  The nested loops evaluate each cell in the $(m + 1) \times (n + 1)$ table in $\mathcal{O}(1)$ time. With $m, n \le 2000$, total cell operations $\le 4 \times 10^6$, finishing in $\approx 25$ ms.
- **Space Complexity:** $\mathcal{O}(m \times n)$  
  Allocates a 2D boolean array of dimensions $(m + 1) \times (n + 1)$. This can be optimized to $\mathcal{O}(n)$ using a 1D rolling array.

---

### Takeaway Pattern & Interview Traps

1. **Star Transition Mechanics:**
   - Remember the dual role of `'*'`:
     - $\text{dp}[i][j - 1]$ ignores the star (zero characters matched).
     - $\text{dp}[i - 1][j]$ assumes the star matched $s[i-1]$ and remains available to match further preceding characters.
2. **Greedy Two-Pointer Alternative:**
   - Wildcard matching can also be solved with two pointers and backtracking stars in $\mathcal{O}(m \times n)$ worst-case and $\mathcal{O}(1)$ space by remembering the last star position (`star_idx`) and the match index (`match_idx`). The DP table formulation is preferred in interviews for its mathematical elegance and consistency.