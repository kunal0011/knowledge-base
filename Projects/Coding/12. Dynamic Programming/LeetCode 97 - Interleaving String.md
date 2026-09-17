---
date: "2026-09-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 97: Interleaving String"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - 2d-dp
  - amazon
  - google
  - meta
  - microsoft
  - apple
---

# LeetCode 97: Interleaving String

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber, Bloomberg  
**Difficulty:** Medium  
**Topic:** 2D Dynamic Programming / String Matching

---

### Problem Statement

Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an **interleaving** of `s1` and `s2`.

An **interleaving** of two strings `s` and `t` is a configuration where they are divided into non-empty substrings such that:
- $s = s_1 + s_2 + \dots + s_n$
- $t = t_1 + t_2 + \dots + t_m$
- $|n - m| \le 1$
- The interleaving is $s_1 + t_1 + s_2 + t_2 + \dots$ or $t_1 + s_1 + t_2 + s_2 + \dots$

Notice that $a + b$ is the concatenation of strings $a$ and $b$.

Equivalently, every character of `s1` and `s2` must appear in `s3` in their exact relative order, with no extra or missing characters.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s1: str` (length $m$)
  - `s2: str` (length $n$)
  - `s3: str` (length $k$)
- **Output:** `bool`
- **Constraints:**
  - `0 <= s1.length, s2.length <= 100`
  - `0 <= s3.length <= 200`
  - `s1`, `s2`, and `s3` consist of lowercase English letters.
- **Follow-up:** Could you solve it using only $\mathcal{O}(s_2\text{.length})$ additional memory space?

---

### Key Idea & Intuition

#### 1. Length Feasibility Invariant
First, if $|s_1| + |s_2| \ne |s_3|$, it is mathematically impossible for $s_3$ to be an interleaving. We can return `False` immediately.

#### 2. Dynamic Programming Formulation
Let $dp[i][j]$ be a boolean value representing:
> Can prefix $s_3[0 \dots i + j - 1]$ be formed by interleaving prefix $s_1[0 \dots i - 1]$ of length $i$ and prefix $s_2[0 \dots j - 1]$ of length $j$?

To compute $dp[i][j]$, consider the origin of the last character $s_3[i + j - 1]$:
1. **Option 1 (From $s_1$):**
   The character was taken from $s_1[i-1]$. This is valid if:
   $$dp[i - 1][j] \text{ is True } \land s_1[i - 1] == s_3[i + j - 1]$$
2. **Option 2 (From $s_2$):**
   The character was taken from $s_2[j-1]$. This is valid if:
   $$dp[i][j - 1] \text{ is True } \land s_2[j - 1] == s_3[i + j - 1]$$

Combining both transitions:
$$dp[i][j] = (dp[i-1][j] \land s_1[i-1] == s_3[i+j-1]) \lor (dp[i][j-1] \land s_2[j-1] == s_3[i+j-1])$$

#### 3. Base Cases
- $dp[0][0] = \text{True}$: Empty $s_1$ and empty $s_2$ interleave to form empty $s_3$.
- First Column ($j = 0$): $dp[i][0] = dp[i-1][0] \land (s_1[i-1] == s_3[i-1])$.
- First Row ($i = 0$): $dp[0][j] = dp[0][j-1] \land (s_2[j-1] == s_3[j-1])$.

#### 4. Space Optimization ($\mathcal{O}(\min(m, n))$)
Notice that evaluating row $i$ only requires:
- $dp[i-1][j]$: The value directly above from the previous row (which is the old value of `dp[j]`).
- $dp[i][j-1]$: The value directly to the left in the current row (which is the updated value of `dp[j-1]`).

Thus, we can collapse the 2D grid into a 1D array of size $|s_2| + 1$, updating in-place from left to right.

---

### Solution Approach (Step-by-Step)

1. **Length Check:**
   - If $|s_1| + |s_2| \ne |s_3|$, return `False`.
2. **Dimension Optimization:**
   - If $|s_1| < |s_2|$, swap $s_1$ and $s_2$ so that the 1D DP table only requires $\mathcal{O}(\min(|s_1|, |s_2|))$ space.
3. **Initialize 1D DP Array:**
   - Let $m = |s_1|$ and $n = |s_2|$.
   - Create `dp` of length $n + 1$, initialized to `False`.
   - Set `dp[0] = True`.
4. **Initialize Row 0 (Matching prefix of $s_2$ with prefix of $s_3$):**
   - For $j = 1$ to $n$:
     - `dp[j] = dp[j - 1] and (s2[j - 1] == s3[j - 1])`
5. **Iterate Rows ($i = 1$ to $m$):**
   - First update column 0 ($j = 0$):
     - `dp[0] = dp[0] and (s1[i - 1] == s3[i - 1])`
   - For columns $j = 1$ to $n$:
     - Let $from\_s1 = dp[j] \land (s_1[i - 1] == s_3[i + j - 1])$
     - Let $from\_s2 = dp[j - 1] \land (s_2[j - 1] == s_3[i + j - 1])$
     - `dp[j] = from_s1 or from_s2`
6. **Return Result:**
   - Return `dp[n]`.

---

### Visual Algorithm Walkthrough

#### Example: `s1 = "aabcc"`, `s2 = "dbbca"`, `s3 = "aadbbcbcac"`
$|s_1| = 5, |s_2| = 5, |s_3| = 10$. Length sum matches ($5 + 5 = 10$).

2D DP Table (`T` = True, `.` = False):
```
           ""   d   b   b   c   a
        j=  0   1   2   3   4   5
""   i=0   [T]  .   .   .   .   .
a      1    T   .   .   .   .   .
a      2    T   T   T   T   T   .
b      3    .   T   T   .   T   .
c      4    .   .   T   T   T   T
c      5    .   .   .   T   .  [T]
```

Trace for Cell `dp[5][5]` ($i=5, j=5 \to s_3[9] = 'c'$):
- From top (`dp[4][5]` = True and $s_1[4] == 'c'$ == $s_3[9] == 'c'$): **Valid!**
- Hence `dp[5][5] = True`. Result is `True`.

#### Counter-Example: `s1 = "aabcc"`, `s2 = "dbbca"`, `s3 = "aadbbbaccc"`
- At $i=3, j=3$, required character is $s_3[5] = 'b'$, but transitions lead to dead ends because duplicate `'b'`s consume characters needed downstream.
- The table cannot reach `dp[5][5] = True`. Returns `False`.

---

### Solved Examples with Multiple Inputs

| `s1` | `s2` | `s3` | Valid Interleaving? | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `"aabcc"` | `"dbbca"` | `"aadbbcbcac"` | `True` | Interleaved as `aa` + `dbbc` + `bc` + `a` + `c` |
| `"aabcc"` | `"dbbca"` | `"aadbbbaccc"` | `False` | Cannot preserve ordering of character counts |
| `""` | `""` | `""` | `True` | Base case: all empty strings match |
| `""` | `"abc"` | `"abc"` | `True` | One empty string reduces to identical string matching |
| `"a"` | `"b"` | `"a"` | `False` | $\vert s_1 \vert + \vert s_2 \vert = 2 \ne \vert s_3 \vert = 1$, early exit |
| `"ab"` | `"bc"` | `"babc"` | `True` | Interleaved: `s2[0]` ('b') + `s1[0..1]` ("ab") + `s2[1]` ('c') |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n, k = len(s1), len(s2), len(s3)
        
        # Length check
        if m + n != k:
            return False
        
        # Ensure s2 is the shorter string for O(min(m, n)) space
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m
            
        # dp[j] represents whether s1[:i] and s2[:j] form s3[:i+j]
        dp = [False] * (n + 1)
        dp[0] = True
        
        # Initialize row 0 (using only characters from s2)
        for j in range(1, n + 1):
            dp[j] = dp[j - 1] and (s2[j - 1] == s3[j - 1])
            
        # Fill DP table row by row
        for i in range(1, m + 1):
            # Column 0: using only characters from s1
            dp[0] = dp[0] and (s1[i - 1] == s3[i - 1])
            
            for j in range(1, n + 1):
                from_s1 = dp[j] and (s1[i - 1] == s3[i + j - 1])
                from_s2 = dp[j - 1] and (s2[j - 1] == s3[i + j - 1])
                dp[j] = from_s1 or from_s2
                
        return dp[n]
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    bool isInterleave(std::string s1, std::string s2, std::string s3) {
        int m = s1.length();
        int n = s2.length();
        int k = s3.length();

        if (m + n != k) {
            return false;
        }

        // Optimize space by making s2 the shorter string
        if (m < n) {
            std::swap(s1, s2);
            std::swap(m, n);
        }

        std::vector<bool> dp(n + 1, false);
        dp[0] = true;

        // Base case: matching prefix of s2 with prefix of s3
        for (int j = 1; j <= n; ++j) {
            dp[j] = dp[j - 1] && (s2[j - 1] == s3[j - 1]);
        }

        for (int i = 1; i <= m; ++i) {
            dp[0] = dp[0] && (s1[i - 1] == s3[i - 1]);

            for (int j = 1; j <= n; ++j) {
                bool fromS1 = dp[j] && (s1[i - 1] == s3[i + j - 1]);
                bool fromS2 = dp[j - 1] && (s2[j - 1] == s3[i + j - 1]);
                dp[j] = fromS1 || fromS2;
            }
        }

        return dp[n];
    }
};
```

#### Java 17
```java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        int m = s1.length();
        int n = s2.length();
        int k = s3.length();

        if (m + n != k) {
            return false;
        }

        // Optimize space: ensure s2 is shorter
        if (m < n) {
            return isInterleave(s2, s1, s3);
        }

        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        // Base row: using only characters from s2
        for (int j = 1; j <= n; j++) {
            dp[j] = dp[j - 1] && (s2.charAt(j - 1) == s3.charAt(j - 1));
        }

        for (int i = 1; i <= m; i++) {
            dp[0] = dp[0] && (s1.charAt(i - 1) == s3.charAt(i - 1));

            for (int j = 1; j <= n; j++) {
                boolean fromS1 = dp[j] && (s1.charAt(i - 1) == s3.charAt(i + j - 1));
                boolean fromS2 = dp[j - 1] && (s2.charAt(j - 1) == s3.charAt(i + j - 1));
                dp[j] = fromS1 || fromS2;
            }
        }

        return dp[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \cdot n)$, where $m = |s_1|$ and $n = |s_2|$. We have two nested loops of size $m$ and $n$. In each iteration, we perform constant time $\mathcal{O}(1)$ operations. Since $m, n \le 100$, maximum operations are $\approx 10,000$, executing in under $2$ ms.
- **Space Complexity:** $\mathcal{O}(\min(m, n))$ auxiliary space. By swapping strings if needed, the 1D dynamic programming array only requires $\min(m, n) + 1$ boolean entries, satisfying the follow-up memory constraint.

---

### Takeaway Pattern & Interview Traps

1. **Greedy Trapping:**
   - When both $s_1[i-1]$ and $s_2[j-1]$ match $s_3[i+j-1]$, a greedy choice cannot decide which string to consume without looking into future characters. DP or memoized recursion is essential to explore both branches.
2. **Length Guard:**
   - Always check $|s_1| + |s_2| == |s_3|$ before initializing DP tables; otherwise, out-of-bounds or false positive edge cases may occur.
3. **1D Rolling Array Updating Order:**
   - In 1D DP, notice that `dp[j]` before being updated holds the value from the previous row ($dp[i-1][j]$), while `dp[j-1]` has already been updated for the current row ($dp[i][j-1]$). Scanning left-to-right matches the recurrence naturally without needing a temporary array.