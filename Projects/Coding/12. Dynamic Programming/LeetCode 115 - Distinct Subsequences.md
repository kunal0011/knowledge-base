---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 115: Distinct Subsequences"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - amazon
  - google
  - bloomberg
  - microsoft
---

# LeetCode 115: Distinct Subsequences

**Target Companies:** Amazon, Google, Bloomberg, Microsoft, Uber  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / String  

---

### Problem Statement

Given two strings `s` and `t`, return *the number of distinct subsequences of `s` which equals `t`*.

The test cases are generated so that the answer fits on a 32-bit signed integer.

---

### Input & Output Formats & Constraints

- **Input:** Two strings `s` and `t` consisting of uppercase and lowercase English letters.
- **Output:** An integer representing the number of distinct subsequences of `s` that match `t`.
- **Constraints:**
  - `1 <= s.length, t.length <= 1000`
  - `s` and `t` consist of English letters.
  - The answer fits in a 32-bit signed integer.

---

### Key Idea & Intuition

#### Prefix-Matching State Invariant
We define $dp[i][j]$ as the number of distinct subsequences of prefix $s[0..i-1]$ that equal prefix $t[0..j-1]$.
- Dimensions: $(m + 1) \times (n + 1)$, where $m = |s|$ and $n = |t|$.
- **Base Case 1 ($j = 0$):** An empty target string $t$ can be formed from any prefix of $s$ in exactly **one way** (by deleting all characters in $s$):
  $$dp[i][0] = 1 \quad \forall 0 \le i \le m$$
- **Base Case 2 ($i = 0, j > 0$):** A non-empty target string $t$ cannot be formed from an empty source string $s$:
  $$dp[0][j] = 0 \quad \forall 1 \le j \le n$$

#### State Transitions
For characters $s[i-1]$ and $t[j-1]$:
1. **Characters Match ($s[i-1] == t[j-1]$):**
   We have two valid, mutually exclusive options:
   - **Option A (Match):** Pair $s[i-1]$ with $t[j-1]$. The number of ways to do this equals the number of ways to form $t[0..j-2]$ from $s[0..i-2]$, which is $dp[i-1][j-1]$.
   - **Option B (Skip):** Choose not to use $s[i-1]$ in this match. The number of ways equals $dp[i-1][j]$.
   $$\text{Summing both choices: } dp[i][j] = dp[i-1][j-1] + dp[i-1][j]$$
2. **Characters Do Not Match ($s[i-1] \ne t[j-1]$):**
   Character $s[i-1]$ cannot be paired with $t[j-1]$. We are forced to skip it:
   $$dp[i][j] = dp[i-1][j]$$

#### Space Optimization to 1D Array
Notice that calculating row $i$ only requires entries from the previous row $i - 1$.
By traversing $j$ **in reverse** from $n$ down to $1$:
$$dp[j] \leftarrow dp[j] + (dp[j-1] \text{ if } s[i-1] == t[j-1] \text{ else } 0)$$
We compress the space complexity from $\mathcal{O}(m \times n)$ down to $\mathcal{O}(n)$.

---

### Solution Approach (Step-by-Step)

1. **Quick Pruning:**
   - If $|s| < |t|$, return `0` immediately.
2. **Initialize 1D Array:**
   - Array `dp` of size $n + 1$ filled with zeros.
   - Set $dp[0] = 1$ (representing the empty target string).
3. **Iterate Through Source and Target:**
   - For each character $c$ in $s$:
     - For $j$ from $n$ down to 1:
       - If $c == t[j-1]$:
         - $dp[j] += dp[j-1]$
4. **Return:**
   - Return $dp[n]$.

---

### Visual Algorithm Walkthrough

#### Trace for `s = "rabbbit"`, `t = "rabbit"`
```
m = 7 ('r','a','b','b','b','i','t'), n = 6 ('r','a','b','b','i','t')

Table Evolution:
        ""   r   a   b   b   i   t
  ""  [  1,  0,  0,  0,  0,  0,  0 ]
  r   [  1,  1,  0,  0,  0,  0,  0 ]  ('r' matches 'r')
  a   [  1,  1,  1,  0,  0,  0,  0 ]  ('a' matches 'a')
  b   [  1,  1,  1,  1,  0,  0,  0 ]  ('b' matches 1st 'b')
  b   [  1,  1,  1,  2,  1,  0,  0 ]  ('b' matches 1st or 2nd 'b': dp[b2] = 1+0=1; dp[b1] = 1+1=2)
  b   [  1,  1,  1,  3,  3,  0,  0 ]  ('b' matches 1st or 2nd 'b': dp[b2] = 1+2=3; dp[b1] = 2+1=3)
  i   [  1,  1,  1,  3,  3,  3,  0 ]  ('i' matches 'i': dp[i] = 0 + 3 = 3)
  t   [  1,  1,  1,  3,  3,  3,  3 ]  ('t' matches 't': dp[t] = 0 + 3 = 3)

Final Answer: dp[6] = 3.
The three 'b's in s ("ra[bb]bit", "ra[b]b[b]it", "rab[bb]it") yield 3 distinct ways!
```

---

### Solved Examples with Multiple Inputs

| Source $s$ | Target $t$ | Distinct Subsequences | Output |
|---|---|---|---|
| `"rabbbit"` | `"rabbit"` | Indices: `{0,1,2,3,5,6}`, `{0,1,2,4,5,6}`, `{0,1,3,4,5,6}` | `3` |
| `"babgbag"` | `"bag"` | Indices: `{0,1,4}`, `{0,3,4}`, `{1,3,4}`, `{2,3,4}`, `{0,1,6}`... | `5` |
| `"abc"` | `"def"` | No common characters | `0` |
| `"aaa"` | `"a"` | Each of the 3 `'a'`s independently | `3` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        if m < n:
            return 0
            
        # dp[j] stores number of subsequences of s prefix matching t[0..j-1]
        dp = [0] * (n + 1)
        dp[0] = 1  # Empty target string has 1 match
        
        for c in s:
            # Traverse backwards to preserve previous row's values
            for j in range(n, 0, -1):
                if c == t[j - 1]:
                    dp[j] += dp[j - 1]
                    
        return dp[n]
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    int numDistinct(const std::string& s, const std::string& t) {
        int m = static_cast<int>(s.size());
        int n = static_cast<int>(t.size());
        if (m < n) return 0;
        
        // Use unsigned long long to avoid intermediate integer overflow
        std::vector<unsigned long long> dp(n + 1, 0);
        dp[0] = 1;
        
        for (char c : s) {
            for (int j = n; j >= 1; --j) {
                if (c == t[j - 1]) {
                    dp[j] += dp[j - 1];
                }
            }
        }
        
        return static_cast<int>(dp[n]);
    }
};
```

#### Java 17
```java
class Solution {
    public int numDistinct(String s, String t) {
        int m = s.length();
        int n = t.length();
        if (m < n) return 0;
        
        // Use double or long to prevent intermediate overflow before final result
        int[] dp = new int[n + 1];
        dp[0] = 1;
        
        // Space-optimized 1D DP with reverse inner loop
        for (int i = 0; i < m; i++) {
            char c = s.charAt(i);
            for (int j = n; j >= 1; j--) {
                if (c == t.charAt(j - 1)) {
                    dp[j] += dp[j - 1];
                }
            }
        }
        
        return dp[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$, where $m = |s|$ and $n = |t|$. Each character in $s$ updates at most $n$ states in reverse with $\mathcal{O}(1)$ operations.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space using the 1D rolling array.

---

### Takeaway Pattern & Interview Traps

1. **Reverse Iteration in 1D Space Optimization:** When transitioning $dp[j] = dp[j] + dp[j-1]$, moving $j$ from left to right would overwrite $dp[j-1]$ with its new value before $dp[j]$ has a chance to read its previous value. Moving $j$ in reverse ($n \to 1$) keeps $dp[j-1]$ untouched.
2. **Intermediate Overflow in C++:** Even though LeetCode guarantees the final answer fits in a 32-bit signed integer, intermediate table values in paths that don't reach $dp[m][n]$ can exceed `INT_MAX`. Using `unsigned long long` in C++ protects against runtime overflow exceptions.
3. **Difference from LCS:** LCS computes the *length* of the longest common subsequence ($\max$). LC 115 computes the *count* of distinct matching ways ($\sum$).