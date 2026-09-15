---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 516: Longest Palindromic Subsequence"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - interval-dp
  - string
  - lcs
  - amazon
  - meta
  - google
---

# LeetCode 516: Longest Palindromic Subsequence

**Target Companies:** Amazon, Meta, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Interval Dynamic Programming / String Subsequence / LCS Equivalence / Space Optimization  

---

### Problem Statement

Given a string `s`, find the longest palindromic subsequence's length in `s`.

A **subsequence** is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

---

### Input & Output Formats & Constraints

- **Input:** `s: str` — String consisting of lowercase English letters.
- **Output:** `int` — Length of the longest palindromic subsequence.
- **Constraints:**
  - $1 \le s.\text{length} \le 1000$

---

### Key Idea & Intuition

1. **Substrings vs. Subsequences:**
   - In LeetCode 5 (Longest Palindromic Substring), characters must be strictly contiguous.
   - Here, characters can be skipped, allowing us to pick any symmetric subset of characters in $s$.

2. **Interval DP Formulation:**
   - Let $\text{dp}[i][j]$ be the length of the longest palindromic subsequence in the substring $s[i \dots j]$ ($0 \le i \le j < n$).
   - **Base Cases:**
     - A single character is a palindrome of length 1: $\text{dp}[i][i] = 1$.
     - When $i > j$: length is $0$.
   - **Transitions:**
     - **Case 1: Outer characters match ($s[i] == s[j]$):**
       - Both endpoints can be matched together to bookend any palindromic subsequence found in the inner substring $s[i+1 \dots j-1]$:
         $$\text{dp}[i][j] = 2 + \text{dp}[i + 1][j - 1]$$
     - **Case 2: Outer characters do not match ($s[i] \ne s[j]$):**
       - The characters $s[i]$ and $s[j]$ cannot both be part of the same palindromic subsequence's boundaries.
       - The optimal solution is the maximum of excluding $s[i]$ or excluding $s[j]$:
         $$\text{dp}[i][j] = \max(\text{dp}[i + 1][j], \ \text{dp}[i][j - 1])$$

3. **LCS Equivalence (Dual Perspective):**
   - The Longest Palindromic Subsequence of string $s$ is strictly equivalent to finding the **Longest Common Subsequence (LCS)** between $s$ and its reverse $s^R$:
     $$\text{LPS}(s) = \text{LCS}(s, s^R)$$

4. **1D Space Optimization:**
   - Notice that to compute row $i$, we only need values from row $i + 1$ and the cell diagonally to the bottom-left.
   - We can compress the $(n \times n)$ table into a single 1D array of size $n$, reducing space from $\mathcal{O}(n^2)$ to $\mathcal{O}(n)$.

---

### Solution Approach (Step-by-Step)

1. **Initialize 1D Array:**
   - `dp = [0] * n`.
2. **Reverse Outer Traversal:**
   - Iterate $i$ backwards from $n - 1$ down to $0$:
     - Set `dp[i] = 1` (base case: single character $s[i \dots i]$).
     - `prev = 0` (stores $\text{dp}[i + 1][j - 1]$ from the previous state).
     - For $j$ from $i + 1$ to $n - 1$:
       - `temp = dp[j]` (saves $\text{dp}[i + 1][j]$ before overwriting).
       - If $s[i] == s[j]$:
         - `dp[j] = 2 + prev`.
       - Else:
         - `dp[j] = max(dp[j], dp[j - 1])`.
       - `prev = temp`.
3. **Return:**
   - Return `dp[n - 1]`.

---

### Visual Algorithm Walkthrough

For `s = "bbbab"`, $n = 5$:

```
Initial 2D view:
        b  b  b  a  b
      [ 0, 1, 2, 3, 4 ]
b (0)   1  2  3  3  4
b (1)      1  2  2  3
b (2)         1  1  3
a (3)            1  1
b (4)               1

Trace for i = 0, j = 4:
  s[0] == 'b' and s[4] == 'b' -> match!
  dp[0][4] = 2 + dp[1][3]
  dp[1][3] was max(dp[2][3], dp[1][2]) = max(1, 2) = 2
  dp[0][4] = 2 + 2 = 4

Optimal Subsequence: "bbbb" (length 4).
```

---

### Solved Examples with Multiple Inputs

| Case | `s` | Longest Palindromic Subsequence | Result | Explanation |
|---|---|---|---|---|
| **Standard** | `"bbbab"` | `"bbbb"` | `4` | Skip `'a'` at index 3 |
| **Two Characters** | `"cbbd"` | `"bb"` | `2` | Skip `'c'` and `'d'` |
| **All Identical** | `"aaaaa"` | `"aaaaa"` | `5` | Entire string is a palindrome |
| **Single Character** | `"z"` | `"z"` | `1` | Base case length 1 |
| **Strictly Alternating** | `"abcba"` | `"abcba"` | `5` | Already a palindrome |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Space Optimized $\mathcal{O}(n)$)
```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [0] * n
        
        # Traverse i backwards from n - 1 down to 0
        for i in range(n - 1, -1, -1):
            dp[i] = 1
            prev = 0  # Represents dp[i+1][j-1]
            for j in range(i + 1, n):
                temp = dp[j]
                if s[i] == s[j]:
                    dp[j] = 2 + prev
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
                
        return dp[n - 1]
```

#### 2. C++ (C++17 / STL — Space Optimized $\mathcal{O}(n)$)
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int longestPalindromeSubseq(std::string s) {
        int n = s.size();
        std::vector<int> dp(n, 0);

        for (int i = n - 1; i >= 0; --i) {
            dp[i] = 1;
            int prev = 0; // Represents dp[i+1][j-1]

            for (int j = i + 1; j < n; ++j) {
                int temp = dp[j];
                if (s[i] == s[j]) {
                    dp[j] = 2 + prev;
                } else {
                    dp[j] = std::max(dp[j], dp[j - 1]);
                }
                prev = temp;
            }
        }

        return dp[n - 1];
    }
};
```

#### 3. Java (Modern, Typed — Space Optimized $\mathcal{O}(n)$)
```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[] dp = new int[n];

        for (int i = n - 1; i >= 0; i--) {
            dp[i] = 1;
            int prev = 0; // Represents dp[i+1][j-1]

            for (int j = i + 1; j < n; j++) {
                int temp = dp[j];
                if (s.charAt(i) == s.charAt(j)) {
                    dp[j] = 2 + prev;
                } else {
                    dp[j] = Math.max(dp[j], dp[j - 1]);
                }
                prev = temp;
            }
        }

        return dp[n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^2)$  
  The nested loops evaluate all $\frac{n(n + 1)}{2}$ sub-intervals in $\mathcal{O}(1)$ time per cell. With $n \le 1000$, total operations $\approx 5 \times 10^5$, executing in under $15$ ms.
- **Space Complexity:** $\mathcal{O}(n)$  
  Space optimization reduces memory from an $\mathcal{O}(n^2)$ matrix down to a single 1D array of length $n$ ($\le 1000$ integers $\approx 4$ KB).

---

### Takeaway Pattern & Interview Traps

1. **Subsequence vs Substring Recurrence:**
   - In Palindromic Substring (LC 5), if characters mismatch, $\text{dp}[i][j] = \text{false}$ (broken continuity).
   - In Palindromic Subsequence (LC 516), if characters mismatch, we take $\max(\text{dp}[i + 1][j], \text{dp}[i][j - 1])$ (skipping non-contributing characters).
2. **The Diagonal Value Preservation Trap:**
   - In 1D DP compression, before overwriting `dp[j]`, you must cache its old value into `temp` so it can serve as `prev` ($\text{dp}[i+1][j-1]$) in the next iteration of $j$.