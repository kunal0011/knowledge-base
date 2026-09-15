---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 132: Palindrome Partitioning II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - amazon
  - google
  - meta
  - bloomberg
---

# LeetCode 132: Palindrome Partitioning II

**Target Companies:** Amazon, Google, Meta, Bloomberg, Microsoft  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / String  

---

### Problem Statement

Given a string `s`, partition `s` such that every substring of the partition is a **palindrome**.

Return *the **minimum cuts** needed for a palindrome partitioning of `s`*.

---

### Input & Output Formats & Constraints

- **Input:** A string `s` ($1 \le |s| \le 2000$) consisting of lowercase English letters.
- **Output:** An integer representing the minimum number of cuts required.
- **Constraints:**
  - `1 <= s.length <= 2000`
  - `s` consists of lowercase English letters only.

---

### Key Idea & Intuition

#### Why Backtracking Fails
In LeetCode 131, we generated all valid partitions. Because $N \le 16$, the exponential count $\mathcal{O}(2^N)$ was tolerable.
Here, $N \le 2000$. An exponential DFS will immediately result in Time Limit Exceeded (TLE). We only need to find the **minimum number of cuts**, which is an optimization problem ideally suited for Dynamic Programming.

#### Two-Stage DP Formulation
1. **Stage 1: Palindrome Table ($is\_pal[i][j]$):**
   - $is\_pal[i][j]$ is `true` if $s[i..j]$ is a palindrome:
     $$is\_pal[i][j] = (s[i] == s[j]) \land (j - i \le 2 \lor is\_pal[i + 1][j - 1])$$
2. **Stage 2: Minimum Cuts DP ($dp[i]$):**
   - Let $dp[i]$ be the minimum cuts required to partition prefix $s[0..i]$ into palindromes.
   - **Case A (No Cut Needed):** If $s[0..i]$ is already a palindrome ($is\_pal[0][i] == \text{true}$):
     $$dp[i] = 0$$
   - **Case B (Partitioning Prefix):** We consider all possible last cuts at position $j$ ($1 \le j \le i$):
     - If the suffix $s[j..i]$ is a palindrome ($is\_pal[j][i] == \text{true}$), then placing a cut immediately before index $j$ requires $dp[j - 1] + 1$ cuts.
     - Taking the minimum across all valid $j$:
       $$dp[i] = \min_{\substack{1 \le j \le i \\ is\_pal[j][i] == \text{true}}} (dp[j - 1] + 1)$$
   - Base initialization: A string of length $i + 1$ can always be partitioned using at most $i$ cuts (cutting every single character), so $dp[i] = i$.

---

### Solution Approach (Step-by-Step)

1. **Precompute Palindromes:**
   - Create 2D boolean array $is\_pal$ of size $n \times n$.
   - Iterate $i$ from $n - 1$ down to 0:
     - For $j$ from $i$ to $n - 1$:
       - $is\_pal[i][j] = (s[i] == s[j]) \land (j - i \le 2 \lor is\_pal[i + 1][j - 1])$.
2. **Compute 1D Cuts DP:**
   - Create 1D array $dp$ of size $n$, where $dp[i] = i$.
   - For $i$ from 0 to $n - 1$:
     - If $is\_pal[0][i]$:
       - $dp[i] = 0$.
     - Else:
       - For $j$ from 1 to $i$:
         - If $is\_pal[j][i]$:
           - $dp[i] = \min(dp[i], dp[j - 1] + 1)$.
3. **Return:**
   - Return $dp[n - 1]$.

---

### Visual Algorithm Walkthrough

#### Trace for `s = "aab"`
```
String indices:
  0: 'a', 1: 'a', 2: 'b'

Step 1: Palindrome Table (is_pal):
  is_pal[0][0] = True ("a")
  is_pal[0][1] = True ("aa")
  is_pal[0][2] = False ("aab")
  is_pal[1][1] = True ("a")
  is_pal[1][2] = False ("ab")
  is_pal[2][2] = True ("b")

Step 2: DP Minimum Cuts Table:
i = 0 ("a"):
  is_pal[0][0] is True -> dp[0] = 0

i = 1 ("aa"):
  is_pal[0][1] is True -> dp[1] = 0

i = 2 ("aab"):
  is_pal[0][2] is False.
  Try candidate cuts j:
  - j = 1: suffix s[1..2] = "ab". is_pal[1][2] is False.
  - j = 2: suffix s[2..2] = "b". is_pal[2][2] is True!
    dp[2] = min(dp[2], dp[1] + 1) = min(2, 0 + 1) = 1

Result: dp[2] = 1 (Cut after 'aa': ["aa", "b"]).
```

---

### Solved Examples with Multiple Inputs

| Input String $s$ | Entire String Palindrome? | Optimal Cuts Breakdown | Minimum Cuts |
|---|---|---|---|
| `"aab"` | No | `"aa" | "b"` | `1` |
| `"a"` | Yes | Single character | `0` |
| `"ab"` | No | `"a" | "b"` | `1` |
| `"racecar"` | Yes | No cuts needed | `0` |
| `"ababbbabbababa"` | No | Substrings: `"a" | "b" | "abbba" | "bb" | "ababa"` | `3` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def minCut(self, s: str) -> int:
        n: int = len(s)
        
        # Stage 1: Precompute all palindromic substrings
        is_pal: list[list[bool]] = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or is_pal[i + 1][j - 1]):
                    is_pal[i][j] = True
                    
        # Stage 2: 1D DP to find minimum cuts
        dp: list[int] = [i for i in range(n)]
        
        for i in range(n):
            if is_pal[0][i]:
                dp[i] = 0
            else:
                for j in range(1, i + 1):
                    if is_pal[j][i]:
                        dp[i] = min(dp[i], dp[j - 1] + 1)
                        
        return dp[n - 1]
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int minCut(const std::string& s) {
        int n = static_cast<int>(s.size());
        
        // Stage 1: Palindrome Table
        std::vector<std::vector<bool>> is_pal(n, std::vector<bool>(n, false));
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i; j < n; ++j) {
                if (s[i] == s[j] && (j - i <= 2 || is_pal[i + 1][j - 1])) {
                    is_pal[i][j] = true;
                }
            }
        }
        
        // Stage 2: Min Cuts DP
        std::vector<int> dp(n);
        for (int i = 0; i < n; ++i) {
            dp[i] = i; // maximum possible cuts
            if (is_pal[0][i]) {
                dp[i] = 0;
            } else {
                for (int j = 1; j <= i; ++j) {
                    if (is_pal[j][i]) {
                        dp[i] = std::min(dp[i], dp[j - 1] + 1);
                    }
                }
            }
        }
        
        return dp[n - 1];
    }
};
```

#### Java 17
```java
class Solution {
    public int minCut(String s) {
        int n = s.length();
        
        // Stage 1: Palindrome precomputation
        boolean[][] isPal = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (s.charAt(i) == s.charAt(j) && (j - i <= 2 || isPal[i + 1][j - 1])) {
                    isPal[i][j] = true;
                }
            }
        }
        
        // Stage 2: 1D DP for minimum cuts
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = i;
            if (isPal[0][i]) {
                dp[i] = 0;
            } else {
                for (int j = 1; j <= i; j++) {
                    if (isPal[j][i]) {
                        dp[i] = Math.min(dp[i], dp[j - 1] + 1);
                    }
                }
            }
        }
        
        return dp[n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^2)$, where $n = |s|$. Populating the palindrome table takes $\frac{n(n + 1)}{2} = \mathcal{O}(n^2)$ time. Finding the minimum cuts takes $\sum_{i=1}^{n} i = \mathcal{O}(n^2)$ time. For $n \le 2000$, $\approx 2 \times 10^6$ operations, executing in under $30$ ms.
- **Space Complexity:** $\mathcal{O}(n^2)$ auxiliary space for the 2D boolean palindrome matrix, and $\mathcal{O}(n)$ space for the 1D cuts array.

---

### Takeaway Pattern & Interview Traps

1. **Space Optimization via Expand Around Center:**
   Instead of storing an explicit $\mathcal{O}(n^2)$ table, we can expand outward from every center $(i, i)$ and $(i, i + 1)$, directly updating $dp[right]$ from $dp[left - 1] + 1$. This reduces space from $\mathcal{O}(n^2)$ to $\mathcal{O}(n)$ while preserving $\mathcal{O}(n^2)$ time.
2. **Number of Cuts vs Number of Palindromes:** The problem asks for the number of *cuts*, not the number of substrings. A string with $K$ palindromes has $K - 1$ cuts. If the prefix itself is a palindrome, cuts needed is $0$.
3. **Difference from LC 131:** LC 131 requires backtracking because all actual partition paths must be returned. LC 132 is a scalar minimization problem that must use DP to avoid exponential blowup.