---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 72: Edit Distance"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - wagner-fischer
  - google
  - amazon
  - meta
---

# LeetCode 72: Edit Distance

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, Uber  
**Difficulty:** Medium  
**Topic:** 2D Dynamic Programming / String Alignment / Wagner-Fischer Algorithm / Space Optimization  

---

### Problem Statement

Given two strings `word1` and `word2`, return the **minimum number of operations** required to convert `word1` to `word2`.

You have the following three operations permitted on a word:

- **Insert** a character
- **Delete** a character
- **Replace** a character

---

### Input & Output Formats & Constraints

- **Input:**
  - `word1: str` — Source string of length $m$.
  - `word2: str` — Target string of length $n$.
- **Output:**
  - `int` — Minimum number of edit operations (Levenshtein distance).
- **Constraints:**
  - $0 \le \text{word1.length}, \text{word2.length} \le 500$
  - `word1` and `word2` consist of lowercase English letters.

---

### Key Idea & Intuition

1. **Prefix Alignment Subproblems (Wagner-Fischer Algorithm):**
   - Let $\text{dp}[i][j]$ denote the minimum edit distance to transform the prefix $\text{word1}[0 \dots i-1]$ into the prefix $\text{word2}[0 \dots j-1]$.
   - Dimensions: $(m + 1) \times (n + 1)$.

2. **Base Cases:**
   - Transforming an empty string into $\text{word2}[0 \dots j-1]$ requires inserting all $j$ characters:
     $$\text{dp}[0][j] = j \quad (\forall 0 \le j \le n)$$
   - Transforming $\text{word1}[0 \dots i-1]$ into an empty string requires deleting all $i$ characters:
     $$\text{dp}[i][0] = i \quad (\forall 0 \le i \le m)$$

3. **State Transitions:**
   - When examining characters $\text{word1}[i-1]$ and $\text{word2}[j-1]$:
     - **Case 1: Characters Match ($\text{word1}[i-1] == \text{word2}[j-1]$):**
       - No operation is required; the edit cost is inherited directly from the diagonal:
         $$\text{dp}[i][j] = \text{dp}[i-1][j-1]$$
     - **Case 2: Characters Mismatch ($\text{word1}[i-1] \ne \text{word2}[j-1]$):**
       - We consider all three allowable operations and take the minimum:
         1. **Insert:** Match $\text{word2}[j-1]$ by inserting it into `word1`. The cost is $1 + \text{dp}[i][j-1]$.
         2. **Delete:** Delete $\text{word1}[i-1]$. The cost is $1 + \text{dp}[i-1][j]$.
         3. **Replace:** Replace $\text{word1}[i-1]$ with $\text{word2}[j-1]$. The cost is $1 + \text{dp}[i-1][j-1]$.
       - Transition:
         $$\text{dp}[i][j] = 1 + \min(\text{dp}[i][j-1], \ \text{dp}[i-1][j], \ \text{dp}[i-1][j-1])$$

4. **1D Space Optimization:**
   - To compute row $i$, we only ever need values from the current row and the previous row ($i - 1$).
   - We can compress the table into a 1D array of size $n + 1$ by caching the previous diagonal value $\text{dp}[i-1][j-1]$ in a temporary variable, reducing space from $\mathcal{O}(m \times n)$ to $\mathcal{O}(n)$.

---

### Solution Approach (Step-by-Step)

1. **Table Allocation:**
   - Create 1D array `dp` of size $n + 1$ where `dp[j] = j` for $j \in [0, n]$.
2. **Iterate Rows ($i = 1 \dots m$):**
   - Save the old `dp[0]` as `prev_diag = i - 1`.
   - Update `dp[0] = i` (representing $i$ deletions).
   - For $j$ from $1$ to $n$:
     - Store current `dp[j]` before overwrite: `temp = dp[j]`.
     - If $\text{word1}[i-1] == \text{word2}[j-1]$:
       - `dp[j] = prev_diag`.
     - Else:
       - `dp[j] = 1 + min(dp[j - 1], dp[j], prev_diag)`.
     - `prev_diag = temp`.
3. **Return:**
   - Return `dp[n]`.

---

### Visual Algorithm Walkthrough

For `word1 = "horse"` and `word2 = "ros"`:

```
        ""   r   o   s
""    [  0,  1,  2,  3 ]
h     [  1,  1,  2,  3 ]
o     [  2,  2,  1,  2 ]
r     [  3,  2,  2,  2 ]
s     [  4,  3,  3,  2 ]
e     [  5,  4,  4,  3 ]

Cell Tracing:
- (0, 0) = 0
- 'h' vs 'r': mismatch -> 1 + min(0, 1, 1) = 1 (Replace 'h' with 'r')
- 'o' vs 'o': match -> inherits diagonal (1)
- 'r' vs 'r': match -> inherits diagonal (2)
- 'e' vs 's': mismatch -> 1 + min(left=4, top=2, diag=3) = 1 + 2 = 3

Optimal Sequence of 3 Operations:
1. horse -> rorse (replace 'h' with 'r')
2. rorse -> rose  (remove 'r')
3. rose  -> ros   (remove 'e')
Minimum operations = 3.
```

---

### Solved Examples with Multiple Inputs

| Case | `word1` | `word2` | Edit Distance | Operations Applied |
|---|---|---|---|---|
| **Standard** | `"horse"` | `"ros"` | `3` | Replace 'h' $\to$ 'r', delete 'r', delete 'e' |
| **Multi-Step** | `"intention"` | `"execution"` | `5` | Replace 'i' $\to$ 'e', replace 'n' $\to$ 'x', replace 't' $\to$ 'e', insert 'u', replace 'n' $\to$ 'n' |
| **Identical Strings** | `"apple"` | `"apple"` | `0` | No operations required |
| **One Empty** | `""` | `"abc"` | `3` | Insert 3 characters |
| **Single Character Change** | `"cat"` | `"car"` | `1` | Replace 't' with 'r' |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — 1D Space Optimized)
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        
        # Ensure word2 is the shorter string to minimize space
        if m < n:
            return self.minDistance(word2, word1)
            
        dp = list(range(n + 1))
        
        for i in range(1, m + 1):
            prev_diag = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if word1[i - 1] == word2[j - 1]:
                    dp[j] = prev_diag
                else:
                    dp[j] = 1 + min(dp[j - 1], dp[j], prev_diag)
                prev_diag = temp
                
        return dp[n]
```

#### 2. C++ (C++17 / STL — 1D Space Optimized)
```cpp
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int minDistance(std::string word1, std::string word2) {
        int m = word1.size();
        int n = word2.size();

        if (m < n) return minDistance(word2, word1);

        std::vector<int> dp(n + 1);
        std::iota(dp.begin(), dp.end(), 0);

        for (int i = 1; i <= m; ++i) {
            int prev_diag = dp[0];
            dp[0] = i;

            for (int j = 1; j <= n; ++j) {
                int temp = dp[j];
                if (word1[i - 1] == word2[j - 1]) {
                    dp[j] = prev_diag;
                } else {
                    dp[j] = 1 + std::min({dp[j - 1], dp[j], prev_diag});
                }
                prev_diag = temp;
            }
        }

        return dp[n];
    }
};
```

#### 3. Java (Modern, Typed — 1D Space Optimized)
```java
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();

        if (m < n) return minDistance(word2, word1);

        int[] dp = new int[n + 1];
        for (int j = 0; j <= n; j++) {
            dp[j] = j;
        }

        for (int i = 1; i <= m; i++) {
            int prevDiag = dp[0];
            dp[0] = i;

            for (int j = 1; j <= n; j++) {
                int temp = dp[j];
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[j] = prevDiag;
                } else {
                    dp[j] = 1 + Math.min(prevDiag, Math.min(dp[j - 1], dp[j]));
                }
                prevDiag = temp;
            }
        }

        return dp[n];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$  
  The nested loops iterate $(m \times n)$ times. Each cell performs $\mathcal{O}(1)$ basic comparisons. For $m, n \le 500$, total operations $\le 2.5 \times 10^5$, executing in $< 5$ ms.
- **Space Complexity:** $\mathcal{O}(\min(m, n))$  
  By swapping arguments so that $n = \min(m, n)$, the 1D rolling array requires only $\min(m, n) + 1 \le 501$ integers ($\approx 2$ KB).

---

### Takeaway Pattern & Interview Traps

1. **Mapping Neighbors to Operations:**
   - $\text{dp}[i - 1][j]$: **Delete** from `word1` (vertical step).
   - $\text{dp}[i][j - 1]$: **Insert** into `word1` (horizontal step).
   - $\text{dp}[i - 1][j - 1]$: **Replace** in `word1` (diagonal step).
2. **Space Optimization `prev_diag` Caching:**
   - In 1D DP, `dp[j]` before update represents $\text{dp}[i - 1][j]$, and `dp[j - 1]` after update represents $\text{dp}[i][j - 1]$. The diagonal $\text{dp}[i - 1][j - 1]$ must be cached into `prev_diag` before `dp[j - 1]` overwrites it.