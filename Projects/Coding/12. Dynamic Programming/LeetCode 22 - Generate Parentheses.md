---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 22: Generate Parentheses"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - backtracking
  - string
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 22: Generate Parentheses

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Adobe  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Backtracking / String  

---

### Problem Statement

Given `n` pairs of parentheses, write a function to *generate all combinations of well-formed parentheses*.

---

### Input & Output Formats & Constraints

- **Input:** An integer `n` ($1 \le n \le 8$).
- **Output:** A list of strings `List[str]` containing all valid combinations of $n$ pairs of parentheses.
- **Constraints:**
  - `1 <= n <= 8`

---

### Key Idea & Intuition

#### 1. The Catalan Number Structure
The total number of valid parentheses expressions with $n$ pairs is given by the $n^{\text{th}}$ Catalan number:
$$C_n = \frac{1}{n + 1} \binom{2n}{n}$$
For $n = 1, 2, 3, 4, \dots$, the counts are $1, 2, 5, 14, 42, 132, 429, 1430$.

#### 2. Structural Dynamic Programming Formulation
Every valid non-empty parenthesis string begins with an opening bracket `'('`. That initial bracket must be closed by some specific matching `')'`.
Thus, any valid expression of $n$ pairs can be uniquely decomposed as:
$$\text{Expression} = \text{"("} + \text{inside} + \text{")"} + \text{outside}$$
where:
- `inside` is a valid parenthesis string containing $k$ pairs ($0 \le k < n$).
- `outside` is a valid parenthesis string containing the remaining $n - 1 - k$ pairs.

Because every valid string has a unique position for the matching closing parenthesis of the very first character, this decomposition generates every valid combination **exactly once** without generating duplicates!
- Base Case: $dp[0] = [""\text{ }]$ (an empty string for 0 pairs).
- Transition for $i \in [1, n]$:
  $$dp[i] = \bigcup_{k=0}^{i-1} \Big\{ \text{"("} + l + \text{")"} + r \;\Big|\; l \in dp[k], \, r \in dp[i - 1 - k] \Big\}$$

---

### Solution Approach (Step-by-Step)

1. **Initialize DP Table:**
   - Create list of lists `dp` of length $n + 1$.
   - Base case: `dp[0] = [""]`.
2. **Bottom-Up Construction:**
   - For total pairs $i$ from 1 to $n$:
     - For $k$ from 0 to $i - 1$:
       - For each string `left` in $dp[k]$:
         - For each string `right` in $dp[i - 1 - k]$:
           - Append `"(" + left + ")" + right` to $dp[i]$.
3. **Return:**
   - Return $dp[n]$.

---

### Visual Algorithm Walkthrough

#### Trace for $n = 3$
```
dp[0] = [""]
dp[1] = ["()"]

Compute dp[2] (i = 2):
  k = 0: "(" + dp[0] + ")" + dp[1] -> "()" + "()" = "()()"
  k = 1: "(" + dp[1] + ")" + dp[0] -> "(())" + "" = "(())"
  dp[2] = ["()()", "(())"]

Compute dp[3] (i = 3):
  k = 0: "(" + dp[0] + ")" + dp[2]
         "(" + "" + ")" + "()()" -> "()()()"
         "(" + "" + ")" + "(())" -> "()(())"
  k = 1: "(" + dp[1] + ")" + dp[1]
         "(" + "()" + ")" + "()" -> "(())()"
  k = 2: "(" + dp[2] + ")" + dp[0]
         "(" + "()()" + ")" + "" -> "(()())"
         "(" + "(())" + ")" + "" -> "((()))"

Final Result dp[3]:
["()()()", "()(())", "(())()", "(()())", "((()))"] (All 5 valid Catalan combinations).
```

---

### Solved Examples with Multiple Inputs

| $n$ | Catalan Count $C_n$ | Generated Combinations |
|---|---|---|
| `1` | 1 | `["()"]` |
| `2` | 2 | `["()()", "(())"]` |
| `3` | 5 | `["()()()", "()(())", "(())()", "(()())", "((()))"]` |
| `4` | 14 | 14 valid permutations of length 8 |

---

### Multi-Language Implementations

#### Python 3 (Dynamic Programming)
```python
class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        # dp[i] stores all valid parentheses strings of length i pairs
        dp: list[list[str]] = [[] for _ in range(n + 1)]
        dp[0] = [""]
        
        for i in range(1, n + 1):
            for k in range(i):
                for left in dp[k]:
                    for right in dp[i - 1 - k]:
                        dp[i].append(f"({left}){right}")
                        
        return dp[n]
```

#### C++17 (Catalan DP)
```cpp
#include <vector>
#include <string>

class Solution {
public:
    std::vector<std::string> generateParenthesis(int n) {
        std::vector<std::vector<std::string>> dp(n + 1);
        dp[0] = {""};

        for (int i = 1; i <= n; ++i) {
            for (int k = 0; k < i; ++k) {
                for (const std::string& left : dp[k]) {
                    for (const std::string& right : dp[i - 1 - k]) {
                        dp[i].push_back("(" + left + ")" + right);
                    }
                }
            }
        }

        return dp[n];
    }
};
```

#### Java 17 (Catalan DP)
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<String> generateParenthesis(int n) {
        List<List<String>> dp = new ArrayList<>();
        for (int i = 0; i <= n; i++) {
            dp.add(new ArrayList<>());
        }
        dp.get(0).add("");

        for (int i = 1; i <= n; i++) {
            for (int k = 0; k < i; k++) {
                for (String left : dp.get(k)) {
                    for (String right : dp.get(i - 1 - k)) {
                        dp.get(i).add("(" + left + ")" + right);
                    }
                }
            }
        }

        return dp.get(n);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}\left(\frac{4^n}{\sqrt{n}}\right) = \mathcal{O}(C_n \cdot n)$, where $C_n = \frac{1}{n+1}\binom{2n}{n}$ is the $n^{\text{th}}$ Catalan number. For $n = 8$, $C_8 = 1430$, taking $< 2$ milliseconds.
- **Space Complexity:** $\mathcal{O}\left(\frac{4^n}{\sqrt{n}}\right)$ to store the generated strings in the DP table.

---

### Takeaway Pattern & Interview Traps

1. **Backtracking Alternative:** In interviews, backtracking tracking `open` and `close` counts is also popular. Demonstrating the DP Catalan decomposition `"(" + dp[k] + ")" + dp[i - 1 - k]` showcases deep structural knowledge of grammar generation.
2. **Guaranteed Uniqueness:** Because $k$ uniquely fixes the position of the closing parenthesis for the very first opening parenthesis, collisions between iterations cannot occur. No hash set is required for deduplication.