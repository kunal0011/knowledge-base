---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 32: Longest Valid Parentheses"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - stack
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 32: Longest Valid Parentheses

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg, ByteDance  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / String / Stack  

---

### Problem Statement

Given a string containing just the characters `'('` and `')'`, return *the length of the longest valid (well-formed) parentheses substring*.

---

### Input & Output Formats & Constraints

- **Input:** A string `s` ($0 \le |s| \le 3 \times 10^4$) consisting of `'('` and `')'`.
- **Output:** An integer representing the length of the longest valid parentheses substring.
- **Constraints:**
  - `0 <= s.length <= 3 * 10^4`
  - `s[i]` is `'('` or `')'`.

---

### Key Idea & Intuition

#### Prefix/Ending-Position DP Invariant
Any valid parentheses substring must end with a closing bracket `')'`. An opening bracket `'('` can never be the final character of a valid substring.
Therefore, we define:
$$dp[i] = \text{length of the longest valid parentheses substring ending at index } i$$

If $s[i] == \text{'('}$, then $dp[i] = 0$.
When $s[i] == \text{')'}$, there are two structural cases depending on the preceding character $s[i - 1]$:

#### Case 1: Direct Pair `...()` ($s[i - 1] == \text{'('}$)
The two characters $s[i-1]$ and $s[i]$ form an immediate valid pair `"()"`.
Its length of $2$ can be concatenated with whatever valid parentheses substring ended immediately prior to index $i - 1$:
$$dp[i] = 2 + (dp[i - 2] \text{ if } i \ge 2 \text{ else } 0)$$

#### Case 2: Nested/Chained Substring `...))` ($s[i - 1] == \text{')'}$)
If $s[i - 1]$ was also a `')'`, it may have ended a valid substring of length $dp[i - 1]$.
The character that could potentially match $s[i]$ must lie immediately before that entire valid block:
$$\text{matching index } j = i - dp[i - 1] - 1$$
If $j \ge 0$ and $s[j] == \text{'('}$, then:
1. $s[j]$ and $s[i]$ form an outer envelope around the valid block $s[j+1..i-1]$ of length $dp[i - 1] + 2$.
2. This newly formed valid block can further concatenate with any valid substring ending immediately before index $j$ (at $j - 1$):
   $$dp[i] = dp[i - 1] + 2 + (dp[j - 1] \text{ if } j \ge 1 \text{ else } 0)$$

---

### Solution Approach (Step-by-Step)

1. **Base Check:**
   - If $|s| < 2$, return `0`.
2. **Initialize DP Array:**
   - Create array `dp` of length $|s|$ initialized with 0.
   - `max_len = 0`.
3. **Linear Sweep:**
   - For $i$ from 1 to $|s| - 1$:
     - If $s[i] == \text{')'}$:
       - If $s[i - 1] == \text{'('}$:
         - $dp[i] = 2 + (dp[i - 2] \text{ if } i \ge 2 \text{ else } 0)$
       - Else ($s[i - 1] == \text{')'}$):
         - $j = i - dp[i - 1] - 1$
         - If $j \ge 0$ and $s[j] == \text{'('}$:
           - $dp[i] = dp[i - 1] + 2 + (dp[j - 1] \text{ if } j \ge 1 \text{ else } 0)$
       - $max\_len = \max(max\_len, dp[i])$
4. **Return:**
   - Return `max_len`.

---

### Visual Algorithm Walkthrough

#### Trace for `s = ")()())"`
```
Indices:   0    1    2    3    4    5
Chars:     )    (    )    (    )    )
Initial: dp = [0, 0, 0, 0, 0, 0]

i = 1 ('('): dp[1] = 0
i = 2 (')'):
  s[1] == '(' -> Case 1: dp[2] = 2 + dp[0] = 2 + 0 = 2. max_len = 2

i = 3 ('('): dp[3] = 0
i = 4 (')'):
  s[3] == '(' -> Case 1: dp[4] = 2 + dp[2] = 2 + 2 = 4. max_len = 4
  (Valid substring from index 1 to 4: "()()")

i = 5 (')'):
  s[4] == ')' -> Case 2:
  j = 5 - dp[4] - 1 = 5 - 4 - 1 = 0
  s[0] is ')' != '(' -> Cannot match! dp[5] = 0

Final dp: [0, 0, 2, 0, 4, 0]
Result: max_len = 4 ("()()").
```

---

### Solved Examples with Multiple Inputs

| `s` | Substring Match Structure | DP Values at `')'` | Output |
|---|---|---|---|
| `")()())"` | `"()" + "()"` | $dp[2]=2, dp[4]=4$ | `4` |
| `"(()"` | Enclosed pair | $dp[2]=2$ | `2` |
| `""` | Empty string | No evaluation | `0` |
| `"()(()"` | `"()"` followed by incomplete | $dp[1]=2, dp[4]=2$ | `2` |
| `"((()))"` | Deeply nested | $dp[3]=2, dp[4]=4, dp[5]=6$ | `6` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n: int = len(s)
        if n < 2:
            return 0
            
        dp: list[int] = [0] * n
        max_len: int = 0
        
        for i in range(1, n):
            if s[i] == ')':
                # Case 1: Immediate pair "()"
                if s[i - 1] == '(':
                    dp[i] = 2 + (dp[i - 2] if i >= 2 else 0)
                # Case 2: Nested/chained pattern "))"
                else:
                    j = i - dp[i - 1] - 1
                    if j >= 0 and s[j] == '(':
                        dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j >= 1 else 0)
                        
                max_len = max(max_len, dp[i])
                
        return max_len
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int longestValidParentheses(const std::string& s) {
        int n = static_cast<int>(s.size());
        if (n < 2) return 0;

        std::vector<int> dp(n, 0);
        int max_len = 0;

        for (int i = 1; i < n; ++i) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = 2 + (i >= 2 ? dp[i - 2] : 0);
                } else {
                    int j = i - dp[i - 1] - 1;
                    if (j >= 0 && s[j] == '(') {
                        dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);
                    }
                }
                max_len = std::max(max_len, dp[i]);
            }
        }

        return max_len;
    }
};
```

#### Java 17
```java
class Solution {
    public int longestValidParentheses(String s) {
        int n = s.length();
        if (n < 2) {
            return 0;
        }

        int[] dp = new int[n];
        int maxLen = 0;

        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = 2 + (i >= 2 ? dp[i - 2] : 0);
                } else {
                    int j = i - dp[i - 1] - 1;
                    if (j >= 0 && s.charAt(j) == '(') {
                        dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);
                    }
                }
                maxLen = Math.max(maxLen, dp[i]);
            }
        }

        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |s|$. We iterate through the string in a single linear pass with constant-time lookup and boundary updates.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the 1D DP table.

---

### Takeaway Pattern & Interview Traps

1. **Two-Pass $\mathcal{O}(1)$ Space Alternative:**
   - Scan left-to-right counting `left` and `right`. If `right == left`, record `2 * left`. If `right > left`, reset both to 0.
   - Scan right-to-left with the symmetric rule to handle excess left brackets (e.g. `"(()"`). This achieves $\mathcal{O}(N)$ time and strictly $\mathcal{O}(1)$ space!
2. **Connecting with $dp[j - 1]$:** When resolving Case 2 ($s[j] == \text{'('}$), remember to add $dp[j - 1]$. Forgetting $dp[j - 1]$ produces incorrect results for expressions like `"()(())"` because it fails to connect the newly resolved envelope with the preceding valid component.