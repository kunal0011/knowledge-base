---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 22: Generate Parentheses"
tags:
  - leetcode
  - coding
  - backtracking
  - dynamic-programming
  - string
  - amazon
  - google
---

# LeetCode 22: Generate Parentheses

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Catalan Numbers / Prefix Invariants  

---

### Problem Statement

Given `n` pairs of parentheses, write a function to *generate all combinations of well-formed parentheses*.

---

### Input & Output Formats & Constraints

- **Input:** `n: int` (number of pairs)
- **Output:** `List[str]` containing all valid parenthesized expressions.
- **Constraints:**
  - $1 \le n \le 8$

---

### Key Idea & Intuition

- **The Dyck Path / Parentheses Prefix Invariant:**
  - A string of length $2n$ consisting of $n$ `'('` and $n$ `')'` is valid if and only if **in every prefix**, the number of opening brackets is greater than or equal to the number of closing brackets:
    $$\text{count}('(') \ge \text{count}(')') \quad \text{for all prefixes}$$
    and at the end:
    $$\text{count}('(') == \text{count}(')') == n$$
- **Branching Decision Rules:**
  - At any step in the recursion, we maintain `open_count` and `close_count`:
    1. **Can we add `'('`?** Yes, if `open_count < n`.
    2. **Can we add `')'`?** Yes, if `close_count < open_count`.
  - By following these two rules strictly, **every path explored in the recursion tree leads to a valid expression**. No invalid strings are ever formed, and no post-generation validation or deduplication is required.
- **Number of Valid Combinations (Catalan Number):**
  - The number of valid combinations of $n$ pairs of parentheses is given by the $n$-th Catalan number:
    $$C_n = \frac{1}{n + 1} \binom{2n}{n} = \frac{(2n)!}{(n + 1)! \, n!}$$
  - For $n = 3$, $C_3 = 5$. For $n = 8$, $C_8 = 1430$.

---

### Solution Approach (Step-by-Step)

1. Initialize `results = []` and a character buffer `current = []`.
2. Define recursive function `backtrack(open_count, close_count)`:
   - **Base Case:** If `len(current) == 2 * n`:
     - Append `"".join(current)` to `results`.
     - Return.
   - **Choice 1 (Add Open):** If `open_count < n`:
     - `current.append('(')`
     - `backtrack(open_count + 1, close_count)`
     - `current.pop()` (backtrack)
   - **Choice 2 (Add Close):** If `close_count < open_count`:
     - `current.append(')')`
     - `backtrack(open_count, close_count + 1)`
     - `current.pop()` (backtrack)
3. Call `backtrack(0, 0)` and return `results`.

---

### Visual Algorithm Walkthrough

For $n = 3$:

```
                                  backtrack(0, 0, "")
                                           |
                                  backtrack(1, 0, "(")
                                /                     \
                     backtrack(2, 0, "((")          backtrack(1, 1, "()")
                     /                  \                     |
           backtrack(3, 0, "(((")   backtrack(2, 1, "(()") backtrack(2, 1, "()(")
                 |                   /             \             /            \
           backtrack(3, 1)     backtrack(3, 1) backtrack(2,2) backtrack(3,1) backtrack(2,2)
                 |                   |              |            |             |
           "((()))"               "(()())"       "(())()"     "()(())"      "()()()"
```

Notice that invalid prefixes such as `")"` or `"())"` are never generated because `close_count < open_count` prohibits them at the root.

---

### Solved Examples with Multiple Inputs

| Test Case | `n` | Catalan Count $C_n$ | Generated Combinations | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `1` | $C_1 = 1$ | `"()"` | `["()"]` |
| **Example 2** | `2` | $C_2 = 2$ | `"(())"`, `"()()"` | `["(())", "()()"]` |
| **Example 3** | `3` | $C_3 = 5$ | `"((()))"`, `"(()())"`, `"(())()"`, `"()(())"`, `"()()()"` | `["((()))","(()())","(())()","()(())","()()()"]` |
| **Example 4** | `4` | $C_4 = 14$ | 14 valid expressions | 14 items |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        """
        Generates all valid combinations of n pairs of parentheses.
        Prunes invalid prefixes at construction time using open and close counters.
        """
        results: List[str] = []
        path: List[str] = []

        def backtrack(open_count: int, close_count: int) -> None:
            if len(path) == 2 * n:
                results.append("".join(path))
                return

            if open_count < n:
                path.append('(')
                backtrack(open_count + 1, close_count)
                path.pop()  # Backtrack

            if close_count < open_count:
                path.append(')')
                backtrack(open_count, close_count + 1)
                path.pop()  # Backtrack

        backtrack(0, 0)
        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::vector<std::string> generateParenthesis(int n) {
        std::vector<std::string> results;
        std::string path;
        path.reserve(2 * n);
        backtrack(0, 0, n, path, results);
        return results;
    }

private:
    void backtrack(int open_count, int close_count, int n,
                   std::string& path, std::vector<std::string>& results) {
        if (static_cast<int>(path.size()) == 2 * n) {
            results.push_back(path);
            return;
        }

        if (open_count < n) {
            path.push_back('(');
            backtrack(open_count + 1, close_count, n, path, results);
            path.pop_back(); // Backtrack
        }

        if (close_count < open_count) {
            path.push_back(')');
            backtrack(open_count, close_count + 1, n, path, results);
            path.pop_back(); // Backtrack
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> results = new ArrayList<>();
        StringBuilder path = new StringBuilder(2 * n);
        backtrack(0, 0, n, path, results);
        return results;
    }

    private void backtrack(int openCount, int closeCount, int n, 
                          StringBuilder path, List<String> results) {
        if (path.length() == 2 * n) {
            results.add(path.toString());
            return;
        }

        if (openCount < n) {
            path.append('(');
            backtrack(openCount + 1, closeCount, n, path, results);
            path.deleteCharAt(path.length() - 1); // Backtrack
        }

        if (closeCount < openCount) {
            path.append(')');
            backtrack(openCount, closeCount + 1, n, path, results);
            path.deleteCharAt(path.length() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}\left(\frac{4^n}{\sqrt{n}}\right) = \mathcal{O}(C_n \cdot 2n)$.
  - The number of valid paths equals the $n$-th Catalan number $C_n = \frac{1}{n+1}\binom{2n}{n} \approx \frac{4^n}{n\sqrt{\pi n}}$.
  - Each valid combination of length $2n$ is generated and copied in $\mathcal{O}(n)$ time.
  - For $n = 8$, $C_8 = 1430$, yielding $1430 \times 16 \approx 2.3 \times 10^4$ operations ($\ll 1 \text{ ms}$).
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space for the recursion stack and character buffer of length $2n$.

---

### Takeaway Pattern & Interview Traps

- **Rule of Constructive Backtracking:** Never generate all $2^{2n}$ candidate binary strings and then test for validity with a stack. Instead, enforce the mathematical validity invariants (`open < n` and `close < open`) at each step to prune invalid branches before they are ever created.
- **Catalan Number Recognition:** Whenever a problem asks for balanced parentheses, non-intersecting chords, mountain ranges, or binary search tree topologies of size $n$, the state space is strictly governed by the Catalan number $C_n$.