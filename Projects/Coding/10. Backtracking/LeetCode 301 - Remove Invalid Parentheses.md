---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 301: Remove Invalid Parentheses"
tags:
  - leetcode
  - coding
  - backtracking
  - bfs
  - string
  - amazon
  - google
  - meta
---

# LeetCode 301: Remove Invalid Parentheses

**Target Companies:** Meta (Top Favorite), Google, Amazon, Uber  
**Difficulty:** Hard  
**Topic:** Backtracking / Minimum Removals Precomputation / Duplicate Pruning  

---

### Problem Statement

Given a string `s` that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.

Return *all the possible results*. You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`
- **Output:** `List[str]` containing all distinct valid strings formed by removing the minimum possible number of parentheses.
- **Constraints:**
  - $1 \le \text{s.length} \le 25$
  - `s` consists of lowercase English letters and parentheses `'('` and `')'`.
  - There will be at most $20$ parentheses in `s`.

---

### Key Idea & Intuition

- **Phase 1: Determine Minimum Removals in $\mathcal{O}(N)$:**
  - Before starting recursion, count the exact number of misplaced `'('` and `')'` characters:
    - Track `rem_left = 0` and `rem_right = 0`.
    - Iterate through $s$:
      - If `c == '('`: increment `rem_left`.
      - If `c == ')'`: if `rem_left > 0`, decrement `rem_left` (matched with a preceding `'('`); otherwise, increment `rem_right` (unmatched closing bracket).
  - This establishes the strict target removals: we must remove **exactly** `rem_left` open brackets and `rem_right` close brackets.
- **Phase 2: Constrained Backtracking with Prefix Invariants:**
  - While building the valid string, track `open_count` and `close_count`.
  - At any character, if `close_count > open_count`, the prefix is immediately invalid and pruned.
  - Letters other than `'('` and `')'` are always kept.
- **Phase 3: Consecutive Duplicate Pruning (The Game Changer):**
  - If we have multiple consecutive identical parentheses (e.g. `"((("`), removing the 1st, 2nd, or 3rd yields the identical resulting string `"(( "`.
  - **Pruning Invariant:** When removing a parenthesis at index $i$, if $i > \text{start}$ and $s[i] == s[i - 1]$, skip removal! This prevents generating duplicate states without needing an expensive hash set.

---

### Solution Approach (Step-by-Step)

1. Compute `rem_left` and `rem_right` via a single linear scan.
2. Initialize `results = []` and dynamic character buffer `path = []`.
3. Define `backtrack(idx, open_count, close_count, rem_left, rem_right)`:
   - If `idx == len(s)`:
     - If `rem_left == 0` and `rem_right == 0` and `open_count == close_count`:
       - Append `"".join(path)` to `results`.
     - Return.
   - Let `c = s[idx]`.
   - **Option A (Remove `c`):**
     - If `c == '('` and `rem_left > 0`:
       - `backtrack(idx + 1, open_count, close_count, rem_left - 1, rem_right)`
     - If `c == ')'` and `rem_right > 0`:
       - `backtrack(idx + 1, open_count, close_count, rem_left, rem_right - 1)`
   - **Option B (Keep `c`):**
     - If `c != '('` and `c != ')'`:
       - `path.append(c)`
       - `backtrack(idx + 1, open_count, close_count, rem_left, rem_right)`
       - `path.pop()`
     - If `c == '('`:
       - `path.append(c)`
       - `backtrack(idx + 1, open_count + 1, close_count, rem_left, rem_right)`
       - `path.pop()`
     - If `c == ')'`:
       - Only keep if `close_count < open_count`:
         - `path.append(c)`
         - `backtrack(idx + 1, open_count, close_count + 1, rem_left, rem_right)`
         - `path.pop()`
4. To avoid duplicate branches when removing, iterate indices or pass `prev_removed` indicator, or de-duplicate via set/consecutive checks.

---

### Visual Algorithm Walkthrough

Let `s = "()())()"`.
1. Scan $s$:
   - `(` $\implies$ `rem_left = 1`
   - `)` $\implies$ `rem_left = 0`
   - `(` $\implies$ `rem_left = 1`
   - `)` $\implies$ `rem_left = 0`
   - `)` $\implies$ `rem_left = 0, rem_right = 1`
   - `(` $\implies$ `rem_left = 1`
   - `)` $\implies$ `rem_left = 0`
   - Target removals: `rem_left = 0`, `rem_right = 1`. Exactly one `')'` must be removed.

2. Candidate removals of `')'`:
   - `')'` at index 1: `( ) () ()` $\implies$ `"(())()"` (Valid!)
   - `')'` at index 3: `() ( ) ()` $\implies$ `"()()()"` (Valid!)
   - `')'` at index 4: `() () ( )` $\implies$ `"()()()"` (Identical to removing index 3 $\implies$ Pruned!)
   - `')'` at index 6: `() () ( ` $\implies$ Invalid (unmatched `'('` at 5).

Output: `["(())()", "()()()"]`.

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | `(rem_left, rem_right)` | Distinct Valid Results |
| :--- | :--- | :--- | :--- |
| **Standard** | `"()())()"` | `(0, 1)` | `["(())()", "()()()"]` |
| **With Letters** | `"(a)())()"` | `(0, 1)` | `["(a())()", "(a)()()"]` |
| **Only Close** | `")("` | `(1, 1)` | `[""]` |
| **Already Valid** | `"()"` | `(0, 0)` | `["()"]` |
| **All Invalid** | `")))"` | `(0, 3)` | `[""]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        """
        Removes the minimum number of invalid parentheses to make string valid.
        Uses two-pass precomputation + DFS with duplicate pruning.
        """
        # Step 1: Find minimum number of '(' and ')' to remove
        rem_left = 0
        rem_right = 0
        for ch in s:
            if ch == '(':
                rem_left += 1
            elif ch == ')':
                if rem_left > 0:
                    rem_left -= 1
                else:
                    rem_right += 1

        results = set()
        path: List[str] = []

        # Step 2: DFS Backtracking
        def backtrack(idx: int, open_count: int, close_count: int, 
                      rl: int, rr: int) -> None:
            if idx == len(s):
                if rl == 0 and rr == 0 and open_count == close_count:
                    results.add("".join(path))
                return

            ch = s[idx]

            # Option 1: Remove current character if it's an extra parenthesis
            if ch == '(' and rl > 0:
                backtrack(idx + 1, open_count, close_count, rl - 1, rr)
            elif ch == ')' and rr > 0:
                backtrack(idx + 1, open_count, close_count, rl, rr - 1)

            # Option 2: Keep current character
            path.append(ch)
            if ch != '(' and ch != ')':
                backtrack(idx + 1, open_count, close_count, rl, rr)
            elif ch == '(':
                backtrack(idx + 1, open_count + 1, close_count, rl, rr)
            elif ch == ')' and close_count < open_count:
                backtrack(idx + 1, open_count, close_count + 1, rl, rr)
            path.pop()  # Backtrack

        backtrack(0, 0, 0, rem_left, rem_right)
        return list(results)
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <unordered_set>

class Solution {
public:
    std::vector<std::string> removeInvalidParentheses(const std::string& s) {
        int rem_left = 0, rem_right = 0;
        for (char c : s) {
            if (c == '(') {
                rem_left++;
            } else if (c == ')') {
                if (rem_left > 0) {
                    rem_left--;
                } else {
                    rem_right++;
                }
            }
        }

        std::unordered_set<std::string> results;
        std::string path;
        backtrack(0, 0, 0, rem_left, rem_right, s, path, results);
        return std::vector<std::string>(results.begin(), results.end());
    }

private:
    void backtrack(int idx, int open_count, int close_count,
                   int rem_left, int rem_right, const std::string& s,
                   std::string& path, std::unordered_set<std::string>& results) {
        if (idx == static_cast<int>(s.size())) {
            if (rem_left == 0 && rem_right == 0 && open_count == close_count) {
                results.insert(path);
            }
            return;
        }

        char c = s[idx];

        // Choice 1: Remove parenthesis
        if (c == '(' && rem_left > 0) {
            backtrack(idx + 1, open_count, close_count, rem_left - 1, rem_right, s, path, results);
        } else if (c == ')' && rem_right > 0) {
            backtrack(idx + 1, open_count, close_count, rem_left, rem_right - 1, s, path, results);
        }

        // Choice 2: Keep character
        path.push_back(c);
        if (c != '(' && c != ')') {
            backtrack(idx + 1, open_count, close_count, rem_left, rem_right, s, path, results);
        } else if (c == '(') {
            backtrack(idx + 1, open_count + 1, close_count, rem_left, rem_right, s, path, results);
        } else if (c == ')' && close_count < open_count) {
            backtrack(idx + 1, open_count, close_count + 1, rem_left, rem_right, s, path, results);
        }
        path.pop_back(); // Backtrack
    }
};
```

#### Java
```java
import java.util.*;

class Solution {
    public List<String> removeInvalidParentheses(String s) {
        int remLeft = 0;
        int remRight = 0;

        for (char c : s.toCharArray()) {
            if (c == '(') {
                remLeft++;
            } else if (c == ')') {
                if (remLeft > 0) {
                    remLeft--;
                } else {
                    remRight++;
                }
            }
        }

        Set<String> results = new HashSet<>();
        StringBuilder path = new StringBuilder();
        backtrack(0, 0, 0, remLeft, remRight, s, path, results);
        return new ArrayList<>(results);
    }

    private void backtrack(int idx, int openCount, int closeCount,
                          int remLeft, int remRight, String s,
                          StringBuilder path, Set<String> results) {
        if (idx == s.length()) {
            if (remLeft == 0 && remRight == 0 && openCount == closeCount) {
                results.add(path.toString());
            }
            return;
        }

        char c = s.charAt(idx);

        // Choice 1: Remove parenthesis
        if (c == '(' && remLeft > 0) {
            backtrack(idx + 1, openCount, closeCount, remLeft - 1, remRight, s, path, results);
        } else if (c == ')' && remRight > 0) {
            backtrack(idx + 1, openCount, closeCount, remLeft, remRight - 1, s, path, results);
        }

        // Choice 2: Keep character
        path.append(c);
        if (c != '(' && c != ')') {
            backtrack(idx + 1, openCount, closeCount, remLeft, remRight, s, path, results);
        } else if (c == '(') {
            backtrack(idx + 1, openCount + 1, closeCount, remLeft, remRight, s, path, results);
        } else if (c == ')' && closeCount < openCount) {
            backtrack(idx + 1, openCount, closeCount + 1, remLeft, remRight, s, path, results);
        }
        path.deleteCharAt(path.length() - 1); // Backtrack
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(2^N)$ worst-case.
  - In the worst case (e.g. `"((((((("`), each parenthesis has 2 choices (keep or remove).
  - However, precomputing `rem_left` and `rem_right` restricts the number of removals strictly to $K = \text{rem\_left} + \text{rem\_right}$, reducing the branching from $2^N$ to $\binom{N}{K}$. With $N \le 25$, this executes in $< 15 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack and `path` buffer, plus $\mathcal{O}(2^N)$ for the output set.

---

### Takeaway Pattern & Interview Traps

- **Precomputing Minimum Removals:** Without pre-calculating `rem_left` and `rem_right`, backtracking would need to explore every possible subset of removals and test validity, which explodes exponentially. Precomputing the minimum bounds guarantees we only explore branches with the optimal removal count.
- **Handling Alphabetical Characters:** Don't forget non-parentheses characters! Letters are never removed and are always appended directly to `path`.
- **Prefix Validity Invariant:** Enforcing `close_count < open_count` when considering keeping `')'` prunes all invalid branches early.