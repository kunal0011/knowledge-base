---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 1249: Minimum Remove to Make Valid Parentheses"
tags:
  - leetcode
  - coding
  - stack
  - string
  - amazon
  - google
---

# LeetCode 1249: Minimum Remove to Make Valid Parentheses

**Target Companies:** Meta, Amazon, Google, Bloomberg, TikTok  
**Difficulty:** Medium  
**Topic:** Stack / String Filtering / Balance Counting

---

### Problem Statement

Given a string `s` of `'('`, `')'` and lowercase English characters.

Your task is to remove the minimum number of parentheses ( `'('` or `')'`, in any positions ) so that the resulting parentheses string is valid and return **any** valid string.

Formally, a parentheses string is valid if and only if:
1. It is the empty string, contains only lowercase characters, or
2. It can be written as `AB` (`A` concatenated with `B`), where `A` and `B` are valid strings, or
3. It can be written as `(A)`, where `A` is a valid string.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str`, where $1 \le \text{len}(s) \le 10^5$. Each character is `'('`, `')'`, or a lowercase English letter.
- **Output:**
  - `str`: Any valid string resulting from removing the minimum number of parentheses.
- **Constraints:**
  - $1 \le s.\text{length} \le 10^5$.
  - Must achieve minimum possible removals.

---

### Key Idea & Intuition

Any character that is not a parenthesis (`'('` or `')'`) is a letter and must never be removed.

A parenthesis is invalid in only two circumstances:
1. **Unmatched closing parenthesis `')'`**: A `')'` is encountered at index $i$, but there is no currently open, unmatched `'('` preceding it. This `')'` must be removed.
2. **Unmatched opening parenthesis `'('`**: At the end of parsing the entire string, an `'('` was opened at index $j$, but never closed by a subsequent `')'`. This `'('` must be removed.

#### Core Technique: Stack of Indices
By storing the **indices** of `'('` in a LIFO stack:
- When encountering `'('`: push its index onto `stack`.
- When encountering `')'`:
  - If `stack` is non-empty: pop from `stack` (the current `')'` successfully matches the most recent unmatched `'('`).
  - If `stack` is empty: this `')'` has no matching `'('`, so its index must be marked for deletion.
- After iterating through the string, all indices remaining in `stack` correspond to unmatched `'('` characters and must also be marked for deletion.
- Finally, reconstruct the string by omitting the marked indices.

---

### Solution Approach (Step-by-Step)

1. **Initialize State:**
   - `stack`: a stack of integer indices storing unmatched `'('`.
   - `to_remove`: a boolean array of size $N$ (or a hash set of indices) to flag characters to exclude.

2. **Single-Pass Scan:**
   - For each index $i$ and character $c = s[i]$:
     - If $c == \text{'('}$: push $i$ onto `stack`.
     - If $c == \text{')'}$:
       - If `stack` is non-empty: `stack.pop()`.
       - Else: mark `to_remove[i] = true`.

3. **Flush Remaining Open Parentheses:**
   - While `stack` is non-empty:
     - Mark `to_remove[stack.pop()] = true`.

4. **Reconstruct Output String:**
   - Build a new string (using a character array or `StringBuilder`) appending $s[i]$ for all $i$ where `to_remove[i] == false`.

---

### Visual Algorithm Walkthrough

Consider $s = \text{"a)b(c)d(e"}$:

```
Index:    0   1   2   3   4   5   6   7   8
Char:     a   )   b   (   c   )   d   (   e

i = 0 ('a'): Letter -> Keep.
i = 1 (')'): Stack empty -> No open '('! Mark index 1 for removal.
             to_remove = {1}
i = 2 ('b'): Letter -> Keep.
i = 3 ('('): Push index 3 to stack. Stack = [3]
i = 4 ('c'): Letter -> Keep.
i = 5 (')'): Stack has [3] -> Matched! Pop 3. Stack = []
i = 6 ('d'): Letter -> Keep.
i = 7 ('('): Push index 7 to stack. Stack = [7]
i = 8 ('e'): Letter -> Keep.

End of string reached.
Flush remaining indices in stack:
  Stack contains [7] -> Mark index 7 for removal.
  to_remove = {1, 7}

Reconstruct string skipping indices {1, 7}:
  Index 0: 'a'
  Index 1: SKIP ')'
  Index 2: 'b'
  Index 3: '('
  Index 4: 'c'
  Index 5: ')'
  Index 6: 'd'
  Index 7: SKIP '('
  Index 8: 'e'

Result: "ab(c)de"
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Characters

- **Input:** `s = "lee(t(c)o)de)"`
- **Tracing:**

| Index | Char | Stack State | Invalid Set | Note |
|:---:|:---:|:---:|:---:|:---:|
| 3 | `(` | `[3]` | `{}` | First open |
| 5 | `(` | `[3, 5]` | `{}` | Second open |
| 7 | `)` | `[3]` | `{}` | Matches index 5 |
| 9 | `)` | `[]` | `{}` | Matches index 3 |
| 12 | `)` | `[]` | `{12}` | Unmatched closing |

- **Output:** `"lee(t(c)o)de"`

#### Example 2: Consecutive Unmatched Opening Parentheses

- **Input:** `s = "a(b(c)d"`
- **Tracing:**
  - Stack holds indices of `(` at $1$ and $3$.
  - Index $5$ `')'` matches index $3$. Stack remains with `[1]`.
  - At end, index $1$ is unmatched. Remove index $1$.
- **Output:** `"ab(c)d"`

#### Example 3: All Invalid Parentheses

- **Input:** `s = "))(("`
- **Output:** `""` (both `)` have no open partner, both `(` are never closed).

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List, Set

class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack: List[int] = []
        invalid_indices: Set[int] = set()

        # Step 1: Find unmatched closing parentheses and track open indices
        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            elif ch == ')':
                if stack:
                    stack.pop()
                else:
                    invalid_indices.add(i)

        # Step 2: Any '(' left in stack was never closed
        invalid_indices.update(stack)

        # Step 3: Reconstruct string omitting invalid indices
        result = [ch for i, ch in enumerate(s) if i not in invalid_indices]
        return "".join(result)
```

#### C++17

```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::string minRemoveToMakeValid(std::string s) {
        int n = static_cast<int>(s.size());
        std::vector<int> stack;
        std::vector<bool> to_remove(n, false);

        for (int i = 0; i < n; ++i) {
            if (s[i] == '(') {
                stack.push_back(i);
            } else if (s[i] == ')') {
                if (!stack.empty()) {
                    stack.pop_back();
                } else {
                    to_remove[i] = true;
                }
            }
        }

        // Mark any remaining unmatched '('
        for (int idx : stack) {
            to_remove[idx] = true;
        }

        // Reconstruct string
        std::string result;
        result.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (!to_remove[i]) {
                result.push_back(s[i]);
            }
        }

        return result;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public String minRemoveToMakeValid(String s) {
        int n = s.length();
        Deque<Integer> stack = new ArrayDeque<>();
        boolean[] toRemove = new boolean[n];

        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(i);
            } else if (c == ')') {
                if (!stack.isEmpty()) {
                    stack.pop();
                } else {
                    toRemove[i] = true;
                }
            }
        }

        // Mark remaining unmatched '('
        while (!stack.isEmpty()) {
            toRemove[stack.pop()] = true;
        }

        // Build valid result string
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            if (!toRemove[i]) {
                sb.append(s.charAt(i));
            }
        }

        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - A single pass of length $N$ scans each character.
  - Stack operations (push/pop) take $\mathcal{O}(1)$ each.
  - Reconstructing the final string visits each character once, taking $\mathcal{O}(N)$.
  - Total Time: $\mathcal{O}(N)$ where $N$ is the length of string `s`.
- **Space Complexity:** $\mathcal{O}(N)$
  - The stack holds at most $N$ indices in the worst case (e.g., all `'('`).
  - The boolean flag array / set stores at most $N$ markers.
  - Total Space: $\mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Stack for Index Tracking:**
   - Storing **indices** rather than character values in the stack is the key insight. This allows pinpointing exactly which occurrences of `'('` were left hanging when the string ends.
2. **Boolean Array vs Set:**
   - In C++ and Java, a boolean array `bool to_remove[n]` gives $\mathcal{O}(1)$ random access cache locality, which is significantly faster and uses less memory than a hash set of boxed integers.
3. **Multiple Valid Answers:**
   - Any valid string with the minimal number of deletions is accepted. For example, for `")("`, removing both yields `""`, which is uniquely optimal. For `"))(("`, deleting all 4 is optimal.