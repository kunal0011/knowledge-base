---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 921: Minimum Add to Make Parentheses Valid"
tags:
  - leetcode
  - coding
  - greedy
  - string
  - stack
  - meta
  - amazon
  - bloomberg
  - google
---

# LeetCode 921: Minimum Add to Make Parentheses Valid

**Target Companies:** Meta, Amazon, Bloomberg, Microsoft, Google  
**Difficulty:** Medium  
**Topic:** Greedy / String / Stack  

---

### Problem Statement

A parentheses string is **valid** if and only if:
1. It is the empty string,
2. It can be written as `AB` (`A` concatenated with `B`), where `A` and `B` are valid strings, or
3. It can be written as `(A)`, where `A` is a valid string.

You are given a parentheses string `s`. In one move, you can insert a parenthesis at any position of the string.

Return *the minimum number of moves required to make `s` valid*.

---

### Input & Output Formats & Constraints

- **Input:** A string `s` ($1 \le |s| \le 1000$) composed solely of characters `'('` and `')'`.
- **Output:** An integer representing the minimum number of insertions required.
- **Constraints:**
  - `1 <= s.length <= 1000`
  - `s[i]` is either `'('` or `')'`.

---

### Key Idea & Intuition

#### Balance Invariant & Two Types of Deficits
A parentheses string is valid if and only if:
1. **Prefix Invariant:** In every prefix of the string, the number of closing brackets `')'` never exceeds the number of opening brackets `'('`.
2. **Total Balance Invariant:** The total count of `'('` equals the total count of `')'`.

When traversing `s` from left to right, we encounter two distinct categories of violations:
1. **Underflow Deficit (Unmatched `')'`):**
   - If we encounter a `')'` when there are currently no available unmatched `'('` to pair with, this `')'` can never be matched by any future `'('`.
   - We are forced to insert an opening parenthesis `'('` before it.
   - We record this immediately in `inserts_needed`.
2. **Overflow Deficit (Unmatched `'('`):**
   - If we have remaining unmatched `'('` after processing the entire string, each of these must be closed by inserting a corresponding `')'` at the end.
   - This count is simply our remaining `open_count`.

#### Why Greedy Counting is Optimal
Because an unmatched `')'` can never be salvaged by future characters, resolving it immediately with an inserted `'('` is necessary and minimal. Similarly, unmatched `'('` must each receive their own `')'`. The total insertions required is strictly:
$$\text{Total Moves} = \text{inserts\_needed} + \text{open\_count}$$

This avoids allocating an explicit stack, running in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space.

---

### Solution Approach (Step-by-Step)

1. **Initialize Counters:**
   - `open_count = 0` (unmatched `'('` available for pairing)
   - `inserts_needed = 0` (unmatched `')'` requiring an inserted `'('`)
2. **Linear Scan:**
   - Iterate through each character `ch` in `s`:
     - If `ch == '('`:
       - Increment `open_count += 1`.
     - Else (`ch == ')'`):
       - If `open_count > 0`:
         - Decrement `open_count -= 1` (valid pair formed).
       - Else:
         - Increment `inserts_needed += 1` (orphan `')'`).
3. **Combine Deficits:**
   - Return `inserts_needed + open_count`.

---

### Visual Algorithm Walkthrough

#### Trace for `s = "()))(("`
```
Index:         0    1    2    3    4    5
Char:          (    )    )    )    (    (

Initial: open_count = 0, inserts_needed = 0

i = 0, char '(':
  open_count = 1, inserts_needed = 0

i = 1, char ')':
  open_count > 0 -> Matched!
  open_count = 0, inserts_needed = 0

i = 2, char ')':
  open_count == 0 -> No open bracket available!
  Must insert '(' before this ')'
  inserts_needed = 1, open_count = 0

i = 3, char ')':
  open_count == 0 -> No open bracket available!
  Must insert '(' before this ')'
  inserts_needed = 2, open_count = 0

i = 4, char '(':
  open_count = 1, inserts_needed = 2

i = 5, char '(':
  open_count = 2, inserts_needed = 2

End of String:
- inserts_needed = 2 (two '(' needed to fix early ')')
- open_count = 2 (two ')' needed to close trailing '(')

Total moves = 2 + 2 = 4.
Valid string formed: "(())()(())" (4 additions).
```

---

### Solved Examples with Multiple Inputs

| Input `s` | Step-by-Step Simulation | Unmatched `')'` (`inserts_needed`) | Unmatched `'('` (`open_count`) | Output |
|---|---|---|---|---|
| `"())"` | `'(' \to 1; ')' \to 0; ')' \to` underflow | `1` | `0` | `1` |
| `"((("` | `'(' \to 1; '(' \to 2; '(' \to 3` | `0` | `3` | `3` |
| `"()))(("` | Trace shown above | `2` | `2` | `4` |
| `"()"` | Perfectly balanced | `0` | `0` | `0` |
| `""` | Empty string | `0` | `0` | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        open_count: int = 0
        inserts_needed: int = 0
        
        for ch in s:
            if ch == '(':
                open_count += 1
            else:  # ch == ')'
                if open_count > 0:
                    open_count -= 1
                else:
                    inserts_needed += 1
                    
        return inserts_needed + open_count
```

#### C++17
```cpp
#include <string>

class Solution {
public:
    int minAddToMakeValid(const std::string& s) {
        int open_count = 0;
        int inserts_needed = 0;
        
        for (char ch : s) {
            if (ch == '(') {
                open_count++;
            } else { // ch == ')'
                if (open_count > 0) {
                    open_count--;
                } else {
                    inserts_needed++;
                }
            }
        }
        
        return inserts_needed + open_count;
    }
};
```

#### Java 17
```java
class Solution {
    public int minAddToMakeValid(String s) {
        int openCount = 0;
        int insertsNeeded = 0;
        
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                openCount++;
            } else { // ch == ')'
                if (openCount > 0) {
                    openCount--;
                } else {
                    insertsNeeded++;
                }
            }
        }
        
        return insertsNeeded + openCount;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |s|$. We iterate through the string in a single linear pass with constant-time updates.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Unlike stack-based parentheses problems, we only store two integer counters (`open_count` and `inserts_needed`).

---

### Takeaway Pattern & Interview Traps

1. **Stack vs. Counter:** While general bracket-matching problems with multiple types (`()`, `[]`, `{}`) require an explicit $\mathcal{O}(N)$ stack to enforce LIFO type matching, a single bracket type only requires tracking balance. Mentioning this reduction highlights algorithmic maturity.
2. **Underflow vs. Overflow:** Be careful to separate `inserts_needed` (underflow caused by `')'`) from `open_count` (overflow caused by `'('`). They represent insertions in opposite directions.
3. **Relation to LeetCode 1249 & 1541:** LC 1249 asks to *remove* minimum parentheses to make valid, while LC 1541 requires each `'('` to match two consecutive `')'`. The two-counter greedy tracking pattern forms the core foundation for both.