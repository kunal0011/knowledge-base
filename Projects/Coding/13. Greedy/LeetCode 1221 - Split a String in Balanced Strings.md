---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1221: Split a String in Balanced Strings"
tags:
  - leetcode
  - coding
  - greedy
  - string
  - counting
  - amazon
  - google
---

# LeetCode 1221: Split a String in Balanced Strings

**Target Companies:** Amazon, Google, Microsoft, Apple  
**Difficulty:** Easy  
**Topic:** Greedy / String / Counting  

---

### Problem Statement

**Balanced** strings are those that have an equal quantity of `'L'` and `'R'` characters.

Given a **balanced** string `s`, split it into some number of substrings such that:
- Each substring is balanced.

Return the **maximum number of balanced substrings** you can obtain.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($2 \le |s| \le 1000$, guaranteed to be balanced).
- **Output:**
  - `int` — maximum number of contiguous balanced substrings.
- **Constraints:**
  - $2 \le \text{s.length} \le 1000$
  - `s[i]` is either `'L'` or `'R'`.
  - `s` is a balanced string.

---

### Key Idea & Intuition

The goal is to maximize the number of balanced contiguous partitions.
Consider scanning `s` from left to right while tracking the net balance:
$$\text{balance} = \text{count}('R') - \text{count}('L')$$
(or equivalently `+1` for `'R'` and `-1` for `'L'`).

#### The Greedy Earliest-Cut Invariant:
Whenever $\text{balance} == 0$, the prefix processed so far has an equal number of `'R'`s and `'L'`s, meaning it forms a valid balanced substring.
- Should we immediately cut the string here?
- **Yes!** Suppose a balanced prefix $P$ is found. If we do not cut at $P$ and instead wait for a larger balanced prefix $P'$, then $P' = P + Q$. Since $P'$ is balanced and $P$ is balanced, the suffix $Q$ must also be balanced!
- Cutting $P$ immediately gives us 1 balanced piece from $P$ plus any balanced pieces we can extract from $Q$. Merging them would give only 1 piece, which can never be better than cutting them into at least 2 pieces.
- Therefore, splitting **as early as possible** (the moment $\text{balance} == 0$) is mathematically guaranteed to maximize the total number of balanced substrings.

---

### Solution Approach (Step-by-Step)

1. Initialize `balance = 0` and `ans = 0`.
2. Iterate through each character `ch` in `s`:
   - If `ch == 'R'`, increment `balance += 1`.
   - Else (`ch == 'L'`), decrement `balance -= 1`.
   - If `balance == 0`:
     - Increment `ans += 1` (a complete balanced component has ended).
3. Return `ans`.

---

### Visual Algorithm Walkthrough

For `s = "RLRRLLRLRL"`:

```
Index:   0    1    2    3    4    5    6    7    8    9
Char:    R    L    R    R    L    L    R    L    R    L

i=0 ('R'): balance = +1
i=1 ('L'): balance =  0 -> balance == 0! CUT 1: "RL"  (ans = 1)
i=2 ('R'): balance = +1
i=3 ('R'): balance = +2
i=4 ('L'): balance = +1
i=5 ('L'): balance =  0 -> balance == 0! CUT 2: "RRLL" (ans = 2)
i=6 ('R'): balance = +1
i=7 ('L'): balance =  0 -> balance == 0! CUT 3: "RL"  (ans = 3)
i=8 ('R'): balance = +1
i=9 ('L'): balance =  0 -> balance == 0! CUT 4: "RL"  (ans = 4)

Total balanced substrings = 4 ("RL", "RRLL", "RL", "RL").
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "RLRRLLRLRL"`
- **Output:** `4`

#### Example 2:
- **Input:** `s = "RLRRRLLRLL"`
- **Tracing:**
  - Cut 1 at index 1: `"RL"`
  - Cut 2 at index 9: `"RRRLLRLL"`
- **Output:** `2`

#### Example 3:
- **Input:** `s = "LLLLRRRR"`
- **Tracing:** Balance only reaches 0 at the very end (index 7).
- **Output:** `1`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def balancedStringSplit(self, s: str) -> int:
        balance = 0
        ans = 0
        
        for ch in s:
            if ch == 'R':
                balance += 1
            else:
                balance -= 1
                
            if balance == 0:
                ans += 1
                
        return ans
```

#### C++17
```cpp
#include <string>

class Solution {
public:
    int balancedStringSplit(const std::string& s) {
        int balance = 0;
        int ans = 0;
        
        for (char ch : s) {
            if (ch == 'R') {
                balance++;
            } else {
                balance--;
            }
            
            if (balance == 0) {
                ans++;
            }
        }
        
        return ans;
    }
};
```

#### Java 17
```java
class Solution {
    public int balancedStringSplit(String s) {
        int balance = 0;
        int ans = 0;
        
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == 'R') {
                balance++;
            } else {
                balance--;
            }
            
            if (balance == 0) {
                ans++;
            }
        }
        
        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - We traverse the string of length $n$ once, performing $\mathcal{O}(1)$ counter increments and comparisons per character.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Uses only two integer counters `balance` and `ans`.

---

### Takeaway Pattern & Interview Traps

- **The Greedy Splitting Invariant:** When seeking to maximize the number of valid independent blocks, cutting at the earliest valid boundary is optimal if and only if the valid property is additive/closed under subtraction ($A$ and $A+B$ valid $\implies$ $B$ valid).
- **Guaranteed Balance:** The problem guarantees the input string is already globally balanced, so `balance` will always equal $0$ at the final character.