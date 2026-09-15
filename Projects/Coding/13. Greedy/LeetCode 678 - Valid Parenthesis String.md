---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 678: Valid Parenthesis String"
tags:
  - leetcode
  - coding
  - greedy
  - string
  - stack
  - amazon
  - google
---

# LeetCode 678: Valid Parenthesis String

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, ByteDance  
**Difficulty:** Medium  
**Topic:** Greedy / String / Range Tracking / Monotonic State  

---

### Problem Statement

Given a string `s` containing only three types of characters: `'('`, `')'` and `'*'`, return `true` if `s` is **valid**.

The following rules define a valid string:
1. Any left parenthesis `'('` must have a corresponding right parenthesis `')'`.
2. Any right parenthesis `')'` must have a corresponding left parenthesis `'('`.
3. Left parenthesis `'('` must go before the corresponding right parenthesis `')'`.
4. `'*'` could be treated as a single right parenthesis `')'` or a single left parenthesis `'('` or an empty string `""`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 100$).
- **Output:**
  - `bool` — `true` if there exists at least one assignment of `'*'` making `s` valid, else `false`.
- **Constraints:**
  - $1 \le \text{s.length} \le 100$
  - `s[i]` is `'('`, `')'` or `'*'`.

---

### Key Idea & Intuition

Without `'*'`, we would only need a single integer counter `open_count`:
- `'('` increments `open_count += 1`
- `')'` decrements `open_count -= 1`
- String is valid if `open_count` never drops below $0$, and ends at exactly $0$.

With `'*'`, each star can change `open_count` by $+1$ (`'('`), $-1$ (`')'`), or $0$ (`""`).
This means at any index, the number of open parentheses is not a single number, but a **continuous interval of possible values**:
$$[\text{cmin}, \text{cmax}]$$
where:
- $\text{cmin}$: the **minimum** number of open parentheses possible.
- $\text{cmax}$: the **maximum** number of open parentheses possible.

#### Updating the Interval:
For each character `ch`:
1. If `ch == '('`:
   - Both minimum and maximum open counts increase by 1:
     $$\text{cmin} \mathrel{+}= 1, \quad \text{cmax} \mathrel{+}= 1$$
2. If `ch == ')'`:
   - Both minimum and maximum open counts decrease by 1:
     $$\text{cmin} \mathrel{-}= 1, \quad \text{cmax} \mathrel{-}= 1$$
3. If `ch == '*'`:
   - Treating `'*'` as `')'` minimizes open count $\rightarrow \text{cmin} \mathrel{-}= 1$.
   - Treating `'*'` as `'('` maximizes open count $\rightarrow \text{cmax} \mathrel{+}= 1$.
   - Treating `'*'` as `""` leaves it unchanged (lies within $[\text{cmin}, \text{cmax}]$).

#### Boundary Checks:
- If $\text{cmax} < 0$:
  - Even if every single star was treated as a `'('`, there are still too many `')'` characters. The prefix is irrecoverably invalid $\rightarrow$ return `False`.
- Lower bound clamp:
  - $\text{cmin} = \max(\text{cmin}, 0)$. We never allow $\text{cmin}$ to remain negative because we can always choose to treat some stars as empty strings `""` or `'('` rather than `')'`.
- At the end of the string:
  - Is it possible to have exactly 0 open parentheses?
  - Yes, if and only if $0 \in [\text{cmin}, \text{cmax}]$, which means $\text{cmin} == 0$.

---

### Solution Approach (Step-by-Step)

1. Initialize `cmin = 0` and `cmax = 0`.
2. For each character `ch` in `s`:
   - If `ch == '('`:
     - `cmin += 1`
     - `cmax += 1`
   - Else if `ch == ')'`:
     - `cmin -= 1`
     - `cmax -= 1`
   - Else (`ch == '*'`):
     - `cmin -= 1`
     - `cmax += 1`
   - If `cmax < 0`:
     - Return `False` (excess `')'`).
   - `cmin = max(cmin, 0)` (clamp negative minimum).
3. Return `cmin == 0`.

---

### Visual Algorithm Walkthrough

For `s = "(*))"`:

```
Start: [cmin, cmax] = [0, 0]

Char 0: '('
  cmin = 0 + 1 = 1
  cmax = 0 + 1 = 1
  Range: [1, 1]

Char 1: '*'
  cmin = 1 - 1 = 0 (if '*' is ')')
  cmax = 1 + 1 = 2 (if '*' is '(')
  Range: [0, 2]

Char 2: ')'
  cmin = 0 - 1 = -1 -> clamped to 0
  cmax = 2 - 1 = 1
  cmax >= 0 (OK)
  Range: [0, 1]

Char 3: ')'
  cmin = 0 - 1 = -1 -> clamped to 0
  cmax = 1 - 1 = 0
  cmax >= 0 (OK)
  Range: [0, 0]

End of string:
cmin == 0 -> TRUE!
Valid interpretation: '*' acted as '(' to match: "(()))" -> wait, s had 4 chars: "(*))" with '*' as '(' gives "(())" which is valid!
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "()"`
- **Output:** `true`

#### Example 2:
- **Input:** `s = "(*)"`
- **Output:** `true`

#### Example 3:
- **Input:** `s = "(*))"`
- **Output:** `true`

#### Example 4 (Invalid):
- **Input:** `s = ")*("`
- **Tracing:**
  - $ch=')' \rightarrow cmin=-1 \rightarrow 0, cmax=-1 < 0 \rightarrow$ returns `false`.
- **Output:** `false`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        cmin = 0  # Minimum possible open parentheses
        cmax = 0  # Maximum possible open parentheses
        
        for ch in s:
            if ch == '(':
                cmin += 1
                cmax += 1
            elif ch == ')':
                cmin -= 1
                cmax -= 1
            else:  # '*'
                cmin -= 1  # Treat as ')'
                cmax += 1  # Treat as '('
                
            # If even the maximum possible open count is negative, fail
            if cmax < 0:
                return False
                
            # cmin cannot drop below 0 (a '*' can always be chosen as empty)
            cmin = max(cmin, 0)
            
        return cmin == 0
```

#### C++17
```cpp
#include <string>
#include <algorithm>

class Solution {
public:
    bool checkValidString(const std::string& s) {
        int cmin = 0;
        int cmax = 0;
        
        for (char ch : s) {
            if (ch == '(') {
                cmin++;
                cmax++;
            } else if (ch == ')') {
                cmin--;
                cmax--;
            } else { // '*'
                cmin--;
                cmax++;
            }
            
            if (cmax < 0) {
                return false;
            }
            
            cmin = std::max(cmin, 0);
        }
        
        return cmin == 0;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean checkValidString(String s) {
        int cmin = 0;
        int cmax = 0;
        
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(') {
                cmin++;
                cmax++;
            } else if (ch == ')') {
                cmin--;
                cmax--;
            } else { // '*'
                cmin--;
                cmax++;
            }
            
            if (cmax < 0) {
                return false;
            }
            
            cmin = Math.max(cmin, 0);
        }
        
        return cmin == 0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - A single linear scan through the string of length $n$.
  - All arithmetic and range updates take $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Uses only two scalar integer registers (`cmin`, `cmax`).

---

### Takeaway Pattern & Interview Traps

- **Range Invariant Technique:** When non-deterministic choices (like wildcards) make branching exponential ($3^N$), ask: *Is the reachable state space a contiguous interval $[L, R]$?* Because wildcards increment/decrement by 1, the set of all reachable open counts forms an unbroken integer range.
- **The Clamp `cmin = max(cmin, 0)`:** Forgetting to clamp `cmin` to 0 will cause false negatives: an earlier `cmin < 0` does not mean the prefix is invalid, since a star could have simply been empty (`""`).