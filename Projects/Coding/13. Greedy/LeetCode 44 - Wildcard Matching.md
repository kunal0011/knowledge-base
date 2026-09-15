---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 44: Wildcard Matching"
tags:
  - leetcode
  - coding
  - greedy
  - two-pointers
  - dynamic-programming
  - string
  - amazon
  - google
---

# LeetCode 44: Wildcard Matching

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, ByteDance  
**Difficulty:** Hard  
**Topic:** Greedy / Two Pointers / String / Dynamic Programming  

---

### Problem Statement

Given an input string (`s`) and a pattern (`p`), implement wildcard pattern matching with support for `'?'` and `'*'` where:
- `'?'` Matches any single character.
- `'*'` Matches any sequence of characters (including the empty sequence).

The matching should cover the **entire** input string (not partial).

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($0 \le |s| \le 2000$).
  - `p`: `str` / `string` ($0 \le |p| \le 2000$).
- **Output:**
  - `bool` — `true` if pattern `p` matches string `s` completely, else `false`.
- **Constraints:**
  - $0 \le \text{s.length}, \text{p.length} \le 2000$
  - `s` contains only lowercase English letters.
  - `p` contains only lowercase English letters, `'?'` or `'*'`.

---

### Key Idea & Intuition

While 2D Dynamic Programming solves this problem in $\mathcal{O}(|s| \times |p|)$ time and $\mathcal{O}(|s| \times |p|)$ space, a **Greedy Two-Pointer with Last-Star Backtracking** algorithm achieves **$\mathcal{O}(1)$ auxiliary space** and runs in near-linear time in practice!

#### The Greedy Last-Star Invariant:
When matching $s$ against $p$:
1. If $p[\text{p\_idx}] == s[\text{s\_idx}]$ or $p[\text{p\_idx}] == '?'$:
   - Direct match. Advance both pointers `s_idx += 1`, `p_idx += 1`.
2. If $p[\text{p\_idx}] == '*' $:
   - A wildcard `*` can match zero or more characters.
   - Greedily assume the `*` matches **zero** characters initially.
   - Record checkpoints: `star_idx = p_idx`, and `s_match = s_idx`.
   - Advance pattern pointer: `p_idx += 1`.
3. If a mismatch occurs:
   - If a previous `'*'` was seen (`star_idx != -1`):
     - The assumption that `'*'` matched fewer characters failed.
     - Backtrack: let `'*'` consume one additional character of $s$ by incrementing `s_match += 1`.
     - Reset `s_idx = s_match`, and restart matching the pattern right after the star: `p_idx = star_idx + 1`.
   - If no previous `'*'` exists:
     - The mismatch cannot be absorbed $\rightarrow$ return `False`.
4. After traversing $s$:
   - Any trailing characters in $p$ must be `'*'` to match the empty suffix.

#### Why We Only Need the Most Recent `'*'` Checkpoint:
If pattern contains multiple stars (e.g. `*abc*def`), any failure after the second star only needs to backtrack to the second star. The second star has already superseded the first star because it can consume any suffix that the first star could have consumed. Thus, storing only the single most recent `star_idx` is mathematically sufficient!

---

### Solution Approach (Step-by-Step)

1. Initialize `s_idx = 0`, `p_idx = 0`, `star_idx = -1`, and `s_match = 0`.
2. While `s_idx < len(s)`:
   - If `p_idx < len(p)` and `p[p_idx] in (s[s_idx], '?')`:
     - Both match: `s_idx += 1`, `p_idx += 1`.
   - Else if `p_idx < len(p)` and `p[p_idx] == '*'`:
     - Record star position: `star_idx = p_idx`, `s_match = s_idx`.
     - Advance pattern: `p_idx += 1`.
   - Else if `star_idx != -1`:
     - Backtrack to star: `s_match += 1`, `s_idx = s_match`, `p_idx = star_idx + 1`.
   - Else:
     - Mismatch without any star: return `False`.
3. Check remaining pattern characters: while `p_idx < len(p)` and `p[p_idx] == '*'`: `p_idx += 1`.
4. Return `p_idx == len(p)`.

---

### Visual Algorithm Walkthrough

For `s = "acdcb"`, `p = "a*c?b"`:

```
s = a  c  d  c  b
p = a  *  c  ?  b

1. s[0]='a', p[0]='a' -> Match! s_idx=1, p_idx=1
2. p[1]='*' -> Star found! star_idx=1, s_match=1, p_idx=2 (try matching 0 chars for '*')
3. s[1]='c', p[2]='c' -> Match! s_idx=2, p_idx=3
4. s[2]='d', p[3]='?' -> Match ('?' matches 'd')! s_idx=3, p_idx=4
5. s[3]='c', p[4]='b' -> MISMATCH ('c' != 'b')!
   Previous star exists at star_idx=1!
   Backtrack:
     s_match = 1 + 1 = 2 ('*' now absorbs s[1]='c')
     s_idx = 2 (points to 'd')
     p_idx = star_idx + 1 = 2 (points to 'c')
6. s[2]='d', p[2]='c' -> MISMATCH ('d' != 'c')!
   Previous star exists at star_idx=1!
   Backtrack:
     s_match = 2 + 1 = 3 ('*' absorbs s[1]='c' and s[2]='d')
     s_idx = 3 (points to second 'c')
     p_idx = 2 (points to 'c')
7. s[3]='c', p[2]='c' -> Match! s_idx=4, p_idx=3
8. s[4]='b', p[3]='?' -> Match ('?' matches 'b')! s_idx=5, p_idx=4
9. End of s reached (s_idx = 5).
   Check p: remaining is p[4]='b', not '*'.
   p_idx != len(p) -> False.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "aa"`, `p = "a"`
- **Output:** `false`

#### Example 2:
- **Input:** `s = "aa"`, `p = "*"`
- **Output:** `true`

#### Example 3:
- **Input:** `s = "cb"`, `p = "?a"`
- **Output:** `false`

#### Example 4:
- **Input:** `s = "adceb"`, `p = "*a*b"`
- **Tracing:** `*` matches `""`, `'a'` matches `'a'`, `*` matches `"dce"`, `'b'` matches `'b'`.
- **Output:** `true`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s_idx, p_idx = 0, 0
        star_idx = -1
        s_match = 0
        
        len_s, len_p = len(s), len(p)
        
        while s_idx < len_s:
            # Case 1: Characters match or pattern has '?'
            if p_idx < len_p and (p[p_idx] == s[s_idx] or p[p_idx] == '?'):
                s_idx += 1
                p_idx += 1
            # Case 2: Pattern has '*', record checkpoint
            elif p_idx < len_p and p[p_idx] == '*':
                star_idx = p_idx
                s_match = s_idx
                p_idx += 1
            # Case 3: Mismatch, but a previous '*' can absorb more characters
            elif star_idx != -1:
                p_idx = star_idx + 1
                s_match += 1
                s_idx = s_match
            # Case 4: Mismatch without any star
            else:
                return False
                
        # Consume any trailing '*' characters in pattern
        while p_idx < len_p and p[p_idx] == '*':
            p_idx += 1
            
        return p_idx == len_p
```

#### C++17
```cpp
#include <string>

class Solution {
public:
    bool isMatch(const std::string& s, const std::string& p) {
        int s_idx = 0, p_idx = 0;
        int star_idx = -1;
        int s_match = 0;
        
        int len_s = static_cast<int>(s.size());
        int len_p = static_cast<int>(p.size());
        
        while (s_idx < len_s) {
            if (p_idx < len_p && (p[p_idx] == s[s_idx] || p[p_idx] == '?')) {
                s_idx++;
                p_idx++;
            } else if (p_idx < len_p && p[p_idx] == '*') {
                star_idx = p_idx;
                s_match = s_idx;
                p_idx++;
            } else if (star_idx != -1) {
                p_idx = star_idx + 1;
                s_match++;
                s_idx = s_match;
            } else {
                return false;
            }
        }
        
        while (p_idx < len_p && p[p_idx] == '*') {
            p_idx++;
        }
        
        return p_idx == len_p;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean isMatch(String s, String p) {
        int sIdx = 0, pIdx = 0;
        int starIdx = -1;
        int sMatch = 0;
        
        int lenS = s.length();
        int lenP = p.length();
        
        while (sIdx < lenS) {
            if (pIdx < lenP && (p.charAt(pIdx) == s.charAt(sIdx) || p.charAt(pIdx) == '?')) {
                sIdx++;
                pIdx++;
            } else if (pIdx < lenP && p.charAt(pIdx) == '*') {
                starIdx = pIdx;
                sMatch = sIdx;
                pIdx++;
            } else if (starIdx != -1) {
                pIdx = starIdx + 1;
                sMatch++;
                sIdx = sMatch;
            } else {
                return false;
            }
        }
        
        while (pIdx < lenP && p.charAt(pIdx) == '*') {
            pIdx++;
        }
        
        return pIdx == lenP;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** Average $\mathcal{O}(|s| + |p|)$, Worst Case $\mathcal{O}(|s| \times |p|)$
  - In ordinary text with wildcards, the pointers advance forward with minimal backtracking, running in linear time.
  - Pathological cases (e.g. `s = "aaaaa"`, `p = "*a*a*a"`) run in at most $\mathcal{O}(|s| \times |p|)$ iterations.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Unlike dynamic programming which requires an $\mathcal{O}(|s| \times |p|)$ table or $\mathcal{O}(|p|)$ rolling row, this greedy two-pointer algorithm uses strictly 4 scalar integer pointers.

---

### Takeaway Pattern & Interview Traps

- **Contrast with Regular Expression Matching (LeetCode 10):**
  - In Wildcard Matching, `'*'` stands alone and matches any sequence, so only the **most recent** `'*'` needs to be remembered.
  - In Regex Matching, `'*'` modifies the preceding character (e.g., `a*`), meaning older wildcards cannot freely subsume subsequent constraints, requiring full DP or backtracking.
- **Trailing Stars:** Always clean up trailing `'*'` characters after `s_idx` reaches the end of the string.