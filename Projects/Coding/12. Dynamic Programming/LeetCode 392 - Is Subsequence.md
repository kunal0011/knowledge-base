---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 392: Is Subsequence"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - two-pointers
  - string
  - binary-search
  - amazon
  - google
  - meta
---

# LeetCode 392: Is Subsequence

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Two Pointers / Dynamic Programming / String Matching / Follow-Up Query Preprocessing  

---

### Problem Statement

Given two strings `s` and `t`, return `true` if `s` is a **subsequence** of `t`, or `false` otherwise.

A **subsequence** of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., `"ace"` is a subsequence of `"abcde"` while `"aec"` is not).

**Follow-up:**  
Suppose there are lots of incoming $S$, say $S_1, S_2, \dots, S_k$ where $k \ge 10^9$, and you want to check one by one to see if $T$ has its subsequence. In this scenario, how would you change your code?

---

### Input & Output Formats & Constraints

- **Input:**
  - `s: str` — Pattern string of length $0 \le |s| \le 100$.
  - `t: str` — Target string of length $0 \le |t| \le 10^4$.
- **Output:**
  - `bool` — `true` if $s$ is a subsequence of $t$, `false` otherwise.
- **Constraints:**
  - $s$ and $t$ consist only of lowercase English letters.

---

### Key Idea & Intuition

1. **Greedy Two Pointers (Standard Query):**
   - For a single query $(s, t)$, a greedy two-pointer match is optimal:
   - Match characters of $s$ from left to right as early as possible in $t$.
   - Why does matching the earliest occurrence always work? Matching a character earlier preserves strictly more remaining characters in $t$ for matching the rest of $s$.
   - If pointer `p_s == len(s)`, all characters in $s$ were matched in order $\implies$ return `true`.

2. **Dynamic Programming Perspective (LCS Variant):**
   - Let $\text{dp}[i][j]$ denote whether $s[0 \dots i-1]$ is a subsequence of $t[0 \dots j-1]$.
   - Recurrence:
     $$\text{dp}[i][j] = \begin{cases} \text{dp}[i - 1][j - 1] & \text{if } s[i - 1] == t[j - 1] \\ \text{dp}[i][j - 1] & \text{otherwise} \end{cases}$$
   - Base case: $\text{dp}[0][j] = \text{true}$ for all $j$, $\text{dp}[i][0] = \text{false}$ for $i > 0$.
   - This takes $\mathcal{O}(|s| \times |t|)$ time, serving as a conceptual bridge to Longest Common Subsequence (LCS).

3. **Follow-Up Invariant ($k \ge 10^9$ queries against fixed $T$):**
   - Re-running two pointers takes $\mathcal{O}(k \times |T|)$, which will time out for large $k$.
   - **Precomputed DP Transition Table:**
     - Build table $\text{next\_pos}[i][c]$: the index of the first occurrence of character $c$ in $T$ at or after index $i$.
     - Built backwards from $i = |T| - 1$ down to $0$ in $\mathcal{O}(26 \times |T|)$ preprocessing time.
     - Each query for $s$ jumps from match to match in $\mathcal{O}(|s|)$ time:
       $$p = \text{next\_pos}[p][s[k]] + 1$$

---

### Solution Approach (Step-by-Step)

#### Approach 1: Two Pointers (Single Query - Standard)
1. Initialize `i = 0` (index into $s$) and `j = 0` (index into $t$).
2. While `i < len(s)` and `j < len(t)`:
   - If `s[i] == t[j]`, increment `i`.
   - Increment `j` on every iteration.
3. Return `i == len(s)`.

#### Approach 2: DP Jump Table (Follow-Up Optimized)
1. Allocate `next_pos` of size $(|T| + 1) \times 26$ filled with $-1$.
2. Fill backwards:
   - For $i = |T| - 1$ down to $0$:
     - Copy row: $\text{next\_pos}[i] = \text{next\_pos}[i + 1]$.
     - Update current char: $\text{next\_pos}[i][T[i] - 'a'] = i$.
3. For each query string $s$:
   - Current index in $T$: `curr = 0`.
   - For each character $ch$ in $s$:
     - Lookup `nxt = next_pos[curr][ch - 'a']`.
     - If `nxt == -1`: return `false`.
     - Advance `curr = nxt + 1`.
   - If all characters match, return `true`.

---

### Visual Algorithm Walkthrough

For `s = "abc"` and `t = "ahbgdc"`:

```
Two-Pointer Execution:
s: a  b  c
   ^
t: a  h  b  g  d  c
   ^
Step 1: s[0] ('a') == t[0] ('a') -> match! i=1, j=1

s: a  b  c
      ^
t: a  h  b  g  d  c
      ^
Step 2: s[1] ('b') != t[1] ('h') -> no match, j=2

s: a  b  c
      ^
t: a  h  b  g  d  c
         ^
Step 3: s[1] ('b') == t[2] ('b') -> match! i=2, j=3

Step 4 & 5: Skip 'g', skip 'd' (j=4, j=5)

s: a  b  c
         ^
t: a  h  b  g  d  c
                  ^
Step 6: s[2] ('c') == t[5] ('c') -> match! i=3, j=6

Loop ends: i == len(s) == 3 -> returns True!
```

---

### Solved Examples with Multiple Inputs

| Case | `s` | `t` | Execution Trace | Result | Explanation |
|---|---|---|---|---|---|
| **True Match** | `"abc"` | `"ahbgdc"` | Matches `'a'` at 0, `'b'` at 2, `'c'` at 5 | `true` | Characters appear in order |
| **False Match** | `"axc"` | `"ahbgdc"` | Matches `'a'`, fails on `'x'` | `false` | `'x'` does not exist in `t` |
| **Empty `s`** | `""` | `"ahbgdc"` | `len(s) == 0`, immediately satisfied | `true` | Empty string is a subsequence of any string |
| **Empty `t`** | `"b"` | `""` | `t` exhausted without matching `'b'` | `false` | Non-empty cannot be formed from empty |
| **Identical Strings** | `"hello"` | `"hello"` | Direct 1-to-1 match | `true` | Entire string matches |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # Standard Two-Pointer Approach: O(|t|) time, O(1) space
        p_s, p_t = 0, 0
        len_s, len_t = len(s), len(t)
        
        while p_s < len_s and p_t < len_t:
            if s[p_s] == t[p_t]:
                p_s += 1
            p_t += 1
            
        return p_s == len_s

    # Follow-Up: Preprocessed Next-Position DP Table for 10^9 queries
    def isSubsequenceFollowUp(self, s: str, next_pos: list) -> bool:
        curr = 0
        for ch in s:
            c_idx = ord(ch) - ord('a')
            if curr >= len(next_pos) or next_pos[curr][c_idx] == -1:
                return False
            curr = next_pos[curr][c_idx] + 1
        return True
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class Solution {
public:
    // Standard Two Pointers: O(|t|) time, O(1) space
    bool isSubsequence(std::string s, std::string t) {
        int p_s = 0, p_t = 0;
        int len_s = s.size(), len_t = t.size();

        while (p_s < len_s && p_t < len_t) {
            if (s[p_s] == t[p_t]) {
                p_s++;
            }
            p_t++;
        }

        return p_s == len_s;
    }

    // Follow-up: Preprocess t in O(26 * |t|) time
    std::vector<std::vector<int>> buildNextPosTable(const std::string& t) {
        int n = t.size();
        std::vector<std::vector<int>> next_pos(n + 1, std::vector<int>(26, -1));

        for (int i = n - 1; i >= 0; --i) {
            next_pos[i] = next_pos[i + 1];
            next_pos[i][t[i] - 'a'] = i;
        }
        return next_pos;
    }

    // Query in O(|s|) time
    bool isSubsequenceFollowUp(const std::string& s, const std::vector<std::vector<int>>& next_pos) {
        int curr = 0;
        for (char ch : s) {
            int c_idx = ch - 'a';
            if (curr >= next_pos.size() || next_pos[curr][c_idx] == -1) {
                return false;
            }
            curr = next_pos[curr][c_idx] + 1;
        }
        return true;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    // Standard Two Pointers: O(|t|) time, O(1) space
    public boolean isSubsequence(String s, String t) {
        int pS = 0, pT = 0;
        int lenS = s.length(), lenT = t.length();

        while (pS < lenS && pT < lenT) {
            if (s.charAt(pS) == t.charAt(pT)) {
                pS++;
            }
            pT++;
        }

        return pS == lenS;
    }

    // Follow-up: Build Next Position Table
    public int[][] buildNextPosTable(String t) {
        int n = t.length();
        int[][] nextPos = new int[n + 1][26];
        Arrays.fill(nextPos[n], -1);

        for (int i = n - 1; i >= 0; i--) {
            for (int c = 0; c < 26; c++) {
                nextPos[i][c] = nextPos[i + 1][c];
            }
            nextPos[i][t.charAt(i) - 'a'] = i;
        }
        return nextPos;
    }

    // Follow-up Query: O(|s|)
    public boolean isSubsequenceFollowUp(String s, int[][] nextPos) {
        int curr = 0;
        for (int i = 0; i < s.length(); i++) {
            int cIdx = s.charAt(i) - 'a';
            if (curr >= nextPos.length || nextPos[curr][cIdx] == -1) {
                return false;
            }
            curr = nextPos[curr][cIdx] + 1;
        }
        return true;
    }
}
```

---

### Complexity Analysis

- **Two Pointers (Standard):**
  - **Time Complexity:** $\mathcal{O}(|t|)$ — Single pass through $t$.
  - **Space Complexity:** $\mathcal{O}(1)$ — Only two integer pointers.
- **Follow-Up (Many Queries):**
  - **Preprocessing Time:** $\mathcal{O}(26 \times |t|)$
  - **Query Time:** $\mathcal{O}(|s|)$ per query string
  - **Space Complexity:** $\mathcal{O}(26 \times |t|)$ for the DP jump table.

---

### Takeaway Pattern & Interview Traps

1. **Greedy Matching Soundness:**
   - The greedy strategy (taking the earliest occurrence of character $s[i]$ in $t$) is mathematically guaranteed to be optimal because it leaves the maximum possible remaining suffix of $t$ for future characters.
2. **The Follow-Up Expectation:**
   - When this problem is asked in senior interviews (especially at Google/Meta), the main question is merely a warmup. Interviewers immediately pivot to the follow-up with $10^9$ queries. Knowing both the DP jump table $\mathcal{O}(26 \times |t|)$ and binary search on bucketed index lists $\mathcal{O}(|s| \log |t|)$ is essential.