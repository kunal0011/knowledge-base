---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1358: Number of Substrings Containing All Three Characters"
tags:
  - leetcode
  - coding
  - sliding-window
  - string
  - hash-table
  - amazon
  - google
---

# LeetCode 1358: Number of Substrings Containing All Three Characters

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Sliding Window / Last Seen Pointers / Substring Counting  

---

### Problem Statement

Given a string `s` consisting only of characters `a`, `b` and `c`.

Return the number of substrings containing **at least one** occurrence of all these characters `a`, `b` and `c`.

---

### Input & Output Formats & Constraints

- **Input:** `s: str` (consisting only of `'a'`, `'b'`, `'c'`)
- **Output:** `int` (count of valid substrings)
- **Constraints:**
  - $3 \le \text{s.length} \le 5 \times 10^4$
  - `s` only consists of `'a'`, `'b'` or `'c'` characters.

---

### Key Idea & Intuition

- **Approach 1: Last Seen Indices ($\mathcal{O}(1)$ Extra Space, Single Linear Scan):**
  - Maintain an array `last_seen` of size 3, storing the most recent index where `'a'`, `'b'`, and `'c'` were encountered (initialized to `-1`).
  - As we iterate through $s$ at index $i$:
    - Update `last_seen[s[i] - 'a'] = i`.
    - Find the minimum of the three last-seen indices:
      $$m = \min(\text{last\_seen}[0], \text{last\_seen}[1], \text{last\_seen}[2])$$
    - If $m \ne -1$, all three characters have appeared. Any starting index $j \in [0, m]$ guarantees that substring $s[j \dots i]$ contains at least one `'a'`, `'b'`, and `'c'`.
    - Hence, ending at index $i$, there are exactly $m + 1$ valid substrings!
    - Summing $(m + 1)$ across all $i$ solves the problem in a single ultra-clean linear pass.

- **Approach 2: Two-Pointer Sliding Window:**
  - Maintain character counts in a window $[L, R]$.
  - Expand $R$: increment `counts[s[R]]`.
  - While all three characters are present (`counts['a'] > 0 and counts['b'] > 0 and counts['c'] > 0`):
    - Every substring starting at $L$ and ending at any index from $R$ to $N - 1$ is valid ($\text{len}(s) - R$ valid extensions).
    - Add $N - R$ to total, decrement `counts[s[L]]`, and increment $L$.

---

### Solution Approach (Step-by-Step)

#### Using the Last-Seen Index Pattern:
1. Initialize `last_seen = [-1, -1, -1]` and `ans = 0`.
2. Loop `i` from $0$ to $\text{len}(s) - 1$:
   - `last_seen[ord(s[i]) - ord('a')] = i`
   - `ans += 1 + min(last_seen)` (only adds positive count once all 3 are $\ge 0$).
3. Return `ans`.

---

### Visual Algorithm Walkthrough

Let `s = "abcabc"`:

```
Indices:   0  1  2  3  4  5
String:    a  b  c  a  b  c

i=0 ('a'): last_seen = [0, -1, -1], min = -1 -> ans += 0
i=1 ('b'): last_seen = [0,  1, -1], min = -1 -> ans += 0
i=2 ('c'): last_seen = [0,  1,  2], min =  0 -> valid starts: [0]
           Substrings ending at 2: "abc" (1) -> ans = 1

i=3 ('a'): last_seen = [3,  1,  2], min =  1 -> valid starts: [0, 1]
           Substrings ending at 3: "abca", "bca" (2) -> ans = 1 + 2 = 3

i=4 ('b'): last_seen = [3,  4,  2], min =  2 -> valid starts: [0, 1, 2]
           Substrings ending at 4: "abcab", "bcab", "cab" (3) -> ans = 3 + 3 = 6

i=5 ('c'): last_seen = [3,  4,  5], min =  3 -> valid starts: [0, 1, 2, 3]
           Substrings ending at 5: "abcabc", "bcabc", "cabc", "abc" (4) -> ans = 6 + 4 = 10

Total Valid Substrings: 10
```

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | Valid Substrings Sample | Output |
| :--- | :--- | :--- | :--- |
| **Example 1** | `"abcabc"` | `"abc"`, `"abca"`, `"abcab"`, `"abcabc"`, `"bca"`, etc. | `10` |
| **Example 2** | `"aaacb"` | `"aaacb"`, `"aacb"`, `"acb"` | `3` |
| **Example 3** | `"abc"` | Exactly 1 substring: `"abc"` | `1` |
| **Missing Char** | `"aaaaa"` | No `'b'` or `'c'` present | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        """
        Calculates the number of substrings containing at least one 'a', 'b', and 'c'.
        Uses the last-seen index tracking method for single-pass O(N) time and O(1) space.
        """
        last_seen = [-1, -1, -1]
        ans = 0

        for i, ch in enumerate(s):
            last_seen[ord(ch) - ord('a')] = i
            # If all 3 characters have appeared, min(last_seen) >= 0
            ans += 1 + min(last_seen)

        return ans
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int numberOfSubstrings(const std::string& s) {
        int last_seen[3] = {-1, -1, -1};
        int ans = 0;

        for (int i = 0; i < static_cast<int>(s.size()); ++i) {
            last_seen[s[i] - 'a'] = i;
            int min_idx = std::min({last_seen[0], last_seen[1], last_seen[2]});
            ans += 1 + min_idx;
        }

        return ans;
    }
};
```

#### Java
```java
class Solution {
    public int numberOfSubstrings(String s) {
        int[] lastSeen = {-1, -1, -1};
        int ans = 0;
        int n = s.length();

        for (int i = 0; i < n; i++) {
            lastSeen[s.charAt(i) - 'a'] = i;
            int minIdx = Math.min(lastSeen[0], Math.min(lastSeen[1], lastSeen[2]));
            ans += 1 + minIdx;
        }

        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{s.length}$.
  - We iterate through the string exactly once.
  - At each index, finding the minimum of 3 integers takes $\mathcal{O}(1)$ time.
  - Runs in $< 5 \text{ ms}$ for $N = 5 \times 10^4$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since `last_seen` uses an array of size 3.

---

### Takeaway Pattern & Interview Traps

- **Last-Seen Array for Fixed Alphabets:** When tracking occurrences of a fixed small alphabet (like `'a', 'b', 'c'`), recording the most recent index of each character allows immediate $\mathcal{O}(1)$ counting of valid prefixes ending at the current index.
- **Prefix Invariant:** If $s[m \dots i]$ contains all required characters, then every substring $s[j \dots i]$ with $0 \le j \le m$ is also guaranteed to contain them, contributing exactly $m + 1$ solutions.