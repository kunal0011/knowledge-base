---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 5: Longest Palindromic Substring"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - two-pointers
  - interval-dp
  - amazon
  - google
  - meta
---

# LeetCode 5: Longest Palindromic Substring

**Target Companies:** Amazon (Top #1), Microsoft, Google, Meta, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming (Interval DP) / Expand Around Center / String Processing  

---

### Problem Statement

Given a string `s`, return the **longest palindromic substring** in `s`.

A **palindromic string** is a string that reads the same backward as forward.

---

### Input & Output Formats & Constraints

- **Input:** `s: str` — String consisting of digits and English letters.
- **Output:** `str` — The longest contiguous palindromic substring.
- **Constraints:**
  - $1 \le s.\text{length} \le 1000$
  - `s` consists of only digits and English letters.

---

### Key Idea & Intuition

1. **Approach 1: Expand Around Center ($\mathcal{O}(N^2)$ Time, $\mathcal{O}(1)$ Space — Production Standard):**
   - A palindrome mirrors around its center.
   - For a string of length $n$, there are $2n - 1$ possible centers:
     - $n$ single-character centers (odd-length palindromes like `"aba"` centered at `'b'`).
     - $n - 1$ between-character centers (even-length palindromes like `"abba"` centered between `'b'` and `'b'`).
   - For each center, expand outward two pointers `(left, right)` as long as $s[\text{left}] == s[\text{right}]$.
   - This takes $\mathcal{O}(1)$ auxiliary space and avoids allocating an $\mathcal{O}(n^2)$ boolean matrix.

2. **Approach 2: Interval Dynamic Programming ($\mathcal{O}(N^2)$ Time, $\mathcal{O}(N^2)$ Space — Theoretical Foundation):**
   - Let $\text{dp}[i][j]$ be `true` if substring $s[i \dots j]$ is a palindrome.
   - Base Cases:
     - Length 1: $\text{dp}[i][i] = \text{true}$
     - Length 2: $\text{dp}[i][i + 1] = (s[i] == s[i + 1])$
   - Recurrence:
     $$\text{dp}[i][j] = (s[i] == s[j]) \land (j - i \le 2 \lor \text{dp}[i + 1][j - 1])$$
   - Iterate by substring length from $3$ to $n$.

---

### Solution Approach (Step-by-Step)

#### Expand Around Center (Recommended)
1. **Handle Base Case:**
   - If `len(s) <= 1`, return `s`.
2. **Define Helper Function `expand(left, right)`:**
   - While `left >= 0` and `right < len(s)` and `s[left] == s[right]`:
     - `left -= 1`, `right += 1`
   - Return substring `s[left + 1 : right]` (or return pair `(left + 1, right - left - 1)`).
3. **Iterate All Centers:**
   - For $i$ from $0$ to $n - 1$:
     - Check odd center: `expand(i, i)`.
     - Check even center: `expand(i, i + 1)`.
     - Update global longest palindrome if a longer one is found.
4. **Return:**
   - Return the longest palindrome substring found.

---

### Visual Algorithm Walkthrough

For `s = "babad"`:

```
Center i = 0 ('b'):
  Odd: expand(0, 0) -> "b" (len 1)
  Even: expand(0, 1) -> s[0] != s[1] -> ""

Center i = 1 ('a'):
  Odd: expand(1, 1) -> s[0]=='b', s[2]=='b' (match!)
       expand(-1, 3) -> out of bounds.
       Result: "bab" (len 3, start=0)
  Even: expand(1, 2) -> s[1] != s[2] -> ""

Center i = 2 ('b'):
  Odd: expand(2, 2) -> s[1]=='a', s[3]=='a' (match!)
       expand(0, 4) -> s[0]=='b', s[4]=='d' (mismatch)
       Result: "aba" (len 3, start=1)
  Even: expand(2, 3) -> s[2] != s[3] -> ""

Center i = 3 ('a'):
  Odd: expand(3, 3) -> "a"
  Even: expand(3, 4) -> ""

Center i = 4 ('d'):
  Odd: expand(4, 4) -> "d"

Maximum Length = 3.
Returns: "bab" (or "aba", both valid).
```

---

### Solved Examples with Multiple Inputs

| Case | `s` | Palindromic Candidates | Result | Explanation |
|---|---|---|---|---|
| **Odd Palindrome** | `"babad"` | `"bab"`, `"aba"` | `"bab"` (or `"aba"`) | Length 3 substring |
| **Even Palindrome** | `"cbbd"` | `"bb"` | `"bb"` | Length 2 substring |
| **All Identical** | `"aaaa"` | Entire string | `"aaaa"` | Entire string is a palindrome |
| **Single Character** | `"a"` | `"a"` | `"a"` | Single character is trivially palindromic |
| **Strictly Unique** | `"abcdef"` | Each single char | `"a"` | Any length 1 character |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Expand Around Center $\mathcal{O}(1)$ Space)
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s or len(s) <= 1:
            return s
            
        start, max_len = 0, 1
        n = len(s)
        
        def expand(left: int, right: int) -> int:
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            # Length of valid palindrome is right - left - 1
            return right - left - 1

        for i in range(n):
            len1 = expand(i, i)       # Odd length center
            len2 = expand(i, i + 1)   # Even length center
            curr_len = max(len1, len2)
            
            if curr_len > max_len:
                max_len = curr_len
                # Calculate new start index
                start = i - (curr_len - 1) // 2
                
        return s[start : start + max_len]
```

#### 2. C++ (C++17 / STL — Expand Around Center $\mathcal{O}(1)$ Space)
```cpp
#include <string>
#include <algorithm>

class Solution {
public:
    std::string longestPalindrome(std::string s) {
        int n = s.size();
        if (n <= 1) return s;

        int start = 0;
        int max_len = 1;

        auto expand = [&](int left, int right) -> int {
            while (left >= 0 && right < n && s[left] == s[right]) {
                left--;
                right++;
            }
            return right - left - 1;
        };

        for (int i = 0; i < n; ++i) {
            int len1 = expand(i, i);
            int len2 = expand(i, i + 1);
            int curr_len = std::max(len1, len2);

            if (curr_len > max_len) {
                max_len = curr_len;
                start = i - (curr_len - 1) / 2;
            }
        }

        return s.substr(start, max_len);
    }
};
```

#### 3. Java (Modern, Typed — Expand Around Center $\mathcal{O}(1)$ Space)
```java
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.length() <= 1) return s;

        int n = s.length();
        int start = 0;
        int maxLen = 1;

        for (int i = 0; i < n; i++) {
            int len1 = expand(s, i, i);
            int len2 = expand(s, i, i + 1);
            int currLen = Math.max(len1, len2);

            if (currLen > maxLen) {
                maxLen = currLen;
                start = i - (currLen - 1) / 2;
            }
        }

        return s.substring(start, start + maxLen);
    }

    private int expand(String s, int left, int right) {
        int n = s.length();
        while (left >= 0 && right < n && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N^2)$  
  There are $2N - 1$ centers. In the worst case (e.g. string of all identical characters `"aaaa"`), expanding outward from each center takes $\mathcal{O}(N)$ comparisons, yielding $\mathcal{O}(N^2)$ total time. In practice, non-palindromic characters mismatch early, making this solution extremely fast ($< 15$ ms).
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space  
  Unlike 2D DP which consumes $\mathcal{O}(N^2)$ heap memory, Expand Around Center requires only index pointers.

---

### Takeaway Pattern & Interview Traps

1. **Center Calculation Formula:**
   - For a palindrome of length $L$ centered around index $i$:
     $$\text{start} = i - \left\lfloor \frac{L - 1}{2} \right\rfloor$$
   - This single unified formula works identically for both odd and even lengths!
     - Odd: $i = 1, L = 3 \implies \text{start} = 1 - (2 // 2) = 0$.
     - Even: $i = 1, L = 4 \implies \text{start} = 1 - (3 // 2) = 0$.
2. **Linear Time (Manacher's Algorithm):**
   - For competitive programming or follow-up discussions, Manacher's Algorithm achieves strictly $\mathcal{O}(N)$ time by exploiting previously computed symmetry boundaries. However, in standard coding interviews, Expand Around Center is the industry-preferred solution due to its brevity and bug-free implementation.