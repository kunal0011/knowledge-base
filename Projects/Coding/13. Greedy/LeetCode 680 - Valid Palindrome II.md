---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 680: Valid Palindrome II"
tags:
  - leetcode
  - coding
  - greedy
  - two-pointers
  - string
  - amazon
  - google
---

# LeetCode 680: Valid Palindrome II

**Target Companies:** Meta (Signature Top 1), Amazon, Google, Microsoft, Apple, Bloomberg  
**Difficulty:** Easy  
**Topic:** Greedy / Two Pointers / String  

---

### Problem Statement

Given a string `s`, return `true` if the `s` can be palindrome after deleting **at most one** character from it.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 10^5$, lowercase English letters).
- **Output:**
  - `bool` — `true` if `s` can become a palindrome by removing 0 or 1 character, else `false`.
- **Constraints:**
  - $1 \le \text{s.length} \le 10^5$
  - `s` consists of lowercase English letters.

---

### Key Idea & Intuition

In a standard two-pointer palindrome check:
- Maintain pointers $l = 0$ and $r = n - 1$.
- While $s[l] == s[r]$, both characters match symmetrically and must be preserved. We increment $l \mathrel{+}= 1$ and decrement $r \mathrel{-}= 1$.

#### The Greedy Branching on First Mismatch:
When we encounter the **first mismatch** ($s[l] \ne s[r]$):
- Because we are allowed to delete **at most one character** across the entire string, the deleted character **must** be either $s[l]$ or $s[r]$!
  - If we delete any other character inside the string, the mismatch between $s[l]$ and $s[r]$ would remain unresolved.
- Thus, there are only two possible paths:
  1. **Delete $s[l]$:** Check if the remaining substring $s[l + 1 \dots r]$ is a pure palindrome.
  2. **Delete $s[r]$:** Check if the remaining substring $s[l \dots r - 1]$ is a pure palindrome.
- If either candidate substring is a palindrome, return `True`.
- If neither is a palindrome, then no single deletion can fix the string $\rightarrow$ return `False`.

Because we branch into at most two linear verification checks that never branch again, the runtime remains strictly $\mathcal{O}(n)$ with $\mathcal{O}(1)$ space.

---

### Solution Approach (Step-by-Step)

1. Helper function `is_palindrome_range(i, j)`:
   - While $i < j$:
     - If `s[i] != s[j]`: return `False`
     - `i += 1`, `j -= 1`
   - Return `True`.
2. Initialize `l = 0`, `r = len(s) - 1`.
3. While `l < r`:
   - If `s[l] == s[r]`:
     - `l += 1`, `r -= 1`
   - Else:
     - Return `is_palindrome_range(l + 1, r) or is_palindrome_range(l, r - 1)`.
4. Return `True` (already a palindrome without any deletions).

---

### Visual Algorithm Walkthrough

For `s = "abca"`:

```
l = 0 ('a'), r = 3 ('a'):
  s[0] == s[3] ('a' == 'a') -> Match!
  l = 1, r = 2

l = 1 ('b'), r = 2 ('c'):
  s[1] != s[2] ('b' != 'c') -> MISMATCH!

Branch 1 (Delete s[l] = 'b'):
  Check substring s[2..2] = "c":
  Single character -> trivially a palindrome! -> TRUE!

Branch 1 succeeded -> Return True immediately.
(Result: removing 'b' yields "aca", or removing 'c' yields "aba").
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "aba"`
- **Output:** `true` (Already a palindrome)

#### Example 2:
- **Input:** `s = "abca"`
- **Output:** `true` (Delete 'b' or 'c')

#### Example 3:
- **Input:** `s = "abc"`
- **Tracing:** Mismatch at (0, 2). Deleting 'a' leaves "bc" (not palindrome). Deleting 'c' leaves "ab" (not palindrome).
- **Output:** `false`

#### Example 4 (Near Center Mismatch):
- **Input:** `s = "deeee"`
- **Tracing:** $s[0]='d' \ne s[4]='e'$. Deleting 'd' leaves "eeee" (palindrome).
- **Output:** `true`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def is_palindrome_range(i: int, j: int) -> bool:
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
            
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                # Try skipping left character or right character
                return is_palindrome_range(l + 1, r) or is_palindrome_range(l, r - 1)
            l += 1
            r -= 1
            
        return True
```

#### C++17
```cpp
#include <string>

class Solution {
private:
    bool isPalindromeRange(const std::string& s, int i, int j) {
        while (i < j) {
            if (s[i] != s[j]) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }

public:
    bool validPalindrome(const std::string& s) {
        int l = 0;
        int r = static_cast<int>(s.size()) - 1;
        
        while (l < r) {
            if (s[l] != s[r]) {
                return isPalindromeRange(s, l + 1, r) || isPalindromeRange(s, l, r - 1);
            }
            l++;
            r--;
        }
        
        return true;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean validPalindrome(String s) {
        int l = 0;
        int r = s.length() - 1;
        
        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) {
                return isPalindromeRange(s, l + 1, r) || isPalindromeRange(s, l, r - 1);
            }
            l++;
            r--;
        }
        
        return true;
    }
    
    private boolean isPalindromeRange(String s, int i, int j) {
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - The main loop traverses inward until the first mismatch.
  - The two helper palindrome checks scan at most the remaining substring of length $< n$.
  - At most $2n$ comparisons total are made.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Passing indices avoids substring memory allocations.

---

### Takeaway Pattern & Interview Traps

- **Why We Don't Need DP for At Most 1 Deletion:** Since at most 1 deletion is allowed, we only branch once at the first point of divergence. Branching at most once gives 2 linear scans ($\mathcal{O}(2n) = \mathcal{O}(n)$), completely avoiding $O(n^2)$ edit-distance DP!
- **Index Slicing vs Pointer Range:** In Python, doing `s[l+1:r+1] == s[l+1:r+1][::-1]` allocates strings in memory $\mathcal{O}(n)$ space. Using helper index functions keeps auxiliary space strictly $\mathcal{O}(1)$.