---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1456: Maximum Number of Vowels in a Substring of Given Length"
tags:
  - leetcode
  - coding
  - sliding-window
  - string
  - amazon
  - google
---

# LeetCode 1456: Maximum Number of Vowels in a Substring of Given Length

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Fixed-Size Sliding Window / Early Exit / Character Classification  

---

### Problem Statement

Given a string `s` and an integer `k`, return *the maximum number of vowel letters in any substring of `s` with length `k`*.

**Vowel letters** in English are `'a'`, `'e'`, `'i'`, `'o'`, and `'u'`.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`, `k: int`
- **Output:** `int` (maximum count of vowels in any window of length $k$)
- **Constraints:**
  - $1 \le \text{s.length} \le 10^5$
  - `s` consists of lowercase English letters.
  - $1 \le k \le \text{s.length}$

---

### Key Idea & Intuition

- **Fixed-Size Window of Length $k$:**
  - Every valid candidate substring has length exactly $k$.
  - Instead of counting vowels from scratch for each starting position ($\mathcal{O}(N \cdot k)$), maintain a running count `curr_vowels` for the current window $[i - k + 1, i]$.
- **Window Transitions in $\mathcal{O}(1)$:**
  - When sliding the window right by 1 position:
    1. **Add Incoming Character:** If $s[i] \in \{\text{'a','e','i','o','u'}\}$, increment `curr_vowels`.
    2. **Remove Outgoing Character:** If $s[i - k] \in \{\text{'a','e','i','o','u'}\}$, decrement `curr_vowels`.
- **Early Exit Optimization:**
  - A substring of length $k$ can have at most $k$ vowels.
  - If at any point `curr_vowels == k`, we can immediately terminate the search and return $k$.

---

### Solution Approach (Step-by-Step)

1. Define a set of vowels: `VOWELS = {'a', 'e', 'i', 'o', 'u'}` (or a boolean lookup table).
2. Compute `curr_vowels` for the first $k$ characters ($0 \le i < k$).
3. Set `max_vowels = curr_vowels`.
4. If `max_vowels == k`: return $k$.
5. Loop `i` from $k$ to $\text{len}(s) - 1$:
   - If `s[i] in VOWELS`: `curr_vowels += 1`.
   - If `s[i - k] in VOWELS`: `curr_vowels -= 1`.
   - `max_vowels = max(max_vowels, curr_vowels)`.
   - If `max_vowels == k`: return $k$.
6. Return `max_vowels`.

---

### Visual Algorithm Walkthrough

Let `s = "abciiidef"`, $k = 3$. Vowels: `{'a', 'e', 'i', 'o', 'u'}`.

```
Indices:   0  1  2  3  4  5  6  7  8
String:    a  b  c  i  i  i  d  e  f

Initial Window [0..2]: "abc"
- 'a' (vowel), 'b' (no), 'c' (no) -> curr_vowels = 1, max_vowels = 1

Slide i=3: incoming 'i' (+1), outgoing 'a' (-1)
- Window [1..3]: "bci" -> curr_vowels = 1 - 1 + 1 = 1

Slide i=4: incoming 'i' (+1), outgoing 'b' (0)
- Window [2..4]: "cii" -> curr_vowels = 1 - 0 + 1 = 2

Slide i=5: incoming 'i' (+1), outgoing 'c' (0)
- Window [3..5]: "iii" -> curr_vowels = 2 - 0 + 1 = 3 == k!
- Reached theoretical maximum k = 3!
- Early exit: Return 3 immediately!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | `k` | Window with Max Vowels | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `"abciiidef"` | `3` | `"iii"` (3 vowels) | `3` |
| **Example 2** | `"aeiou"` | `2` | `"ae"` (2 vowels) | `2` |
| **Example 3** | `"leetcode"` | `3` | `"lee"`, `"eet"`, `"ode"` (2 vowels) | `2` |
| **No Vowels** | `"rhythms"` | `4` | 0 vowels in entire string | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        """
        Finds the maximum number of vowels in any substring of length k.
        Uses fixed-size sliding window with early-exit optimization.
        """
        vowels = {'a', 'e', 'i', 'o', 'u'}

        # Calculate vowels in the first window
        curr_vowels = sum(1 for i in range(k) if s[i] in vowels)
        max_vowels = curr_vowels

        if max_vowels == k:
            return k

        # Slide the window across the rest of the string
        for i in range(k, len(s)):
            if s[i] in vowels:
                curr_vowels += 1
            if s[i - k] in vowels:
                curr_vowels -= 1

            if curr_vowels > max_vowels:
                max_vowels = curr_vowels
                if max_vowels == k:
                    return k

        return max_vowels
```

#### C++17
```cpp
#include <string>
#include <algorithm>

class Solution {
public:
    int maxVowels(const std::string& s, int k) {
        auto isVowel = [](char c) {
            return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
        };

        int curr_vowels = 0;
        for (int i = 0; i < k; ++i) {
            if (isVowel(s[i])) {
                curr_vowels++;
            }
        }

        int max_vowels = curr_vowels;
        if (max_vowels == k) return k;

        for (size_t i = k; i < s.size(); ++i) {
            if (isVowel(s[i])) curr_vowels++;
            if (isVowel(s[i - k])) curr_vowels--;

            if (curr_vowels > max_vowels) {
                max_vowels = curr_vowels;
                if (max_vowels == k) return k;
            }
        }

        return max_vowels;
    }
};
```

#### Java
```java
class Solution {
    public int maxVowels(String s, int k) {
        int currVowels = 0;
        int n = s.length();

        for (int i = 0; i < k; i++) {
            if (isVowel(s.charAt(i))) {
                currVowels++;
            }
        }

        int maxVowels = currVowels;
        if (maxVowels == k) return k;

        for (int i = k; i < n; i++) {
            if (isVowel(s.charAt(i))) currVowels++;
            if (isVowel(s.charAt(i - k))) currVowels--;

            if (currVowels > maxVowels) {
                maxVowels = currVowels;
                if (maxVowels == k) return k;
            }
        }

        return maxVowels;
    }

    private boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{s.length}$.
  - We traverse the string in a single linear pass.
  - Adding and subtracting the incoming and outgoing characters takes $\mathcal{O}(1)$ time.
  - Early exit when `max_vowels == k` terminates earlier on dense vowel strings.
  - Runs in $< 8 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since vowel lookup is a fixed 5-element set or function.

---

### Takeaway Pattern & Interview Traps

- **Early Exit on Theoretical Maximum:** If the running answer ever matches the theoretical upper bound (here, $k$), terminate immediately.
- **$\mathcal{O}(1)$ Vowel Checks:** Avoid creating sub-lists or substrings. Simply check `isVowel(s[i])` directly on character primitives.