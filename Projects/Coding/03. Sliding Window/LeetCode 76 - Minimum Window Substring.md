---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 76: Minimum Window Substring"
tags:
  - leetcode
  - coding
  - sliding-window
  - hash-table
  - string
  - amazon
  - google
---

# LeetCode 76: Minimum Window Substring

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber, Airbnb  
**Difficulty:** Hard  
**Topic:** Sliding Window / Hash Table / String  

---

### Problem Statement

Given two strings `s` and `t` of lengths `m` and `n` respectively, return the **minimum window substring** of `s` such that every character in `t` (**including duplicates**) is included in the window. If there is no such substring, return the empty string `""`.

The testcases will be generated such that the answer is **unique**.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 10^5$).
  - `t`: `str` / `string` ($1 \le |t| \le 10^5$).
- **Output:**
  - `str` / `string` — the shortest contiguous substring of `s` containing all characters of `t` with their required counts, or `""` if impossible.
- **Constraints:**
  - $m == \text{s.length}$, $n == \text{t.length}$
  - $1 \le m, n \le 10^5$
  - `s` and `t` consist of uppercase and lowercase English letters.

---

### Key Idea & Intuition

The task requires finding the shortest contiguous window $[l, r]$ in `s` such that for every character $c \in t$, $\text{count}_{window}(c) \ge \text{count}_{t}(c)$.

#### Two-Pointer Sliding Window with `formed` Counter:
1. Precompute the target frequency map `target_counts` of characters in string `t`.
2. Let `required = len(target_counts)` be the number of **unique** characters in `t` whose frequency requirements must be met.
3. Maintain a dynamic sliding window $[l, r]$ across `s`, and a frequency map `window_counts` of characters currently inside the window.
4. Maintain an integer `formed`, tracking how many distinct characters in the current window meet or exceed their required target frequency:
   - When introducing $s[r]$ into the window:
     - `window_counts[s[r]] += 1`
     - If $s[r]$ is in `target_counts` and `window_counts[s[r]] == target_counts[s[r]]`:
       - `formed += 1`
5. Once the window is **valid** (`formed == required`):
   - Record the window substring if its length $r - l + 1$ is smaller than our previous minimum.
   - **Shrink from the left:** Evict $s[l]$:
     - If $s[l]$ is in `target_counts` and `window_counts[s[l]] == target_counts[s[l]]`:
       - `formed -= 1`
     - `window_counts[s[l]] -= 1`
     - `l += 1`
   - Repeat while `formed == required`.
6. Both $l$ and $r$ traverse $s$ at most once $\implies \mathcal{O}(|s| + |t|)$ time.

---

### Solution Approach (Step-by-Step)

1. If $|s| < |t|$ or $|t| == 0$, return `""`.
2. Compute `target = Counter(t)` and `required = len(target)`.
3. Initialize `window = {}`, `formed = 0`, `l = 0`.
4. Initialize `min_len = infinity`, `best_l = 0`, `best_r = 0`.
5. For `r, char` in `enumerate(s)`:
   - `window[char] = window.get(char, 0) + 1`
   - If `char in target and window[char] == target[char]`:
     - `formed += 1`
   - While `l <= r and formed == required`:
     - If `r - l + 1 < min_len`:
       - `min_len = r - l + 1`
       - `best_l = l`
       - `best_r = r`
     - Evict `s[l]`:
       - If `s[l] in target and window[s[l]] == target[s[l]]`:
         - `formed -= 1`
       - `window[s[l]] -= 1`
       - `l += 1`
6. Return `""` if `min_len == infinity` else `s[best_l : best_r + 1]`.

---

### Visual Algorithm Walkthrough

For `s = "ADOBECODEBANC"`, `t = "ABC"`:
`target = {'A': 1, 'B': 1, 'C': 1}`, `required = 3`

```
r=0 ('A'): window={'A':1}, formed=1
r=1 ('D'): window={'A':1, 'D':1}, formed=1
r=2 ('O'): window={'A':1, 'D':1, 'O':1}, formed=1
r=3 ('B'): window={'A':1, 'D':1, 'O':1, 'B':1}, formed=2
r=4 ('E'): ... formed=2
r=5 ('C'): window includes 'A':1, 'B':1, 'C':1 -> formed=3 == required!
  Valid window [0..5] ("ADOBEC"), length 6. min_len = 6.
  Shrink left:
    Evict s[0]='A': formed drops to 2. left=1. Window invalid.

... r advances to 10 ('A'), then 12 ('C'):
At r = 12 ('C'):
  Window covers [9..12] -> "BANC"
  Contains 'B':1, 'A':1, 'N':1, 'C':1 -> formed = 3!
  Valid window [9..12], length 4 < 6.
  min_len = 4, best substring = "BANC".
  Shrink left:
    Evict s[9]='B': formed drops to 2. left=10.

End of string.
Result = "BANC".
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "ADOBECODEBANC"`, `t = "ABC"`
- **Output:** `"BANC"`

#### Example 2:
- **Input:** `s = "a"`, `t = "a"`
- **Output:** `"a"`

#### Example 3:
- **Input:** `s = "a"`, `t = "aa"`
- **Output:** `""` (Window must contain 2 'a's, but `s` only has 1)

---

### Multi-Language Implementations

#### Python 3
```python
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t or len(s) < len(t):
            return ""
            
        target = Counter(t)
        required = len(target)
        
        window = {}
        formed = 0
        left = 0
        
        min_len = float("inf")
        best_left = 0
        best_right = 0
        
        for right, char in enumerate(s):
            window[char] = window.get(char, 0) + 1
            
            if char in target and window[char] == target[char]:
                formed += 1
                
            # Contract window from left while maintaining validity
            while left <= right and formed == required:
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    best_left = left
                    best_right = right
                    
                left_char = s[left]
                if left_char in target and window[left_char] == target[left_char]:
                    formed -= 1
                window[left_char] -= 1
                left += 1
                
        return "" if min_len == float("inf") else s[best_left : best_right + 1]
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <climits>

class Solution {
public:
    std::string minWindow(const std::string& s, const std::string& t) {
        int m = static_cast<int>(s.size());
        int n = static_cast<int>(t.size());
        if (m < n || n == 0) return "";
        
        std::vector<int> target(128, 0);
        int required = 0;
        for (char c : t) {
            if (target[static_cast<unsigned char>(c)] == 0) {
                required++;
            }
            target[static_cast<unsigned char>(c)]++;
        }
        
        std::vector<int> window(128, 0);
        int formed = 0;
        int left = 0;
        int min_len = INT_MAX;
        int best_left = 0;
        
        for (int right = 0; right < m; ++right) {
            unsigned char r_char = static_cast<unsigned char>(s[right]);
            window[r_char]++;
            
            if (target[r_char] > 0 && window[r_char] == target[r_char]) {
                formed++;
            }
            
            while (left <= right && formed == required) {
                if (right - left + 1 < min_len) {
                    min_len = right - left + 1;
                    best_left = left;
                }
                
                unsigned char l_char = static_cast<unsigned char>(s[left]);
                if (target[l_char] > 0 && window[l_char] == target[l_char]) {
                    formed--;
                }
                window[l_char]--;
                left++;
            }
        }
        
        return min_len == INT_MAX ? "" : s.substr(best_left, min_len);
    }
};
```

#### Java 17
```java
class Solution {
    public String minWindow(String s, String t) {
        if (s == null || t == null || s.length() < t.length() || t.length() == 0) {
            return "";
        }
        
        int[] target = new int[128];
        int required = 0;
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (target[c] == 0) {
                required++;
            }
            target[c]++;
        }
        
        int[] window = new int[128];
        int formed = 0;
        int left = 0;
        int minLen = Integer.MAX_VALUE;
        int bestLeft = 0;
        
        for (int right = 0; right < s.length(); right++) {
            char rChar = s.charAt(right);
            window[rChar]++;
            
            if (target[rChar] > 0 && window[rChar] == target[rChar]) {
                formed++;
            }
            
            while (left <= right && formed == required) {
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    bestLeft = left;
                }
                
                char lChar = s.charAt(left);
                if (target[lChar] > 0 && window[lChar] == target[lChar]) {
                    formed--;
                }
                window[lChar]--;
                left++;
            }
        }
        
        return minLen == Integer.MAX_VALUE ? "" : s.substring(bestLeft, bestLeft + minLen);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(|s| + |t|)$
  - Building the frequency table for `t` takes $\mathcal{O}(|t|)$ time.
  - The pointers `left` and `right` traverse string `s` at most once each.
  - Frequency table queries and comparisons are $\mathcal{O}(1)$ with array / hash table.
  - Total time is strictly linear $\mathcal{O}(|s| + |t|)$.
- **Space Complexity:** $\mathcal{O}(|\Sigma|)$ auxiliary space
  - Where $|\Sigma| \le 128$ is the size of the ASCII alphabet (letters and casing).

---

### Takeaway Pattern & Interview Traps

- **The `formed == required` Counter Trick:** Comparing two hash maps at each step takes $\mathcal{O}(|\Sigma|)$. By tracking the exact count of unique characters whose condition is met via `formed`, checking window validity takes $\mathcal{O}(1)$ time.
- **Strict Equality for Counter Triggers:**
  - `if window[char] == target[char]: formed += 1`
  - `if window[char] == target[char]: formed -= 1`
  Triggering only on **exact equality** ensures `formed` increments and decrements exactly once per character type regardless of excess occurrences!