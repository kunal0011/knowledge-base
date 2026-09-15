---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 424: Longest Repeating Character Replacement"
tags:
  - leetcode
  - coding
  - sliding-window
  - string
  - hash-table
  - amazon
  - google
---

# LeetCode 424: Longest Repeating Character Replacement

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber, Bloomberg  
**Difficulty:** Medium  
**Topic:** Sliding Window / Hash Table / String  

---

### Problem Statement

You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most `k` times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 10^5$, consists of uppercase English letters).
  - `k`: `int` ($0 \le k \le |s|$).
- **Output:**
  - `int` — maximum length of a contiguous substring containing identical characters after at most $k$ replacements.
- **Constraints:**
  - $1 \le \text{s.length} \le 10^5$
  - `s` consists of only uppercase English letters.
  - $0 \le k \le \text{s.length}$

---

### Key Idea & Intuition

#### Window Validity Invariant:
Consider any substring of length $L = r - l + 1$.
If we want to make all characters in this window identical, the optimal strategy is to keep the most frequent character in this window unchanged, and replace all other characters.
Let `max_freq` be the frequency of the most common character in the window $[l, r]$.
The number of characters requiring replacement is:
$$\text{replacements} = (r - l + 1) - \text{max\_freq}$$

The window is **valid** if and only if:
$$(r - l + 1) - \text{max\_freq} \le k$$

#### The Monotonic `max_freq` Trick ($\mathcal{O}(n)$ without rescanning 26 chars):
When a window becomes invalid ($(r - l + 1) - \text{max\_freq} > k$), we advance the left pointer $l$.
Do we need to recompute `max_freq` by scanning all 26 character counts after evicting $s[l]$?
**No!**
Here is why:
- We are searching for the **maximum** window length.
- A new window can only beat our previous best length if its `max_freq` is strictly larger than the historical `max_freq` we have already seen.
- If the true maximum frequency in the shrunken window is smaller, the window cannot possibly break the global record anyway.
- Therefore, we only update `max_freq = max(max_freq, count[s[r]])`. We never need to decrement `max_freq`.
This guarantees that each character is processed in $\mathcal{O}(1)$ time, yielding an optimal $\mathcal{O}(n)$ algorithm.

---

### Solution Approach (Step-by-Step)

1. Maintain an integer array `freq = [0] * 26` to store frequencies of characters in the current window.
2. Initialize `left = 0`, `max_freq = 0`, and `max_len = 0`.
3. Loop `right` from $0$ to $n - 1$:
   - Increment `freq[s[right] - 'A'] += 1`.
   - Update `max_freq = max(max_freq, freq[s[right] - 'A'])`.
   - Check if current window $[left, right]$ violates the condition:
     $$\text{window\_len} - \text{max\_freq} > k$$
   - If invalid:
     - Decrement `freq[s[left] - 'A'] -= 1`.
     - Increment `left += 1`.
   - Update `max_len = max(max_len, right - left + 1)`.
4. Return `max_len` (or simply `n - left` in non-shrinking window variant).

---

### Visual Algorithm Walkthrough

For `s = "AABABBA"`, `k = 1`:

```
Indices:   0   1   2   3   4   5   6
String:    A   A   B   A   B   B   A

r=0 ('A'): freq={A:1}, max_freq=1. len=1, 1-1=0 <= 1. Valid. max_len=1
r=1 ('A'): freq={A:2}, max_freq=2. len=2, 2-2=0 <= 1. Valid. max_len=2
r=2 ('B'): freq={A:2, B:1}, max_freq=2. len=3, 3-2=1 <= 1. Valid. max_len=3
r=3 ('A'): freq={A:3, B:1}, max_freq=3. len=4, 4-3=1 <= 1. Valid. max_len=4 ("AABA")
r=4 ('B'): freq={A:3, B:2}, max_freq=3. len=5, 5-3=2 > 1. INVALID!
  Evict s[0]='A': freq={A:2, B:2}, left becomes 1.
  Window [1..4] ("ABAB"), len=4.
r=5 ('B'): freq={A:2, B:3}, max_freq=3. len=5, 5-3=2 > 1. INVALID!
  Evict s[1]='A': freq={A:1, B:3}, left becomes 2.
  Window [2..5] ("BABB"), len=4.
r=6 ('A'): freq={A:2, B:3}, max_freq=3. len=5, 5-3=2 > 1. INVALID!
  Evict s[2]='B': freq={A:2, B:2}, left becomes 3.
  Window [3..6] ("ABBA"), len=4.

Loop finishes.
Final Result = 4 (e.g., "AABA" with 1 replacement -> "AAAA").
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "ABAB"`, `k = 2`
- **Output:** `4` (Replace both 'A's or both 'B's to form "BBBB" or "AAAA")

#### Example 2:
- **Input:** `s = "AABABBA"`, `k = 1`
- **Output:** `4`

#### Example 3 ($k = 0$, Longest Run of Identical Characters):
- **Input:** `s = "ABBCDDDDE"`, `k = 0`
- **Output:** `4` (Substring "DDDD")

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        freq = [0] * 26
        left = 0
        max_freq = 0
        max_len = 0
        
        for right, char in enumerate(s):
            c_idx = ord(char) - ord('A')
            freq[c_idx] += 1
            max_freq = max(max_freq, freq[c_idx])
            
            # If remaining characters to change exceed k, slide left forward
            if (right - left + 1) - max_freq > k:
                freq[ord(s[left]) - ord('A')] -= 1
                left += 1
                
            max_len = max(max_len, right - left + 1)
            
        return max_len
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int characterReplacement(const std::string& s, int k) {
        std::vector<int> freq(26, 0);
        int left = 0;
        int max_freq = 0;
        int max_len = 0;
        int n = static_cast<int>(s.size());
        
        for (int right = 0; right < n; ++right) {
            int c_idx = s[right] - 'A';
            freq[c_idx]++;
            max_freq = std::max(max_freq, freq[c_idx]);
            
            // Window is invalid when non-majority characters exceed k
            if ((right - left + 1) - max_freq > k) {
                freq[s[left] - 'A']--;
                left++;
            }
            
            max_len = std::max(max_len, right - left + 1);
        }
        
        return max_len;
    }
};
```

#### Java 17
```java
class Solution {
    public int characterReplacement(String s, int k) {
        int[] freq = new int[26];
        int left = 0;
        int maxFreq = 0;
        int maxLen = 0;
        int n = s.length();
        
        for (int right = 0; right < n; right++) {
            int cIdx = s.charAt(right) - 'A';
            freq[cIdx]++;
            maxFreq = Math.max(maxFreq, freq[cIdx]);
            
            // If characters needing replacement > k, shift left pointer
            if ((right - left + 1) - maxFreq > k) {
                freq[s.charAt(left) - 'A']--;
                left++;
            }
            
            maxLen = Math.max(maxLen, right - left + 1);
        }
        
        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - The right pointer iterates across the string of length $n$ once.
  - The left pointer advances at most $n$ times.
  - Frequency array updates and `max_freq` comparisons are $\mathcal{O}(1)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The frequency table size is fixed at $26$ uppercase English characters.

---

### Takeaway Pattern & Interview Traps

- **Why We Never Need to Decrement `max_freq`:** A classic interview follow-up question is: *"Why don't we recompute `max_freq` when evicting `s[left]`?"* The answer is that a smaller `max_freq` could only validate a *shorter* window, but we only care about achieving a window *strictly larger* than our previous best. Hence, keeping an optimistic upper bound on `max_freq` is completely safe and saves $\mathcal{O}(26)$ per shrink!
- **`if` vs `while` for Shrinking:** Because `right` increments by 1 in each step, the window length grows by at most 1. Thus, at most 1 invalid state can occur per step, meaning a single `if` statement suffices rather than a `while` loop.