---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 395: Longest Substring with At Least K Repeating Characters"
tags:
  - leetcode
  - coding
  - sliding-window
  - divide-and-conquer
  - string
  - amazon
  - google
---

# LeetCode 395: Longest Substring with At Least K Repeating Characters

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Baidu  
**Difficulty:** Medium  
**Topic:** Sliding Window / Divide and Conquer / String  

---

### Problem Statement

Given a string `s` and an integer `k`, return the length of the longest substring of `s` such that the frequency of each character in this substring is greater than or equal to `k`.

If no such substring exists, return `0`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 10^4$)
  - `k`: `int` ($1 \le k \le 10^5$)
- **Output:**
  - `int` — the maximum length of a valid substring where every character present appears $\ge k$ times.
- **Constraints:**
  - $1 \le \text{s.length} \le 10^4$
  - `s` consists of only lowercase English letters.
  - $1 \le k \le 10^5$

---

### Key Idea & Intuition

At first glance, a direct sliding window seems inapplicable because the condition "every unique character has frequency $\ge k$" is **non-monotonic**: expanding the window may introduce a new character and break the property, while shrinking might drop a character below $k$.

There are two primary optimal paradigms to solve this:

#### Approach 1: Divide and Conquer (Splitting on Invalid Characters)
1. Count the frequency of all characters in the current substring $s$.
2. Any character $c$ whose total frequency in $s$ is $< k$ **can never** be part of any valid substring.
3. Therefore, $c$ acts as an impassable barrier (delimiter). We split $s$ at occurrences of $c$, and recursively solve for each contiguous chunk between delimiters.
4. If no character has frequency $< k$, the entire string $s$ is valid, and its length is the answer.

#### Approach 2: Sliding Window by Constrained Unique Character Count
Why did standard sliding window fail? Because the number of unique characters in the window was unconstrained.
If we **fix the target number of unique characters** $u \in [1, 26]$:
- We maintain a sliding window $[l, r]$ containing at most $u$ unique characters.
- When `unique_chars > u`, we advance $l$ until `unique_chars <= u`.
- We maintain:
  - `unique_count`: number of distinct characters in the current window.
  - `count_at_least_k`: number of distinct characters in the current window whose frequency is $\ge k$.
- Whenever `unique_count == u` and `count_at_least_k == u`, every character in the window appears at least $k$ times! We update `max_len = max(max_len, r - l + 1)`.
- Repeating this for all possible unique targets $u \in [1, 26]$ yields a strict $\mathcal{O}(26 \cdot n) = \mathcal{O}(n)$ time and $\mathcal{O}(1)$ space solution.

---

### Solution Approach (Step-by-Step: Constrained Sliding Window)

1. Find the total number of distinct characters in string $s$, say `max_unique`.
2. Initialize `max_len = 0`.
3. For `target_unique` from $1$ to `max_unique`:
   - Initialize `counts = [0] * 26`, `unique_count = 0`, `count_at_least_k = 0`, and `l = 0`.
   - For `r` from $0$ to $n - 1$:
     - Let `c = s[r]`. If `counts[c] == 0`, increment `unique_count += 1`.
     - Increment `counts[c] += 1`.
     - If `counts[c] == k`, increment `count_at_least_k += 1`.
     - While `unique_count > target_unique`:
       - Let `left_c = s[l]`.
       - If `counts[left_c] == k`, decrement `count_at_least_k -= 1`.
       - Decrement `counts[left_c] -= 1`.
       - If `counts[left_c] == 0`, decrement `unique_count -= 1`.
       - Increment `l += 1`.
     - If `unique_count == target_unique` and `unique_count == count_at_least_k`:
       - `max_len = max(max_len, r - l + 1)`.
4. Return `max_len`.

---

### Visual Algorithm Walkthrough

For `s = "aaabb"`, `k = 3`:

```
Distinct characters in s: {'a', 'b'} -> target_unique in [1, 2]

--- Target Unique = 1 ---
r=0 ('a'): counts['a']=1, unique=1, >=k: 0
r=1 ('a'): counts['a']=2, unique=1, >=k: 0
r=2 ('a'): counts['a']=3, unique=1, >=k: 1 ('a' >= 3)
  unique == 1 and count_at_least_k == 1 -> Valid window [0..2] ("aaa"), len = 3!
r=3 ('b'): counts['b']=1, unique=2 > 1 -> Shrink left until unique <= 1:
  Evicts 'a', 'a', 'a'. left reaches 3.
  Window is "b", len 1, count_at_least_k = 0.
r=4 ('b'): counts['b']=2, unique=1, count_at_least_k = 0.
Max len for target 1 = 3.

--- Target Unique = 2 ---
Window expands over 'a', 'a', 'a', 'b', 'b':
At r=4:
  counts['a'] = 3 (>= 3)
  counts['b'] = 2 (< 3)
  unique = 2, count_at_least_k = 1 != 2 (b has only 2 occurrences).
  No valid window of 2 unique chars where both have count >= 3.

Global Result: max_len = 3 ("aaa").
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "aaabb"`, `k = 3`
- **Output:** `3` (Substrings: `"aaa"`)

#### Example 2:
- **Input:** `s = "ababbc"`, `k = 2`
- **Tracing:**
  - 'c' appears only 1 time ($< 2$), so 'c' splits the string into `"ababb"`.
  - In `"ababb"`, 'a' appears 2 times, 'b' appears 3 times. Both $\ge 2$. Length is 5.
- **Output:** `5`

#### Example 3 (No character meets requirement):
- **Input:** `s = "abcdef"`, `k = 2`
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3 (Constrained Sliding Window)
```python
from collections import Counter

class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        max_unique = len(set(s))
        max_len = 0
        n = len(s)
        
        for target_unique in range(1, max_unique + 1):
            counts = [0] * 26
            unique_count = 0
            count_at_least_k = 0
            left = 0
            
            for right in range(n):
                c_idx = ord(s[right]) - ord('a')
                if counts[c_idx] == 0:
                    unique_count += 1
                counts[c_idx] += 1
                if counts[c_idx] == k:
                    count_at_least_k += 1
                    
                while unique_count > target_unique:
                    left_idx = ord(s[left]) - ord('a')
                    if counts[left_idx] == k:
                        count_at_least_k -= 1
                    counts[left_idx] -= 1
                    if counts[left_idx] == 0:
                        unique_count -= 1
                    left += 1
                    
                if unique_count == target_unique and unique_count == count_at_least_k:
                    max_len = max(max_len, right - left + 1)
                    
        return max_len
```

#### C++17 (Constrained Sliding Window)
```cpp
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>

class Solution {
public:
    int longestSubstring(const std::string& s, int k) {
        int n = static_cast<int>(s.size());
        std::unordered_set<char> unique_chars(s.begin(), s.end());
        int max_unique = static_cast<int>(unique_chars.size());
        int max_len = 0;
        
        for (int target_unique = 1; target_unique <= max_unique; ++target_unique) {
            std::vector<int> counts(26, 0);
            int unique_count = 0;
            int count_at_least_k = 0;
            int left = 0;
            
            for (int right = 0; right < n; ++right) {
                int r_idx = s[right] - 'a';
                if (counts[r_idx] == 0) ++unique_count;
                counts[r_idx]++;
                if (counts[r_idx] == k) ++count_at_least_k;
                
                while (unique_count > target_unique) {
                    int l_idx = s[left] - 'a';
                    if (counts[l_idx] == k) --count_at_least_k;
                    counts[l_idx]--;
                    if (counts[l_idx] == 0) --unique_count;
                    ++left;
                }
                
                if (unique_count == target_unique && unique_count == count_at_least_k) {
                    max_len = std::max(max_len, right - left + 1);
                }
            }
        }
        
        return max_len;
    }
};
```

#### Java 17 (Constrained Sliding Window)
```java
import java.util.HashSet;
import java.util.Set;

class Solution {
    public int longestSubstring(String s, int k) {
        Set<Character> uniqueChars = new HashSet<>();
        for (char c : s.toCharArray()) {
            uniqueChars.add(c);
        }
        int maxUnique = uniqueChars.size();
        int maxLen = 0;
        int n = s.length();
        
        for (int targetUnique = 1; targetUnique <= maxUnique; targetUnique++) {
            int[] counts = new int[26];
            int uniqueCount = 0;
            int countAtLeastK = 0;
            int left = 0;
            
            for (int right = 0; right < n; right++) {
                int rIdx = s.charAt(right) - 'a';
                if (counts[rIdx] == 0) uniqueCount++;
                counts[rIdx]++;
                if (counts[rIdx] == k) countAtLeastK++;
                
                while (uniqueCount > targetUnique) {
                    int lIdx = s.charAt(left) - 'a';
                    if (counts[lIdx] == k) countAtLeastK--;
                    counts[lIdx]--;
                    if (counts[lIdx] == 0) uniqueCount--;
                    left++;
                }
                
                if (uniqueCount == targetUnique && uniqueCount == countAtLeastK) {
                    maxLen = Math.max(maxLen, right - left + 1);
                }
            }
        }
        
        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(26 \cdot n) = \mathcal{O}(n)$
  - There are at most 26 distinct lowercase English letters, so the outer loop executes at most 26 times.
  - In each iteration, both pointers `left` and `right` traverse the string at most once, and character count checks are $\mathcal{O}(1)$.
  - Total time: $26 \times \mathcal{O}(n) = \mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The frequency table size is fixed at $26$ entries regardless of input length.

---

### Takeaway Pattern & Interview Traps

- **Restoring Monotonicity via Parameterization:** When sliding window monotonicity fails because of competing constraints, ask: *Can I hold one dimension constant (e.g., number of unique characters)?* Once constrained to a fixed number of unique characters, the window condition becomes strictly monotonic.
- **Divide & Conquer Alternative:** When asked to implement recursively, identify any character with total frequency $< k$. Such characters can never participate in any valid substring, splitting the problem cleanly at their indices.