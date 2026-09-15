---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 438: Find All Anagrams in a String"
tags:
  - leetcode
  - coding
  - sliding-window
  - hash-table
  - string
  - amazon
  - google
---

# LeetCode 438: Find All Anagrams in a String

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Sliding Window / Hash Table / String  

---

### Problem Statement

Given two strings `s` and `p`, return an array of all the start indices of `p`'s anagrams in `s`. You may return the answer in **any order**.

An **anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 3 \times 10^4$).
  - `p`: `str` / `string` ($1 \le |p| \le 3 \times 10^4$).
- **Output:**
  - `List[int]` / `vector<int>` / `int[]` — all 0-indexed start indices where a substring of `s` is an anagram of `p`.
- **Constraints:**
  - $1 \le \text{s.length}, \text{p.length} \le 3 \times 10^4$
  - `s` and `p` consist of lowercase English letters.

---

### Key Idea & Intuition

An anagram of string `p` must have the exact same length $k = |p|$ and the exact same character frequencies as `p`.

This defines a **fixed-size sliding window** of size $k$:
1. If $|s| < |p|$, return `[]` immediately.
2. Maintain two frequency tables of size 26: `p_count` and `s_count`.
3. Rather than comparing the entire 26-element array at every step (which would cost $26 \times |s|$ operations), we can maintain a variable `matches` that tracks how many of the 26 characters have identical counts in both `s_count` and `p_count`.
4. As the window slides:
   - When introducing incoming character $s[i]$:
     - Check its count before incrementing vs `p_count`.
     - Increment count in `s_count`.
     - Check if the change created a match or broke a match.
   - When evicting outgoing character $s[i - k]$:
     - Check its count before decrementing vs `p_count`.
     - Decrement count in `s_count`.
     - Check if the eviction created a match or broke a match.
   - If `matches == 26`, the current window is an anagram; append start index $i - k + 1$ to results.

This yields an ultra-fast, strictly $\mathcal{O}(1)$ update per slide.

---

### Solution Approach (Step-by-Step)

1. Let $m = |s|$ and $k = |p|$. If $m < k$, return `[]`.
2. Initialize `p_count = [0] * 26` and `s_count = [0] * 26`.
3. Populate `p_count` and the first $k$ characters of `s_count`.
4. Initialize `matches = 0`. For each character $c \in [0, 25]$, if `s_count[c] == p_count[c]`, increment `matches += 1`.
5. Initialize `result = []`. If `matches == 26`, append $0$ to `result`.
6. Slide the window from $i = k$ to $m - 1$:
   - **Add $s[i]$ (entering window):**
     - Let $idx = \text{ord}(s[i]) - \text{ord}('a')$.
     - If `s_count[idx] == p_count[idx]`, it will no longer match $\rightarrow$ `matches -= 1`.
     - `s_count[idx] += 1`.
     - If `s_count[idx] == p_count[idx]`, it now matches $\rightarrow$ `matches += 1`.
   - **Remove $s[i - k]$ (leaving window):**
     - Let $out\_idx = \text{ord}(s[i - k]) - \text{ord}('a')$.
     - If `s_count[out_idx] == p_count[out_idx]`, it will no longer match $\rightarrow$ `matches -= 1`.
     - `s_count[out_idx] -= 1`.
     - If `s_count[out_idx] == p_count[out_idx]`, it now matches $\rightarrow$ `matches += 1`.
   - If `matches == 26`, append $i - k + 1$ to `result`.
7. Return `result`.

---

### Visual Algorithm Walkthrough

For `s = "cbaebabacd"`, `p = "abc"` ($k = 3$):
Target frequencies `p_count`: `{a: 1, b: 1, c: 1, others: 0}`

```
Window 0: [0..2] -> "cba"
  s_count: {a: 1, b: 1, c: 1, others: 0}
  s_count == p_count! matches = 26.
  -> Append index 0. Result: [0]

Slide to i = 3 ('e'), evict i - 3 = 0 ('c'):
  Add 'e': s_count['e'] becomes 1 (was 0 == p_count) -> matches = 25
  Evict 'c': s_count['c'] becomes 0 (was 1 == p_count) -> matches = 24
  matches = 24 != 26.

Slide to i = 4 ('b'), evict i - 3 = 1 ('b'):
  Add 'b', evict 'b' -> net 0 change. matches = 24.

Slide to i = 5 ('a'), evict i - 3 = 2 ('a'):
  Add 'a', evict 'a' -> net 0 change. matches = 24.

Slide to i = 6 ('b'), evict i - 3 = 3 ('e'):
  Add 'b': s_count['b'] becomes 2 -> matches drops
  Evict 'e': s_count['e'] becomes 0 == p_count['e'] -> matches restored

... When window reaches [6..8] -> "bac":
  s_count: {a: 1, b: 1, c: 1, others: 0}
  matches = 26!
  -> Append index 6. Result: [0, 6]

Final Result: [0, 6]
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "cbaebabacd"`, `p = "abc"`
- **Output:** `[0, 6]`

#### Example 2:
- **Input:** `s = "abab"`, `p = "ab"`
- **Tracing:**
  - $i=0$: "ab" -> match! Add 0.
  - $i=1$: "ba" -> match! Add 1.
  - $i=2$: "ab" -> match! Add 2.
- **Output:** `[0, 1, 2]`

#### Example 3 ($|s| < |p|$):
- **Input:** `s = "a"`, `p = "ab"`
- **Output:** `[]`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        m, k = len(s), len(p)
        if m < k:
            return []
            
        p_count = [0] * 26
        s_count = [0] * 26
        
        for i in range(k):
            p_count[ord(p[i]) - ord('a')] += 1
            s_count[ord(s[i]) - ord('a')] += 1
            
        matches = sum(1 for c in range(26) if s_count[c] == p_count[c])
        result: List[int] = []
        
        if matches == 26:
            result.append(0)
            
        for i in range(k, m):
            in_idx = ord(s[i]) - ord('a')
            out_idx = ord(s[i - k]) - ord('a')
            
            # Add incoming character s[i]
            if s_count[in_idx] == p_count[in_idx]:
                matches -= 1
            s_count[in_idx] += 1
            if s_count[in_idx] == p_count[in_idx]:
                matches += 1
                
            # Evict outgoing character s[i - k]
            if s_count[out_idx] == p_count[out_idx]:
                matches -= 1
            s_count[out_idx] -= 1
            if s_count[out_idx] == p_count[out_idx]:
                matches += 1
                
            if matches == 26:
                result.append(i - k + 1)
                
        return result
```

#### C++17
```cpp
#include <vector>
#include <string>

class Solution {
public:
    std::vector<int> findAnagrams(const std::string& s, const std::string& p) {
        int m = static_cast<int>(s.size());
        int k = static_cast<int>(p.size());
        if (m < k) return {};
        
        std::vector<int> p_count(26, 0);
        std::vector<int> s_count(26, 0);
        
        for (int i = 0; i < k; ++i) {
            p_count[p[i] - 'a']++;
            s_count[s[i] - 'a']++;
        }
        
        int matches = 0;
        for (int c = 0; c < 26; ++c) {
            if (s_count[c] == p_count[c]) ++matches;
        }
        
        std::vector<int> result;
        if (matches == 26) {
            result.push_back(0);
        }
        
        for (int i = k; i < m; ++i) {
            int in_idx = s[i] - 'a';
            int out_idx = s[i - k] - 'a';
            
            // Add incoming
            if (s_count[in_idx] == p_count[in_idx]) --matches;
            s_count[in_idx]++;
            if (s_count[in_idx] == p_count[in_idx]) ++matches;
            
            // Evict outgoing
            if (s_count[out_idx] == p_count[out_idx]) --matches;
            s_count[out_idx]--;
            if (s_count[out_idx] == p_count[out_idx]) ++matches;
            
            if (matches == 26) {
                result.push_back(i - k + 1);
            }
        }
        
        return result;
    }
};
```

#### Java 17
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> result = new ArrayList<>();
        int m = s.length();
        int k = p.length();
        if (m < k) return result;
        
        int[] pCount = new int[26];
        int[] sCount = new int[26];
        
        for (int i = 0; i < k; i++) {
            pCount[p.charAt(i) - 'a']++;
            sCount[s.charAt(i) - 'a']++;
        }
        
        int matches = 0;
        for (int c = 0; c < 26; c++) {
            if (sCount[c] == pCount[c]) matches++;
        }
        
        if (matches == 26) {
            result.add(0);
        }
        
        for (int i = k; i < m; i++) {
            int inIdx = s.charAt(i) - 'a';
            int outIdx = s.charAt(i - k) - 'a';
            
            // Add incoming
            if (sCount[inIdx] == pCount[inIdx]) matches--;
            sCount[inIdx]++;
            if (sCount[inIdx] == pCount[inIdx]) matches++;
            
            // Evict outgoing
            if (sCount[outIdx] == pCount[outIdx]) matches--;
            sCount[outIdx]--;
            if (sCount[outIdx] == pCount[outIdx]) matches++;
            
            if (matches == 26) {
                result.add(i - k + 1);
            }
        }
        
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(|s| + |p|)$
  - Initializing frequency counts takes $\mathcal{O}(|p|)$ time.
  - Sliding the window performs strictly $\mathcal{O}(1)$ comparisons and integer updates per step.
  - Overall time is linear in $|s|$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The arrays `s_count` and `p_count` are fixed at size 26.
  - Output list is not counted towards auxiliary space.

---

### Takeaway Pattern & Interview Traps

- **Delta Matches vs. Array Equality:** Comparing two 26-element arrays takes $26 \times |s|$ operations. While still linear, tracking the integer count of matching buckets (`matches == 26`) reduces per-step overhead to pure $\mathcal{O}(1)$ and runs $5\times$ faster in practice.
- **Twin Problem:** Notice that **LeetCode 567 (Permutation in String)** is identical to LeetCode 438, except it asks for a boolean (`True`/`False` whether any anagram exists) rather than collecting all start indices!