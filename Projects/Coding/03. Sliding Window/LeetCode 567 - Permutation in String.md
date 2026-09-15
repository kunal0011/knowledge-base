---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 567: Permutation in String"
tags:
  - leetcode
  - coding
  - sliding-window
  - hash-table
  - two-pointers
  - string
  - amazon
  - google
---

# LeetCode 567: Permutation in String

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, ByteDance, Bloomberg  
**Difficulty:** Medium  
**Topic:** Sliding Window / Hash Table / String  

---

### Problem Statement

Given two strings `s1` and `s2`, return `true` if `s2` contains a permutation of `s1`, or `false` otherwise.

In other words, return `true` if one of `s1`'s permutations is the substring of `s2`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s1`: `str` / `string` ($1 \le |s1| \le 10^4$)
  - `s2`: `str` / `string` ($1 \le |s2| \le 10^4$)
- **Output:**
  - `bool` — `true` if a contiguous substring of `s2` is an exact permutation of `s1`, else `false`.
- **Constraints:**
  - $1 \le \text{s1.length}, \text{s2.length} \le 10^4$
  - `s1` and `s2` consist of lowercase English letters.

---

### Key Idea & Intuition

A permutation of `s1` must have length equal to $k = |s1|$ and must match character frequencies of `s1` exactly.

This means we must search for a contiguous substring in `s2` of length $k$ whose character frequency counts equal those of `s1`.
1. If $|s2| < |s1|$, no such substring can exist $\rightarrow$ return `false`.
2. We maintain a fixed sliding window of size $k$ across `s2`.
3. To achieve optimal $\mathcal{O}(1)$ update time per character, maintain a count `matches` denoting the number of character positions (out of 26 lowercase English letters) where `s1_count[c] == s2_count[c]`.
4. As the window advances by one character:
   - Add the new incoming character $s2[i]$: adjust `matches` accordingly.
   - Evict the outgoing character $s2[i - k]$: adjust `matches` accordingly.
   - If `matches == 26`, an exact permutation is found $\rightarrow$ return `true` immediately.
5. If the loop completes without `matches == 26`, return `false`.

---

### Solution Approach (Step-by-Step)

1. Let $k = |s1|$ and $n = |s2|$. If $n < k$, return `False`.
2. Initialize two frequency arrays of size 26: `s1_count` and `s2_count`.
3. Populate `s1_count` from `s1` and initialize `s2_count` from the first $k$ characters of `s2`.
4. Count initial character matches across all 26 letters:
   $$\text{matches} = \sum_{c=0}^{25} \mathbb{I}(\text{s1\_count}[c] == \text{s2\_count}[c])$$
5. If `matches == 26`, return `True`.
6. Slide the window from index $i = k$ to $n - 1$:
   - Let `in_char = s2[i]` and `out_char = s2[i - k]`.
   - Update frequency and `matches` for `in_char`.
   - Update frequency and `matches` for `out_char`.
   - If `matches == 26`, return `True`.
7. If no window matched, return `False`.

---

### Visual Algorithm Walkthrough

For `s1 = "ab"`, `s2 = "eidbaooo"` ($k = 2$):
Target: `{a: 1, b: 1, others: 0}`

```
Window size = 2

i = 0..1: window "ei"
  s2_count: {e: 1, i: 1, others: 0}
  matches = 24 (a and b don't match, e and i don't match)

i = 2: add 'd', evict 'e' -> window "id"
  matches != 26

i = 3: add 'b', evict 'i' -> window "db"
  matches != 26

i = 4: add 'a', evict 'd' -> window "ba"
  s2_count: {a: 1, b: 1, others: 0}
  s1_count: {a: 1, b: 1, others: 0}
  matches = 26!
  -> FOUND VALID PERMUTATION!
  Return True immediately.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s1 = "ab"`, `s2 = "eidbaooo"`
- **Tracing:** Window `"ba"` at index 3 matches `s1`.
- **Output:** `true`

#### Example 2:
- **Input:** `s1 = "ab"`, `s2 = "eidboaoo"`
- **Tracing:** Windows: "ei", "id", "db", "bo", "oa", "ao", "oo". None match `{a: 1, b: 1}`.
- **Output:** `false`

#### Example 3 ($|s1| > |s2|$):
- **Input:** `s1 = "hello"`, `s2 = "o"`
- **Output:** `false`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        k, n = len(s1), len(s2)
        if n < k:
            return False
            
        s1_count = [0] * 26
        s2_count = [0] * 26
        
        for i in range(k):
            s1_count[ord(s1[i]) - ord('a')] += 1
            s2_count[ord(s2[i]) - ord('a')] += 1
            
        matches = sum(1 for c in range(26) if s1_count[c] == s2_count[c])
        if matches == 26:
            return True
            
        for i in range(k, n):
            in_idx = ord(s2[i]) - ord('a')
            out_idx = ord(s2[i - k]) - ord('a')
            
            # Incorporate incoming character
            if s2_count[in_idx] == s1_count[in_idx]:
                matches -= 1
            s2_count[in_idx] += 1
            if s2_count[in_idx] == s1_count[in_idx]:
                matches += 1
                
            # Evict outgoing character
            if s2_count[out_idx] == s1_count[out_idx]:
                matches -= 1
            s2_count[out_idx] -= 1
            if s2_count[out_idx] == s1_count[out_idx]:
                matches += 1
                
            if matches == 26:
                return True
                
        return False
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    bool checkInclusion(const std::string& s1, const std::string& s2) {
        int k = static_cast<int>(s1.size());
        int n = static_cast<int>(s2.size());
        if (n < k) return false;
        
        std::vector<int> s1_count(26, 0);
        std::vector<int> s2_count(26, 0);
        
        for (int i = 0; i < k; ++i) {
            s1_count[s1[i] - 'a']++;
            s2_count[s2[i] - 'a']++;
        }
        
        int matches = 0;
        for (int c = 0; c < 26; ++c) {
            if (s1_count[c] == s2_count[c]) ++matches;
        }
        
        if (matches == 26) return true;
        
        for (int i = k; i < n; ++i) {
            int in_idx = s2[i] - 'a';
            int out_idx = s2[i - k] - 'a';
            
            // Add incoming
            if (s2_count[in_idx] == s1_count[in_idx]) --matches;
            s2_count[in_idx]++;
            if (s2_count[in_idx] == s1_count[in_idx]) ++matches;
            
            // Evict outgoing
            if (s2_count[out_idx] == s1_count[out_idx]) --matches;
            s2_count[out_idx]--;
            if (s2_count[out_idx] == s1_count[out_idx]) ++matches;
            
            if (matches == 26) return true;
        }
        
        return false;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int k = s1.length();
        int n = s2.length();
        if (n < k) return false;
        
        int[] s1Count = new int[26];
        int[] s2Count = new int[26];
        
        for (int i = 0; i < k; i++) {
            s1Count[s1.charAt(i) - 'a']++;
            s2Count[s2.charAt(i) - 'a']++;
        }
        
        int matches = 0;
        for (int c = 0; c < 26; c++) {
            if (s1Count[c] == s2Count[c]) matches++;
        }
        
        if (matches == 26) return true;
        
        for (int i = k; i < n; i++) {
            int inIdx = s2.charAt(i) - 'a';
            int outIdx = s2.charAt(i - k) - 'a';
            
            // Add incoming
            if (s2Count[inIdx] == s1Count[inIdx]) matches--;
            s2Count[inIdx]++;
            if (s2Count[inIdx] == s1Count[inIdx]) matches++;
            
            // Evict outgoing
            if (s2Count[outIdx] == s1Count[outIdx]) matches--;
            s2Count[outIdx]--;
            if (s2Count[outIdx] == s1Count[outIdx]) matches++;
            
            if (matches == 26) return true;
        }
        
        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(|s1| + |s2|)$
  - Building initial counts takes $\mathcal{O}(|s1|)$.
  - Sliding the window does $\mathcal{O}(1)$ operations per character of $s2$.
  - Early returns upon finding first match.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Two fixed-size arrays of size 26 for English alphabet counts.

---

### Takeaway Pattern & Interview Traps

- **Fixed Window vs. Variable Window:** Whenever the problem statement mentions "permutation of a string $P$", the window size is **fixed** at $|P|$.
- **Early Exit:** Because we only need an existence check (`bool`), returning immediately once `matches == 26` avoids searching the rest of $s2$.