---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 242: Valid Anagram"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - hash-table
  - string
  - amazon
  - google
  - meta
---

# LeetCode 242: Valid Anagram

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber  
**Difficulty:** Easy  
**Topic:** Frequency Array / Character Counter / Hash Table

---

### Problem Statement

Given two strings `s` and `t`, return `true` if `t` is an **anagram** of `s`, and `false` otherwise.

An **anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`, `t: str`
- **Output:** `bool`
- **Constraints:**
  - $1 \le \text{s.length}, \text{t.length} \le 5 \times 10^4$
  - `s` and `t` consist of lowercase English letters.
- **Follow up:** What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

---

### Key Idea & Intuition

#### 1. Length Invariant:
If $|s| \ne |t|$, they cannot possibly be anagrams. Return `false` immediately.

#### 2. Character Frequency Balance:
Two strings of identical length are anagrams if and only if each character appears with the exact same count in both strings:
- Because the characters are limited to lowercase English letters (`'a'` to `'z'`), we do not need a heavy hash table. A fixed-size array of length 26 suffices.
- As we iterate through strings $s$ and $t$:
  - Increment the count for character $s[i]$.
  - Decrement the count for character $t[i]$.
- At the end, if all 26 frequency counts are exactly $0$, the strings are anagrams. If any count is non-zero, they are not.

#### 3. Follow-up (Unicode Characters):
If characters can be arbitrary Unicode code points, replace the fixed 26-element array with a hash table (`HashMap<Character, Integer>`) to handle the unbounded character set dynamically.

---

### Solution Approach (Step-by-Step)

1. **Length Check:**
   - If `len(s) != len(t)`, return `False`.
2. **Frequency Array Allocation:**
   - Allocate an integer array `counts` of size 26 initialized to 0.
3. **Single Simultaneous Pass:**
   - For $i$ from 0 to $len(s) - 1$:
     - `counts[s[i] - 'a']++`
     - `counts[t[i] - 'a']--`
4. **Verification Pass:**
   - Check if all values in `counts` are 0.
   - If any value is non-zero, return `False`.
   - If all are zero, return `True`.

---

### Visual Algorithm Walkthrough

#### Example: `s = "anagram"`, `t = "nagaram"`

```
Length Check: len(s) = 7 == len(t) = 7 (Pass)

Simultaneous Pass:
i=0: s[0]='a' (+1), t[0]='n' (-1) -> counts['a']=1, counts['n']=-1
i=1: s[1]='n' (+1), t[1]='a' (-1) -> counts['a']=0, counts['n']=0
i=2: s[2]='a' (+1), t[2]='g' (-1) -> counts['a']=1, counts['g']=-1
i=3: s[3]='g' (+1), t[3]='a' (-1) -> counts['a']=0, counts['g']=0
i=4: s[4]='r' (+1), t[4]='r' (-1) -> counts['r']=0
i=5: s[5]='a' (+1), t[5]='a' (-1) -> counts['a']=0
i=6: s[6]='m' (+1), t[6]='m' (-1) -> counts['m']=0

All 26 buckets are 0 -> Return True.
```

---

### Solved Examples with Multiple Inputs

| `s` | `t` | Length Check | Final Frequency State | Output |
| :--- | :--- | :--- | :--- | :--- |
| `"anagram"` | `"nagaram"` | Equal (7) | All counts 0 | `True` |
| `"rat"` | `"car"` | Equal (3) | `'c': -1`, `'t': +1` | `False` |
| `"a"` | `"ab"` | $1 \ne 2$ | Early exit | `False` |
| `"ab"` | `"a"` | $2 \ne 1$ | Early exit | `False` |
| `"listen"` | `"silent"` | Equal (6) | All counts 0 | `True` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
            
        counts = [0] * 26
        for c1, c2 in zip(s, t):
            counts[ord(c1) - ord('a')] += 1
            counts[ord(c2) - ord('a')] -= 1
            
        return all(c == 0 for c in counts)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class Solution {
public:
    bool isAnagram(std::string s, std::string t) {
        if (s.size() != t.size()) {
            return false;
        }

        int counts[26] = {0};
        for (size_t i = 0; i < s.size(); ++i) {
            counts[s[i] - 'a']++;
            counts[t[i] - 'a']--;
        }

        for (int c : counts) {
            if (c != 0) return false;
        }
        return true;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }

        int[] counts = new int[26];
        for (int i = 0; i < s.length(); i++) {
            counts[s.charAt(i) - 'a']++;
            counts[t.charAt(i) - 'a']--;
        }

        for (int c : counts) {
            if (c != 0) return false;
        }
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$, where $n = |s| = |t|$. We do a single pass of length $n$ updating counts, followed by inspecting 26 integers ($\mathcal{O}(1)$).
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. The frequency array has a fixed size of 26 regardless of the input length $n$.

---

### Takeaway Pattern & Interview Traps

1. **Sorting vs Array Counting:**
   - Sorting both strings takes $\mathcal{O}(n \log n)$ time. The frequency array is strictly $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ space, which is optimal.
2. **Immediate Length Check:**
   - Checking `len(s) != len(t)` at the beginning avoids unnecessary iterations and allows parallel character iteration without index out-of-bounds guards.
3. **Handling Unicode Characters (Interview Follow-Up):**
   - If characters are Unicode, use a hash map instead of a fixed 26-element array, which yields $\mathcal{O}(n)$ time and $\mathcal{O}(k)$ space where $k$ is the number of distinct Unicode characters.
