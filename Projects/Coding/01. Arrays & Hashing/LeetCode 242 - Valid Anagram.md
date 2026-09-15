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
  - amazon
  - google
---

# LeetCode 242: Valid Anagram

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Easy  
**Topic:** Frequency Array / Character Counter

---

### Problem Statement

Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`, `t: str`
- **Output:** `bool`
- **Constraints:**
  - $1 \le \text{s.length}, \text{t.length} \le 5 \times 10^4$
  - `s` and `t` consist of lowercase English letters.

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
        if (s.size() != t.size()) return false;

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
        if (s.length() != t.length()) return false;

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

- **Time Complexity:** $O(N)$ where $N = \text{len}(s)$.
- **Space Complexity:** $O(1)$ auxiliary space using fixed 26-element array.
