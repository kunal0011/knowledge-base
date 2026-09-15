---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 567: Permutation in String"
tags:
  - leetcode
  - coding
  - two-pointers
  - sliding-window
  - amazon
  - google
---

# LeetCode 567: Permutation in String

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Fixed Sliding Window / 26-Character Array Counter

---

### Problem Statement

Given two strings `s1` and `s2`, return `true` if `s2` contains a permutation of `s1`, or `false` otherwise.

In other words, return `true` if one of `s1`'s permutations is the substring of `s2`.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        n1, n2 = len(s1), len(s2)
        if n1 > n2:
            return False
            
        c1, c2 = [0] * 26, [0] * 26
        for i in range(n1):
            c1[ord(s1[i]) - ord('a')] += 1
            c2[ord(s2[i]) - ord('a')] += 1
            
        if c1 == c2:
            return True
            
        for i in range(n1, n2):
            c2[ord(s2[i]) - ord('a')] += 1
            c2[ord(s2[i - n1]) - ord('a')] -= 1
            if c1 == c2:
                return True
                
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class Solution {
public:
    bool checkInclusion(std::string s1, std::string s2) {
        int n1 = s1.size(), n2 = s2.size();
        if (n1 > n2) return false;

        std::vector<int> c1(26, 0), c2(26, 0);
        for (int i = 0; i < n1; ++i) {
            c1[s1[i] - 'a']++;
            c2[s2[i] - 'a']++;
        }
        if (c1 == c2) return true;

        for (int i = n1; i < n2; ++i) {
            c2[s2[i] - 'a']++;
            c2[s2[i - n1] - 'a']--;
            if (c1 == c2) return true;
        }
        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int n1 = s1.length(), n2 = s2.length();
        if (n1 > n2) return false;

        int[] c1 = new int[26];
        int[] c2 = new int[26];

        for (int i = 0; i < n1; i++) {
            c1[s1.charAt(i) - 'a']++;
            c2[s2.charAt(i) - 'a']++;
        }
        if (Arrays.equals(c1, c2)) return true;

        for (int i = n1; i < n2; i++) {
            c2[s2.charAt(i) - 'a']++;
            c2[s2.charAt(i - n1) - 'a']--;
            if (Arrays.equals(c1, c2)) return true;
        }
        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N_2)$ — Constant 26-element array comparison on each character shift.
- **Space Complexity:** $O(1)$ — Two fixed 26-element integer buffers.
