---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 242: Valid Anagram"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 242: Valid Anagram

---

### Problem Statement

Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise.

---

### Key Observation

* An anagram has the exact same character frequencies as the original string.
* We can count character occurrences using a frequency array or hash map.

---

### Core Technique: Character Frequency Counting

---

### Python 3 Solution (with typing)

```python
from collections import Counter

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)
```

---

### Worked-Out Example

```python
s = "anagram", t = "nagaram"
Counter(s) = {'a': 3, 'n': 1, 'g': 1, 'r': 1, 'm': 1}
Counter(t) = {'a': 3, 'n': 1, 'g': 1, 'r': 1, 'm': 1}
Counters match -> return True
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1) (fixed 26 lowercase alphabet characters)`

---

### Takeaway Pattern

Compare frequency tables when verifying permutations or anagrams.