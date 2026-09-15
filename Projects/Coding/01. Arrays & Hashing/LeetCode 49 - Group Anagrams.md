---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 49: Group Anagrams"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 49: Group Anagrams

---

### Problem Statement

Given an array of strings `strs`, group the anagrams together. You can return the answer in any order.

---

### Key Observation

* All anagrams share the same canonical signature (e.g. sorted string or a 26-element character count tuple).
* We can use this signature as a Hash Map key mapping to lists of matching words.

---

### Core Technique: Hash Map with Frequency Tuple Keys

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        groups = defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            groups[tuple(count)].append(s)
        return list(groups.values())
```

---

### Worked-Out Example

```python
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
Signature for "eat", "tea", "ate" -> (1, 0, ..., 1, ..., 1) -> ["eat", "tea", "ate"]
Signature for "tan", "nat" -> ["tan", "nat"]
Signature for "bat" -> ["bat"]
Result: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n * k) where n is number of strings, k is max string length`
* **Space Complexity:** `O(n * k)`

---

### Takeaway Pattern

Use immutable tuples of frequency counts or sorted representations as hash keys to group equivalent items.