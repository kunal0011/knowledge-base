---
date: "2026-08-29"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 68: Text Justification"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
---

# LeetCode 68: Text Justification

**Target Companies:** Google (Signature Hard), Amazon

---

### Problem Statement

Given an array of strings `words` and a width `maxWidth`, format the text such that each line has exactly `maxWidth` characters and is fully (left and right) justified.

---

### Key Observation

* Pack as many words as possible into each line greedily (accounting for at least 1 space between words).
* For intermediate lines, distribute extra spaces as evenly as possible. Left slots get remainder spaces first: `extra_spaces // num_slots` + 1 if `slot < remainder`.
* Single word lines and the final line are strictly left-justified (words separated by 1 space, padded with spaces at end).

---

### Core Technique: Greedy Line Fitting with Modular Space Distribution

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res = []
        i = 0
        n = len(words)
        
        while i < n:
            # 1. Determine words for current line
            line_len = len(words[i])
            j = i + 1
            while j < n and line_len + 1 + len(words[j]) <= maxWidth:
                line_len += 1 + len(words[j])
                j += 1
                
            line_words = words[i:j]
            num_words = len(line_words)
            
            # 2. Format line
            # Case A: Last line or single-word line -> Left justify
            if j == n or num_words == 1:
                line_str = " ".join(line_words)
                line_str += " " * (maxWidth - len(line_str))
            else:
                # Case B: Fully justify
                total_letters = sum(len(w) for w in line_words)
                total_spaces = maxWidth - total_letters
                slots = num_words - 1
                base_space = total_spaces // slots
                extra_space = total_spaces % slots
                
                line_parts = []
                for k in range(slots):
                    line_parts.append(line_words[k])
                    spaces_to_add = base_space + (1 if k < extra_space else 0)
                    line_parts.append(" " * spaces_to_add)
                line_parts.append(line_words[-1])
                line_str = "".join(line_parts)
                
            res.append(line_str)
            i = j
            
        return res
```

---

### Worked-Out Example

```python
words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Line 1: ["This", "is", "an"] -> 8 letters, 8 spaces across 2 slots -> "This    is    an"
Line 2: ["example", "of", "text"] -> 13 letters, 3 spaces across 2 slots (2, 1) -> "example  of text"
Line 3: ["justification."] -> Last line, left justify -> "justification.  "
```

---

### Complexity Analysis

* **Time Complexity:** `O(N) where N is total characters across words`
* **Space Complexity:** `O(maxWidth)`

---

### Takeaway Pattern

Break text justification into: 1) greedy word grouping, 2) slot modulo space math, 3) left-justified edge case.