---
date: "2026-08-29"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 394: Decode String"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 394: Decode String

**Target Companies:** Google, Amazon, Meta

---

### Problem Statement

Given an encoded string, return its decoded string: `k[encoded_string]` means the `encoded_string` inside brackets is repeated `k` times. Nesting is allowed.

---

### Key Observation

* Use a Stack storing tuples of `(previous_string, repeat_count)` when encountering `'['`.
* Accumulate digits to parse multi-digit counts (e.g. `"100["`).
* When encountering `']'`, pop the multiplier and preceding string, then update `curr_str = prev_str + curr_str * count`.

---

### Core Technique: Stack with Context Frame Restoration

---

### Python 3 Solution (with typing)

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []  # stores (prev_str, k)
        curr_str = ""
        curr_k = 0
        
        for ch in s:
            if ch.isdigit():
                curr_k = curr_k * 10 + int(ch)
            elif ch == '[':
                stack.append((curr_str, curr_k))
                curr_str = ""
                curr_k = 0
            elif ch == ']':
                prev_str, k = stack.pop()
                curr_str = prev_str + curr_str * k
            else:
                curr_str += ch
                
        return curr_str
```

---

### Worked-Out Example

```python
s = "3[a2[c]]"
'3', '[' -> stack = [("", 3)], curr_str = ""
'a' -> curr_str = "a"
'2', '[' -> stack = [("", 3), ("a", 2)], curr_str = ""
'c' -> curr_str = "c"
']' -> pop ("a", 2) -> curr_str = "a" + "c" * 2 = "acc"
']' -> pop ("", 3) -> curr_str = "" + "acc" * 3 = "accaccacc"
```

---

### Complexity Analysis

* **Time Complexity:** `O(output_length)`
* **Space Complexity:** `O(output_length)`

---

### Takeaway Pattern

Nested recursive structures like brackets are cleanly evaluated by saving parent state to a stack on `[` and resolving on `]`.