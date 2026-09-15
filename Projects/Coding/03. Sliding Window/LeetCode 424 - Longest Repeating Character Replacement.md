---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 424: Longest Repeating Character Replacement"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 424: Longest Repeating Character Replacement

**LeetCode 424 – Longest Repeating Character Replacement**.

---

## 🔹 Problem Restatement

You’re given a string `s` and an integer `k`.  
You can replace at most `k` characters in the string.  
Return the **length of the longest substring** that can be obtained where all characters are the same.

---

## 🔹 Example

```
s = "AABABBA", k = 1
```

* Change the middle `"B"` to `"A"` → `"AAAA"`.
* Length = 4.

✅ Answer = 4.

---

## 🔹 Key Observations

1. We want the **longest window** such that at most `k` replacements make it uniform.
2. Within a window, the number of changes needed =  
   `window_size - count_of_most_frequent_char_in_window`.
3. Valid window condition:  
   `window_size - max_freq <= k`.

---

## 🔹 Approaches

### **1. Brute Force**

* For each substring, check if it can be turned into a uniform string with ≤ `k` changes.
* Count frequencies → check condition.

⏱ **O(n³)** → too slow.  
❌ Not acceptable.

---

### **2. Sliding Window + Frequency Count (Optimal)**

* Maintain a sliding window `[l, r]`.
* Track character frequencies inside.
* Track `max_count` = max frequency of any character in the window.
* If `window_size - max_count > k`, shrink from left.
* Otherwise, update answer.

⏱ **O(n)**  
✅ Optimal.

---

## 🔹 Implementation

```python
def characterReplacement(s: str, k: int) -> int:
    from collections import defaultdict
    
    count = defaultdict(int)
    l = 0
    max_count = 0
    res = 0
    
    for r in range(len(s)):
        count[s[r]] += 1
        max_count = max(max_count, count[s[r]])
        
        # if more than k replacements needed
        while (r - l + 1) - max_count > k:
            count[s[l]] -= 1
            l += 1
        
        res = max(res, r - l + 1)
    
    return res
```

---

## 🔹 Example Walkthrough

`s = "AABABBA", k = 1`

* Expand window, track counts:

  * `"AAB"` → window size 3, max\_count=2, need 1 change → valid.
  * `"AABA"` → size 4, max\_count=3, need 1 change → valid.
  * `"AABAB"` → size 5, max\_count=3, need 2 changes > 1 → shrink left.

Answer = 4.

---

## 🔹 Which Works?

* **Brute force** → too slow.
* **Sliding window with frequency** → efficient and correct.
* **Binary search on length + check** also possible, but overkill.