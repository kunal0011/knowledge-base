---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 395: Longest Substring with At Least K Repeating Characters"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 395: Longest Substring with At Least K Repeating Characters

**LeetCode 395 – Longest Substring with At Least K Repeating Characters**.

---

## 🔹 Problem Restatement

Given a string `s` and integer `k`, return the length of the **longest substring** where **every character appears at least `k`times**.

---

## 🔹 Key Observations

1. We need **contiguous substring**.
2. If a character appears **fewer than k times** in the whole string, it **cannot** be part of the valid substring → it splits the problem into smaller pieces.
3. This naturally suggests **divide & conquer**.
4. Alternative: **sliding window by unique character count** also works.

---

## 🔹 Approaches

### **1. Brute Force**

* Generate all substrings.
* Count frequency, check if all ≥ k.
* Track max length.

⏱ **O(n³)** (generating + counting).  
❌ Too slow (n ≤ 10⁴).

---

### **2. Divide & Conquer (Recursive Splitting)**

* Count frequencies of all characters.
* If all frequencies ≥ k → whole string is valid → return length.
* Else, pick a "bad" character (freq < k).
* Split string by this character → recursively solve each part.
* Take max of results.

⏱ **O(26·n)** average (each level removes bad chars).  
✅ Efficient and accepted.

**Code:**

```python
def longestSubstring(s: str, k: int) -> int:
    if not s:
        return 0
    
    from collections import Counter
    count = Counter(s)
    
    for ch in count:
        if count[ch] < k:  # bad char splits problem
            return max(longestSubstring(t, k) for t in s.split(ch))
    
    return len(s)  # all good
```

---

### **3. Sliding Window by Unique Characters**

* We know answer must be substring with some number of **unique characters**.
* Try `target_unique = 1 … 26`:

  * Use sliding window to find longest substring with **exactly target\_unique unique chars** and **each appears ≥ k**.
* Update max length.

⏱ **O(26·n)**.  
✅ Also efficient.

**Code:**

```python
def longestSubstring(s: str, k: int) -> int:
    n = len(s)
    res = 0
    
    for target_unique in range(1, 27):
        freq = {}
        l = r = 0
        unique = count_at_least_k = 0
        
        while r < n:
            # expand
            freq[s[r]] = freq.get(s[r], 0) + 1
            if freq[s[r]] == 1:
                unique += 1
            if freq[s[r]] == k:
                count_at_least_k += 1
            r += 1
            
            # shrink if too many unique
            while unique > target_unique:
                freq[s[l]] -= 1
                if freq[s[l]] == 0:
                    unique -= 1
                if freq[s[l]] == k - 1:
                    count_at_least_k -= 1
                l += 1
            
            # valid window
            if unique == count_at_least_k:
                res = max(res, r - l)
    
    return res
```

---

## 🔹 Example Walkthrough

`s = "aaabb", k = 3`

* Count = {a:3, b:2}
* `b` is bad (< 3). Split → ["aaa", ""]
* `"aaa"` → valid (all freq ≥ 3) → length = 3.  
  ✅ Answer = 3.