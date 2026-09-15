---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1456: Maximum Number of Vowels in a Substring of Given Length"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1456: Maximum Number of Vowels in a Substring of Given Length

**Leetcode 1456: Maximum Number of Vowels in a Substring of Given Length**.

---

## 📌 Problem Recap

We’re given:

* A string `s`
* An integer `k`

We need the **maximum number of vowels** in any substring of length `k`.

---

## ✅ Approach 1: Brute Force

### Idea:

* For each substring of length `k`, count vowels.

### Code:

```python
def maxVowels_bruteforce(s, k):
    vowels = set("aeiou")
    n = len(s)
    ans = 0
    for i in range(n - k + 1):
        cnt = sum(1 for ch in s[i:i+k] if ch in vowels)
        ans = max(ans, cnt)
    return ans
```

### Complexity:

* Time: **O(n·k)**
* Space: **O(1)**

❌ Too slow for `n = 10^5`.

---

## ✅ Approach 2: Sliding Window (Optimal)

### Idea:

* Use a sliding window of length `k`.
* Maintain count of vowels inside the window.
* When window slides:

  * Add new char if vowel.
  * Remove old char if vowel.
* Track max count.

### Code:

```python
def maxVowels(s, k):
    vowels = set("aeiou")
    n = len(s)
    
    # first window
    cnt = sum(1 for ch in s[:k] if ch in vowels)
    ans = cnt
    
    for i in range(k, n):
        if s[i] in vowels:
            cnt += 1
        if s[i-k] in vowels:
            cnt -= 1
        ans = max(ans, cnt)
    
    return ans
```

### Complexity:

* Time: **O(n)**
* Space: **O(1)**

✅✅ Best solution.

---

## ✅ Approach 3: Prefix Sum (Alternative)

### Idea:

* Compute prefix array where `prefix[i] = number of vowels in s[:i]`.
* Then number of vowels in substring `s[i:i+k] = prefix[i+k] - prefix[i]`.
* Take max over all windows.

### Code:

```python
def maxVowels_prefix(s, k):
    vowels = set("aeiou")
    n = len(s)
    prefix = [0] * (n+1)
    
    for i in range(n):
        prefix[i+1] = prefix[i] + (1 if s[i] in vowels else 0)
    
    ans = 0
    for i in range(n-k+1):
        ans = max(ans, prefix[i+k] - prefix[i])
    return ans
```

### Complexity:

* Time: **O(n)**
* Space: **O(n)**

✅ Works fine, but extra memory.

---

# 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(n·k) | ❌ | Too slow |
| Sliding Window | O(n), O(1) | ✅✅ | **Best solution** |
| Prefix Sum | O(n), O(n) | ✅ | Alternative |