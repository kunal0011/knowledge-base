---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1297: Maximum Number of Occurrences of a Substring"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1297: Maximum Number of Occurrences of a Substring

**Leetcode 1297: Maximum Number of Occurrences of a Substring**.

---

## 📌 Problem Recap

You are given:

* A string `s`
* Integers `maxLetters`, `minSize`, `maxSize`

You need to find the **maximum number of occurrences** of any substring of `s` such that:

1. The substring’s length is between `minSize` and `maxSize`.
2. The substring contains at most `maxLetters` **unique characters**.

---

## ✅ Key Observations

1. The substring length that really matters is **`minSize`**, not `maxSize`.

   * If a longer substring is valid, its shorter prefix of length `minSize` is also valid and occurs at least as many times.
   * So, the answer can be found by only checking substrings of length `minSize`.  
     👉 This reduces search space drastically.
2. We need to count frequencies of valid substrings efficiently.

---

## ✅ Approach 1: Brute Force (All Substrings)

### Idea:

* Generate all substrings with length between `minSize` and `maxSize`.
* Count unique characters.
* Track frequency if valid.

### Code:

```python
from collections import Counter

def maxFreq_bruteforce(s, maxLetters, minSize, maxSize):
    n = len(s)
    counter = Counter()
    for i in range(n):
        for l in range(minSize, maxSize+1):
            if i + l <= n:
                sub = s[i:i+l]
                if len(set(sub)) <= maxLetters:
                    counter[sub] += 1
    return max(counter.values() or [0])
```

### Complexity:

* Time: **O(n \* (maxSize-minSize+1) \* substring\_length)**
* Space: **O(n)**

### Verdict:

❌ TLE (too many substrings when `n = 10^5`).

---

## ✅ Approach 2: Sliding Window + HashMap (Optimal)

### Idea:

* Only consider substrings of length `minSize`.
* Use sliding window of size `minSize`.
* Count distinct letters in window.
* If valid, update frequency map.

### Code:

```python
from collections import defaultdict

def maxFreq(s, maxLetters, minSize, maxSize):
    n = len(s)
    counter = defaultdict(int)
    
    for i in range(n - minSize + 1):
        sub = s[i:i+minSize]
        if len(set(sub)) <= maxLetters:
            counter[sub] += 1
    
    return max(counter.values() or [0])
```

### Complexity:

* Time: **O(n \* minSize)** (because `set(sub)` recomputes each time).
* Space: **O(n)**.

### Why it works:

* Only `minSize` substrings matter.
* Count their frequencies if unique char constraint holds.

### Verdict:

✅ Works, but `len(set(sub))` makes it a bit inefficient.

---

## ✅ Approach 3: Sliding Window + Char Count Optimization

### Idea:

* Instead of building `set(sub)` every time:

  * Maintain a sliding frequency count of characters in the current window.
  * Maintain number of distinct letters.
* Update efficiently when sliding.

### Code:

```python
from collections import defaultdict

def maxFreq_optimized(s, maxLetters, minSize, maxSize):
    counter = defaultdict(int)
    char_count = defaultdict(int)
    
    distinct = 0
    left = 0
    
    # First window
    for right in range(minSize):
        char_count[s[right]] += 1
        if char_count[s[right]] == 1:
            distinct += 1
    if distinct <= maxLetters:
        counter[s[0:minSize]] += 1
    
    # Slide window
    for right in range(minSize, len(s)):
        char_count[s[left]] -= 1
        if char_count[s[left]] == 0:
            distinct -= 1
        left += 1
        
        char_count[s[right]] += 1
        if char_count[s[right]] == 1:
            distinct += 1
        
        if distinct <= maxLetters:
            counter[s[left:right+1]] += 1
    
    return max(counter.values() or [0])
```

### Complexity:

* Time: **O(n)** (sliding window only).
* Space: **O(n)**.

### Why it works:

* We never re-build sets; we just maintain distinct count efficiently.

### Verdict:

✅✅ **Best solution**

---

# 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(n \* substring\_length²) | ❌ | Too slow |
| Count minSize substrings with `set()` | O(n \* minSize) | ✅ | Works but slower |
| Sliding Window + Char Count | O(n) | ✅✅ | **Best solution** |

👉 In interviews, the **optimized sliding window** solution (Approach 3) is expected.