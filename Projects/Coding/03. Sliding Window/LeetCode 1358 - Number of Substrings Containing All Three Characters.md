---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1358: Number of Substrings Containing All Three Characters"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1358: Number of Substrings Containing All Three Characters

**Leetcode 1358: Number of Substrings Containing All Three Characters**.

---

## 📌 Problem Recap

We’re given a string `s` containing only `'a'`, `'b'`, `'c'`.  
We must count how many substrings contain **at least one `'a'`, one `'b'`, and one `'c'`**.

---

## ✅ Approach 1: Brute Force

### Idea:

* Generate all substrings → check if it has all 3 chars.
* Way too slow (`O(n^3)` time for substring generation + checking).

### Code:

```python
def numberOfSubstrings_brute(s):
    n = len(s)
    ans = 0
    for i in range(n):
        freq = {"a":0, "b":0, "c":0}
        for j in range(i, n):
            freq[s[j]] += 1
            if all(freq[ch] > 0 for ch in "abc"):
                ans += 1
    return ans
```

### Complexity:

* Time: **O(n²)**
* Space: **O(1)**

✅ Works for small inputs but too slow for `n = 10^5`.

---

## ✅ Approach 2: Sliding Window (Optimal)

### Idea:

* Maintain a window `[left, right]` with counts of `'a','b','c'`.
* Expand `right`, and once window contains all 3 →

  * Any substring starting at `left` and ending at `right, right+1, …, n-1` will also be valid.
  * So add `n - right` to answer.
* Then shrink from `left`.

### Code:

```python
def numberOfSubstrings(s):
    n = len(s)
    count = {"a":0, "b":0, "c":0}
    ans = 0
    left = 0
    
    for right in range(n):
        count[s[right]] += 1
        
        # shrink until window valid
        while all(count[ch] > 0 for ch in "abc"):
            ans += n - right
            count[s[left]] -= 1
            left += 1
    
    return ans
```

### Complexity:

* Time: **O(n)** (each index enters & exits window once).
* Space: **O(1)**

✅ Efficient and clean.

---

## ✅ Approach 3: Last Seen Index Trick

### Idea:

* Instead of sliding window, track **last seen indices** of `a, b, c`.
* At each index `i`, if all three chars have been seen, then  
  `ans += 1 + min(last['a'], last['b'], last['c'])`.

### Code:

```python
def numberOfSubstrings_lastseen(s):
    last = {"a": -1, "b": -1, "c": -1}
    ans = 0
    
    for i, ch in enumerate(s):
        last[ch] = i
        if -1 not in last.values():  # all seen at least once
            ans += 1 + min(last.values())
    return ans
```

### Complexity:

* Time: **O(n)**
* Space: **O(1)**

✅ Shorter and more elegant.

---

# 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(n²) | ❌ | Too slow |
| Sliding Window | O(n), O(1) | ✅ | Canonical solution |
| Last Seen Index | O(n), O(1) | ✅✅ | Most elegant solution |

---