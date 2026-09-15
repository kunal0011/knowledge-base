---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2024: Maximize the Confusion of an Exam"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 2024: Maximize the Confusion of an Exam

**Leetcode 2024: Maximize the Confusion of an Exam**.

---

## 📌 Problem Recap

We’re given:

* A string `answerKey` containing only `'T'` and `'F'`.
* An integer `k`.

We can change **at most k answers** (`T ↔ F`).  
We want the **length of the longest substring with all characters the same** (all `'T'` or all `'F'`).

---

## ✅ Approach 1: Brute Force

### Idea:

* Try every substring.
* Count how many flips are needed to make it uniform (`T` or `F`).
* If ≤ `k`, update answer.

### Complexity:

* Time: **O(n²)** (checking substrings).
* Too slow for `n = 10⁵`.

❌ Not feasible.

---

## ✅ Approach 2: Sliding Window (Optimal)

### Key Observation:

* Problem = "longest window where we can flip ≤ `k` characters".
* Equivalent to:

  * Longest substring of `T` after flipping ≤ `k` `F`s.
  * Longest substring of `F` after flipping ≤ `k` `T`s.
* Answer = max of both.

---

### Code:

```python
def maxConsecutiveAnswers(answerKey, k):
    def maxWindow(ch):
        left = 0
        flips = 0
        ans = 0
        for right in range(len(answerKey)):
            if answerKey[right] != ch:
                flips += 1
            while flips > k:
                if answerKey[left] != ch:
                    flips -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans
    
    return max(maxWindow('T'), maxWindow('F'))
```

---

### Complexity:

* Time: **O(n)** (two sliding window scans).
* Space: **O(1)**.

✅ Works in linear time.

---

## ✅ Approach 3: Sliding Window with Count (Alternative)

Instead of running twice, we can keep track of counts of `T` and `F` in one pass:

### Code:

```python
def maxConsecutiveAnswers(answerKey, k):
    left = 0
    count = {"T":0, "F":0}
    ans = 0
    
    for right in range(len(answerKey)):
        count[answerKey[right]] += 1
        
        # shrink if window invalid (too many flips)
        while min(count["T"], count["F"]) > k:
            count[answerKey[left]] -= 1
            left += 1
        
        ans = max(ans, right - left + 1)
    
    return ans
```

---

### Why This Works:

* In any window, to make it uniform, you flip the smaller count (`min(count["T"], count["F"])`).
* If that exceeds `k`, shrink from left.

---

## 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(n²) | ❌ | Too slow |
| Sliding Window (two passes) | O(n) | ✅ | Clean & intuitive |
| Sliding Window (one pass) | O(n) | ✅✅ | **Best solution** |