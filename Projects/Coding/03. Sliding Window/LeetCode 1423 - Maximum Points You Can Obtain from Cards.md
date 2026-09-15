---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1423: Maximum Points You Can Obtain from Cards"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1423: Maximum Points You Can Obtain from Cards

**Leetcode 1423: Maximum Points You Can Obtain from Cards**.

---

## 📌 Problem Recap

We’re given an array `cardPoints` and an integer `k`.

We must choose exactly `k` cards **from the start or end** of the array.  
Return the maximum total score we can obtain.

---

## ✅ Approach 1: Brute Force (Try All Splits)

### Idea:

* You can take `i` cards from the front and `k-i` cards from the back (for all `0 ≤ i ≤ k`).
* Compute sum for each possibility, take max.

### Code:

```python
def maxScore_bruteforce(cardPoints, k):
    n = len(cardPoints)
    ans = 0
    for i in range(k+1):  
        left_sum = sum(cardPoints[:i])
        right_sum = sum(cardPoints[n-(k-i):])
        ans = max(ans, left_sum + right_sum)
    return ans
```

### Complexity:

* Time: **O(k²)** (because slicing sums cost `O(k)` each).
* Space: **O(1)**

❌ Works but too slow for large input.

---

## ✅ Approach 2: Prefix + Suffix Precomputation

### Idea:

* Precompute prefix sums (left side) and suffix sums (right side).
* Then try all splits in **O(k)**.

### Code:

```python
def maxScore_prefix_suffix(cardPoints, k):
    n = len(cardPoints)
    
    prefix = [0] * (k+1)
    suffix = [0] * (k+1)
    
    for i in range(1, k+1):
        prefix[i] = prefix[i-1] + cardPoints[i-1]
        suffix[i] = suffix[i-1] + cardPoints[n-i]
    
    ans = 0
    for i in range(k+1):
        ans = max(ans, prefix[i] + suffix[k-i])
    
    return ans
```

### Complexity:

* Time: **O(k)**
* Space: **O(k)**

✅ Works efficiently.

---

## ✅ Approach 3: Sliding Window (Optimal)

### Key Observation:

* Instead of choosing `k` cards, we can think of **removing `n-k` consecutive cards** (a subarray) and keeping the rest.
* Max score = `total_sum - min_subarray_sum(n-k)`.

### Code:

```python
def maxScore(cardPoints, k):
    n = len(cardPoints)
    total = sum(cardPoints)
    window_size = n - k
    
    if window_size == 0:
        return total
    
    # initial window sum
    window_sum = sum(cardPoints[:window_size])
    min_sum = window_sum
    
    for i in range(window_size, n):
        window_sum += cardPoints[i] - cardPoints[i - window_size]
        min_sum = min(min_sum, window_sum)
    
    return total - min_sum
```

### Complexity:

* Time: **O(n)**
* Space: **O(1)**

✅✅ This is the **canonical best solution**.

---

# 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(k²) | ❌ | Too slow |
| Prefix + Suffix | O(k), O(k) | ✅ | Good |
| Sliding Window | O(n), O(1) | ✅✅ | Best solution |