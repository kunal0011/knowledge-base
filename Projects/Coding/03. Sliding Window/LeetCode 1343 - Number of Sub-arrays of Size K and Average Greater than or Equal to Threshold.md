---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1343: Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1343: Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold

**Leetcode 1343: Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold**.

---

## 📌 Problem Recap

We’re given:

* An integer array `arr`
* Two integers `k` (subarray size) and `threshold`

We need to count how many contiguous subarrays of size `k` have an **average ≥ threshold**.

---

## ✅ Approach 1: Brute Force

### Idea:

* For every subarray of size `k`, compute sum and check average.

### Code:

```python
def numOfSubarrays_bruteforce(arr, k, threshold):
    n = len(arr)
    ans = 0
    for i in range(n - k + 1):
        sub_sum = sum(arr[i:i+k])
        if sub_sum / k >= threshold:
            ans += 1
    return ans
```

### Complexity:

* Time: **O(n·k)**
* Space: **O(1)**

### Verdict:

❌ Works but too slow for `n=10^5`.

---

## ✅ Approach 2: Sliding Window (Optimal)

### Idea:

* Instead of recomputing the sum for each window, use sliding window:

  * Add `arr[right]`, remove `arr[left]`.
* Check if sum ≥ `threshold * k` (avoid floating-point).

### Code:

```python
def numOfSubarrays(arr, k, threshold):
    n = len(arr)
    target = threshold * k  # compare sum directly
    
    window_sum = sum(arr[:k])
    ans = 0
    
    if window_sum >= target:
        ans += 1
    
    for i in range(k, n):
        window_sum += arr[i] - arr[i-k]
        if window_sum >= target:
            ans += 1
    
    return ans
```

### Complexity:

* Time: **O(n)**
* Space: **O(1)**

### Why it works:

* Maintains rolling sum of size `k`.
* Avoids recomputation → efficient.

### Verdict:

✅✅ Best solution

---

## ✅ Approach 3: Prefix Sum (Alternative)

### Idea:

* Precompute prefix sums.
* Subarray sum = `prefix[i+k] - prefix[i]`.
* Check if ≥ `threshold * k`.

### Code:

```python
def numOfSubarrays_prefix(arr, k, threshold):
    n = len(arr)
    prefix = [0] * (n+1)
    for i in range(n):
        prefix[i+1] = prefix[i] + arr[i]
    
    target = threshold * k
    ans = 0
    
    for i in range(n-k+1):
        if prefix[i+k] - prefix[i] >= target:
            ans += 1
    return ans
```

### Complexity:

* Time: **O(n)**
* Space: **O(n)**

### Verdict:

✅ Works fine, but uses extra memory.

---

# 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(n·k) | ❌ | Too slow |
| Sliding Window | O(n), O(1) | ✅✅ | **Best solution** |
| Prefix Sum | O(n), O(n) | ✅ | Alternative |