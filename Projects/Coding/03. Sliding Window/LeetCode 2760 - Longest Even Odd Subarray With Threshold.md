---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2760: Longest Even Odd Subarray With Threshold"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 2760: Longest Even Odd Subarray With Threshold

**LeetCode 2760 – Longest Even Odd Subarray With Threshold** in detail with all approaches.

---

## 🔹 Problem Restatement

You are given:

* An integer array `nums`.
* An integer `threshold`.

Find the **longest subarray** that:

1. Starts with an **even** number.
2. Alternates between **even and odd** numbers.
3. Every element is ≤ `threshold`.

Return the length of the longest such subarray. If none exists, return `0`.

---

## 🔹 Key Observations

1. The subarray must **start with an even number**.
2. Once it starts, it must follow a strict **even → odd → even → odd ...** alternating pattern.
3. All numbers must be ≤ `threshold`.
4. The problem reduces to: **scan linearly, reset when conditions break**.

---

## 🔹 Approaches

### **1. Brute Force (Check all subarrays)**

* Try all subarrays starting at `i`.
* Verify conditions (even start, alternation, ≤ threshold).
* Track max length.

⏱ Complexity: **O(n²)**  
✅ Works (but too slow for `n=1000`).

---

### **2. Greedy Linear Scan (Optimal)**

* Traverse from left to right.
* Maintain `cur_len` = length of current valid subarray.
* If current number breaks the rule:

  * Reset `cur_len` (start new subarray if current number is even and ≤ threshold).
* Update `ans` with maximum length seen.

⏱ Complexity: **O(n)**  
✅ Works efficiently.

---

## 🔹 Implementation

### Brute Force (for clarity)

```python
def longestAlternatingSubarray(nums, threshold):
    n = len(nums)
    ans = 0

    for i in range(n):
        if nums[i] % 2 == 0 and nums[i] <= threshold:  # must start with even
            length = 1
            for j in range(i+1, n):
                if nums[j] > threshold:  # violates threshold
                    break
                if nums[j] % 2 != nums[j-1] % 2:  # alternates parity
                    length += 1
                    ans = max(ans, length)
                else:
                    break
            ans = max(ans, length)

    return ans
```

---

### Greedy Linear (Final Solution)

```python
def longestAlternatingSubarray(nums, threshold):
    n = len(nums)
    ans = 0
    cur_len = 0

    for i in range(n):
        if nums[i] > threshold:  # invalid number
            cur_len = 0
            continue

        if cur_len == 0:  # potential start
            if nums[i] % 2 == 0:
                cur_len = 1
                ans = max(ans, cur_len)
        else:
            if nums[i] % 2 != nums[i-1] % 2:  # alternates
                cur_len += 1
                ans = max(ans, cur_len)
            else:  # restart
                cur_len = 1 if nums[i] % 2 == 0 else 0
                ans = max(ans, cur_len)

    return ans
```

---

## 🔹 Example Walkthrough

Input:

```text
nums = [3, 2, 5, 4], threshold = 5
```

Steps:

* `3` (odd, ≤ 5) → invalid start.
* `2` (even, ≤ 5) → start → `cur_len = 1`.
* `5` (odd, ≤ 5, alternates) → `cur_len = 2`.
* `4` (even, ≤ 5, alternates) → `cur_len = 3`.

✅ Answer = 3.

---

## 🔹 Which Works / Fails?

* **Brute force O(n²)** → Works but slower.
* **Greedy O(n)** → Works efficiently.
* **DP not needed** since greedy already handles pattern.