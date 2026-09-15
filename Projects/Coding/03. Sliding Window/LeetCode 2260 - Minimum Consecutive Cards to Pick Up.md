---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2260: Minimum Consecutive Cards to Pick Up"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 2260: Minimum Consecutive Cards to Pick Up

**LeetCode 2260 – Minimum Consecutive Cards to Pick Up**.

---

## 🔹 Problem Restatement

You’re given an array `cards` (length `n`), where `cards[i]` is the value of the `i`-th card.  
Return the **minimum length of a subarray** containing at least **two equal cards**.  
If no such subarray exists, return `-1`.

---

## 🔹 Key Observations

1. We’re looking for the **shortest distance between two equal numbers** in the array.
2. If a number appears multiple times, the **closest two appearances** determine the shortest subarray containing them.
3. So we need to track the **last seen index** of each number while scanning.

---

## 🔹 Approaches

### **1. Brute Force**

* Try all pairs `(i, j)` with `cards[i] == cards[j]`.
* Track minimum `j - i + 1`.

⏱ Complexity: **O(n²)** (too slow for `n = 10⁵`).  
❌ Not acceptable.

---

### **2. Hash Map for Last Seen Index (Optimal)**

* Maintain a dictionary `last_seen` that maps card value → last index.
* Traverse array:

  * If card was seen before, compute length = `i - last_seen[val] + 1`.
  * Update minimum.
  * Update `last_seen[val] = i`.
* If no duplicates found, return `-1`.

⏱ Complexity: **O(n)**  
✅ Works optimally.  
💾 Space: **O(n)** for hashmap.

---

## 🔹 Implementation

```python
def minimumCardPickup(cards):
    last_seen = {}
    ans = float("inf")

    for i, val in enumerate(cards):
        if val in last_seen:
            ans = min(ans, i - last_seen[val] + 1)
        last_seen[val] = i  # update latest position

    return -1 if ans == float("inf") else ans
```

---

## 🔹 Example Walkthrough

Input:

```
cards = [3, 4, 2, 3, 4, 7]
```

Steps:

* `3` → not seen before → store {3:0}.
* `4` → store {3:0, 4:1}.
* `2` → store {3:0, 4:1, 2:2}.
* `3` → seen at index 0 → length = 3-0+1 = 4 → ans=4 → update {3:3}.
* `4` → seen at index 1 → length = 4-1+1 = 4 → ans=4 → update {4:4}.
* `7` → new → store {…, 7:5}.

✅ Answer = 4.

---

## 🔹 Which Solutions Work?

* **Brute force O(n²)** → Works but too slow for large `n`.
* **Hash map O(n)** → Correct & efficient (best solution).
* **Sorting approach** (sort by value, then check neighbors) → O(n log n), works but unnecessary since hashmap is faster.