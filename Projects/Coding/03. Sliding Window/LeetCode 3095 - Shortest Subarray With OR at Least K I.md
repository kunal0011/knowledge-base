---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 3095: Shortest Subarray With OR at Least K I"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 3095: Shortest Subarray With OR at Least K I

**3095 – Shortest Subarray With OR at Least K I** step by step, with **all possible approaches**, including which work and why others don’t.

---

## Problem Restatement

You’re given an integer array `nums` and an integer `k`.  
Find the length of the **shortest subarray** such that the **bitwise OR** of its elements is at least `k`. If no such subarray exists, return `-1`.

---

## Key Observations

1. **Bitwise OR is monotonic** → Once a bit is set to `1`, it cannot be unset by adding more elements.
2. That means extending a subarray will never decrease the OR value.
3. Small constraints (`n ≤ 50` in this problem) → Brute force is possible.
4. Larger constraints would require sliding window or bit tracking.

---

## Approaches

### 1. **Brute Force – Check All Subarrays**

* For each starting index `i`, compute OR while extending to `j`.
* If OR ≥ k, track the minimum length.
* Time complexity: **O(n²)**, fine for `n ≤ 50`.

✅ Works  
❌ Won’t scale for large `n` (but problem constraints are small).

**Python code:**

```python
def minimumSubarrayLength(nums, k):
    n = len(nums)
    ans = float("inf")
    
    for i in range(n):
        cur = 0
        for j in range(i, n):
            cur |= nums[j]  # OR accumulate
            if cur >= k:
                ans = min(ans, j - i + 1)
                break  # no need to extend further, OR only grows
    
    return -1 if ans == float("inf") else ans
```

---

### 2. **Early Stopping Optimization (Better Brute Force)**

* Same as above, but stop immediately when OR ≥ k for a given `i`.
* Saves time, still **O(n²)** in worst case, but faster in practice.

✅ Works  
✅ Accepted within constraints  
❌ Still quadratic

---

### 3. **Sliding Window with Bit Count**

* Idea: Maintain a window `[l, r]` with bit frequency counts.
* Each time we extend `r`, update OR.
* If OR ≥ k, try shrinking `l`.
* Works like sliding window for sum, but trickier since OR is **not invertible** easily (removing a number might unset bits).
* To handle this, maintain **count of set bits per position (0–31)**.

Steps:

1. Maintain array `bit_count[32]` = how many numbers in window have bit set.
2. Add/remove numbers → update `bit_count`.
3. Reconstruct OR from `bit_count` each time.
4. Shrink left pointer while OR ≥ k.

Time complexity: **O(32n) = O(n)**.  
Space: **O(32) = O(1)**.

✅ Works  
✅ Much faster (linear)  
🚀 Useful if constraints were larger.

**Python code:**

```python
def minimumSubarrayLength(nums, k):
    n = len(nums)
    ans = float("inf")
    bit_count = [0] * 32

    def get_or():
        val = 0
        for b in range(32):
            if bit_count[b] > 0:
                val |= (1 << b)
        return val

    l = 0
    for r in range(n):
        # add nums[r]
        for b in range(32):
            if nums[r] & (1 << b):
                bit_count[b] += 1

        while l <= r and get_or() >= k:
            ans = min(ans, r - l + 1)
            # remove nums[l]
            for b in range(32):
                if nums[l] & (1 << b):
                    bit_count[b] -= 1
            l += 1

    return -1 if ans == float("inf") else ans
```

---

### 4. **Binary Search + OR Prefix**

* Compute prefix OR array → but problem: OR is not invertible, so prefix OR doesn’t help directly like prefix sums.
* You can binary search subarray length `L` and check if any subarray of length `L` has OR ≥ k.
* Checking OR for all subarrays of length `L` is still `O(n * L)` → inefficient.

❌ Won’t work efficiently here.  
Only brute force feasible for small `n`.

---

## Which Solutions Work?

* **Brute Force O(n²)** → Works (best for constraints).
* **Sliding Window + Bit Count O(32n)** → Also works (overkill for this problem, but best if `n` were large).
* **Prefix OR + Binary Search** → Doesn’t help since OR is not invertible.
* **Greedy** (like sums) → Doesn’t work because OR only grows, but removing left side may break OR ≥ k condition unpredictably without bit tracking.

---

## Final Takeaway

* For **LeetCode 3095 (n ≤ 50)** → Use **Brute Force O(n²)** (simplest, fast enough).
* For **generalized larger constraints** → Use **Sliding Window with Bit Count O(n)**.

---

👉 Do you want me to also **prove with an example why sliding window without bit counts fails** (i.e., naive OR tracking when shrinking left)?