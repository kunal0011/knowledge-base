---
date: "2025-12-12"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1248: Count Number of Nice Subarrays"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1248: Count Number of Nice Subarrays

**Leetcode 1248: Count Number of Nice Subarrays**.

---

## 📌 Problem Recap

* You’re given an integer array `nums` and an integer `k`.
* A subarray is **nice** if it contains exactly `k` odd numbers.
* Return the **number of nice subarrays**.

---

## ✅ Approach 1: Brute Force (Check All Subarrays)

### Idea:

* Enumerate all subarrays `[i..j]`.
* Count odd numbers inside.
* If count == k → increment answer.

### Code:

```python
def numberOfSubarrays_bruteforce(nums, k):
    n = len(nums)
    ans = 0
    for i in range(n):
        odd_count = 0
        for j in range(i, n):
            if nums[j] % 2 == 1:
                odd_count += 1
            if odd_count == k:
                ans += 1
            elif odd_count > k:
                break
    return ans
```

### Complexity:

* Time: **O(n²)**
* Space: **O(1)**

### Why it works:

* Correct but too slow when `n = 10^5`.

### Verdict:

❌ TLE for big inputs.

---

## ✅ Approach 2: Prefix Sum + HashMap (Optimal)

### Idea:

* Convert `nums` → `parity`: `1` if odd, `0` if even.
* Problem reduces to: **count subarrays with sum == k**.
* Use prefix sum with a hashmap to count.

### Code:

```python
from collections import defaultdict

def numberOfSubarrays_prefix(nums, k):
    count = defaultdict(int)
    count[0] = 1  # base case
    prefix = 0
    ans = 0
    
    for num in nums:
        prefix += (num % 2)  # +1 if odd, +0 if even
        ans += count[prefix - k]
        count[prefix] += 1
    return ans
```

### Complexity:

* Time: **O(n)**
* Space: **O(n)**

### Why it works:

* Subarray sum trick: if `prefix[j] - prefix[i] == k`, then `[i+1..j]` has exactly `k` odds.
* HashMap tracks frequency of prefix sums.

### Verdict:

✅✅ Best solution

---

## ✅ Approach 3: Two-Pointer / Sliding Window

### Idea:

* Count subarrays with **at most k odds**.
* Count subarrays with **at most (k-1) odds**.
* Answer = `atMost(k) - atMost(k-1)`.

### Helper Function:

```python
def atMost(nums, k):
    left = 0
    ans = 0
    odd = 0
    
    for right in range(len(nums)):
        if nums[right] % 2 == 1:
            odd += 1
        while odd > k:
            if nums[left] % 2 == 1:
                odd -= 1
            left += 1
        ans += right - left + 1
    return ans
```

### Main:

```python
def numberOfSubarrays_sliding(nums, k):
    return atMost(nums, k) - atMost(nums, k-1)
```

### Complexity:

* Time: **O(n)**
* Space: **O(1)**

### Why it works:

* Sliding window counts all subarrays ending at `right` with at most `k` odds.
* Subtracting ensures we only keep exactly `k`.

### Verdict:

✅✅ Another best solution

### Approach 4: Sliding Window (Two Pointers) - Space Optimized

This is the most efficient approach regarding space complexity ($O(1)$ auxiliary space).

**Key Logic:**

1. Expand a `right` pointer. If we hit an odd number, increment `odd_count`.
2. Once `odd_count == k`, we have a valid window.
3. We need to count how many valid "starting positions" (prefixes) exist for this window.

   * While `odd_count == k`, we shrink from the `left`.
   * We count how many elements we pop off.
   * Since the element at `nums[right]` made the count valid, every time we successfully shrink `left` (removing evens until we remove an odd), we find a valid subarray.
4. **Handling trailing evens:** If the next numbers (as `right` increases) are even, the number of valid subarrays *remains the same* as the previous step. We don't need to re-scan `left`; we just re-add the `sub_count` we calculated.

**Python Code:**

Python

```python
def numberOfSubarrays_sliding(nums, k):
    left = 0
    result = 0
    odd_count = 0
    sub_count = 0  # Counts valid sub-segments for current 'right'
    
    for right in range(len(nums)):
        # If current number is odd
        if nums[right] % 2 == 1:
            odd_count += 1
            sub_count = 0 # Reset because we encountered a new odd number
            
        # While we have exactly k odds, calculate valid starts
        while odd_count == k:
            # Look at the number at 'left'
            if nums[left] % 2 == 1:
                odd_count -= 1
            sub_count += 1
            left += 1
            
        # Add the count of valid subarrays ending at 'right'
        result += sub_count
        
    return result
```

**Example Explanation:**

* `nums = [2, 1, 1, 2]`, `k = 2`
* **Iteration:**

  * `r=0 (2)`: odd=0. res+=0.
  * `r=1 (1)`: odd=1. res+=0.
  * `r=2 (1)`: odd=2. **Reset sub\_count=0.**

    * **While Loop:**

      * `l=0 (2)`: even. `sub_count=1`. `l=1`.
      * `l=1 (1)`: odd. `odd_count=1`. `sub_count=2`. `l=2`.
    * `res += 2`.
  * `r=3 (2)`: odd=1 (from previous step). Condition `odd_count == k` fails immediately?

    * *Correction in logic trace*: In the code, when `odd_count` drops to 1 inside the while loop, the loop breaks. The `sub_count` is preserved as 2.
    * Because `nums[3]` is even, `odd_count` remains 1? **Wait, strictly looking at the code:**
    * Actually, for the standard sliding window logic above, there is a nuance. The `sub_count` variable remembers how many valid prefixes we found *the last time we hit k odds*.
    * If `nums[r]` is even, we just add `sub_count` again.
    * Wait, my code above decreases `odd_count` to `k-1`. So the next even number won't trigger the while loop, but it will add `sub_count` to result.
    * `r=3 (2)`: Even. `odd_count` is 1. While loop skipped. `res += sub_count (2)`. Total = 4.
    * *Is 4 correct?* Subarrays: `[2,1,1]`, `[1,1]`, `[2,1,1,2]`, `[1,1,2]`. Yes.

**Complexity:**

* **Time:** $O(N)$ (Each element is visited at most twice, by left and right pointers).
* **Space:** $O(1)$ (No extra data structures used).

---

# 🎯 Final Summary

| Approach | Complexity | Works? | Notes |
| --- | --- | --- | --- |
| Brute Force | O(n²) | ❌ | Too slow |
| Prefix Sum + HashMap | O(n), O(n) | ✅✅ | Elegant, widely used |
| Sliding Window | O(n), O(1) | ✅✅ | Efficient, math trick |

👉 The **canonical solutions** are **Prefix Sum + HashMap** or **Sliding Window (atMost trick)**.

---