---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 862, "Shortest Subarray with Sum at Least K”,"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 862, "Shortest Subarray with Sum at Least K”,

LeetCode 862, **"Shortest Subarray with Sum at Least K****”**,

 the most optimal approach uses a **monotonic deque** (double-ended queue) combined with prefix sums. This achieves an **O(n)** time complexity.

Optimal Approach (Monotonic Queue + Prefix Sum):

1. **Prefix Sum Array:**

* Construct a prefix sum array prefix\_sum, where prefix\_sum[i] is the sum of the first i elements.

2. **Monotonic Deque:**

* Use a deque to store indices of prefix\_sum in increasing order.
* This helps maintain candidates for the shortest subarray.

3. **Algorithm Steps:**

* Iterate through the prefix\_sum:

* **Check for valid subarrays:** While the current prefix sum minus the first element in the deque is ≥ K, update the result.
* **Maintain monotonicity:** Remove indices from the back if the current prefix sum is less than or equal to the prefix sum at the deque’s back.
* Add the current index to the deque.

4. **Return:**

* The minimum length found, or -1 if no valid subarray exists.

Code Implementation (Python):

from collections import deque

def shortestSubarray(nums, k):

    n = len(nums)

    prefix\_sum = [0] \* (n + 1)

    for i in range(n):

        prefix\_sum[i + 1] = prefix\_sum[i] + nums[i]

    result = n + 1

    dq = deque()

    for i in range(n + 1):

        while dq and prefix\_sum[i] - prefix\_sum[dq[0]] >= k:

            result = min(result, i - dq.popleft())

        while dq and prefix\_sum[i] <= prefix\_sum[dq[-1]]:

            dq.pop()

        dq.append(i)

    return result if result != n + 1 else -1

Complexity Analysis:

* **Time Complexity:** O(n) — Each element is added and removed from the deque at most once.
* **Space Complexity:** O(n) for the prefix sum array and deque.

Key Insights:

* Monotonic deque optimizes redundant checks.
* Prefix sums simplify sum calculations.

This approach handles both positive and negative numbers efficiently.