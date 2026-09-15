---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 658: FInd k Closest element"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 658: FInd k Closest element

Leetcode 658 (FInd k Closest element):

The most optimal solution for LeetCode 658, **Find K Closest Elements**, uses a **binary search** approach with a time complexity of **O(log(n - k) + k)** and space complexity of **O(1)** (excluding the output).

Approach: Binary Search

1. **Observation:**

* The k closest elements will form a **contiguous subarray** in the sorted array.

2. **Binary Search:**

* Perform binary search on the **starting index** of this subarray.
* Set left = 0 and right = len(arr) - k.

3. **Condition:**

* Compare the distances of x from arr[mid] and arr[mid + k].
* If x - arr[mid] > arr[mid + k] - x, shift to the right (left = mid + 1), else shift to the left (right = mid).

4. **Result:**

* After the loop, the subarray from left to left + k is the answer.

Code:

from typing import List

def findClosestElements(arr: List[int], k: int, x: int) -> List[int]:

    left, right = 0, len(arr) - k

    while left < right:

        mid = (left + right) // 2

        if x - arr[mid] > arr[mid + k] - x:

            left = mid + 1

        else:

            right = mid

    return arr[left:left + k]

Example:

arr = [1, 2, 3, 4, 5]

k = 4

x = 3

print(findClosestElements(arr, k, x))  # Output: [1, 2, 3, 4]

Complexity Analysis:

* **Time:** O(log(n - k) + k)

* Binary search: O(log(n - k))
* Collecting k elements: O(k)

* **Space:** O(1) (excluding output)

This solution is highly efficient, especially for large arrays.