---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 930, Binary Subarrays With Sum"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 930, Binary Subarrays With Sum

LeetCode 930, **Binary Subarrays With Sum**

 utilizes the **prefix sum with a hashmap** approach, achieving **O(n)** time complexity and **O(n)** space complexity.

Approach:

* Use a prefix sum to keep track of the cumulative sum as you iterate through the array.
* Store the frequency of each prefix sum in a hashmap.
* For each new prefix sum curr\_sum, the number of subarrays with the target sum goal equals the count of curr\_sum - goal seen so far.

Python Implementation:

from collections import Counter

def numSubarraysWithSum(nums, goal):

    prefix\_sum\_count = Counter()

    prefix\_sum\_count[0] = 1

    curr\_sum = 0

    result = 0

    for num in nums:

        curr\_sum += num

        result += prefix\_sum\_count[curr\_sum - goal]

        prefix\_sum\_count[curr\_sum] += 1

    return result

Example:

nums = [1, 0, 1, 0, 1]

goal = 2

print(numSubarraysWithSum(nums, goal))  # Output: 4

Complexity:

* **Time:** O(n) — Each element is processed once.
* **Space:** O(n) — For the hashmap to store prefix sums.