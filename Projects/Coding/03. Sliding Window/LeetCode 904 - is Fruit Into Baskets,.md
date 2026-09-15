---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 904: is "Fruit Into Baskets,"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 904: is "Fruit Into Baskets,

LeetCode 904 is "Fruit Into Baskets,”

which asks for the length of the longest subarray with at most two distinct elements.

Optimal Solution (Sliding Window Approach):

def totalFruit(fruits):

    from collections import defaultdict

    count = defaultdict(int)

    left = 0

    max\_fruits = 0

    for right in range(len(fruits)):

        count[fruits[right]] += 1

        while len(count) > 2:

            count[fruits[left]] -= 1

            if count[fruits[left]] == 0:

                del count[fruits[left]]

            left += 1

        max\_fruits = max(max\_fruits, right - left + 1)

    return max\_fruits

Explanation:

* **Sliding Window:** Maintain a window [left, right] that contains at most two distinct fruit types.
* **HashMap (**count**):** Track the count of each fruit type within the window.
* **Shrink Window:** When more than two distinct fruits are present, move the left pointer to maintain the constraint.
* **Track Maximum:** Update max\_fruits with the maximum window size encountered.

Complexity:

* **Time:** O(N) — each element is added and removed from the window at most once.
* **Space:** O(1) — at most 2 distinct fruit types tracked in the hashmap.