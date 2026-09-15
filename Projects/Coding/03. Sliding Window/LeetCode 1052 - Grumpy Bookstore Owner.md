---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1052: Grumpy Bookstore Owner"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 1052: Grumpy Bookstore Owner

LeetCode 1052 (Grumpy Bookstore Owner) using the sliding window technique:

from typing import List

def maxSatisfied(customers: List[int], grumpy: List[int], minutes: int) -> int:

    # Base satisfaction from customers when the owner is not grumpy

    base\_satisfaction = sum(c for c, g in zip(customers, grumpy) if g == 0)

    # Additional satisfaction from using the secret technique

    additional\_satisfaction = 0

    current\_gain = 0

    # Calculate the gain for the first 'minutes' window

    for i in range(minutes):

        if grumpy[i] == 1:

            current\_gain += customers[i]

    additional\_satisfaction = current\_gain

    # Slide the window through the rest of the array

    for i in range(minutes, len(customers)):

        if grumpy[i] == 1:

            current\_gain += customers[i]

        if grumpy[i - minutes] == 1:

            current\_gain -= customers[i - minutes]

        additional\_satisfaction = max(additional\_satisfaction, current\_gain)

    return base\_satisfaction + additional\_satisfaction

# Example usage

customers = [1,0,1,2,1,1,7,5]

grumpy = [0,1,0,1,0,1,0,1]

minutes = 3

print(maxSatisfied(customers, grumpy, minutes))  # Output: 16

**Explanation:**

* Calculate satisfaction from non-grumpy minutes.
* Use a sliding window to find the maximum gain when the secret technique is used.
* Return the sum of base satisfaction and the maximum additional gain.