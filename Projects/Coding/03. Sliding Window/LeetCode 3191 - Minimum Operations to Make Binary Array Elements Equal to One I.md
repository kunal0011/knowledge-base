---
date: "2025-09-13"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 3191: Minimum Operations to Make Binary Array Elements Equal to One I"
tags:
  - leetcode
  - coding
  - sliding-window
---

# LeetCode 3191: Minimum Operations to Make Binary Array Elements Equal to One I

**Leetcode 3191: Minimum Operations to Make Binary Array Elements Equal to One I**.

---

## 📌 Problem Recap

You're given a binary array `nums` consisting of 0s and 1s. The task is to determine the minimum number of operations required to make all elements in the array equal to 1. The allowed operation is:

* **Flip** any 3 consecutive elements in the array.

Flipping means changing a 0 to 1 and a 1 to 0.

If it's impossible to make all elements equal to 1, return `-1`.

---

## ✅ Approach: Greedy Simulation

### Key Insight

To solve this problem efficiently:

1. **Left-to-Right Traversal**: Traverse the array from left to right.
2. **Flip When Necessary**: Whenever you encounter a 0 at position `i`, flip the next two elements (`i+1` and `i+2`) to ensure the current element becomes 1.
3. **Boundary Check**: If you encounter a 0 in the last two positions (`n-2` or `n-1`), it's impossible to flip them because there aren't enough elements to perform the operation. In this case, return `-1`.

### Code Implementation

```python
def minOperations(nums):
    n = len(nums)
    operations = 0

    for i in range(n - 2):
        if nums[i] == 0:
            nums[i + 1] ^= 1
            nums[i + 2] ^= 1
            operations += 1

    # Check if the last two elements are 0
    if nums[n - 2] == 0 or nums[n - 1] == 0:
        return -1

    return operations
```

### Explanation

* **XOR Operation**: The `^= 1` operation flips the bits. If `nums[i]` is 0, it becomes 1; if it's 1, it becomes 0.
* **Greedy Approach**: By flipping the next two elements whenever a 0 is encountered, we ensure that the current element becomes 1 without affecting previously processed elements.

---

## ✅ Time and Space Complexity

* **Time Complexity**: O(n), where n is the length of the array. We traverse the array once.
* **Space Complexity**: O(1), as we use a constant amount of extra space.

---

## ✅ Example Walkthrough

### Input: `[0, 1, 1, 1, 0, 0]`

1. **Initial State**: `[0, 1, 1, 1, 0, 0]`

   * At index 0, `nums[0]` is 0.
   * Flip elements at indices 1 and 2.
   * Array becomes: `[1, 0, 0, 1, 0, 0]`
   * Operations: 1
2. **Next State**: `[1, 0, 0, 1, 0, 0]`

   * At index 1, `nums[1]` is 0.
   * Flip elements at indices 2 and 3.
   * Array becomes: `[1, 1, 1, 0, 0, 0]`
   * Operations: 2
3. **Next State**: `[1, 1, 1, 0, 0, 0]`

   * At index 3, `nums[3]` is 0.
   * Flip elements at indices 4 and 5.
   * Array becomes: `[1, 1, 1, 1, 1, 1]`
   * Operations: 3

**Final Output**: `3`

---

### Input: `[0, 1, 1, 1]`

1. **Initial State**: `[0, 1, 1, 1]`

   * At index 0, `nums[0]` is 0.
   * Flip elements at indices 1 and 2.
   * Array becomes: `[1, 0, 0, 1]`
   * Operations: 1
2. **Next State**: `[1, 0, 0, 1]`

   * At index 1, `nums[1]` is 0.
   * Flip elements at indices 2 and 3.
   * Array becomes: `[1, 1, 1, 0]`
   * Operations: 2

**Final Output**: `-1` (It's impossible to make all elements 1)