---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 457: Circular Array Loop"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 457: Circular Array Loop

Below is a **complete, interview-grade explanation** of **LeetCode 457 – Circular Array Loop**, structured exactly as requested.

---

## 📌 Problem Statement (LeetCode 457)

You are given a **circular integer array** `nums` of length `n`.  
Each element `nums[i]` represents the number of steps to move forward (positive) or backward (negative) from index `i`.

A **valid loop** must satisfy:

1. The loop length is **greater than 1**
2. All movements are in the **same direction** (all positive or all negative)
3. The loop is **circular** (wrap-around allowed)

Return `True` if such a loop exists, otherwise `False`.

---

## 🔑 Key Observations

1. **Circular movement**

   * Index calculation must be done using modulo:

     ```
     next = (current + nums[current]) % n
     ```
2. **Direction consistency**

   * Once direction is chosen (positive or negative), it must remain the same throughout the loop.
   * Mixed directions invalidate the loop.
3. **Self-loop is invalid**

   * A move that points back to itself (cycle length = 1) is **not allowed**.
4. **Cycle detection**

   * This is a classic **cycle detection** problem → use **Floyd’s Tortoise & Hare (two-pointer)** technique.

---

## 🧠 Two Pointer Technique (Why It Works)

* Use:

  * `slow`: moves 1 step at a time
  * `fast`: moves 2 steps at a time
* If a cycle exists, `slow` and `fast` **must meet**
* Abort if:

  * Direction changes
  * A self-loop is detected

This gives:

* **O(n)** time
* **O(1)** space

---

## 🧩 Algorithm

For each index `i`:

1. Set direction (`is_forward`)
2. Initialize `slow = i`, `fast = i`
3. Move pointers while direction remains consistent
4. If `slow == fast`:

   * Ensure it is **not a self-loop**
   * Return `True`
5. Otherwise continue
6. If all indices checked → return `False`

---

## 🧪 Worked Example

### Input

```text
nums = [2, -1, 1, 2, 2]
```

### Step-by-step

* Start at index `0`, direction = positive
* Moves:

  ```
  0 → 2 → 3 → 0
  ```
* Loop length = 3
* Direction consistent
* Valid cycle found ✅

**Output:** `True`

---

## 🧑‍💻 Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        n = len(nums)

        def next_index(curr: int) -> int:
            return (curr + nums[curr]) % n

        for i in range(n):
            slow = i
            fast = i
            is_forward = nums[i] > 0

            while True:
                # move slow pointer once
                next_slow = next_index(slow)
                if nums[next_slow] > 0 != is_forward:
                    break

                # move fast pointer twice
                next_fast = next_index(fast)
                if nums[next_fast] > 0 != is_forward:
                    break

                next_fast2 = next_index(next_fast)
                if nums[next_fast2] > 0 != is_forward:
                    break

                slow = next_slow
                fast = next_fast2

                # cycle detected
                if slow == fast:
                    # check for self-loop
                    if slow == next_index(slow):
                        break
                    return True

        return False
```

---

## ⏱️ Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(1)** |

---

## 🎯 Key Takeaway

* This problem is **cycle detection with constraints**
* The **two-pointer technique** is optimal because:

  * No extra memory
  * Clean handling of circular movement
* Always enforce:

  * Same direction
  * Loop length > 1

---

If you want, I can also provide:

* A **dry-run table**
* **Why marking visited works but is inferior**
* **Comparison with DFS approach**

Just tell me.