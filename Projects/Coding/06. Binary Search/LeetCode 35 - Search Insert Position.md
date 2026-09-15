---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 35: Search Insert Position"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 35: Search Insert Position

---

### Problem Statement

You are given a **sorted array of distinct integers** `nums` and a **target** integer `target`.

Return the **index** if the target is found.  
If not, return the **index where it would be inserted** in order.

**Constraints**

* `1 <= nums.length <= 10^4`
* `-10^4 <= nums[i], target <= 10^4`
* `nums` is sorted in **strictly increasing order**
* Time complexity requirement: **O(log n)**

---

### Key Observation

Because the array is:

* **Sorted**
* **Has unique elements**

this is a **classic binary search variant**.

The crucial insight is:

> Even if the target is **not present**, binary search naturally converges to the **correct insertion index**.

At the end of binary search:

* The `left` pointer always points to the **smallest index where `nums[left] >= target`**
* If all elements are smaller, `left` points to `len(nums)`

Therefore:

* **Return `left` unconditionally**

---

### Binary Search Technique Used

This problem uses **lower bound binary search**:

Goal:

> Find the first index `i` such that `nums[i] >= target`

Binary search invariants:

* All elements **left of `left`** are `< target`
* All elements **right of `right`** are `> target`

Pointer movement:

* If `nums[mid] < target` → discard left half
* Else → discard right half but keep `mid`

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = left + (right - left) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return left
```

---

### Worked-Out Example

#### Example 1

```text
nums = [1, 3, 5, 6]
target = 5
```

Step-by-step:

```
left = 0, right = 3
mid = 1 → nums[1] = 3 < 5 → left = 2

left = 2, right = 3
mid = 2 → nums[2] = 5 == target → return 2
```

**Answer:** `2`

---

#### Example 2

```text
nums = [1, 3, 5, 6]
target = 2
```

Step-by-step:

```
left = 0, right = 3
mid = 1 → nums[1] = 3 > 2 → right = 0

left = 0, right = 0
mid = 0 → nums[0] = 1 < 2 → left = 1

loop ends
```

Return `left = 1`

**Insertion position:** index `1`

---

#### Example 3

```text
nums = [1, 3, 5, 6]
target = 7
```

Binary search progression:

```
left eventually becomes 4
```

**Insertion position:** index `4` (after last element)

---

### Why Returning `left` Always Works

After the loop:

* `right < left`
* `left` is the **first position where target can be placed**
* This satisfies both:

  * Found case
  * Not-found insertion case

This is the same logic used in **lower bound** implementations in STL and standard libraries.

---

### Complexity Analysis

* **Time Complexity:** `O(log n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Whenever you see:

* Sorted array
* “Find position / insert index”
* Logarithmic constraint

Think:

> **Binary Search → Lower Bound**

This pattern appears frequently in problems like:

* First/last occurrence
* Ceiling/floor element
* Insert position variants

If you want, I can also show:

* Recursive version
* Template-based lower bound pattern
* How this maps to `bisect_left` in Python