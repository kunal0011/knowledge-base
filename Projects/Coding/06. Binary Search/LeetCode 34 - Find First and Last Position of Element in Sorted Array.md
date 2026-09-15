---
date: "2025-12-23"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 34: Find First and Last Position of Element in Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 34: Find First and Last Position of Element in Sorted Array

Below is a complete, structured explanation of **Problem 34 – Find First and Last Position of Element in Sorted Array**.

---

## 1. Problem Statement

You are given a **sorted array of integers** `nums` (in non-decreasing order) and an integer `target`.

Your task is to return the **starting index** and **ending index** of `target` in `nums`.

* If `target` does not exist in the array, return `[-1, -1]`.
* The algorithm **must run in O(log n)** time.

---

## 2. Key Observation

1. The array is **sorted**, which immediately suggests **Binary Search**.
2. A standard binary search finds **any one occurrence**, but this problem requires:

   * the **leftmost (first) occurrence**
   * the **rightmost (last) occurrence**
3. Therefore:

   * Perform **two binary searches**

     * one biased toward the **left**
     * one biased toward the **right**

This preserves `O(log n)` complexity.

---

## 3. Binary Search Technique Used

### Modified Binary Search Pattern

Instead of stopping when `nums[mid] == target`, we:

* **Left boundary search**

  * Record `mid` as a candidate
  * Continue searching **left half**
* **Right boundary search**

  * Record `mid` as a candidate
  * Continue searching **right half**

### Why this works

Binary search guarantees logarithmic narrowing of the search space.  
By continuing after finding `target`, we force the algorithm to locate the **extreme boundaries**.

---

## 4. Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:

        def find_left() -> int:
            left, right = 0, len(nums) - 1
            index = -1

            while left <= right:
                mid = (left + right) // 2
                if nums[mid] >= target:
                    right = mid - 1
                else:
                    left = mid + 1

                if nums[mid] == target:
                    index = mid
            return index

        def find_right() -> int:
            left, right = 0, len(nums) - 1
            index = -1

            while left <= right:
                mid = (left + right) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid - 1

                if nums[mid] == target:
                    index = mid
            return index

        return [find_left(), find_right()]
```

---

## 5. Worked-Out Example

### Input

```text
nums = [5, 7, 7, 8, 8, 10]
target = 8
```

---

### Step 1: Find Leftmost Index

| left | right | mid | nums[mid] | action |
| --- | --- | --- | --- | --- |
| 0 | 5 | 2 | 7 | move right |
| 3 | 5 | 4 | 8 | record index=4, move left |
| 3 | 3 | 3 | 8 | record index=3, move left |

**Leftmost index = 3**

---

### Step 2: Find Rightmost Index

| left | right | mid | nums[mid] | action |
| --- | --- | --- | --- | --- |
| 0 | 5 | 2 | 7 | move right |
| 3 | 5 | 4 | 8 | record index=4, move right |
| 5 | 5 | 5 | 10 | move left |

**Rightmost index = 4**

---

### Final Output

```
[3, 4]
```

---

## 6. Complexity Analysis

* **Time Complexity**:  
  `O(log n)` (two binary searches)
* **Space Complexity**:  
  `O(1)` (constant extra space)

---

## 7. Takeaway Pattern

This problem is a **classic binary search boundary pattern**:

> When asked for **first / last occurrence** in a sorted array →  
> **Binary search with directional bias**

This pattern appears frequently in range and boundary-search problems.

---

If you want, I can also:

* derive this from a **single generalized binary search**
* explain **why normal binary search fails**
* map this problem to other similar problems for practice

  Below is the same **LeetCode 34** solution rewritten using **Python’s `bisect` module**, which internally uses **binary search**and fully satisfies the `O(log n)` requirement.

  ---

  ## 1. Problem Statement (Recap)

  Given a **sorted integer array** `nums` and a value `target`, return the **first and last position** of `target`.

  * If `target` is not present → return `[-1, -1]`
  * Required time complexity: **O(log n)**

  ---

  ## 2. Key Observation (Bisect Insight)

  Python’s `bisect` module gives us exactly what we need:

  * `bisect_left(nums, target)`

    * returns the **first index where `target` can be inserted**
  * `bisect_right(nums, target)`

    * returns the **first index where an element > target can be inserted**

  ### Critical insight

  If `target` exists:

  * **first occurrence** = `bisect_left`
  * **last occurrence** = `bisect_right - 1`

  If `target` does **not** exist:

  * `bisect_left` will point to an index where `nums[i] != target`

  ---

  ## 3. Binary Search Technique Used

  This is a **library-backed binary search boundary technique**:

  | Function | Purpose |
  | --- | --- |
  | `bisect_left` | Finds left boundary |
  | `bisect_right` | Finds right boundary |

  Each call runs in **O(log n)**.

  ---

  ## 4. Python 3 Solution (with typing)

  ```python
  from typing import List
  import bisect

  class Solution:
      def searchRange(self, nums: List[int], target: int) -> List[int]:
          left = bisect.bisect_left(nums, target)

          # target not present
          if left == len(nums) or nums[left] != target:
              return [-1, -1]

          right = bisect.bisect_right(nums, target) - 1
          return [left, right]
  ```

  ---

  ## 5. Worked-Out Example

  ### Input

  ```text
  nums = [5, 7, 7, 8, 8, 10]
  target = 8
  ```

  ---

  ### Step-by-step Execution

  #### Step 1: `bisect_left(nums, 8)`

  ```
  Insertion point = 3
  nums[3] == 8  → valid
  ```

  #### Step 2: `bisect_right(nums, 8)`

  ```
  Insertion point = 5
  Last index = 5 - 1 = 4
  ```

  ---

  ### Output

  ```
  [3, 4]
  ```

  ---

  ## 6. Edge Case Example

  ### Input

  ```text
  nums = [2, 2, 2, 2]
  target = 3
  ```

  * `bisect_left(nums, 3) = 4`
  * `left == len(nums)` → target not present

  ### Output

  ```
  [-1, -1]
  ```

  ---

  ## 7. Complexity Analysis

  | Metric | Value |
  | --- | --- |
  | Time | `O(log n)` |
  | Space | `O(1)` |

  ---

  ## 8. When to Prefer `bisect`

  Use `bisect` when:

  * You want **clean, readable boundary logic**
  * The array is **already sorted**
  * Language/library support is allowed

  Avoid it when:

  * You are implementing binary search **from scratch**
  * Interview explicitly restricts library usage

  ---

  ## 9. Pattern Takeaway

  > **Range in sorted array**  
  > → `bisect_left` + `bisect_right - 1`

  This pattern directly generalizes to:

  * count of occurrences
  * insertion ranges
  * lower/upper bound problems

  ---

  If you want, I can also:

  * convert this into a **count occurrences** problem
  * show the **C++ STL equivalent**
  * compare **manual binary search vs bisect** in interviews