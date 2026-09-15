---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 75: Sort Colors"
tags:
  - leetcode
  - coding
  - two-pointers
  - amazon
  - google
---

# LeetCode 75: Sort Colors (Dutch National Flag)

**Target Companies:** Amazon (Top Classic), Google, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Two Pointers / 3-Way Partitioning

---

### Problem Statement

Given an array `nums` with $n$ objects colored red, white, or blue, sort them **in-place** so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers `0`, `1`, and `2` to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function and in **one pass** using only **constant extra space**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` containing only values in $\{0, 1, 2\}$.
- **Output:** None (in-place modification of `nums`).
- **Constraints:**
  - $n == 	ext{nums.length}$
  - $1 \le n \le 300$
  - $	ext{nums}[i] \in \{0, 1, 2\}$

---

### Key Idea & Intuition

- **Dijkstra's Dutch National Flag Algorithm:**
  - We want to partition the array into three contiguous zones:
    1. Zone 0 (Red): `[0 ... low - 1]`
    2. Zone 1 (White): `[low ... mid - 1]`
    3. Unexplored: `[mid ... high]`
    4. Zone 2 (Blue): `[high + 1 ... n - 1]`
- **Pointer Semantics:**
  - `low`: Next position where a `0` should be placed.
  - `mid`: Current element being inspected.
  - `high`: Next position where a `2` should be placed.
- **Why One-Pass works:**
  - When `nums[mid] == 0`: Swap `nums[low]` and `nums[mid]`, increment both `low++` and `mid++` (since the swapped-in element from `low` was guaranteed to be `1` or already inspected).
  - When `nums[mid] == 1`: Already in correct middle zone, simply advance `mid++`.
  - When `nums[mid] == 2`: Swap `nums[mid]` and `nums[high]`, decrement `high--`. **Do NOT advance `mid`**, because the newly swapped-in element at `mid` from `high` has not been inspected yet!

---

### Solution Approach (Step-by-Step)

1. Initialize `low = 0`, `mid = 0`, `high = len(nums) - 1`.
2. While `mid <= high`:
   - If `nums[mid] == 0`:
     - Swap `nums[low]` with `nums[mid]`.
     - `low += 1`, `mid += 1`.
   - Else if `nums[mid] == 1`:
     - `mid += 1`.
   - Else (`nums[mid] == 2`):
     - Swap `nums[mid]` with `nums[high]`.
     - `high -= 1` (do not advance `mid`).
3. Terminate when `mid > high`.

---

### Visual Algorithm Walkthrough

```
Initial: nums = [2, 0, 2, 1, 1, 0]
                 ^              ^
                mid,low        high

Step 1: nums[mid] == 2 -> Swap nums[mid], nums[high], high--
        nums = [0, 0, 2, 1, 1, 2], mid=0, low=0, high=4

Step 2: nums[mid] == 0 -> Swap nums[low], nums[mid], low++, mid++
        nums = [0, 0, 2, 1, 1, 2], mid=1, low=1, high=4

Step 3: nums[mid] == 0 -> Swap nums[low], nums[mid], low++, mid++
        nums = [0, 0, 2, 1, 1, 2], mid=2, low=2, high=4

Step 4: nums[mid] == 2 -> Swap nums[mid], nums[high], high--
        nums = [0, 0, 1, 1, 2, 2], mid=2, low=2, high=3

Step 5: nums[mid] == 1 -> mid++
        nums = [0, 0, 1, 1, 2, 2], mid=3, low=2, high=3

Step 6: nums[mid] == 1 -> mid++
        nums = [0, 0, 1, 1, 2, 2], mid=4, low=2, high=3 (mid > high: STOP)
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Array
- **Input:** `nums = [2, 0, 1]`
- **Trace:**
  - `low=0, mid=0, high=2`: `nums[0]==2` -> Swap with `nums[2]`, `high=1`. Array: `[1, 0, 2]`.
  - `low=0, mid=0, high=1`: `nums[0]==1` -> `mid=1`. Array: `[1, 0, 2]`.
  - `low=0, mid=1, high=1`: `nums[1]==0` -> Swap with `nums[0]`, `low=1, mid=2`. Array: `[0, 1, 2]`.
  - `mid > high` (2 > 1) -> Finished.
- **Output:** `[0, 1, 2]`

#### Example 2: Already Sorted Array
- **Input:** `nums = [0, 1, 2]`
- **Trace:**
  - `nums[0]==0` -> `low=1, mid=1`
  - `nums[1]==1` -> `mid=2`
  - `nums[2]==2` -> `high=1`, `mid=2`
  - `mid > high` -> Finished.
- **Output:** `[0, 1, 2]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        low, mid, high = 0, 0, len(nums) - 1
        
        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:  # nums[mid] == 2
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    void sortColors(std::vector<int>& nums) {
        int low = 0, mid = 0, high = static_cast<int>(nums.size()) - 1;
        
        while (mid <= high) {
            if (nums[mid] == 0) {
                std::swap(nums[low++], nums[mid++]);
            } else if (nums[mid] == 1) {
                mid++;
            } else {
                std::swap(nums[mid], nums[high--]);
            }
        }
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public void sortColors(int[] nums) {
        int low = 0, mid = 0, high = nums.length - 1;
        
        while (mid <= high) {
            if (nums[mid] == 0) {
                int temp = nums[low];
                nums[low] = nums[mid];
                nums[mid] = temp;
                low++;
                mid++;
            } else if (nums[mid] == 1) {
                mid++;
            } else {
                int temp = nums[mid];
                nums[mid] = nums[high];
                nums[high] = temp;
                high--;
            }
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Every element is examined at most once; in each step, either `mid` increases or `high` decreases.
- **Space Complexity:** $O(1)$ — Strict in-place swap using three pointer variables.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Three-Way Partitioning (Dutch National Flag). Critical for Quicksort optimization (handling many duplicate pivot elements).
- **Trap:** Incrementing `mid` after swapping with `high`: NEVER increment `mid` when swapping `nums[mid]` with `nums[high]`, because the incoming element from `high` could be `0` or `2` and needs re-evaluation!
