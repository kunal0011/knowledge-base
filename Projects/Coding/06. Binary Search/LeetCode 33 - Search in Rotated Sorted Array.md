---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 33: Search in Rotated Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
  - arrays
  - amazon
  - google
  - meta
---

# LeetCode 33: Search in Rotated Sorted Array

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Modified Binary Search on Rotated Sorted Invariant

---

### Problem Statement

There is an integer array `nums` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, `nums` is possibly rotated at an unknown pivot index $k$ ($1 \le k < \text{nums.length}$) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`.

Given the array `nums` after the possible rotation and an integer `target`, return *the index of `target` if it is in `nums`, or `-1` if it is not in `nums`*.

You must write an algorithm with **$\mathcal{O}(\log n)$ runtime complexity**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `target: int`
- **Output:** `int` (index of target or -1)
- **Constraints:**
  - $1 \le \text{nums.length} \le 5000$
  - $-10^4 \le \text{nums}[i], \text{target} \le 10^4$
  - All values of `nums` are **strictly unique**.
  - `nums` is an ascending array that has been possibly rotated.

---

### Key Idea & Intuition

#### The One-Sorted-Half Invariant:
In any rotated sorted array with distinct elements, if we divide the array at midpoint `mid`:
$$\text{At least one of the two halves } [left, mid] \text{ or } [mid, right] \text{ is strictly monotonically sorted!}$$

We can easily determine which half is sorted by comparing `nums[left]` with `nums[mid]`:
1. **Case 1: `nums[left] <= nums[mid]` (Left Half is Sorted)**
   - All elements in $[left, mid]$ are in standard ascending order.
   - We check if `target` falls cleanly within the sorted left boundary:
     $$nums[left] \le target < nums[mid]$$
   - If yes: search left (`right = mid - 1`).
   - Else: target cannot be in the left half $\implies$ search right (`left = mid + 1`).
2. **Case 2: `nums[left] > nums[mid]` (Right Half is Sorted)**
   - The pivot lies in the left half, meaning all elements in $[mid, right]$ are in standard ascending order.
   - We check if `target` falls cleanly within the sorted right boundary:
     $$nums[mid] < target \le nums[right]$$
   - If yes: search right (`left = mid + 1`).
   - Else: target cannot be in the right half $\implies$ search left (`right = mid - 1`).

At each step, we eliminate half of the search space, maintaining strict $\mathcal{O}(\log n)$ time.

---

### Solution Approach (Step-by-Step)

1. **Initialize Pointers:**
   - `left = 0`, `right = len(nums) - 1`.
2. **Binary Search While `left <= right`:**
   - Compute `mid = left + (right - left) // 2`.
   - If `nums[mid] == target`, return `mid`.
   - **Check Left Half Sorted (`nums[left] <= nums[mid]`):**
     - If `nums[left] <= target < nums[mid]`:
       - `right = mid - 1`
     - Else:
       - `left = mid + 1`
   - **Else Right Half Sorted (`nums[left] > nums[mid]`):**
     - If `nums[mid] < target <= nums[right]`:
       - `left = mid + 1`
     - Else:
       - `right = mid - 1`
3. **Target Not Found:**
   - If loop terminates without match, return `-1`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [4, 5, 6, 7, 0, 1, 2]`, `target = 0`

```
Search space:
Indices:  0  1  2  3  4  5  6
Values:  [4, 5, 6, 7, 0, 1, 2]

Iteration 1:
  left = 0 (4), right = 6 (2)
  mid = 3, nums[mid] = 7
  Target = 0 != 7.
  Check sorted half:
    nums[left] (4) <= nums[mid] (7) -> Left half [4, 5, 6, 7] is SORTED!
  Is target in left range [4, 7)?
    4 <= 0 < 7 is FALSE!
    So target must be in right half -> left = mid + 1 = 4.

Iteration 2:
  left = 4 (0), right = 6 (2)
  mid = 5, nums[mid] = 1
  Target = 0 != 1.
  Check sorted half:
    nums[left] (0) <= nums[mid] (1) -> Left half [0, 1] is SORTED!
  Is target in left range [0, 1)?
    0 <= 0 < 1 is TRUE!
    So target is in left half -> right = mid - 1 = 4.

Iteration 3:
  left = 4 (0), right = 4 (0)
  mid = 4, nums[mid] = 0 == target!
  MATCH FOUND at index 4!
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | `target` | Halves Evaluated | Output Index | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `[4, 5, 6, 7, 0, 1, 2]` | 0 | Left sorted, then right searched | `4` | Target in right rotated segment |
| `[4, 5, 6, 7, 0, 1, 2]` | 3 | Left sorted, right searched $\to$ not found | `-1` | Target absent |
| `[1]` | 0 | Single element mismatch | `-1` | Boundary length 1 |
| `[1]` | 1 | Single element match | `0` | Boundary length 1 |
| `[5, 1, 3]` | 5 | Right sorted $[1, 3]$, target in left $[5]$ | `0` | Small rotated array |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
                
            # Check if left half is monotonically sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Otherwise, right half is monotonically sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
                    
        return -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int search(std::vector<int>& nums, int target) {
        int left = 0, right = static_cast<int>(nums.size()) - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;

            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;

            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\log n)$. The search space is strictly divided in half at every step, guaranteeing at most $\lceil \log_2 n \rceil$ iterations.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only three pointer variables (`left`, `right`, `mid`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Strict Inequality on `mid`:**
   - Notice the range conditions: `nums[left] <= target < nums[mid]`. Because `nums[mid] == target` is already checked at the top of the loop, `mid` is excluded with `< nums[mid]`.
2. **`nums[left] <= nums[mid]` Equality Check:**
   - The `<=` is necessary because when `left == mid` (a two-element window), `nums[left] == nums[mid]`, which represents a 1-element sorted left half.
3. **Contrast with LeetCode 81 (Duplicates Allowed):**
   - If duplicates are allowed (`nums[left] == nums[mid] == nums[right]`), we cannot determine which half is sorted. In that case, worst-case time degrades to $\mathcal{O}(n)$ because we must shrink boundaries linearly (`left++`, `right--`).
