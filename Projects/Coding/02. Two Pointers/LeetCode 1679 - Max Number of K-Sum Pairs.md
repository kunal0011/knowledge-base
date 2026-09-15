---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 1679: Max Number of K-Sum Pairs"
tags:
  - leetcode
  - coding
  - two-pointers
  - hash-table
  - greedy
  - sorting
  - amazon
  - google
---

# LeetCode 1679: Max Number of K-Sum Pairs

**Target Companies:** Amazon, Google, Meta, Apple  
**Difficulty:** Medium  
**Topic:** Two Pointers on Sorted Array / Hash Map Complement

---

### Problem Statement

You are given an integer array `nums` and an integer `k`.

In one operation, you can pick two numbers from the array whose sum equals `k` and remove them from the array.

Return *the maximum number of operations you can perform on the array*.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` (maximum pairs summing to $k$)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $1 \le \text{nums}[i] \le 10^9$
  - $1 \le k \le 10^9$

---

### Key Idea & Intuition

Each operation consumes two distinct elements whose sum is $k$. We want to find the maximum disjoint pair matching.

#### Approach 1: Two Pointers with Sorting ($\mathcal{O}(n \log n)$ time, $\mathcal{O}(1)$ space)
- Sort `nums` in ascending order.
- Place two pointers: `left = 0`, `right = len(nums) - 1`.
- Compute `total = nums[left] + nums[right]`:
  - If `total == k`: A valid pair is found! Increment `ops += 1`, and advance both pointers: `left += 1`, `right -= 1`.
  - If `total < k`: The sum is too small. Increment `left += 1` to increase the sum.
  - If `total > k`: The sum is too large. Decrement `right -= 1` to decrease the sum.

#### Approach 2: Hash Map Frequencies ($\mathcal{O}(n)$ time, $\mathcal{O}(n)$ space)
- Count frequencies in a hash map.
- For each number $x$:
  - If $x == k - x$: we can form $\lfloor count[x] / 2 \rfloor$ pairs.
  - If $x < k - x$: we can form $\min(count[x], count[k - x])$ pairs.
- Summing these yields $\mathcal{O}(n)$ time.

---

### Solution Approach (Step-by-Step)

1. **Sort the Array:**
   - Sort `nums` in ascending order.
2. **Two Pointers Convergence:**
   - Initialize `left = 0`, `right = len(nums) - 1`, `ops = 0`.
   - While `left < right`:
     - `total = nums[left] + nums[right]`
     - If `total == k`:
       - `ops += 1`
       - `left += 1`
       - `right -= 1`
     - Else if `total < k`:
       - `left += 1`
     - Else:
       - `right -= 1`
3. **Return Operations:**
   - Return `ops`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [1, 2, 3, 4]`, `k = 5`

```
Sorted: [1, 2, 3, 4]
left = 0 (1), right = 3 (4)

Iteration 1:
  nums[left] + nums[right] = 1 + 4 = 5 == k!
  ops = 1
  left = 1 (2), right = 2 (3)

Iteration 2:
  nums[left] + nums[right] = 2 + 3 = 5 == k!
  ops = 2
  left = 2, right = 1 (left >= right, terminate)

Result = 2 pairs.
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | `k` | Sorted Array | Matching Pairs | Output |
| :--- | :--- | :--- | :--- | :--- |
| `[1, 2, 3, 4]` | 5 | `[1, 2, 3, 4]` | `(1, 4), (2, 3)` | `2` |
| `[3, 1, 3, 4, 3]` | 6 | `[1, 3, 3, 3, 4]` | `(3, 3)` (one 3 left unpaired) | `1` |
| `[2, 2, 2, 2]` | 4 | `[2, 2, 2, 2]` | `(2, 2), (2, 2)` | `2` |
| `[1, 5, 6]` | 4 | `[1, 5, 6]` | None | `0` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        ops = 0
        
        while left < right:
            total = nums[left] + nums[right]
            if total == k:
                ops += 1
                left += 1
                right -= 1
            elif total < k:
                left += 1
            else:
                right -= 1
                
        return ops
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxOperations(std::vector<int>& nums, int k) {
        std::sort(nums.begin(), nums.end());
        int left = 0, right = static_cast<int>(nums.size()) - 1;
        int ops = 0;

        while (left < right) {
            int total = nums[left] + nums[right];
            if (total == k) {
                ops++;
                left++;
                right--;
            } else if (total < k) {
                left++;
            } else {
                right--;
            }
        }
        return ops;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int maxOperations(int[] nums, int k) {
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        int ops = 0;

        while (left < right) {
            int total = nums[left] + nums[right];
            if (total == k) {
                ops++;
                left++;
                right--;
            } else if (total < k) {
                left++;
            } else {
                right--;
            }
        }
        return ops;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$ using the sorting approach (or $\mathcal{O}(n)$ using a single-pass hash map). In practice, in-place sorting on primitives has low memory overhead and optimal cache locality.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space beyond the sorting stack ($\mathcal{O}(\log n)$).

---

### Takeaway Pattern & Interview Traps

1. **Greedy Matching Soundness:**
   - Pairing the smallest available number with the largest valid complement is always optimal. If `nums[left] + nums[right] == k`, using `nums[left]` with any other number would require a larger complement than `nums[right]`, which cannot exist since `nums[right]` is the largest available element.
2. **Contrast with 2Sum / 3Sum:**
   - In 2Sum, we find one pair. In 3Sum, we find unique values. Here, duplicate numbers can be reused across different pairs (e.g. `[2, 2, 2, 2]` with $k=4$ yields 2 operations), requiring pointer advancement without skipping duplicates.
