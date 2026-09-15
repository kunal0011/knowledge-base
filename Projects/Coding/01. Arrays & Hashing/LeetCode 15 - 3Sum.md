---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 15: 3Sum"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - two-pointers
  - amazon
  - google
---

# LeetCode 15: 3Sum

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple, Microsoft  
**Difficulty:** Medium  
**Topic:** Sorting + Two Pointers (Deduplicated Triplet Search)

---

### Problem Statement

Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[List[int]]` (all unique zero-sum triplets)
- **Constraints:**
  - $3 \le \text{nums.length} \le 3000$
  - $-10^5 \le \text{nums}[i] \le 10^5$

---

### Key Idea & Intuition

- **Why Sort First?**
  - Sorting in $O(N \log N)$ allows us to:
    1. Eliminate duplicate triplets easily by skipping consecutive identical numbers.
    2. Reduce the 3-sum problem to 2-sum using two pointers (`left` and `right`) in $O(N)$ for each fixed first element.
- **Two Pointer Convergence:**
  - Fix the first number `nums[i]`. If `nums[i] > 0`, break early (sum of 3 positive numbers can never be 0).
  - Target for two pointers: `nums[left] + nums[right] == -nums[i]`.
  - If `sum < 0`, advance `left++` to increase sum.
  - If `sum > 0`, decrement `right--` to decrease sum.
  - If `sum == 0`, record triplet, then skip all duplicate `nums[left]` and `nums[right]`.

### Solution Approach (Step-by-Step)

1. **Sort the Input Array:**
   - Sort `nums` in ascending order.
2. **Iterate First Element ($i$ from $0$ to $n - 3$):**
   - **Early pruning:** If `nums[i] > 0`, break immediately because any remaining numbers are positive and can never sum to zero.
   - **Skip duplicates:** If $i > 0$ and `nums[i] == nums[i - 1]`, skip this index to avoid duplicate triplets.
3. **Initialize Two Pointers:**
   - `left = i + 1`, `right = n - 1`.
4. **Scan with Two Pointers:**
   - Compute `total = nums[i] + nums[left] + nums[right]`.
   - If `total < 0`: Increment `left` to increase the sum.
   - If `total > 0`: Decrement `right` to decrease the sum.
   - If `total == 0`:
     - Record `[nums[i], nums[left], nums[right]]`.
     - Skip identical numbers for `left` while `nums[left] == nums[left + 1]`.
     - Skip identical numbers for `right` while `nums[right] == nums[right - 1]`.
     - Advance both `left++` and `right--`.
5. **Return Collected Triplets.**

---

### Visual Algorithm Walkthrough

```
nums = [-1, 0, 1, 2, -1, -4]
Sorted: [-4, -1, -1, 0, 1, 2]

i=0 (num = -4): target = 4. Two pointers cannot sum to 4.
i=1 (num = -1): target = 1.
   left=2 (-1), right=5 (2): -1 + 2 = 1 == target! -> Triplet: [-1, -1, 2]
   left=3 (0),  right=4 (1):  0 + 1 = 1 == target! -> Triplet: [-1, 0, 1]
i=2 (num = -1): nums[2] == nums[1] -> SKIP duplicate!
i=3 (num = 0):  target = 0. left=4 (1), right=5 (2) -> sum=3 > 0.
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | Sorted Array | Triplets Identified | Explanation |
| :--- | :--- | :--- | :--- |
| `[-1, 0, 1, 2, -1, -4]` | `[-4, -1, -1, 0, 1, 2]` | `[[-1, -1, 2], [-1, 0, 1]]` | Two unique combinations sum to 0 |
| `[0, 1, 1]` | `[0, 1, 1]` | `[]` | Minimum sum is $0 + 1 + 1 = 2 > 0$ |
| `[0, 0, 0]` | `[0, 0, 0]` | `[[0, 0, 0]]` | Single unique zero-sum triplet |
| `[-2, 0, 1, 1, 2]` | `[-2, 0, 1, 1, 2]` | `[[-2, 0, 2], [-2, 1, 1]]` | Handles duplicate positive elements |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        n = len(nums)
        
        for i in range(n - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            left, right = i + 1, n - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                    
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
        std::sort(nums.begin(), nums.end());
        std::vector<std::vector<int>> res;
        int n = nums.size();

        for (int i = 0; i < n - 2; ++i) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;

            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum < 0) {
                    left++;
                } else if (sum > 0) {
                    right--;
                } else {
                    res.push_back({nums[i], nums[left], nums[right]});
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    left++;
                    right--;
                }
            }
        }
        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;

        for (int i = 0; i < n - 2; i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;

            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum < 0) {
                    left++;
                } else if (sum > 0) {
                    right--;
                } else {
                    res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    left++;
                    right--;
                }
            }
        }
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N^2)$ — Sorting takes $O(N \log N)$ and the nested two-pointer loop takes $O(N^2)$.
- **Space Complexity:** $O(1)$ auxiliary space (ignoring sorting recursion stack).

---

### Takeaway Pattern & Interview Traps

1. **Deduplication Without Hash Sets:**
   - Instead of inserting found triplets into a `set` (which adds heavy memory and hashing overhead), sorting allows clean deduplication:
     - Outer loop: `if (i > 0 && nums[i] == nums[i-1]) continue;`
     - Inner two pointers: advance past duplicates only **after** recording a valid triplet.
2. **Early Pruning Invariant:**
   - Once `nums[i] > 0`, since the array is sorted ascending, all subsequent elements are also positive. The sum of three positive numbers can never be zero, so we can `break` early.
3. **Pointers Movement Timing:**
   - Both `left` and `right` must be incremented/decremented simultaneously when a triplet is found, and then duplicate values skipped. Skipping beforehand or updating only one pointer leads to infinite loops or missed triplets.

