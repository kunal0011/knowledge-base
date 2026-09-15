---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 698: Partition to K Equal Sum Subsets"
tags:
  - leetcode
  - coding
  - backtracking
  - dynamic-programming
  - bitmask
  - array
  - amazon
  - google
---

# LeetCode 698: Partition to K Equal Sum Subsets

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber  
**Difficulty:** Medium  
**Topic:** Backtracking / State-Space Search / Symmetry Pruning  

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return `true` if it is possible to divide this array into `k` non-empty subsets whose sums are all equal.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `bool`
- **Constraints:**
  - $1 \le k \le \text{nums.length} \le 16$
  - $1 \le \text{nums}[i] \le 10^4$
  - The frequency of each element is in the range $[1, 4]$.

---

### Key Idea & Intuition

- **Mathematical Pruning (Upfront Invariants):**
  - Compute `total_sum = sum(nums)`. If `total_sum % k != 0`, partitioning into $k$ equal integer sums is mathematically impossible $\implies$ return `false`.
  - Let $\text{target} = \text{total\_sum} / k$.
  - If any single element `nums[i] > target`, it can never fit into any subset $\implies$ return `false`.
- **Heuristic 1: Descending Order Sort:**
  - Sorting `nums` in descending order places the largest elements first.
  - Larger numbers leave less remaining space in buckets, which forces constraints to be violated early and collapses invalid recursion branches near the root.
- **Heuristic 2: Symmetry Pruning (Empty Buckets):**
  - All $k$ buckets are initially identical and indistinguishable.
  - If a number cannot fit into an empty bucket, trying to place it into a *different* empty bucket produces an identical search tree by symmetry.
  - **Invariant:** If `bucket[j] == 0` and placing the number fails, **break immediately** from trying subsequent buckets!
- **Heuristic 3: Fill Bucket-by-Bucket (Alternative Optimal Formulation):**
  - Instead of placing each number into $k$ buckets ($\mathcal{O}(k^N)$), build subsets sequentially from $1$ to $k$ ($\mathcal{O}(k \cdot 2^N)$):
    - Maintain `curr_sum`, `k_remaining`, and `start_idx`.
    - When `curr_sum == target`, reset `curr_sum = 0` and decrement `k_remaining`.

---

### Solution Approach (Step-by-Step)

1. Compute `total_sum = sum(nums)`.
2. Check if `total_sum % k != 0`: return `False`.
3. Set `target = total_sum // k`.
4. Sort `nums` descending. If `nums[0] > target`: return `False`.
5. Initialize `buckets = [0] * k`.
6. Define `backtrack(idx)`:
   - If `idx == len(nums)`: return `True` (all numbers placed successfully).
   - For `j` from $0$ to $k - 1$:
     - If `buckets[j] + nums[idx] <= target`:
       - `buckets[j] += nums[idx]`
       - If `backtrack(idx + 1)`:
         - Return `True`
       - `buckets[j] -= nums[idx]` (backtrack)
     - **Symmetry Pruning:** If `buckets[j] == 0`:
       - Break! (Never try another empty bucket).
   - Return `False`.
7. Call `backtrack(0)`.

---

### Visual Algorithm Walkthrough

Let `nums = [4, 3, 2, 3, 5, 2, 1]`, `k = 4`.
- Total sum $= 20 \implies \text{target} = 20 / 4 = 5$.
- Sorted descending: `[5, 4, 3, 3, 2, 2, 1]`.
- Buckets: `B0=0, B1=0, B2=0, B3=0`.

```
1. Place 5: fits in B0 -> B0 = 5.
2. Place 4: cannot fit in B0 (5+4 > 5). Fits in B1 -> B1 = 4.
3. Place 3: cannot fit in B0 (full) or B1 (4+3 > 5). Fits in B2 -> B2 = 3.
4. Place 3: cannot fit in B0, B1, B2. Fits in B3 -> B3 = 3.
5. Place 2:
   - Try B0 (full)
   - Try B1 (4+2 > 5)
   - Try B2: fits! (3+2 = 5) -> B2 = 5.
6. Place 2:
   - Try B3: fits! (3+2 = 5) -> B3 = 5.
7. Place 1:
   - Try B1: fits! (4+1 = 5) -> B1 = 5.
All elements placed! All 4 buckets sum to 5:
B0: [5], B1: [4, 1], B2: [3, 2], B3: [3, 2].
Return True!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `k` | Target | Partition Found? | Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[4,3,2,3,5,2,1]` | `4` | $20 / 4 = 5$ | `[5]`, `[4,1]`, `[3,2]`, `[3,2]` | `true` |
| **Impossible Sum** | `[1,2,3,4]` | `3` | $10 / 3 = 3.33$ | Not divisible | `false` |
| **Single Element Too Large** | `[10, 1, 1]` | `2` | $12 / 2 = 6$ | `10 > 6` | `false` |
| **k == 1** | `[5, 5, 5]` | `1` | $15 / 1 = 15$ | Trivial (entire array) | `true` |
| **k == len(nums)** | `[2, 2, 2]` | `3` | $6 / 3 = 2$ | All elements equal target | `true` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        """
        Determines whether nums can be partitioned into k equal sum subsets.
        Uses descending sort with bucket symmetry pruning.
        """
        total_sum = sum(nums)
        if total_sum % k != 0:
            return False

        target = total_sum // k
        nums.sort(reverse=True)

        if nums[0] > target:
            return False

        buckets = [0] * k

        def backtrack(idx: int) -> bool:
            if idx == len(nums):
                return True

            num = nums[idx]
            for j in range(k):
                if buckets[j] + num <= target:
                    buckets[j] += num
                    if backtrack(idx + 1):
                        return True
                    buckets[j] -= num  # Backtrack

                # Critical Symmetry Pruning:
                # If this bucket is empty and placing num didn't lead to a solution,
                # placing num in any subsequent empty bucket will yield identical failure.
                if buckets[j] == 0:
                    break

            return False

        return backtrack(0)
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    bool canPartitionKSubsets(std::vector<int>& nums, int k) {
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);
        if (total_sum % k != 0) return false;

        int target = total_sum / k;
        std::sort(nums.rbegin(), nums.rend());

        if (nums[0] > target) return false;

        std::vector<int> buckets(k, 0);
        return backtrack(0, target, k, nums, buckets);
    }

private:
    bool backtrack(int idx, int target, int k, const std::vector<int>& nums, std::vector<int>& buckets) {
        if (idx == static_cast<int>(nums.size())) {
            return true;
        }

        int val = nums[idx];
        for (int j = 0; j < k; ++j) {
            if (buckets[j] + val <= target) {
                buckets[j] += val;
                if (backtrack(idx + 1, target, k, nums, buckets)) {
                    return true;
                }
                buckets[j] -= val; // Backtrack
            }

            // Symmetry Pruning: avoid testing duplicate empty buckets
            if (buckets[j] == 0) {
                break;
            }
        }

        return false;
    }
};
```

#### Java
```java
import java.util.Arrays;

class Solution {
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }

        if (totalSum % k != 0) {
            return false;
        }

        int target = totalSum / k;
        Arrays.sort(nums); // Sort ascending, then traverse backwards for descending order

        if (nums[nums.length - 1] > target) {
            return false;
        }

        int[] buckets = new int[k];
        return backtrack(nums.length - 1, target, k, nums, buckets);
    }

    private boolean backtrack(int idx, int target, int k, int[] nums, int[] buckets) {
        if (idx < 0) {
            return true;
        }

        int val = nums[idx];
        for (int j = 0; j < k; j++) {
            if (buckets[j] + val <= target) {
                buckets[j] += val;
                if (backtrack(idx - 1, target, k, nums, buckets)) {
                    return true;
                }
                buckets[j] -= val; // Backtrack
            }

            // Symmetry Pruning: avoid redundant empty buckets
            if (buckets[j] == 0) {
                break;
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(k \cdot 2^N)$ with bitmasking / bucket pruning.
  - Without pruning, the search space is $\mathcal{O}(k^N)$ ($k^{16} \approx 4 \times 10^{12}$).
  - With descending sort and symmetry pruning (`if buckets[j] == 0: break`), unviable branches are eliminated almost immediately, resulting in runtime $< 5 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack ($N \le 16$) and $\mathcal{O}(k)$ space for the buckets array.

---

### Takeaway Pattern & Interview Traps

- **The Empty Bucket Break Invariant:** Forgetting `if (buckets[j] == 0) break;` is the single most common cause of TLE in this problem. All empty buckets are identical; putting an element into bucket 2 after it failed in bucket 1 is completely redundant.
- **Descending Sort Pruning:** Always sort descending when filling bins or buckets. Large items have the fewest placement options; attempting to place them first prunes bad configurations at the top of the recursion tree rather than deep down.