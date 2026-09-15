---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1005: Maximize Sum Of Array After K Negations"
tags:
  - leetcode
  - coding
  - greedy
  - sorting
  - heap
  - array
  - amazon
  - google
---

# LeetCode 1005: Maximize Sum Of Array After K Negations

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Greedy / Sorting / Priority Queue  

---

### Problem Statement

Given an integer array `nums` and an integer `k`, modify the array in the following way:
- Choose an index `i` and replace `nums[i]` with `-nums[i]`.

You should apply this process exactly `k` times. You may choose the same index `i` multiple times.

Return the **maximum possible sum** of the array after modifying it in this way.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 10^4$).
  - `k`: `int` ($1 \le k \le 10^4$).
- **Output:**
  - `int` — maximum achievable sum of elements after exactly $k$ negations.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $-100 \le \text{nums}[i] \le 100$
  - $1 \le k \le 10^4$

---

### Key Idea & Intuition

#### Greedy Priority:
1. **Negating Negatives Maximizes Gains:**
   - Negating a negative number $-x$ ($x > 0$) adds $2x$ to the total sum.
   - To maximize this gain, we greedily flip the **most negative** numbers first (i.e. numbers with largest absolute value, sorted in ascending order).
2. **Exhausting Remaining $k$:**
   - If we have flipped all negative numbers and $k > 0$ operations still remain:
     - Flipping the same element twice leaves it unchanged: $-(-x) = x$.
     - Thus, pairs of flips cancel out. Only the **parity** of the remaining $k$ ($k \pmod 2$) matters!
     - If remaining $k$ is even, we can flip any element an even number of times for net zero change.
     - If remaining $k$ is odd, we are forced to perform one net flip on a non-negative number. To minimize the reduction in sum, we must flip the element with the **smallest absolute value**.

#### Approaches:
- **Sorting Approach:** Sort array by absolute value descending (or standard ascending), flip negative numbers, then flip the minimum element if $k \pmod 2 == 1$. Takes $\mathcal{O}(n \log n)$ time.
- **Min-Heap Approach:** Push all numbers into a min-heap. Pop minimum, negate it, push it back, repeated $k$ times. Takes $\mathcal{O}(n + k \log n)$ time.
- **Counting Sort / Frequency Array:** Since $-100 \le \text{nums}[i] \le 100$, frequency counting achieves $\mathcal{O}(n + 200) = \mathcal{O}(n)$ time and $\mathcal{O}(1)$ space!

---

### Solution Approach (Step-by-Step: Sorting)

1. Sort `nums` in ascending order.
2. Iterate through `nums`:
   - If `nums[i] < 0` and `k > 0`:
     - `nums[i] = -nums[i]`
     - `k -= 1`
3. Sum the modified array: `total_sum = sum(nums)`.
4. If $k > 0$ and $k \pmod 2 == 1$:
   - Find the minimum element in `nums` (which is now all non-negative): `min_val = min(nums)`.
   - Subtract $2 \times \text{min\_val}$ from `total_sum` (since `min_val` was previously added in `total_sum`, flipping it from $+v$ to $-v$ subtracts $2v$).
5. Return `total_sum`.

---

### Visual Algorithm Walkthrough

For `nums = [4, 2, 3]`, `k = 1`:

```
All elements are positive: [4, 2, 3]
k = 1 is odd.
Minimum element is 2.
Flip 2 -> -2.
Array becomes: [4, -2, 3]
Sum = 4 - 2 + 3 = 5.
```

For `nums = [3, -1, 0, 2]`, `k = 3`:

```
1. Sorted: [-1, 0, 2, 3]
2. Flip negatives:
   nums[0] = -1 < 0 -> flip to 1. k becomes 2.
3. Remaining k = 2 (even).
   Even operations cancel out (e.g., flip 0 twice -> 0).
4. Final array: [1, 0, 2, 3]
   Sum = 1 + 0 + 2 + 3 = 6.
```

For `nums = [2, -3, -1, 5, -4]`, `k = 2`:

```
1. Sorted: [-4, -3, -1, 2, 5]
2. Flip largest negatives first:
   -4 -> 4 (k becomes 1)
   -3 -> 3 (k becomes 0)
3. k = 0, no remaining operations.
4. Final array: [4, 3, -1, 2, 5]
   Sum = 4 + 3 - 1 + 2 + 5 = 13.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [4, 2, 3]`, `k = 1`
- **Output:** `5`

#### Example 2:
- **Input:** `nums = [3, -1, 0, 2]`, `k = 3`
- **Output:** `6`

#### Example 3:
- **Input:** `nums = [2, -3, -1, 5, -4]`, `k = 2`
- **Output:** `13`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        
        # Step 1: Flip negatives to positives
        for i in range(n):
            if nums[i] < 0 and k > 0:
                nums[i] = -nums[i]
                k -= 1
                
        total_sum = sum(nums)
        
        # Step 2: If remaining k is odd, subtract 2 * minimum element
        if k % 2 == 1:
            total_sum -= 2 * min(nums)
            
        return total_sum
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <numeric>

class Solution {
public:
    int largestSumAfterKNegations(std::vector<int>& nums, int k) {
        std::sort(nums.begin(), nums.end());
        int n = static_cast<int>(nums.size());
        
        // Greedily negate negative values
        for (int i = 0; i < n && k > 0; ++i) {
            if (nums[i] < 0) {
                nums[i] = -nums[i];
                --k;
            }
        }
        
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);
        
        // If remaining k is odd, flip smallest positive element
        if (k % 2 == 1) {
            int min_val = *std::min_element(nums.begin(), nums.end());
            total_sum -= 2 * min_val;
        }
        
        return total_sum;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int largestSumAfterKNegations(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length;
        
        // Flip negative numbers
        for (int i = 0; i < n && k > 0; i++) {
            if (nums[i] < 0) {
                nums[i] = -nums[i];
                k--;
            }
        }
        
        int totalSum = 0;
        int minVal = Integer.MAX_VALUE;
        for (int num : nums) {
            totalSum += num;
            minVal = Math.min(minVal, num);
        }
        
        // Deduct 2 * minVal if remaining k is odd
        if (k % 2 == 1) {
            totalSum -= 2 * minVal;
        }
        
        return totalSum;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$
  - Sorting `nums` takes $\mathcal{O}(n \log n)$.
  - Linear scan and summation take $\mathcal{O}(n)$.
  - Finding the minimum takes $\mathcal{O}(n)$.
  *(Can be optimized to $\mathcal{O}(n)$ using bucket sort / counting array since $|nums[i]| \le 100$).*
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Sorting is performed in-place (or $\mathcal{O}(\log n)$ stack space depending on language sort implementation).

---

### Takeaway Pattern & Interview Traps

- **Why `- 2 * min_val`:** Since `min_val` was already included in `total_sum` as $+v$, changing its sign to $-v$ produces a delta of $-2v$.
- **Zero is a "Free Sink":** If the array contains $0$, remaining flips can be absorbed by $0$ at zero cost ($k \pmod 2$ does not diminish the sum at all because $\min = 0$).