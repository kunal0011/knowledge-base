---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 611: Valid Triangle Number"
tags:
  - leetcode
  - coding
  - two-pointers
  - sorting
  - binary-search
  - amazon
  - google
---

# LeetCode 611: Valid Triangle Number

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Sorted Two Pointers / Triangle Inequality Invariant

---

### Problem Statement

Given an integer array `nums`, return the number of triplets chosen from the array that can make triangles if we take them as side lengths of a triangle.

Three sides $a, b, c$ form a triangle if and only if:
- $a + b > c$
- $a + c > b$
- $b + c > a$

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (number of valid triplets)
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $0 \le \text{nums}[i] \le 1000$

---

### Key Idea & Intuition

#### 1. Mathematical Simplification via Sorting:
In general, verifying a triangle requires checking all 3 inequalities. However, if the sides are **sorted in ascending order**:
$$a \le b \le c$$
Notice that since $c \ge b$ and $a > 0$, we automatically have:
$$a + c > b \quad \text{and} \quad b + c > a$$
Both conditions are guaranteed! Therefore, the triangle condition reduces strictly to a single check:
$$a + b > c$$

#### 2. Reverse Two-Pointer Technique ($\mathcal{O}(n^2)$):
- Sort `nums` in ascending order.
- Fix the **largest** side $c = \text{nums}[k]$, iterating $k$ backwards from $n - 1$ down to 2.
- For each fixed $k$, set two pointers: `left = 0` and `right = k - 1`.
- If $\text{nums}[left] + \text{nums}[right] > \text{nums}[k]$:
  - Because the array is sorted, any element between $left$ and $right - 1$ paired with $\text{nums}[right]$ will ALSO have sum $> \text{nums}[k]$!
  - Therefore, all pairs $(i, right)$ for $left \le i < right$ form valid triangles with $\text{nums}[k]$.
  - Add $(right - left)$ to our total count in $\mathcal{O}(1)$ time!
  - Decrement `right -= 1` to search for smaller sums.
- Else ($\text{nums}[left] + \text{nums}[right] \le \text{nums}[k]$):
  - The sum is too small. Increment `left += 1` to increase the sum.
- This checks all triplets with fixed largest side $k$ in $\mathcal{O}(k)$ time, giving an overall $\mathcal{O}(n^2)$ algorithm.

---

### Solution Approach (Step-by-Step)

1. **Sort the Array:**
   - Sort `nums` ascending: $\mathcal{O}(n \log n)$.
2. **Iterate Largest Side $k$:**
   - Loop $k$ from $n - 1$ down to 2:
     - Set `left = 0`, `right = k - 1`.
     - While `left < right`:
       - If `nums[left] + nums[right] > nums[k]`:
         - `count += (right - left)`
         - `right -= 1`
       - Else:
         - `left += 1`
3. **Return Count:**
   - Return `count`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [2, 2, 3, 4]` ($N = 4$)
Already sorted: `[2, 2, 3, 4]`

```
Fix k = 3 (nums[k] = 4):
  left = 0 (2), right = 2 (3)
  2 + 3 = 5 > 4 (Condition met!)
  Number of valid pairs with right=2: right - left = 2 - 0 = 2.
  Pairs: (nums[0], nums[2], nums[3]) -> (2, 3, 4)
         (nums[1], nums[2], nums[3]) -> (2, 3, 4)
  count = 2
  right decrements to 1.

  left = 0 (2), right = 1 (2)
  2 + 2 = 4 NOT > 4!
  left increments to 1. Loop terminates (left == right).

Fix k = 2 (nums[k] = 3):
  left = 0 (2), right = 1 (2)
  2 + 2 = 4 > 3 (Condition met!)
  Number of valid pairs with right=1: right - left = 1 - 0 = 1.
  Pair: (nums[0], nums[1], nums[2]) -> (2, 2, 3)
  count = 2 + 1 = 3
  right decrements to 0. Loop terminates.

Final Result: 3 valid triplets.
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | Sorted Array | Valid Triplets | Output |
| :--- | :--- | :--- | :--- |
| `[2, 2, 3, 4]` | `[2, 2, 3, 4]` | `(2,3,4), (2,3,4), (2,2,3)` | `3` |
| `[4, 2, 3, 4]` | `[2, 3, 4, 4]` | `(2,3,4), (2,3,4), (2,4,4), (3,4,4)` | `4` |
| `[0, 0, 0]` | `[0, 0, 0]` | $0 + 0 \ngtr 0$ (no valid triangles) | `0` |
| `[1, 1, 3]` | `[1, 1, 3]` | $1 + 1 \ngtr 3$ | `0` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        count = 0
        n = len(nums)
        
        # k is the index of the longest side c
        for k in range(n - 1, 1, -1):
            left, right = 0, k - 1
            while left < right:
                if nums[left] + nums[right] > nums[k]:
                    # For current right, any left' in [left, right-1] also satisfies nums[left'] + nums[right] > nums[k]
                    count += (right - left)
                    right -= 1
                else:
                    left += 1
                    
        return count
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int triangleNumber(std::vector<int>& nums) {
        std::sort(nums.begin(), nums.end());
        int count = 0;
        int n = nums.size();

        for (int k = n - 1; k >= 2; --k) {
            int left = 0, right = k - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[k]) {
                    count += (right - left);
                    right--;
                } else {
                    left++;
                }
            }
        }

        return count;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int triangleNumber(int[] nums) {
        Arrays.sort(nums);
        int count = 0;
        int n = nums.length;

        for (int k = n - 1; k >= 2; k--) {
            int left = 0, right = k - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[k]) {
                    count += (right - left);
                    right--;
                } else {
                    left++;
                }
            }
        }

        return count;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^2)$. Sorting `nums` requires $\mathcal{O}(n \log n)$. The outer loop for $k$ runs $\approx n$ times, and within each step the two pointers `left` and `right` converge in $\mathcal{O}(n)$ steps. Overall time is $\mathcal{O}(n \log n + n^2) = \mathcal{O}(n^2)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space beyond the sorting stack ($\mathcal{O}(\log n)$).

---

### Takeaway Pattern & Interview Traps

1. **Why Fix the Largest Side $k$ Instead of Smallest Side?**
   - If we fix the smallest side $i$, then for pointers $j$ and $k$, $nums[i] + nums[j] > nums[k]$ requires finding how many $k$ are smaller than a sum. Because $k$ is on the upper end, moving pointers is not monotonic. Fixing the largest side $k$ makes the sum monotonic with respect to `left` and `right`.
2. **Zeros in the Input:**
   - Sides can be $0$ ($0 \le \text{nums}[i] \le 1000$). Sides of length 0 will never satisfy $0 + b > b$ because $b > b$ is false, so triangles with 0-length edges are naturally excluded.
