---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 611: Valid Triangle Number"
tags:
  - leetcode
  - coding
  - greedy
  - two-pointers
  - sorting
  - array
  - amazon
  - google
---

# LeetCode 611: Valid Triangle Number

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Two Pointers / Sorting / Array  

---

### Problem Statement

Given an integer array `nums`, return the number of triplets chosen from the array that can make triangles if we take them as side lengths of a triangle.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 1000$).
- **Output:**
  - `int` — total count of triplets $(i, j, k)$ with $i < j < k$ that form valid non-degenerate triangles.
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $0 \le \text{nums}[i] \le 1000$

---

### Key Idea & Intuition

By Euclidean geometry, three lengths $a, b, c$ form a non-degenerate triangle if and only if:
$$a + b > c, \quad a + c > b, \quad b + c > a$$

#### The Sorting Simplification:
If we sort the array in ascending order such that $a \le b \le c$:
- $a + c > b$ is automatically true since $c \ge b$ and $a > 0$.
- $b + c > a$ is automatically true since $c \ge a$ and $b > 0$.
- Therefore, we only need to verify the single condition:
  $$a + b > c$$

#### Two-Pointer Greedy Counting ($\mathcal{O}(n^2)$):
1. Sort `nums` in ascending order.
2. Fix the **largest side** $c = \text{nums}[k]$ by iterating $k$ from $n - 1$ down to $2$.
3. Use two pointers for the remaining two smaller sides:
   - `left = 0` (smallest candidate)
   - `right = k - 1` (second largest candidate)
4. While `left < right`:
   - If $\text{nums}[left] + \text{nums}[right] > \text{nums}[k]$:
     - Because `nums` is sorted, every index $i \in [left, right - 1]$ satisfies $\text{nums}[i] + \text{nums}[right] \ge \text{nums}[left] + \text{nums}[right] > \text{nums}[k]$.
     - Thus, all $(right - left)$ pairs $(left, right), (left+1, right), \dots, (right-1, right)$ form valid triangles with $\text{nums}[k]$!
     - We add $right - left$ to our answer.
     - We then decrement `right -= 1` to test the next second-largest candidate.
   - Else ($\text{nums}[left] + \text{nums}[right] \le \text{nums}[k]$):
     - The sum is too small. To increase the sum, increment `left += 1`.

---

### Solution Approach (Step-by-Step)

1. Sort `nums` ascending.
2. Initialize `count = 0` and $n = \text{len}(nums)$.
3. Loop $k$ from $n - 1$ down to $2$:
   - Initialize `left = 0`, `right = k - 1`.
   - While `left < right`:
     - If `nums[left] + nums[right] > nums[k]`:
       - `count += right - left`
       - `right -= 1`
     - Else:
       - `left += 1`
4. Return `count`.

---

### Visual Algorithm Walkthrough

For `nums = [2, 2, 3, 4]`:
Already sorted: `[2, 2, 3, 4]`, $n = 4$.

```
Fix k = 3 (c = nums[3] = 4):
  left = 0 (2), right = 2 (3)
  nums[left] + nums[right] = 2 + 3 = 5 > 4 (VALID!)
  All pairs with right=2 work:
    (left=0, right=2) -> [2, 3, 4]
    (left=1, right=2) -> [2, 3, 4]
  count += right - left = 2 - 0 = 2.
  right decrements to 1.

  left = 0 (2), right = 1 (2):
  nums[left] + nums[right] = 2 + 2 = 4 <= 4 (NOT > 4)
  left increments to 1.
  left == right -> terminate k=3.

Fix k = 2 (c = nums[2] = 3):
  left = 0 (2), right = 1 (2)
  nums[left] + nums[right] = 2 + 2 = 4 > 3 (VALID!)
  count += right - left = 1 - 0 = 1 ([2, 2, 3])
  right decrements to 0.
  left == right -> terminate k=2.

Total valid triangles = 2 + 1 = 3.
Triplets: (2, 3, 4), (2, 3, 4), (2, 2, 3).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [2, 2, 3, 4]`
- **Output:** `3`

#### Example 2:
- **Input:** `nums = [4, 2, 3, 4]`
- **Tracing:**
  - Sorted: `[2, 3, 4, 4]`
  - $k=3$ (4): (2, 4) gives 2; (3, 4) gives 1 $\rightarrow$ 3
  - $k=2$ (4): (2, 3) gives 1 $\rightarrow$ 1
  - Total = $3 + 1 = 4$.
- **Output:** `4`

#### Example 3 (Contains Zeros):
- **Input:** `nums = [0, 1, 1, 1]`
- **Tracing:** Triangles with 0 cannot have $a + b > c$ since $0 + 1 \ngtr 1$. Triplet (1, 1, 1) gives 1.
- **Output:** `1`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        count = 0
        n = len(nums)
        
        # Fix the largest side k from right to left
        for k in range(n - 1, 1, -1):
            left = 0
            right = k - 1
            
            while left < right:
                if nums[left] + nums[right] > nums[k]:
                    # All elements from left to right - 1 can pair with right
                    count += right - left
                    right -= 1
                else:
                    left += 1
                    
        return count
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int triangleNumber(std::vector<int>& nums) {
        std::sort(nums.begin(), nums.end());
        int count = 0;
        int n = static_cast<int>(nums.size());
        
        for (int k = n - 1; k >= 2; --k) {
            int left = 0;
            int right = k - 1;
            
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

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int triangleNumber(int[] nums) {
        Arrays.sort(nums);
        int count = 0;
        int n = nums.length;
        
        for (int k = n - 1; k >= 2; k--) {
            int left = 0;
            int right = k - 1;
            
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

- **Time Complexity:** $\mathcal{O}(n^2)$
  - Sorting takes $\mathcal{O}(n \log n)$.
  - The outer loop runs $n - 2$ times.
  - The inner two-pointer loop runs at most $k \le n$ steps.
  - Total time: $\mathcal{O}(n^2)$. With $n \le 1000$, $\approx 5 \times 10^5$ operations, running in $\approx 5\text{ ms}$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - In-place sorting and scalar pointer variables.

---

### Takeaway Pattern & Interview Traps

- **Fix Largest vs. Fix Smallest:** In 3Sum, fixing the smallest element is standard. But in Triangle Number, **fixing the largest side** ($k$) allows us to add `right - left` candidates in $\mathcal{O}(1)$ whenever the condition holds. If you fix the smallest element, advancing the pointer doesn't yield an immediate contiguous interval count without binary search ($\mathcal{O}(n^2 \log n)$).
- **Zero Values:** Array can contain zeros. If side $= 0$, it can never satisfy $a + b > c$ with $c \ge b$. Sorting places zeros at the beginning, where they naturally fail the `nums[left] + nums[right] > nums[k]` condition and increment `left`.