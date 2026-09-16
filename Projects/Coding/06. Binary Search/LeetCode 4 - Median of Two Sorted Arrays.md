---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 4: Median of Two Sorted Arrays"
tags:
  - leetcode
  - coding
  - binary-search
  - arrays
  - divide-and-conquer
  - google
  - amazon
  - meta
---

# LeetCode 4: Median of Two Sorted Arrays

**Target Companies:** Google (Signature Classic Hard), Amazon, Meta, Apple, Microsoft, Goldman Sachs  
**Difficulty:** Hard  
**Topic:** Binary Search on Partition Boundary / Divide & Conquer

---

### Problem Statement

Given two sorted arrays `nums1` and `nums2` of size $m$ and $n$ respectively, return the **median** of the two sorted arrays.

The overall run time complexity should be **$\mathcal{O}(\log (m+n))$**.

---

### Input & Output Formats & Constraints

- **Input:** `nums1: List[int]`, `nums2: List[int]`
- **Output:** `float` (the median of the combined sorted sequence)
- **Constraints:**
  - $0 \le m \le 1000$
  - $0 \le n \le 1000$
  - $1 \le m + n \le 2000$
  - $-10^6 \le \text{nums1}[i], \text{nums2}[i] \le 10^6$

---

### Key Idea & Intuition

#### 1. The Partition Invariant:
Finding the median is equivalent to partitioning both arrays into two sets—a **Left Part** and a **Right Part**—such that:
1. $\text{len}(\text{Left Part}) == \text{len}(\text{Right Part})$ (or $\text{len}(\text{Left Part}) == \text{len}(\text{Right Part}) + 1$ if total length is odd).
2. Every element in the Left Part is $\le$ every element in the Right Part.

Let partition index $i$ split `nums1` into `nums1[0..i-1]` (left) and `nums1[i..m-1]` (right).  
Let partition index $j$ split `nums2` into `nums2[0..j-1]` (left) and `nums2[j..n-1]` (right).

To ensure the Left Part has half of the total elements:
$$j = \left\lfloor \frac{m + n + 1}{2} \right\rfloor - i$$

#### 2. Partition Validation:
Since `nums1` and `nums2` are already individually sorted, we only need to verify the cross-array boundaries:
- $\text{maxLeft1} \le \text{minRight2}$ (largest left element of `nums1` $\le$ smallest right element of `nums2`)
- $\text{maxLeft2} \le \text{minRight1}$ (largest left element of `nums2` $\le$ smallest right element of `nums1`)

If both conditions hold:
- If total length $(m + n)$ is **odd**:
  $$\text{median} = \max(\text{maxLeft1}, \text{maxLeft2})$$
- If total length $(m + n)$ is **even**:
  $$\text{median} = \frac{\max(\text{maxLeft1}, \text{maxLeft2}) + \min(\text{minRight1}, \text{minRight2})}{2.0}$$

#### 3. Binary Search on the Shorter Array:
By ensuring $m \le n$ (swapping arrays if necessary), the search range for $i$ is $[0, m]$, which guarantees that $j = \lfloor (m + n + 1) / 2 \rfloor - i \ge 0$ is always non-negative and valid. This achieves an optimal runtime of $\mathcal{O}(\log(\min(m, n)))$.

---

### Solution Approach (Step-by-Step)

1. **Ensure `nums1` is the Shorter Array:**
   - If `len(nums1) > len(nums2)`, recursively return `findMedianSortedArrays(nums2, nums1)`.
2. **Binary Search Initialization:**
   - `m = len(nums1)`, `n = len(nums2)`.
   - `low = 0`, `high = m`.
3. **Partition Search:**
   - While `low <= high`:
     - $i = (low + high) // 2$
     - $j = (m + n + 1) // 2 - i$
     - Set boundaries with infinities for edge cases:
       - $\text{maxLeft1} = -\infty$ if $i == 0$ else $\text{nums1}[i - 1]$
       - $\text{minRight1} = +\infty$ if $i == m$ else $\text{nums1}[i]$
       - $\text{maxLeft2} = -\infty$ if $j == 0$ else $\text{nums2}[j - 1]$
       - $\text{minRight2} = +\infty$ if $j == n$ else $\text{nums2}[j]$
     - **Check Partition Validity:**
       - If $\text{maxLeft1} \le \text{minRight2}$ and $\text{maxLeft2} \le \text{minRight1}$:
         - If $(m + n) \pmod 2 == 1$: return $\max(\text{maxLeft1}, \text{maxLeft2})$.
         - Else: return $(\max(\text{maxLeft1}, \text{maxLeft2}) + \min(\text{minRight1}, \text{minRight2})) / 2.0$.
       - Else if $\text{maxLeft1} > \text{minRight2}$:
         - Partition in `nums1` is too far right $\implies$ search left: `high = i - 1`.
       - Else:
         - Partition in `nums1` is too far left $\implies$ search right: `low = i + 1`.

---

### Visual Algorithm Walkthrough

#### Example: `nums1 = [1, 3]`, `nums2 = [2]` ($m = 2, n = 1$)
1. Swap so `nums1 = [2]`, `nums2 = [1, 3]` ($m = 1, n = 2$).
2. Total = $1 + 2 = 3$ (odd). Left half needs $\lfloor (3 + 1) / 2 \rfloor = 2$ elements.

```
Initial binary search on nums1: low = 0, high = 1
i = (0 + 1) // 2 = 0
j = 2 - 0 = 2

Partitions:
  nums1 Left: []      | nums1 Right: [2]      -> maxLeft1 = -inf, minRight1 = 2
  nums2 Left: [1, 3]  | nums2 Right: []       -> maxLeft2 = 3,    minRight2 = +inf

Check:
  maxLeft1 (-inf) <= minRight2 (+inf)? YES
  maxLeft2 (3) <= minRight1 (2)? NO! (3 > 2)
  maxLeft2 > minRight1 -> i is too small, move right: low = i + 1 = 1

Iteration 2:
low = 1, high = 1
i = 1, j = 2 - 1 = 1

Partitions:
  nums1 Left: [2]     | nums1 Right: []       -> maxLeft1 = 2,    minRight1 = +inf
  nums2 Left: [1]     | nums2 Right: [3]      -> maxLeft2 = 1,    minRight2 = 3

Check:
  maxLeft1 (2) <= minRight2 (3)? YES!
  maxLeft2 (1) <= minRight1 (+inf)? YES!

Valid partition found!
Total length is odd -> median = max(maxLeft1, maxLeft2) = max(2, 1) = 2.0.
```

---

### Solved Examples with Multiple Inputs

| `nums1` | `nums2` | $m+n$ | Left Partition Elements | Right Partition Elements | Median Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `[1, 3]` | `[2]` | 3 (Odd) | `[1, 2]` | `[3]` | `2.0` |
| `[1, 2]` | `[3, 4]` | 4 (Even) | `[1, 2]` | `[3, 4]` | `(2 + 3) / 2 = 2.5` |
| `[0, 0]` | `[0, 0]` | 4 (Even) | `[0, 0]` | `[0, 0]` | `0.0` |
| `[]` | `[1]` | 1 (Odd) | `[1]` | `[]` | `1.0` |
| `[2]` | `[]` | 1 (Odd) | `[2]` | `[]` | `2.0` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
            
        m, n = len(nums1), len(nums2)
        low, high = 0, m
        
        while low <= high:
            i = (low + high) // 2
            j = (m + n + 1) // 2 - i
            
            max_left1 = float('-inf') if i == 0 else nums1[i - 1]
            min_right1 = float('inf') if i == m else nums1[i]
            
            max_left2 = float('-inf') if j == 0 else nums2[j - 1]
            min_right2 = float('inf') if j == n else nums2[j]
            
            if max_left1 <= min_right2 and max_left2 <= min_right1:
                if (m + n) % 2 == 1:
                    return float(max(max_left1, max_left2))
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            elif max_left1 > min_right2:
                high = i - 1
            else:
                low = i + 1
                
        return 0.0
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
        if (nums1.size() > nums2.size()) {
            return findMedianSortedArrays(nums2, nums1);
        }

        int m = nums1.size(), n = nums2.size();
        int low = 0, high = m;

        while (low <= high) {
            int i = low + (high - low) / 2;
            int j = (m + n + 1) / 2 - i;

            int maxLeft1 = (i == 0) ? INT_MIN : nums1[i - 1];
            int minRight1 = (i == m) ? INT_MAX : nums1[i];

            int maxLeft2 = (j == 0) ? INT_MIN : nums2[j - 1];
            int minRight2 = (j == n) ? INT_MAX : nums2[j];

            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                if ((m + n) % 2 == 1) {
                    return std::max(maxLeft1, maxLeft2);
                }
                return (std::max(maxLeft1, maxLeft2) + std::min(minRight1, minRight2)) / 2.0;
            } else if (maxLeft1 > minRight2) {
                high = i - 1;
            } else {
                low = i + 1;
            }
        }
        return 0.0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1);
        }

        int m = nums1.length, n = nums2.length;
        int low = 0, high = m;

        while (low <= high) {
            int i = (low + high) / 2;
            int j = (m + n + 1) / 2 - i;

            int maxLeft1 = (i == 0) ? Integer.MIN_VALUE : nums1[i - 1];
            int minRight1 = (i == m) ? Integer.MAX_VALUE : nums1[i];

            int maxLeft2 = (j == 0) ? Integer.MIN_VALUE : nums2[j - 1];
            int minRight2 = (j == n) ? Integer.MAX_VALUE : nums2[j];

            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                if ((m + n) % 2 == 1) {
                    return Math.max(maxLeft1, maxLeft2);
                }
                return (Math.max(maxLeft1, maxLeft2) + Math.min(minRight1, minRight2)) / 2.0;
            } else if (maxLeft1 > minRight2) {
                high = i - 1;
            } else {
                low = i + 1;
            }
        }
        return 0.0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\log(\min(m, n)))$. By always performing the binary search on the smaller array, the number of iterations is bounded by $\log_2(\min(m, n))$, strictly meeting and exceeding the $\mathcal{O}(\log(m + n))$ requirement.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only scalar pointers and boundary sentinel variables are used.

---

### Takeaway Pattern & Interview Traps

1. **Why Search on the Smaller Array?**
   - Searching on the larger array can cause $j = (m + n + 1) // 2 - i$ to become negative or exceed $n$. Searching on the shorter array guarantees $0 \le j \le n$ for all $0 \le i \le m$.
2. **The $(m + n + 1) // 2$ Integer Floor Division:**
   - Adding $1$ to $(m + n)$ before dividing by 2 ensures that the left partition holds the extra element whenever the total count is odd, so the median is simply $\max(\text{maxLeft1}, \text{maxLeft2})$.
3. **Empty Array Edge Cases:**
   - Handled seamlessly by setting $\text{maxLeft} = -\infty$ and $\text{minRight} = +\infty$ when partition indices are $0$ or equal to array length.
