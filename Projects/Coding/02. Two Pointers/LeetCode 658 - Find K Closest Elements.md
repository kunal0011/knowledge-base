---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 658: Find K Closest Elements"
tags:
  - leetcode
  - coding
  - two-pointers
  - binary-search
  - sliding-window
  - google
  - amazon
  - meta
---

# LeetCode 658: Find K Closest Elements

**Target Companies:** Google, Amazon, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Binary Search on Left Boundary / Two Pointers Sliding Window

---

### Problem Statement

Given a **sorted** integer array `arr`, two integers `k` and `x`, return the `k` closest integers to `x` in the array. The result should also be sorted in ascending order.

An integer `a` is closer to `x` than an integer `b` if:
- `|a - x| < |b - x|`, or
- `|a - x| == |b - x|` and `a < b`

---

### Input & Output Formats & Constraints

- **Input:** `arr: List[int]`, `k: int`, `x: int`
- **Output:** `List[int]` (length $k$, sorted ascending)
- **Constraints:**
  - $1 \le k \le \text{arr.length} \le 10^4$
  - `arr` is sorted in **ascending** order.
  - $-10^4 \le \text{arr}[i], x \le 10^4$

---

### Key Idea & Intuition

Because the array `arr` is already sorted, the $k$ closest elements must form a **contiguous subarray** of length $k$.

#### 1. Two Pointers Shrinking Approach ($\mathcal{O}(n)$):
- Set `left = 0`, `right = len(arr) - 1`.
- While `right - left + 1 > k`:
  - Compare distances to $x$:
    - If $|arr[left] - x| > |arr[right] - x|$, $arr[left]$ is farther $\implies left++$.
    - Else (if $|arr[left] - x| \le |arr[right] - x|$), $arr[right]$ is farther or tied $\implies right--$ (preferring the smaller element).
- Return `arr[left : right + 1]`.

#### 2. Binary Search on the Window's Left Boundary ($\mathcal{O}(\log(n - k) + k)$):
- The starting index $left$ of a contiguous window of size $k$ must lie within $[0, n - k]$.
- For any candidate starting index $mid$, compare the elements on the boundaries:
  - $arr[mid]$ (left boundary candidate)
  - $arr[mid + k]$ (the element immediately outside the window on the right)
- If $x - arr[mid] > arr[mid + k] - x$:
  - This means $arr[mid + k]$ is strictly closer to $x$ than $arr[mid]$.
  - Therefore, the optimal window cannot start at $mid$ (or anywhere to the left of $mid$). We must shift right: `left = mid + 1`.
- Else:
  - $arr[mid]$ is closer or tied with $arr[mid + k]$.
  - The window starting at $mid$ is either optimal or too far right: `right = mid`.
- This locates the exact starting index in $\mathcal{O}(\log(n - k))$ steps!

---

### Solution Approach (Step-by-Step)

1. **Binary Search Initialization:**
   - `left = 0`, `right = len(arr) - k`.
2. **Binary Search Loop:**
   - While `left < right`:
     - `mid = left + (right - left) // 2`.
     - If `x - arr[mid] > arr[mid + k] - x`:
       - `left = mid + 1`.
     - Else:
       - `right = mid`.
3. **Return Slice:**
   - Return `arr[left : left + k]`.

---

### Visual Algorithm Walkthrough

#### Example: `arr = [1, 2, 3, 4, 5]`, `k = 4`, `x = 3` ($n - k = 5 - 4 = 1$)

```
Search space for left boundary: [0, 1]

Iteration 1:
  left = 0, right = 1
  mid = (0 + 1) // 2 = 0
  Compare arr[mid=0] (1) vs arr[mid+k=4] (5) to x = 3:
    Dist left:  x - arr[0] = 3 - 1 = 2
    Dist right: arr[4] - x = 5 - 3 = 2
    Is 2 > 2? False! (Ties favor smaller left element)
    right = mid = 0

Loop terminates: left == right == 0.
Window starts at index 0.
Subarray of length k=4: arr[0:4] = [1, 2, 3, 4].
```

---

### Solved Examples with Multiple Inputs

| Input `arr` | `k` | `x` | Boundary Comparison | Output |
| :--- | :--- | :--- | :--- | :--- |
| `[1, 2, 3, 4, 5]` | 4 | 3 | Tie between 1 and 5 favors smaller values | `[1, 2, 3, 4]` |
| `[1, 2, 3, 4, 5]` | 4 | -1 | $x$ far to left $\to$ leftmost window chosen | `[1, 2, 3, 4]` |
| `[1, 2, 3, 4, 5]` | 4 | 10 | $x$ far to right $\to$ rightmost window chosen | `[2, 3, 4, 5]` |
| `[1, 1, 1, 10, 10, 10]` | 1 | 9 | `arr[mid] = 1` vs `arr[mid+1] = 10` $\to 10$ is closer | `[10]` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - k
        
        while left < right:
            mid = left + (right - left) // 2
            # If arr[mid + k] is closer to x than arr[mid], discard mid and move right
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
                
        return arr[left : left + k]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> findClosestElements(std::vector<int>& arr, int k, int x) {
        int left = 0, right = arr.size() - k;

        while (left < right) {
            int mid = left + (right - left) / 2;
            // Compare distance to x
            if (x - arr[mid] > arr[mid + k] - x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return std::vector<int>(arr.begin() + left, arr.begin() + left + k);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int left = 0, right = arr.length - k;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (x - arr[mid] > arr[mid + k] - x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        List<Integer> result = new ArrayList<>(k);
        for (int i = left; i < left + k; i++) {
            result.add(arr[i]);
        }
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\log(n - k) + k)$. The binary search executes in $\mathcal{O}(\log(n - k))$ iterations, and extracting the $k$ elements into the result list takes $\mathcal{O}(k)$ time. This is significantly faster than standard $\mathcal{O}(n)$ two-pointer shrinking.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space beyond the output array.

---

### Takeaway Pattern & Interview Traps

1. **Why `x - arr[mid] > arr[mid + k] - x` without `abs()`?**
   - If $x$ is between $arr[mid]$ and $arr[mid + k]$, then $x - arr[mid] \ge 0$ and $arr[mid + k] - x \ge 0$, representing standard distances.
   - If $x < arr[mid]$, $x - arr[mid]$ is negative, while $arr[mid + k] - x$ is positive, so the inequality is `False`, correctly pulling the window leftward (`right = mid`).
   - If $x > arr[mid + k]$, $x - arr[mid]$ is positive and larger than the negative $arr[mid + k] - x$, correctly advancing rightward (`left = mid + 1`).
   - Thus, omitting `abs()` handles all boundary conditions seamlessly without extra branching!
2. **Binary Search Window Upper Bound:**
   - The search range is `[0, len(arr) - k]`, not `[0, len(arr) - 1]`. The window cannot begin after index $n - k$.
