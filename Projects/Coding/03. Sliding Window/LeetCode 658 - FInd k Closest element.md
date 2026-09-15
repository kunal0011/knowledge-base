---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 658: Find K Closest Elements"
tags:
  - leetcode
  - coding
  - sliding-window
  - binary-search
  - two-pointers
  - array
  - amazon
  - google
---

# LeetCode 658: Find K Closest Elements

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Sliding Window / Two Pointers / Binary Search  

---

### Problem Statement

Given a **sorted** integer array `arr`, two integers `k` and `x`, return the `k` closest integers to `x` in the array. The result should also be sorted in ascending order.

An integer `a` is closer to `x` than an integer `b` if:
- `|a - x| < |b - x|`, or
- `|a - x| == |b - x|` and `a < b`

---

### Input & Output Formats & Constraints

- **Input:**
  - `arr`: `List[int]` / `vector<int>` / `int[]` — sorted in ascending order ($1 \le \text{arr.length} \le 10^4$).
  - `k`: `int` ($1 \le k \le \text{arr.length}$).
  - `x`: `int` ($-10^4 \le x \le 10^4$).
- **Output:**
  - `List[int]` / `vector<int>` / `int[]` — $k$ closest elements to $x$, sorted in ascending order.
- **Constraints:**
  - $1 \le k \le \text{arr.length} \le 10^4$
  - `arr` is sorted in ascending order.
  - $-10^4 \le \text{arr}[i], x \le 10^4$

---

### Key Idea & Intuition

Because the array `arr` is already sorted, any set of $k$ closest elements must form a **contiguous subarray / sliding window** of size $k$.

There are two primary paradigms:

#### Paradigm 1: Shrinking Two-Pointer Window ($\mathcal{O}(n - k)$ time, $\mathcal{O}(1)$ space)
- Start with a full window spanning $[0, n - 1]$.
- While window length $r - l + 1 > k$:
  - Compare `abs(arr[l] - x)` and `abs(arr[r] - x)`.
  - If `arr[l]` is farther than `arr[r]` (`abs(arr[l] - x) > abs(arr[r] - x)`), increment `l += 1`.
  - Else (if `arr[l]` is closer or equal distance), `arr[r]` is the one to eliminate because smaller elements take tie-break priority $\rightarrow$ decrement `r -= 1`.
- Stop when $r - l + 1 = k$.

#### Paradigm 2: Binary Search on the Window's Starting Index ($\mathcal{O}(\log(n - k) + k)$ time)
- The starting index $L$ of the optimal window must lie in $[0, n - k]$.
- For any candidate starting index `mid`:
  - The window under test is `arr[mid .. mid + k - 1]`.
  - To decide whether to shift the window to the right, compare the left boundary candidate `arr[mid]` with the element that would enter if shifted right: `arr[mid + k]`.
  - If $x - \text{arr}[mid] > \text{arr}[mid + k] - x$, then `arr[mid + k]` is strictly closer to $x$ than `arr[mid]`, which proves the window must start to the right $\rightarrow$ `left = mid + 1`.
  - Otherwise, `arr[mid]` is at least as good as `arr[mid + k]`, so the window cannot start to the right of `mid` $\rightarrow$ `right = mid`.
- Once `left == right`, return `arr[left : left + k]`.

---

### Solution Approach (Step-by-Step: Binary Search)

1. Set `left = 0` and `right = len(arr) - k`.
2. While `left < right`:
   - Compute `mid = left + (right - left) // 2`.
   - Compare distance between `arr[mid]` and `arr[mid + k]` relative to `x`:
     - If `x - arr[mid] > arr[mid + k] - x`:
       - `left = mid + 1`
     - Else:
       - `right = mid`
3. Return the sub-slice `arr[left : left + k]`.

---

### Visual Algorithm Walkthrough

For `arr = [1, 2, 3, 4, 5]`, `k = 4`, `x = 3`:

```
Range for start index: left = 0, right = 5 - 4 = 1

Iteration 1:
  mid = (0 + 1) // 2 = 0
  Compare arr[mid] = arr[0] = 1 with arr[mid + k] = arr[4] = 5
  x - arr[mid]   = 3 - 1 = 2
  arr[mid + k] - x = 5 - 3 = 2
  Is (3 - 1) > (5 - 3)? -> 2 > 2 is FALSE (Tie-break prefers smaller value arr[0]=1)
  -> right = mid = 0

Loop terminates (left == right == 0).
Output slice: arr[0 : 0 + 4] = [1, 2, 3, 4].
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `arr = [1, 2, 3, 4, 5]`, `k = 4`, `x = 3`
- **Output:** `[1, 2, 3, 4]`

#### Example 2:
- **Input:** `arr = [1, 2, 3, 4, 5]`, `k = 4`, `x = -1`
- **Tracing:** `x` is smaller than all elements. Start index is $0$.
- **Output:** `[1, 2, 3, 4]`

#### Example 3:
- **Input:** `arr = [1, 1, 1, 10, 10, 10]`, `k = 1`, `x = 9`
- **Tracing:** `10` has distance $|10 - 9| = 1$, which is closer than $1$ ($|1 - 9| = 8$).
- **Output:** `[10]`

---

### Multi-Language Implementations

#### Python 3 (Binary Search on Window Start)
```python
from typing import List

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left = 0
        right = len(arr) - k
        
        while left < right:
            mid = left + (right - left) // 2
            # Compare distance from x to arr[mid] vs distance from x to arr[mid + k]
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
                
        return arr[left : left + k]
```

#### C++17 (Binary Search on Window Start)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> findClosestElements(const std::vector<int>& arr, int k, int x) {
        int left = 0;
        int right = static_cast<int>(arr.size()) - k;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
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

#### Java 17 (Binary Search on Window Start)
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int left = 0;
        int right = arr.length - k;
        
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

- **Time Complexity:**
  - **Binary Search Approach:** $\mathcal{O}(\log(n - k) + k)$
    - Binary search takes $\mathcal{O}(\log(n - k))$ iterations.
    - Slicing or copying the $k$ elements takes $\mathcal{O}(k)$ time.
  - **Two-Pointer Shrinking Approach:** $\mathcal{O}(n - k)$
    - Shrinks from $n$ elements down to $k$ elements one by one.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space (excluding the output container).

---

### Takeaway Pattern & Interview Traps

- **Why Binary Search on Window Start Works Without Absolute Values:**
  Notice the condition `x - arr[mid] > arr[mid + k] - x`. We do NOT use `abs()`.
  Because `arr` is sorted:
  - If $x$ is between `arr[mid]` and `arr[mid + k]`, `x - arr[mid] >= 0` and `arr[mid + k] - x >= 0`, so the difference directly represents distances.
  - If $x \le arr[mid]$, `x - arr[mid] <= 0` while `arr[mid + k] - x > 0`, so the inequality is false, keeping `right = mid`.
  - If $x \ge arr[mid + k]$, `x - arr[mid] > 0` while `arr[mid + k] - x <= 0`, so the inequality is true, shifting `left = mid + 1`.
  The algebraic sign preserves perfect monotonicity without needing branches or absolute values!