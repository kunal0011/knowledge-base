---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 11: Container With Most Water"
tags:
  - leetcode
  - coding
  - greedy
  - two-pointers
  - array
  - amazon
  - google
---

# LeetCode 11: Container With Most Water

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Two Pointers / Array  

---

### Problem Statement

You are given an integer array `height` of length $n$. There are $n$ vertical lines drawn such that the two endpoints of the $i$-th line are $(i, 0)$ and $(i, \text{height}[i])$.

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the **maximum amount of water** a container can store.

Notice that you may not slant the container.

---

### Input & Output Formats & Constraints

- **Input:**
  - `height`: `List[int]` / `vector<int>` / `int[]` ($2 \le \text{height.length} \le 10^5$).
- **Output:**
  - `int` — the maximum volume/area of water the container can hold.
- **Constraints:**
  - $n == \text{height.length}$
  - $2 \le n \le 10^5$
  - $0 \le \text{height}[i] \le 10^4$

---

### Key Idea & Intuition

The area formed by two lines at indices $l$ and $r$ ($l < r$) is given by:
$$\text{Area}(l, r) = \min(\text{height}[l], \text{height}[r]) \times (r - l)$$

#### The Greedy Proof by Contradiction:
1. Start with the widest possible container: $l = 0$ and $r = n - 1$.
2. Suppose $\text{height}[l] < \text{height}[r]$.
   - Consider any pair $(l, k)$ where $k \in (l, r)$:
     $$\text{Width}(l, k) = k - l < r - l$$
     $$\text{Height}(l, k) = \min(\text{height}[l], \text{height}[k]) \le \text{height}[l] = \min(\text{height}[l], \text{height}[r])$$
   - Therefore, for **every** $k < r$:
     $$\text{Area}(l, k) < \text{Area}(l, r)$$
   - Index $l$ can **never** pair with any other right boundary to produce an area greater than $\text{Area}(l, r)$.
   - Hence, we can safely and permanently discard $l$ by incrementing $l \mathrel{+}= 1$ without missing any optimal solution!
3. By symmetry, if $\text{height}[r] < \text{height}[l]$, we safely decrement $r \mathrel{-}= 1$.
4. If $\text{height}[l] == \text{height}[r]$, moving either pointer (or both) is safe because neither can produce a larger area with any interior line without finding a line strictly taller than both.

This establishes an invariant where at each step, one boundary is discarded with mathematical certainty, converging in exactly $n - 1$ steps.

---

### Solution Approach (Step-by-Step)

1. Initialize `l = 0`, `r = len(height) - 1`, and `max_water = 0`.
2. While `l < r`:
   - Compute current width: `width = r - l`.
   - Compute current area: `area = min(height[l], height[r]) * width`.
   - Update `max_water = max(max_water, area)`.
   - If `height[l] < height[r]`:
     - Increment `l += 1`.
   - Else:
     - Decrement `r -= 1`.
3. Return `max_water`.

---

### Visual Algorithm Walkthrough

For `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]`:

```
Indices:   0   1   2   3   4   5   6   7   8
Heights:  [1,  8,  6,  2,  5,  4,  8,  3,  7]

Step 1: l = 0 (h=1), r = 8 (h=7), width = 8
  Area = min(1, 7) * 8 = 1 * 8 = 8. max_water = 8
  height[0] < height[8] -> increment l to 1.

Step 2: l = 1 (h=8), r = 8 (h=7), width = 7
  Area = min(8, 7) * 7 = 7 * 7 = 49. max_water = 49
  height[8] < height[1] -> decrement r to 7.

Step 3: l = 1 (h=8), r = 7 (h=3), width = 6
  Area = min(8, 3) * 6 = 3 * 6 = 18.
  height[7] < height[1] -> decrement r to 6.

Step 4: l = 1 (h=8), r = 6 (h=8), width = 5
  Area = min(8, 8) * 5 = 8 * 5 = 40.
  Equal -> decrement r to 5.

... Pointers continue to shrink inwards, no area exceeds 49.
Pointers meet at index 1.
Maximum Area = 49.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]`
- **Output:** `49` (Between index 1 and index 8: $\min(8, 7) \times (8 - 1) = 7 \times 7 = 49$)

#### Example 2:
- **Input:** `height = [1, 1]`
- **Output:** `1` (Between index 0 and index 1: $\min(1, 1) \times 1 = 1$)

#### Example 3 (Monotonically Decreasing Heights):
- **Input:** `height = [4, 3, 2, 1, 4]`
- **Tracing:** Index 0 (4) and Index 4 (4) yield $\min(4, 4) \times 4 = 16$.
- **Output:** `16`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        l = 0
        r = len(height) - 1
        max_water = 0
        
        while l < r:
            h_l = height[l]
            h_r = height[r]
            
            # Compute current container capacity
            if h_l < h_r:
                area = h_l * (r - l)
                l += 1
            else:
                area = h_r * (r - l)
                r -= 1
                
            if area > max_water:
                max_water = area
                
        return max_water
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxArea(const std::vector<int>& height) {
        int l = 0;
        int r = static_cast<int>(height.size()) - 1;
        int max_water = 0;
        
        while (l < r) {
            int h_l = height[l];
            int h_r = height[r];
            int area = 0;
            
            if (h_l < h_r) {
                area = h_l * (r - l);
                ++l;
            } else {
                area = h_r * (r - l);
                --r;
            }
            
            max_water = std::max(max_water, area);
        }
        
        return max_water;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxArea(int[] height) {
        int l = 0;
        int r = height.length - 1;
        int maxWater = 0;
        
        while (l < r) {
            int hl = height[l];
            int hr = height[r];
            int area;
            
            if (hl < hr) {
                area = hl * (r - l);
                l++;
            } else {
                area = hr * (r - l);
                r--;
            }
            
            if (area > maxWater) {
                maxWater = area;
            }
        }
        
        return maxWater;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - The pointers `l` and `r` start at opposite ends of the array of size $n$ and advance towards each other by at least 1 at every step.
  - Exactly $n - 1$ steps are performed before they meet.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Only scalar pointer variables and area accumulators are used.

---

### Takeaway Pattern & Interview Traps

- **The Proof by Contradiction:** When asked *"Why is it safe to advance the pointer with the smaller height?"*, articulate that keeping the shorter line while decreasing width cannot possibly increase the area, because the width strictly decreases while the height remains bounded by that shorter line.
- **Micro-Optimization:** Fast-skipping: while advancing $l$, you can do `while l < r and height[l] <= hl: l += 1` to skip lines shorter than the previous boundary without recalculating area.