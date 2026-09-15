---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 42: Trapping Rain Water"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - two-pointers
  - prefix-sum
  - amazon
  - google
  - meta
---

# LeetCode 42: Trapping Rain Water

**Target Companies:** Amazon (Top #1), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Hard  
**Topic:** Dynamic Programming (Prefix/Suffix Max) / Two Pointers / Monotonic Stack  

---

### Problem Statement

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.

---

### Input & Output Formats & Constraints

- **Input:** `height: List[int]` — Array of non-negative integers representing elevation.
- **Output:** `int` — Total volume of trapped rainwater.
- **Constraints:**
  - $n == \text{height.length}$
  - $1 \le n \le 2 \times 10^4$
  - $0 \le \text{height}[i] \le 10^5$

---

### Key Idea & Intuition

1. **Water at Index $i$ (First Principles):**
   - Consider a single column at index $i$. The water trapped directly above column $i$ is bounded by:
     - The tallest bar to its left: $\text{left\_max}[i] = \max_{0 \le j \le i} \text{height}[j]$.
     - The tallest bar to its right: $\text{right\_max}[i] = \max_{i \le j < n} \text{height}[j]$.
   - The effective water surface level above bar $i$ is the lower of these two enclosing boundaries:
     $$\text{water\_level}[i] = \min(\text{left\_max}[i], \text{right\_max}[i])$$
   - Therefore, the trapped water at column $i$ is:
     $$\text{trapped}[i] = \max(0, \min(\text{left\_max}[i], \text{right\_max}[i]) - \text{height}[i])$$

2. **Dynamic Programming Formulation:**
   - Instead of recalculating left and right maxima in $\mathcal{O}(n)$ per element (total $\mathcal{O}(n^2)$), compute two prefix/suffix DP arrays in linear time:
     $$\text{left\_max}[i] = \max(\text{left\_max}[i - 1], \text{height}[i])$$
     $$\text{right\_max}[i] = \max(\text{right\_max}[i + 1], \text{height}[i])$$
   - Both arrays take $\mathcal{O}(n)$ time and $\mathcal{O}(n)$ space.

3. **Space Optimization to $\mathcal{O}(1)$ (Two Pointers):**
   - Notice that if $\text{left\_max} < \text{right\_max}$, the bottleneck for column $left$ is unconditionally dictated by $\text{left\_max}$, regardless of any higher peaks far to the right.
   - We can maintain pointers `left` and `right` with running `left_max` and `right_max`, processing the smaller side inward in $\mathcal{O}(1)$ space.

---

### Solution Approach (Step-by-Step)

#### Approach 1: Two Pointers ($\mathcal{O}(1)$ Space - Optimal)
1. Initialize `left = 0`, `right = n - 1`, `left_max = 0`, `right_max = 0`, `water = 0`.
2. While `left < right`:
   - If `height[left] < height[right]`:
     - If `height[left] >= left_max`, update `left_max = height[left]`.
     - Else, add `left_max - height[left]` to `water`.
     - `left += 1`.
   - Else:
     - If `height[right] >= right_max`, update `right_max = height[right]`.
     - Else, add `right_max - height[right]` to `water`.
     - `right -= 1`.
3. Return `water`.

---

### Visual Algorithm Walkthrough

For `height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]`:

```
Elevation Map:
              #
      #       # #   #
  #   # #   # # # # # #
0 1 0 2 1 0 1 3 2 1 2 1

Prefix Left Max:
[0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]

Suffix Right Max:
[3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1]

min(left_max, right_max):
[0, 1, 1, 2, 2, 2, 2, 3, 2, 2, 2, 1]

Trapped Water (min - height):
i=0:  0 - 0 = 0
i=1:  1 - 1 = 0
i=2:  1 - 0 = 1  (Trapped!)
i=3:  2 - 2 = 0
i=4:  2 - 1 = 1  (Trapped!)
i=5:  2 - 0 = 2  (Trapped!)
i=6:  2 - 1 = 1  (Trapped!)
i=7:  3 - 3 = 0
i=8:  2 - 2 = 0
i=9:  2 - 1 = 1  (Trapped!)
i=10: 2 - 2 = 0
i=11: 1 - 1 = 0

Total Water = 1 + 1 + 2 + 1 + 1 = 6
```

---

### Solved Examples with Multiple Inputs

| Case | `height` | Left / Right Enclosing Peaks | Result | Explanation |
|---|---|---|---|---|
| **Standard** | `[0,1,0,2,1,0,1,3,2,1,2,1]` | Peaks at index 3 (height 2) and index 7 (height 3) | `6` | 6 units of water trapped |
| **Simple Basin** | `[4, 2, 0, 3, 2, 5]` | Left wall 4, Right wall 5 | `9` | Basin holds $(4-2)+(4-0)+(4-3)+(4-2)=2+4+1+2=9$ |
| **Monotonically Increasing** | `[1, 2, 3, 4, 5]` | No right wall taller than any peak | `0` | Water flows off to the right |
| **Flat Terrain** | `[3, 3, 3, 3]` | No depression | `0` | No hollows to hold water |
| **V-Shape** | `[3, 0, 3]` | Left wall 3, Right wall 3 | `3` | Center holds $3 - 0 = 3$ |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Two Pointers Optimal)
```python
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
            
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        water = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water += right_max - height[right]
                right -= 1
                
        return water
```

#### 2. C++ (C++17 / STL — Two Pointers Optimal)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int trap(std::vector<int>& height) {
        if (height.empty()) return 0;

        int left = 0, right = height.size() - 1;
        int left_max = 0, right_max = 0;
        int total_water = 0;

        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= left_max) {
                    left_max = height[left];
                } else {
                    total_water += left_max - height[left];
                }
                left++;
            } else {
                if (height[right] >= right_max) {
                    right_max = height[right];
                } else {
                    total_water += right_max - height[right];
                }
                right--;
            }
        }

        return total_water;
    }
};
```

#### 3. Java (Modern, Typed — Two Pointers Optimal)
```java
class Solution {
    public int trap(int[] height) {
        if (height == null || height.length == 0) return 0;

        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int totalWater = 0;

        while (left < right) {
            if (height[left] < height[right]) {
                if (height[left] >= leftMax) {
                    leftMax = height[left];
                } else {
                    totalWater += leftMax - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightMax) {
                    rightMax = height[right];
                } else {
                    totalWater += rightMax - height[right];
                }
                right--;
            }
        }

        return totalWater;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$  
  Both the two-pointer approach and the 2-pass DP approach visit each bar at most twice, resulting in strictly $\mathcal{O}(n)$ time.
- **Space Complexity:**  
  - Two Pointers: $\mathcal{O}(1)$ auxiliary space.
  - Prefix/Suffix DP: $\mathcal{O}(n)$ auxiliary space to store `left_max` and `right_max`.

---

### Takeaway Pattern & Interview Traps

1. **The Invariant Behind Two Pointers:**
   - Why can we advance `left` when `height[left] < height[right]` without knowing the absolute global maximum of the right side?
   - Because `height[right]` is already greater than `height[left]`, we know $\text{right\_max} \ge \text{height}[right] > \text{height}[left]$. Therefore, $\text{left\_max}$ is guaranteed to be smaller than $\text{right\_max}$, making $\text{left\_max}$ the true limiting boundary.
2. **Alternative Approaches in Interviews:**
   - Be prepared to discuss all three paradigms:
     1. **Dynamic Programming:** Prefix/Suffix Max arrays ($\mathcal{O}(n)$ time, $\mathcal{O}(n)$ space).
     2. **Monotonic Stack:** Tracks bounded troughs horizontally ($\mathcal{O}(n)$ time, $\mathcal{O}(n)$ space).
     3. **Two Pointers:** Processes water vertically from ends inwards ($\mathcal{O}(n)$ time, $\mathcal{O}(1)$ space).