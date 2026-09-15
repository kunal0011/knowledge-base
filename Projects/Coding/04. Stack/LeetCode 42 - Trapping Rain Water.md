---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 42: Trapping Rain Water"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - two-pointers
  - amazon
  - google
---

# LeetCode 42: Trapping Rain Water

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Hard  
**Topic:** Monotonic Stack / Two Pointers / Dynamic Programming

---

### Problem Statement

Given `n` non-negative integers representing an elevation map where the width of each bar is `1`, compute how much water it can trap after raining.

---

### Input & Output Formats & Constraints

- **Input:**
  - `height`: `List[int]`, where $n = \text{len}(height)$. $1 \le n \le 2 \times 10^4$.
  - $0 \le height[i] \le 10^5$.
- **Output:**
  - `int`: Total units of rainwater trapped between the elevations.
- **Constraints:**
  - Width of each bar is $1$.
  - Elevation heights are non-negative.

---

### Key Idea & Intuition

Water can only be trapped in a "valley"—a depression bounded by taller bars on both the left and right.

There are two primary paradigms to compute trapped water:

#### Paradigm 1: Monotonic Stack (Horizontal Layer Trapping)
Instead of accumulating water column-by-column vertically, we accumulate water **layer-by-layer horizontally**:
- Maintain a **monotonic decreasing stack** of bar indices.
- While the current bar `height[i]` is strictly greater than the bar at `stack[-1]`:
  - We have identified a **right boundary** (`height[i]`).
  - Pop `bottom = stack.pop()`. This bar represents the floor of the valley.
  - If the stack is now empty, there is no left boundary; no water can be trapped.
  - Otherwise, the new `stack[-1]` is the **left boundary**.
  - The bounded height is:
    $$\text{bounded\_height} = \min(height[\text{left}], height[i]) - height[\text{bottom}]$$
  - The horizontal span is:
    $$\text{distance} = i - \text{left} - 1$$
  - Trapped water added: $\text{distance} \times \text{bounded\_height}$.
- Push current index $i$ onto the stack.

#### Paradigm 2: Two Pointers ($\mathcal{O}(1)$ Auxiliary Space)
Water trapped at any individual column $i$ is determined strictly by:
$$\text{water}[i] = \max(0, \min(\text{left\_max}, \text{right\_max}) - height[i])$$
With two pointers `left` and `right`:
- Maintain `left_max` and `right_max`.
- If `left_max < right_max`, the bottleneck for `left` is guaranteed to be `left_max`, regardless of future bars. We accumulate water at `left` and increment `left`.
- Otherwise, the bottleneck for `right` is guaranteed to be `right_max`. We accumulate water at `right` and decrement `right`.

---

### Solution Approach (Step-by-Step)

#### Approach 1: Monotonic Stack Algorithm
1. Initialize `stack = []`, `water = 0`.
2. For index $i$ from $0$ to $n - 1$:
   - While `stack` is non-empty and `height[i] > height[stack[-1]]`:
     - `bottom = stack.pop()`
     - If not `stack`: break
     - `left = stack[-1]`
     - `distance = i - left - 1`
     - `bounded_height = min(height[left], height[i]) - height[bottom]`
     - `water += distance * bounded_height`
   - `stack.append(i)`
3. Return `water`.

#### Approach 2: Two Pointers Algorithm ($\mathcal{O}(1)$ Space)
1. Initialize `left = 0, right = n - 1, left_max = 0, right_max = 0, water = 0`.
2. While `left < right`:
   - If `height[left] < height[right]`:
     - If `height[left] >= left_max`: `left_max = height[left]`
     - Else: `water += left_max - height[left]`
     - `left += 1`
   - Else:
     - If `height[right] >= right_max`: `right_max = height[right]`
     - Else: `water += right_max - height[right]`
     - `right -= 1`
3. Return `water`.

---

### Visual Algorithm Walkthrough

Let `height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]`:

```
Elevation Map:
              #
      # . . . # # . #
  # . # # . # # # # # #
-----------------------
0 1 0 2 1 0 1 3 2 1 2 1  (Index)

Stack trace at key valley filling:
- At i=3 (height=2):
  - Stack holds [1, 2] (bars of height 1 and 0).
  - Pop bottom = 2 (height 0).
  - Left boundary = 1 (height 1), Right boundary = 3 (height 2).
  - bounded_height = min(1, 2) - 0 = 1.
  - distance = 3 - 1 - 1 = 1.
  - water += 1 * 1 = 1.

- At i=7 (height=3):
  - Stack fills layers between left boundary index 3 (height 2) and right index 7 (height 3).
  - Horizontal slices trapped: 1 unit + 3 units = 4 units.

Total accumulated trapped water = 6 units.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Classic Profile

- **Input:** `height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]`
- **Output:** `6`

#### Example 2: Simple V-Shaped Basin

- **Input:** `height = [4, 2, 0, 3, 2, 5]`
- **Stack Trace:**
  - Valley floors at 0, 2, 3 trapped between 4 and 5.
  - Total trapped = $2 + 4 + 1 + 2 = 9$.
- **Output:** `9`

#### Example 3: Flat or Monotonic Slopes (No Trapped Water)

- **Input:** `height = [1, 2, 3, 4, 5]` or `height = [5, 4, 3, 2, 1]`
- **Output:** `0` (No valley formed).

---

### Multi-Language Implementations

#### Python 3

##### Monotonic Stack Approach ($\mathcal{O}(N)$ Time, $\mathcal{O}(N)$ Space)
```python
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []  # stores indices of bars in non-increasing height
        water = 0

        for i, h in enumerate(height):
            while stack and h > height[stack[-1]]:
                bottom = stack.pop()
                if not stack:
                    break  # no left boundary
                left = stack[-1]
                distance = i - left - 1
                bounded_height = min(height[left], h) - height[bottom]
                water += distance * bounded_height

            stack.append(i)

        return water
```

##### Optimal Two-Pointer Approach ($\mathcal{O}(N)$ Time, $\mathcal{O}(1)$ Space)
```python
class SolutionTwoPointer:
    def trap(self, height: List[int]) -> int:
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

#### C++17

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    // Monotonic Stack Approach
    int trap(const std::vector<int>& height) {
        std::vector<int> stack;
        int water = 0;
        int n = static_cast<int>(height.size());

        for (int i = 0; i < n; ++i) {
            while (!stack.empty() && height[i] > height[stack.back()]) {
                int bottom = stack.back();
                stack.pop_back();

                if (stack.empty()) break; // No left wall

                int left = stack.back();
                int distance = i - left - 1;
                int bounded_height = std::min(height[left], height[i]) - height[bottom];
                water += distance * bounded_height;
            }
            stack.push_back(i);
        }

        return water;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    // Monotonic Stack Approach
    public int trap(int[] height) {
        Deque<Integer> stack = new ArrayDeque<>();
        int water = 0;
        int n = height.length;

        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int bottom = stack.pop();

                if (stack.isEmpty()) {
                    break;
                }

                int left = stack.peek();
                int distance = i - left - 1;
                int boundedHeight = Math.min(height[left], height[i]) - height[bottom];
                water += distance * boundedHeight;
            }
            stack.push(i);
        }

        return water;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Stack Approach: Every bar index is pushed onto the stack once and popped at most once $\implies \mathcal{O}(N)$ time.
  - Two Pointers Approach: Each step advances either `left` or `right` $\implies \mathcal{O}(N)$ time.
- **Space Complexity:**
  - Stack Approach: $\mathcal{O}(N)$ to store indices in the stack in the worst case (strictly descending heights).
  - Two Pointers Approach: $\mathcal{O}(1)$ auxiliary space using constant scalar variables.

---

### Takeaway Pattern & Interview Traps

1. **Horizontal vs Vertical Trapping:**
   - The Two-Pointer technique accumulates water **vertically** for each single column: $\min(L_{max}, R_{max}) - h[i]$.
   - The Monotonic Stack technique accumulates water **horizontally** in bounded rectangular water slices between left boundary, bottom, and right boundary.
2. **Empty Stack After Pop:**
   - If `stack` becomes empty immediately after popping `bottom`, there is no left wall (`left = stack.peek()`), so no water can be held. You must break without computing area.
3. **Distance Calculation:**
   - The horizontal width is strictly `i - left - 1` (the number of bars strictly between `left` and `i`).