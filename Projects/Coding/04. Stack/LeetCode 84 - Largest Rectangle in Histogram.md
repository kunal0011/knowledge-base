---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 84: Largest Rectangle in Histogram"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - amazon
  - google
---

# LeetCode 84: Largest Rectangle in Histogram

**Target Companies:** Amazon (Top Classic), Google, Microsoft, Meta, Apple, Bloomberg  
**Difficulty:** Hard  
**Topic:** Monotonic Increasing Stack / Histogram Area

---

### Problem Statement

Given an array of integers `heights` representing the histogram's bar height where the width of each bar is `1`, return *the area of the largest rectangle in the histogram*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `heights`: `List[int]`, where $n = \text{len}(heights)$.
- **Output:**
  - `int`: The maximum rectangular area formed by contiguous bars.
- **Constraints:**
  - $1 \le \text{heights.length} \le 10^5$.
  - $0 \le heights[i] \le 10^4$.

---

### Key Idea & Intuition

Any valid rectangle in a histogram is constrained vertically by the **shortest bar** within its horizontal span.

Therefore, for every bar $i$ with height $h = heights[i]$:
- What is the maximum width of a rectangle where bar $i$ is the bottleneck (shortest) bar?
- It extends to the left until it encounters the first bar strictly shorter than $heights[i]$ ($L$).
- It extends to the right until it encounters the first bar strictly shorter than $heights[i]$ ($R$).
- The maximal area with $heights[i]$ as the limiting height is:
  $$\text{Area} = heights[i] \times (R - L - 1)$$

Finding the **first smaller element to the left** and the **first smaller element to the right** for all elements simultaneously is solved in linear $\mathcal{O}(N)$ time using a **Monotonic Increasing Stack**:
1. The stack stores indices of bars in strictly non-decreasing height order.
2. When scanning index $i$ with $heights[i]$:
   - If $heights[i] < heights[stack.top()]$, the bar at $stack.top()$ cannot extend any further to the right!
   - We pop $stack.top()$ as the bottleneck bar $h$.
   - The right boundary $R$ is the current index $i$.
   - The left boundary $L$ is the new top of the stack (or $-1$ if the stack is now empty).
   - Width is $i - L - 1$.
   - Area is $h \times (i - L - 1)$.
3. By appending a dummy sentinel $0$ at the end of the array, all remaining bars in the stack are cleanly flushed out and evaluated before terminating.

---

### Solution Approach (Step-by-Step)

1. Initialize `stack = []` and `max_area = 0`.
2. Loop $i$ from $0$ to $n$ (inclusive, treating $i = n$ as height $0$ to flush):
   - `curr_height = heights[i] if i < n else 0`
   - While `stack` is non-empty and `curr_height < heights[stack[-1]]`:
     - `h = heights[stack.pop()]`
     - `w = i if not stack else (i - stack[-1] - 1)`
     - `max_area = max(max_area, h * w)`
   - `stack.append(i)`
3. Return `max_area`.

---

### Visual Algorithm Walkthrough

Let `heights = [2, 1, 5, 6, 2, 3]`:

```
Indices:    0  1  2  3  4  5  (and virtual 6 with height 0)
Heights:    2  1  5  6  2  3  [0]

-------------------------------------------------------------------------
i = 0, h = 2:
  Push 0. Stack: [0]

i = 1, h = 1:
  1 < 2! Pop 0:
    h = 2, stack empty -> w = 1. Area = 2 * 1 = 2.
  Push 1. Stack: [1]

i = 2, h = 5:
  5 > 1. Push 2. Stack: [1, 2]

i = 3, h = 6:
  6 > 5. Push 3. Stack: [1, 2, 3]

i = 4, h = 2:
  2 < 6! Pop 3:
    h = 6, left = 2. w = 4 - 2 - 1 = 1. Area = 6 * 1 = 6.
  2 < 5! Pop 2:
    h = 5, left = 1. w = 4 - 1 - 1 = 2. Area = 5 * 2 = 10! <-- MAX!
  2 >= 1. Stop popping.
  Push 4. Stack: [1, 4]

i = 5, h = 3:
  3 > 2. Push 5. Stack: [1, 4, 5]

i = 6 (Sentinel h = 0):
  0 < 3! Pop 5: h = 3, left = 4. w = 6 - 4 - 1 = 1. Area = 3 * 1 = 3.
  0 < 2! Pop 4: h = 2, left = 1. w = 6 - 1 - 1 = 4. Area = 2 * 4 = 8.
  0 < 1! Pop 1: h = 1, left = -1. w = 6. Area = 1 * 6 = 6.

Max Area Found: 10 (bars at index 2 and 3: heights [5, 6] with width 2).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Histogram

- **Input:** `heights = [2, 1, 5, 6, 2, 3]`
- **Output:** `10`

#### Example 2: Uniform Height

- **Input:** `heights = [2, 2, 2, 2]`
- **Tracing:** Stack pushes all 4 bars. Sentinel flushes all bars with width 4 $\implies 2 \times 4 = 8$.
- **Output:** `8`

#### Example 3: Strictly Increasing

- **Input:** `heights = [1, 2, 3, 4, 5]`
- **Tracing:** Every bar pushed. Sentinel flushes:
  - $h=5, w=1 \implies 5$
  - $h=4, w=2 \implies 8$
  - $h=3, w=3 \implies 9$
  - $h=2, w=4 \implies 8$
  - $h=1, w=5 \implies 5$
- **Output:** `9`

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []  # stores indices of bars in monotonic increasing order
        max_area = 0
        n = len(heights)

        # Loop up to n (inclusive) to use virtual height 0 as sentinel flush
        for i in range(n + 1):
            curr_height = heights[i] if i < n else 0

            while stack and curr_height < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, h * w)

            stack.append(i)

        return max_area
```

#### C++17

```cpp
#include <vector>
#include <stack>
#include <algorithm>

class Solution {
public:
    int largestRectangleArea(const std::vector<int>& heights) {
        std::vector<int> stack;
        int max_area = 0;
        int n = static_cast<int>(heights.size());

        for (int i = 0; i <= n; ++i) {
            int curr_height = (i < n) ? heights[i] : 0;

            while (!stack.empty() && curr_height < heights[stack.back()]) {
                int h = heights[stack.back()];
                stack.pop_back();
                int w = stack.empty() ? i : (i - stack.back() - 1);
                max_area = std::max(max_area, h * w);
            }

            stack.push_back(i);
        }

        return max_area;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public int largestRectangleArea(int[] heights) {
        Deque<Integer> stack = new ArrayDeque<>();
        int maxArea = 0;
        int n = heights.length;

        for (int i = 0; i <= n; i++) {
            int currHeight = (i < n) ? heights[i] : 0;

            while (!stack.isEmpty() && currHeight < heights[stack.peek()]) {
                int h = heights[stack.pop()];
                int w = stack.isEmpty() ? i : (i - stack.peek() - 1);
                maxArea = Math.max(maxArea, h * w);
            }

            stack.push(i);
        }

        return maxArea;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Each index $0 \dots N$ is pushed onto the stack exactly once and popped at most once.
  - Computing the area during each pop takes $\mathcal{O}(1)$ arithmetic.
  - Overall Time: $\mathcal{O}(N)$, where $N$ is the number of bars.
- **Space Complexity:** $\mathcal{O}(N)$
  - In the worst case (strictly increasing bars), the stack holds all $N$ indices.
  - Overall Space: $\mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **The Sentinel $0$ Technique:**
   - Appending or virtualizing a bar of height $0$ at index $N$ ensures that any unpopped elements remaining in the stack are evaluated and flushed. This avoids writing duplicate cleanup loops after the main loop.
2. **Width Calculation Invariant:**
   - If the stack is empty after popping, the popped bar was the shortest bar seen so far from index $0$ to $i-1$, meaning it spans the entire width $i$.
   - If the stack is not empty, the bar spans from `stack.top() + 1` to `i - 1`, giving width $i - \text{stack.top()} - 1$.