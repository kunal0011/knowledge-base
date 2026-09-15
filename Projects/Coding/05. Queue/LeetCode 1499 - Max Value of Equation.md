---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1499: Max Value of Equation"
tags:
  - leetcode
  - coding
  - queue
  - monotonic-queue
  - math
  - google
  - amazon
---

# LeetCode 1499: Max Value of Equation

**Target Companies:** Google (Top Asked Signature Hard), Amazon  
**Difficulty:** Hard  
**Topic:** Coordinate Algebraic Factorization with Monotonic Decreasing Deque  

---

### Problem Statement

You are given an array `points` containing the coordinates of points on a 2D plane, sorted by the x-values, where `points[i] = [x_i, y_i]` such that $x_i < x_j$ for all $1 \le i < j \le \text{points.length}$. You are also given an integer $k$.

Return the **maximum value of the equation** $y_i + y_j + |x_i - x_j|$ where $|x_i - x_j| \le k$ and $1 \le i < j \le \text{points.length}$.

It is guaranteed that there exists at least one pair of points that satisfy the constraint $|x_i - x_j| \le k$.

---

### Input & Output Formats & Constraints

- **Input:** `points: List[List[int]]`, `k: int`
- **Output:** `int` (maximum value of equation)
- **Constraints:**
  - $2 \le \text{points.length} \le 10^5$
  - $\text{points}[i].\text{length} == 2$
  - $-10^8 \le x_i, y_i \le 10^8$
  - $1 \le k \le 2 \times 10^8$
  - $x_i < x_j$ for all $1 \le i < j \le \text{points.length}$

---

### Key Idea & Intuition

Since the array is strictly sorted by $x$-coordinate and $i < j$, we know that $x_j > x_i$, so $|x_i - x_j| = x_j - x_i$.

We rewrite the equation algebraically by separating terms dependent on $j$ from terms dependent on $i$:
$$y_i + y_j + |x_i - x_j| = y_i + y_j + (x_j - x_i) = (x_j + y_j) + (y_i - x_i)$$

For each fixed point $j = (x_j, y_j)$, the term $(x_j + y_j)$ is fixed. To maximize the whole sum, we need to find an index $i < j$ that:
1. Satisfies the sliding window boundary: $x_j - x_i \le k$.
2. Maximizes the value of $(y_i - x_i)$.

This is the exact structure of a sliding window maximum!
We maintain a **Monotonic Decreasing Deque** of pairs $(y_i - x_i, x_i)$:
- The deque is kept in strictly decreasing order of $(y_i - x_i)$.
- When considering point $j$:
  - Evict expired points from the front where $x_j - x_i > k$.
  - If the deque is non-empty, the front element provides the maximum $(y_i - x_i)$ within the valid range! We update `max_val = max(max_val, x_j + y_j + deque[0].val)`.
  - Maintain monotonicity: pop elements from the back while their stored value $\le y_j - x_j$.
  - Push $(y_j - x_j, x_j)$ to the back.

---

### Solution Approach (Step-by-Step)

1. Initialize `q = deque()` storing pairs `(diff, x)` where `diff = y - x`.
2. Initialize `max_val = -infinity`.
3. For each point `[x, y]` in `points`:
   - **Boundary Eviction:** While `q` and `x - q[0][1] > k`: `q.popleft()`.
   - **Evaluate Best Candidate:** If `q`: `max_val = max(max_val, (x + y) + q[0][0])`.
   - **Maintain Monotonic Decreasing Order:** Let `curr_diff = y - x`.
     - While `q` and `q[-1][0] <= curr_diff`: `q.pop()`.
     - `q.append((curr_diff, x))`.
4. Return `max_val`.

---

### Visual Algorithm Walkthrough

```
points = [[1, 3], [2, 0], [5, 10], [6, -10]], k = 1

Formula: (x_j + y_j) + (y_i - x_i) subject to x_j - x_i <= 1

Point 0: [1, 3]
  diff = 3 - 1 = 2
  q is empty.
  Push (2, 1) -> q = [(2, 1)]

Point 1: [2, 0]
  Check expiry: 2 - 1 = 1 <= 1 (VALID!)
  q[0] is (2, 1) -> max_val = max(-inf, (2 + 0) + 2) = 4.
  diff = 0 - 2 = -2.
  q[-1][0] is 2 > -2 -> push (-2, 2).
  q = [(2, 1), (-2, 2)]

Point 2: [5, 10]
  Check expiry:
    5 - 1 = 4 > 1 -> popleft (2, 1)
    5 - 2 = 3 > 1 -> popleft (-2, 2)
  q is now empty.
  diff = 10 - 5 = 5.
  Push (5, 5) -> q = [(5, 5)]

Point 3: [6, -10]
  Check expiry: 6 - 5 = 1 <= 1 (VALID!)
  q[0] is (5, 5) -> max_val = max(4, (6 + (-10)) + 5) = max(4, -4 + 5) = max(4, 1) = 4.
  diff = -10 - 6 = -16.
  q = [(5, 5), (-16, 6)]

Final Result: 4 (from points [1, 3] and [2, 0]: 3 + 0 + |1 - 2| = 3 + 0 + 1 = 4).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Proximity Window
- **Input:** `points = [[1, 3], [2, 0], [5, 10], [6, -10]]`, `k = 1`
- **Output:** `4`

#### Example 2: Wide Window Range
- **Input:** `points = [[0, 0], [3, 0], [9, 2]]`, `k = 3`
- **Trace:**
  - Pair `([0, 0], [3, 0])`: $|0 - 3| \le 3 \implies 0 + 0 + 3 = 3$.
  - Pair `([3, 0], [9, 2])`: $|3 - 9| = 6 > 3$ (out of range).
- **Output:** `3`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        # q stores tuples: (y - x, x), monotonically decreasing in (y - x)
        q = deque()
        max_val = float('-inf')
        
        for x, y in points:
            # 1. Evict points outside the x-distance window k
            while q and x - q[0][1] > k:
                q.popleft()
                
            # 2. The front has the maximum (y_i - x_i)
            if q:
                max_val = max(max_val, (x + y) + q[0][0])
                
            # 3. Maintain monotonic decreasing order
            curr_diff = y - x
            while q and q[-1][0] <= curr_diff:
                q.pop()
            q.append((curr_diff, x))
            
        return int(max_val)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <deque>
#include <algorithm>
#include <climits>

class Solution {
public:
    int findMaxValueOfEquation(std::vector<std::vector<int>>& points, int k) {
        // Deque stores pair: {y - x, x}
        std::deque<std::pair<int, int>> dq;
        int maxVal = INT_MIN;

        for (const auto& pt : points) {
            int x = pt[0], y = pt[1];

            // 1. Evict expired points
            while (!dq.empty() && x - dq.front().second > k) {
                dq.pop_front();
            }

            // 2. Best candidate is at front
            if (!dq.empty()) {
                maxVal = std::max(maxVal, (x + y) + dq.front().first);
            }

            // 3. Maintain decreasing monotonic order
            int currDiff = y - x;
            while (!dq.empty() && dq.back().first <= currDiff) {
                dq.pop_back();
            }
            dq.push_back({currDiff, x});
        }

        return maxVal;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int findMaxValueOfEquation(int[][] points, int k) {
        // Deque stores int[]{y - x, x}
        Deque<int[]> dq = new ArrayDeque<>();
        int maxVal = Integer.MIN_VALUE;

        for (int[] pt : points) {
            int x = pt[0], y = pt[1];

            // 1. Evict elements outside window k
            while (!dq.isEmpty() && x - dq.peekFirst()[1] > k) {
                dq.pollFirst();
            }

            // 2. Front contains optimal (y_i - x_i)
            if (!dq.isEmpty()) {
                maxVal = Math.max(maxVal, (x + y) + dq.peekFirst()[0]);
            }

            // 3. Maintain monotonic decreasing order
            int currDiff = y - x;
            while (!dq.isEmpty() && dq.peekLast()[0] <= currDiff) {
                dq.pollLast();
            }
            dq.offerLast(new int[]{currDiff, x});
        }

        return maxVal;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Each point is pushed onto the deque once and popped at most once from either front or back.
- **Space Complexity:** $O(N)$ auxiliary space for the deque in the worst case where $(y_i - x_i)$ is strictly decreasing and all points fit within window $k$.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Formula Linear Factorization ($f(i, j) = g(j) + h(i)$) + Sliding Window Monotonic Deque.
- **Trap:** Forgetting that points can have negative coordinates: $x + y + (y_i - x_i)$ can be deeply negative. Always initialize `max_val` to negative infinity (`INT_MIN` / `float('-inf')`), never `0`.