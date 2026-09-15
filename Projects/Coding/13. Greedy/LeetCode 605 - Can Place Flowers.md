---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 605: Can Place Flowers"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - amazon
  - google
---

# LeetCode 605: Can Place Flowers

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Easy  
**Topic:** Greedy / Array / Boundary Conditions  

---

### Problem Statement

You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted in **adjacent** plots.

Given an integer array `flowerbed` containing $0$'s and $1$'s, where $0$ means empty and $1$ means not empty, and an integer $n$, return `true` if $n$ new flowers can be planted in the `flowerbed` without violating the no-adjacent-flowers rule and `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:**
  - `flowerbed`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{flowerbed.length} \le 2 \times 10^4$).
  - `n`: `int` ($0 \le n \le \text{flowerbed.length}$).
- **Output:**
  - `bool` — `true` if at least $n$ flowers can be planted, else `false`.
- **Constraints:**
  - $1 \le \text{flowerbed.length} \le 2 \times 10^4$
  - `flowerbed[i]` is $0$ or $1$.
  - There are no two adjacent flowers in `flowerbed` initially.
  - $0 \le n \le \text{flowerbed.length}$

---

### Key Idea & Intuition

To maximize the total number of flowers planted, we should plant a flower **as early as possible** whenever we encounter a valid plot.

#### The Greedy Planting Invariant:
A flower can be legally planted at index $i$ if and only if all three conditions are met:
1. Current plot is empty: `flowerbed[i] == 0`.
2. Left neighbor is empty or $i$ is the left boundary: `i == 0 or flowerbed[i - 1] == 0`.
3. Right neighbor is empty or $i$ is the right boundary: `i == len - 1 or flowerbed[i + 1] == 0`.

Whenever these conditions hold:
- Planting at index $i$ is always optimal or tied with planting at index $i + 1$:
  - Planting at $i$ blocks $i + 1$, leaving $i + 2$ open for another flower.
  - Skipping $i$ to plant at $i + 1$ blocks both $i$ and $i + 2$, which is strictly worse or equivalent.
- Therefore, we greedily set `flowerbed[i] = 1` and decrement $n \mathrel{-}= 1$.
- If $n \le 0$ at any moment, we can return `True` early.

---

### Solution Approach (Step-by-Step)

1. If $n == 0$, return `True`.
2. Loop $i$ from $0$ to $\text{len}(flowerbed) - 1$:
   - If `flowerbed[i] == 0`:
     - Check if left neighbor is clear: `left_empty = (i == 0 or flowerbed[i - 1] == 0)`
     - Check if right neighbor is clear: `right_empty = (i == len(flowerbed) - 1 or flowerbed[i + 1] == 0)`
     - If `left_empty and right_empty`:
       - `flowerbed[i] = 1`
       - `n -= 1`
       - If `n == 0`: return `True`
3. Return `n <= 0`.

---

### Visual Algorithm Walkthrough

For `flowerbed = [1, 0, 0, 0, 1]`, `n = 1`:

```
Indices:    0   1   2   3   4
Plots:     [1,  0,  0,  0,  1]

i = 0: plot is 1 (occupied) -> continue
i = 1: plot is 0. Left is 1 (occupied) -> cannot plant
i = 2: plot is 0.
       Left is plot 1 (value 0, empty!)
       Right is plot 3 (value 0, empty!)
       -> VALID SPOT! Plant flower!
       flowerbed[2] becomes 1.
       n becomes 1 - 1 = 0.
       n == 0 -> Return True immediately!
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `flowerbed = [1, 0, 0, 0, 1]`, `n = 1`
- **Output:** `true`

#### Example 2:
- **Input:** `flowerbed = [1, 0, 0, 0, 1]`, `n = 2`
- **Tracing:** Only plot 2 is valid. Total plantable = 1 < 2.
- **Output:** `false`

#### Example 3 ($n = 0$):
- **Input:** `flowerbed = [1, 1, 1]`, `n = 0`
- **Output:** `true`

#### Example 4 (Single Plot):
- **Input:** `flowerbed = [0]`, `n = 1`
- **Tracing:** Plot 0 has no neighbors; plant flower $\rightarrow n = 0$.
- **Output:** `true`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        if n == 0:
            return True
            
        m = len(flowerbed)
        for i in range(m):
            if flowerbed[i] == 0:
                left_empty = (i == 0 or flowerbed[i - 1] == 0)
                right_empty = (i == m - 1 or flowerbed[i + 1] == 0)
                
                if left_empty and right_empty:
                    flowerbed[i] = 1
                    n -= 1
                    if n == 0:
                        return True
                        
        return n <= 0
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    bool canPlaceFlowers(std::vector<int>& flowerbed, int n) {
        if (n == 0) return true;
        
        int m = static_cast<int>(flowerbed.size());
        for (int i = 0; i < m; ++i) {
            if (flowerbed[i] == 0) {
                bool left_empty = (i == 0 || flowerbed[i - 1] == 0);
                bool right_empty = (i == m - 1 || flowerbed[i + 1] == 0);
                
                if (left_empty && right_empty) {
                    flowerbed[i] = 1;
                    n--;
                    if (n == 0) {
                        return true;
                    }
                }
            }
        }
        
        return n <= 0;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        if (n == 0) {
            return true;
        }
        
        int m = flowerbed.length;
        for (int i = 0; i < m; i++) {
            if (flowerbed[i] == 0) {
                boolean leftEmpty = (i == 0 || flowerbed[i - 1] == 0);
                boolean rightEmpty = (i == m - 1 || flowerbed[i + 1] == 0);
                
                if (leftEmpty && rightEmpty) {
                    flowerbed[i] = 1;
                    n--;
                    if (n == 0) {
                        return true;
                    }
                }
            }
        }
        
        return n <= 0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m)$ where $m = \text{flowerbed.length}$
  - A single pass through the array. Early returns as soon as $n$ reaches 0.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The array is modified in-place without additional data structures.

---

### Takeaway Pattern & Interview Traps

- **Boundary Guarding Without Padding:** Check boundary conditions dynamically: `(i == 0 || flowerbed[i - 1] == 0)` and `(i == m - 1 || flowerbed[i + 1] == 0)` to avoid allocating a padded copy array $[0] + \text{flowerbed} + [0]$.
- **$n = 0$ Corner Case:** Always handle $n = 0$ immediately, returning `true` even if the flowerbed is packed with 1s.