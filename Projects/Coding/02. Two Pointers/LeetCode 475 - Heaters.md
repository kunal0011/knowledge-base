---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 475: Heaters"
tags:
  - leetcode
  - coding
  - two-pointers
  - binary-search
  - sorting
  - google
  - amazon
---

# LeetCode 475: Heaters

**Target Companies:** Google, Amazon  
**Difficulty:** Medium  
**Topic:** Two Pointers / Greedy Proximity Search  

---

### Problem Statement

Winter is coming! During the contest, your first job is to design a standard heater with a fixed warm radius to warm all the houses.

Every house can be warmed as long as the house is within the heater's warm radius range.

Given the positions of `houses` and `heaters` on a horizontal line, return the **minimum radius** necessary to warm all houses.

---

### Input & Output Formats & Constraints

- **Input:** `houses: List[int]`, `heaters: List[int]`
- **Output:** `int` (minimum heater radius)
- **Constraints:**
  - $1 \le \text{houses.length}, \text{heaters.length} \le 3 \times 10^4$
  - $1 \le \text{houses}[i], \text{heaters}[i] \le 10^9$

---

### Key Idea & Intuition

Every single house $h$ must be covered by *some* heater. To minimize the needed global radius, house $h$ should naturally be served by the heater nearest to it:
$$\text{dist}(h) = \min_{ht \in \text{heaters}} |h - ht|$$

The global radius must be large enough to cover the most isolated house, so:
$$R = \max_{h \in \text{houses}} \left( \min_{ht \in \text{heaters}} |h - ht| \right)$$

If we sort both `houses` and `heaters`:
- As we iterate through sorted houses, the nearest heater never moves backwards.
- We maintain a heater pointer $j$. For each house, we greedily advance $j$ as long as the next heater `heaters[j + 1]` is closer (or equidistant) to the current house than `heaters[j]`.
- This converts the nearest-neighbor search into an $O(N + M)$ two-pointer scan after sorting.

---

### Solution Approach (Step-by-Step)

1. Sort `houses` and `heaters` in ascending order.
2. Initialize `j = 0` (pointer into `heaters`) and `radius = 0`.
3. For each `house` in `houses`:
   - While `j + 1 < len(heaters)` and `abs(heaters[j + 1] - house) <= abs(heaters[j] - house)`:
     - Advance `j += 1`.
   - Update `radius = max(radius, abs(heaters[j] - house))`.
4. Return `radius`.

---

### Visual Algorithm Walkthrough

```
houses = [1, 2, 3, 4], heaters = [1, 4]
Sorted:
  houses:  1   2   3   4
  heaters: 1           4
           j=0         j=1

House 1:
  Compare abs(heaters[0] - 1) = |1 - 1| = 0
          abs(heaters[1] - 1) = |4 - 1| = 3
  Next heater is farther (3 > 0), keep j = 0.
  Distance = 0. radius = max(0, 0) = 0.

House 2:
  Compare abs(heaters[0] - 2) = |1 - 2| = 1
          abs(heaters[1] - 2) = |4 - 2| = 2
  Next heater is farther (2 > 1), keep j = 0.
  Distance = 1. radius = max(0, 1) = 1.

House 3:
  Compare abs(heaters[0] - 3) = |1 - 3| = 2
          abs(heaters[1] - 3) = |4 - 3| = 1
  Next heater is CLOSER (1 <= 2)! Advance j -> 1.
  Distance = 1. radius = max(1, 1) = 1.

House 4:
  j is already at 1 (last heater).
  Distance = |4 - 4| = 0.
  radius = max(1, 0) = 1.

All houses covered with minimum radius = 1!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Symmetric Setup
- **Input:** `houses = [1, 2, 3]`, `heaters = [2]`
- **Trace:**
  - House 1: distance $|1 - 2| = 1$.
  - House 2: distance $|2 - 2| = 0$.
  - House 3: distance $|3 - 2| = 1$.
- **Output:** `1`

#### Example 2: Out of Bounds Extreme
- **Input:** `houses = [1, 5]`, `heaters = [2]`
- **Trace:**
  - House 1: distance $|1 - 2| = 1$.
  - House 5: distance $|5 - 2| = 3$.
- **Output:** `3`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        houses.sort()
        heaters.sort()
        
        j = 0
        radius = 0
        m = len(heaters)
        
        for house in houses:
            # Advance heater pointer while next heater is closer or equidistant
            while j + 1 < m and abs(heaters[j + 1] - house) <= abs(heaters[j] - house):
                j += 1
            radius = max(radius, abs(heaters[j] - house))
            
        return radius
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>
#include <cmath>

class Solution {
public:
    int findRadius(std::vector<int>& houses, std::vector<int>& heaters) {
        std::sort(houses.begin(), houses.end());
        std::sort(heaters.begin(), heaters.end());
        
        int j = 0;
        int radius = 0;
        int m = heaters.size();
        
        for (int house : houses) {
            while (j + 1 < m && std::abs(heaters[j + 1] - house) <= std::abs(heaters[j] - house)) {
                j++;
            }
            radius = std::max(radius, std::abs(heaters[j] - house));
        }
        
        return radius;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(houses);
        Arrays.sort(heaters);
        
        int j = 0;
        int radius = 0;
        int m = heaters.length;
        
        for (int house : houses) {
            while (j + 1 < m && Math.abs(heaters[j + 1] - house) <= Math.abs(heaters[j] - house)) {
                j++;
            }
            radius = Math.max(radius, Math.abs(heaters[j] - house));
        }
        
        return radius;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \log N + M \log M)$ where $N = \text{len(houses)}$ and $M = \text{len(heaters)}$. Sorting dominates. The two-pointer traversal takes $O(N + M)$ because $j$ only advances forward across all iterations.
- **Space Complexity:** $O(1)$ auxiliary space beyond the in-place sorting routines.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Nearest-Neighbor on 1D Coordinate Line via Dual-Sorting & Monotonic Advancing Pointer.
- **Trap:** Using `<` instead of `<=` when comparing next heater distance. Equidistant heaters should be advanced so subsequent houses further to the right don't get stuck comparing against an older left heater.