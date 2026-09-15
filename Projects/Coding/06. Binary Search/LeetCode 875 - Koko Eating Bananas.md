---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 875: Koko Eating Bananas"
tags:
  - leetcode
  - coding
  - binary-search
  - amazon
  - google
---

# LeetCode 875: Koko Eating Bananas

**Target Companies:** Amazon (Top #1 Classic), Google, Meta  
**Difficulty:** Medium  
**Topic:** Binary Search on Answer (Monotonic Predicate)

---

### Problem Statement

Koko loves to eat bananas. There are `n` piles of bananas, the $i$-th pile has `piles[i]` bananas. The guards have gone and will come back in `h` hours.

Koko can decide her bananas-per-hour eating speed of `k`. Each hour, she chooses some pile of bananas and eats `k` bananas from that pile. If the pile has less than `k` bananas, she eats all of them instead and will not eat any more bananas during this hour.

Return the **minimum integer `k`** such that she can eat all the bananas within `h` hours.

---

### Input & Output Formats & Constraints

- **Input:** `piles: List[int]`, `h: int`
- **Output:** `int` (minimum speed $k$)
- **Constraints:**
  - $1 \le \text{piles.length} \le 10^4$
  - $\text{piles.length} \le h \le 10^9$
  - $1 \le \text{piles}[i] \le 10^9$

---

### Key Idea & Intuition

- **Monotonicity:**
  - Let $f(k)$ be the hours needed to eat all bananas at speed $k$.
  - As speed $k$ increases, hours $f(k)$ monotonically decreases!
  - Search range for $k$: $[1, \max(\text{piles})]$.
  - If at speed $k$, hours $\le h$, speed $k$ is feasible; try smaller speeds (`right = mid`).
  - Otherwise, speed $k$ is too slow; must increase speed (`left = mid + 1`).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
import math

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        
        while left < right:
            mid = (left + right) // 2
            hours_needed = sum(math.ceil(p / mid) for p in piles)
            
            if hours_needed <= h:
                right = mid
            else:
                left = mid + 1
                
        return left
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int minEatingSpeed(std::vector<int>& piles, int h) {
        int left = 1, right = *std::max_element(piles.begin(), piles.end());

        while (left < right) {
            int mid = left + (right - left) / 2;
            long long hoursNeeded = 0;
            for (int p : piles) {
                hoursNeeded += (p + mid - 1) / mid;
            }

            if (hoursNeeded <= h) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int left = 1, right = 0;
        for (int p : piles) right = Math.max(right, p);

        while (left < right) {
            int mid = left + (right - left) / 2;
            long hoursNeeded = 0;
            for (int p : piles) {
                hoursNeeded += (p + mid - 1) / mid;
            }

            if (hoursNeeded <= h) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \log(\max(\text{piles})))$ — Binary search over speed range $[1, 10^9]$.
- **Space Complexity:** $O(1)$ auxiliary space.
