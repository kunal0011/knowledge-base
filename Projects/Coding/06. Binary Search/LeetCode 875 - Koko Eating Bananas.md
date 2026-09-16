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
  - arrays
  - amazon
  - google
  - meta
---

# LeetCode 875: Koko Eating Bananas

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Binary Search on Answer / Monotonic Feasibility Predicate

---

### Problem Statement

Koko loves to eat bananas. There are `n` piles of bananas, the $i$-th pile has `piles[i]` bananas. The guards have gone and will come back in `h` hours.

Koko can decide her bananas-per-hour eating speed of `k`. Each hour, she chooses some pile of bananas and eats `k` bananas from that pile. If the pile has less than `k` bananas, she eats all of them instead and will not eat any more bananas during this hour.

Return *the minimum integer `k` such that she can eat all the bananas within `h` hours*.

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

#### 1. Binary Search on Answer (Monotonic Predicate):
Let $T(k)$ be the total time required to finish all piles at speed $k$:
$$T(k) = \sum_{p \in \text{piles}} \left\lceil \frac{p}{k} \right\rceil = \sum_{p \in \text{piles}} \left\lfloor \frac{p + k - 1}{k} \right\rfloor$$
- Notice that as eating speed $k$ increases, total hours $T(k)$ strictly decreases or remains constant:
  $$k_1 < k_2 \implies T(k_1) \ge T(k_2)$$
- Therefore, the feasibility function $P(k) = (T(k) \le h)$ is **monotonic**:
  $$\text{False, False, ..., False, True, True, ..., True}$$
- We want to find the **first `True`** (the minimum speed $k$).

#### 2. Search Space Boundaries:
- Minimum possible speed: $left = 1$ (she must eat at least 1 banana per hour).
- Maximum necessary speed: $right = \max(\text{piles})$ (at speed $\max(\text{piles})$, each pile takes exactly 1 hour, so $T(k) = n \le h$).
- Thus, the search space $[1, \max(\text{piles})]$ is bounded and can be binary searched in $\mathcal{O}(\log(\max(\text{piles})))$ iterations.

---

### Solution Approach (Step-by-Step)

1. **Initialize Search Range:**
   - `left = 1`, `right = max(piles)`.
2. **Binary Search Loop (`while left < right`):**
   - `mid = left + (right - left) // 2`.
   - Calculate total hours needed at speed `mid`:
     $$\text{hours} = \sum_{p \in \text{piles}} \frac{p + \text{mid} - 1}{\text{mid}}$$
   - If $\text{hours} \le h$:
     - Speed `mid` is feasible. Try to find a smaller feasible speed by searching left: `right = mid`.
   - Else:
     - Speed `mid` is too slow ($\text{hours} > h$). We must increase speed: `left = mid + 1`.
3. **Return Answer:**
   - When `left == right`, `left` is the minimum integer speed. Return `left`.

---

### Visual Algorithm Walkthrough

#### Example: `piles = [3, 6, 7, 11]`, `h = 8`
Range: $left = 1, right = \max(piles) = 11$.

```
Iteration 1:
  left = 1, right = 11 -> mid = 6
  Hours for piles [3, 6, 7, 11] at speed 6:
    ceil(3/6)  = 1
    ceil(6/6)  = 1
    ceil(7/6)  = 2
    ceil(11/6) = 2
    Total = 1 + 1 + 2 + 2 = 6 hours <= 8 -> Feasible!
  right = mid = 6

Iteration 2:
  left = 1, right = 6 -> mid = 3
  Hours at speed 3:
    ceil(3/3)  = 1
    ceil(6/3)  = 2
    ceil(7/3)  = 3
    ceil(11/3) = 4
    Total = 1 + 2 + 3 + 4 = 10 hours > 8 -> Too slow!
  left = mid + 1 = 4

Iteration 3:
  left = 4, right = 6 -> mid = 5
  Hours at speed 5:
    ceil(3/5)  = 1
    ceil(6/5)  = 2
    ceil(7/5)  = 2
    ceil(11/5) = 3
    Total = 1 + 2 + 2 + 3 = 8 hours <= 8 -> Feasible!
  right = mid = 5

Iteration 4:
  left = 4, right = 5 -> mid = 4
  Hours at speed 4:
    ceil(3/4)  = 1
    ceil(6/4)  = 2
    ceil(7/4)  = 2
    ceil(11/4) = 3
    Total = 8 hours <= 8 -> Feasible!
  right = mid = 4

left == right == 4 -> Terminate.
Result: k = 4.
```

---

### Solved Examples with Multiple Inputs

| `piles` | `h` | Speed Range $[1, \max]$ | Binary Search Transitions | Output $k$ |
| :--- | :--- | :--- | :--- | :--- |
| `[3, 6, 7, 11]` | 8 | $[1, 11]$ | $6 \to 3 \to 5 \to 4$ | `4` |
| `[30, 11, 23, 4, 20]` | 5 | $[1, 30]$ | $h == n \implies$ must eat each pile in 1 hr $\implies k = \max(piles)$ | `30` |
| `[30, 11, 23, 4, 20]` | 6 | $[1, 30]$ | Evaluates speeds to find minimum fitting in 6 hours | `23` |
| `[1000000000]` | 2 | $[1, 10^9]$ | Two equal splits of $10^9$ | `500000000` |

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
            mid = left + (right - left) // 2
            # Total hours at speed mid
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
        int left = 1;
        int right = *std::max_element(piles.begin(), piles.end());

        while (left < right) {
            int mid = left + (right - left) / 2;
            long long hoursNeeded = 0;
            for (int p : piles) {
                // Integer ceil division: (p + mid - 1) / mid
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
        for (int p : piles) {
            right = Math.max(right, p);
        }

        while (left < right) {
            int mid = left + (right - left) / 2;
            long hoursNeeded = 0;
            for (int p : piles) {
                // Integer ceil division
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

- **Time Complexity:** $\mathcal{O}(n \log(\max(\text{piles})))$, where $n$ is the number of piles. The binary search does $\mathcal{O}(\log(\max(\text{piles})))$ iterations. In each iteration, we do a linear pass of length $n$ summing the ceil divisions. For $\max(\text{piles}) \le 10^9$, $\log_2(10^9) \approx 30$ iterations, so $30 \times 10^4 = 3 \times 10^5$ operations (well under 5 ms).
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space.

---

### Takeaway Pattern & Interview Traps

1. **Integer Overflow in Hours Sum:**
   - When testing small speeds like $mid = 1$ with $10^4$ piles of size $10^9$, the total hours needed can be $10^4 \times 10^9 = 10^{13}$, which exceeds the 32-bit signed integer limit ($2 \times 10^9$). Always use 64-bit integers (`long long` in C++, `long` in Java) for `hoursNeeded`.
2. **Ceil Division without Floating Point:**
   - Use the integer formula `(p + mid - 1) / mid` to avoid slow and precision-prone floating-point `ceil((double)p / mid)`.
