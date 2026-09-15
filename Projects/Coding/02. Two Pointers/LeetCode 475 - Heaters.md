---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 475: Heaters"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 475: Heaters

Below is a complete, interview-ready explanation for **LeetCode 475 – Heaters**, structured exactly as requested.

---

## LeetCode 475 — Heaters

### Problem Statement

You are given two integer arrays:

* `houses`: positions of houses on a number line
* `heaters`: positions of heaters on the same number line

Each heater warms all houses within a **fixed radius R**.

**Goal:**  
Find the **minimum radius R** such that **every house** is within distance `R` of **at least one heater**.

---

### Key Observation

For each house, what matters is:

> **The distance to its nearest heater**

If we compute the minimum distance from every house to a heater, then:

> **The answer is the maximum of those minimum distances**

Why?

* A single radius `R` must cover **all houses**
* The house that is farthest from its closest heater determines the required radius

---

### Why Sorting Helps

Sort both arrays:

* `houses` in increasing order
* `heaters` in increasing order

This allows us to **scan both arrays in one pass**, maintaining the nearest heater for the current house.

---

### Two Pointer Technique (Core Idea)

Maintain a pointer `j` over heaters.

For each house `h`:

1. While moving to the **next heater makes it closer**, advance `j`
2. Compute distance from `h` to the closest heater seen so far
3. Track the **maximum** of these distances

This works because:

* Houses are processed in sorted order
* Heater distances change monotonically

---

### Algorithm Steps

1. Sort `houses` and `heaters`
2. Initialize:

   * `j = 0` (heater pointer)
   * `answer = 0`
3. For each house:

   * Move `j` while next heater is closer
   * Update `answer = max(answer, distance)`
4. Return `answer`

---

### Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        houses.sort()
        heaters.sort()

        j = 0
        radius = 0

        for house in houses:
            # Move heater pointer if next heater is closer
            while (
                j + 1 < len(heaters) and
                abs(heaters[j + 1] - house) <= abs(heaters[j] - house)
            ):
                j += 1

            # Update maximum radius needed
            radius = max(radius, abs(heaters[j] - house))

        return radius
```

---

### Worked Out Example

#### Input

```
houses  = [1, 2, 3, 4]
heaters = [1, 4]
```

#### Step 1: Sort (already sorted)

```
houses  = [1, 2, 3, 4]
heaters = [1, 4]
```

---

#### Step 2: Process Each House

| House | Closest Heater | Distance |
| --- | --- | --- |
| 1 | 1 | 0 |
| 2 | 1 | 1 |
| 3 | 4 | 1 |
| 4 | 4 | 0 |

---

#### Step 3: Final Answer

```
Maximum distance = 1
```

**Output**

```
1
```

---

### Time & Space Complexity

* **Time:** `O(n log n + m log m)`

  * Sorting dominates
  * Two-pointer scan is linear
* **Space:** `O(1)` extra space (ignoring sort cost)

---

### Key Takeaways (Interview Gold)

* This is a **nearest neighbor on a line** problem
* Convert “global coverage” into **per-house minimum distance**
* Sorting + two pointers avoids binary search per house
* Extremely clean and optimal solution

---

If you want:

* Binary search alternative
* Visualization walkthrough
* Edge-case discussion
* Conversion to sliding-window intuition

Let me know.