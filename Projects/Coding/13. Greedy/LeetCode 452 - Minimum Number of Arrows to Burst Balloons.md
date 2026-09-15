---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 452: Minimum Number of Arrows to Burst Balloons"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 452: Minimum Number of Arrows to Burst Balloons

Below is a **complete, structured greedy explanation** for **LeetCode 452 – Minimum Number of Arrows to Burst Balloons**, aligned with interview expectations.

---

## 1. Problem Statement

You are given an array `points`, where  
`points[i] = [x_start, x_end]` represents a **balloon** stretched horizontally from `x_start` to `x_end` (inclusive).

An arrow shot vertically at position `x` will burst **all balloons** where:

```
x_start ≤ x ≤ x_end
```

Your task is to return the **minimum number of arrows** required to burst all balloons.

---

### Example

```text
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
```

---

## 2. Key Observation (Core Greedy Insight)

This problem is a **classic interval overlap greedy problem**.

### Crucial insight:

If multiple balloons **overlap**, a **single arrow** can burst all of them **if placed within their common overlapping region**.

Therefore:

* We want to **maximize the number of balloons burst by one arrow**
* Which means **shoot arrows at optimal positions**

---

## 3. Greedy Strategy

### Step 1: Sort balloons by their **ending coordinate**

```
Sort by x_end (ascending)
```

Why?

* Shooting the arrow at the **earliest possible end** gives maximum chance to cover upcoming balloons.

---

### Step 2: Traverse and Count Arrows

Maintain:

* `arrows = 1`
* `current_arrow_pos = end of first balloon`

For each next balloon:

* If its `start > current_arrow_pos`

  * No overlap → need **new arrow**
  * Update `current_arrow_pos = balloon.end`
* Else

  * Overlaps → same arrow works

---

## 4. Why This Greedy Works

This is equivalent to:

> **Interval Scheduling with minimum resources**

Choosing the earliest end ensures:

* We never miss a future overlap opportunity
* Any other arrow placement would be suboptimal or equivalent

This is a **provably optimal greedy choice**.

---

## 5. Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0

        # Step 1: sort by end coordinate
        points.sort(key=lambda x: x[1])

        arrows = 1
        current_arrow_pos = points[0][1]

        # Step 2: iterate through intervals
        for start, end in points[1:]:
            if start > current_arrow_pos:
                arrows += 1
                current_arrow_pos = end

        return arrows
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
points = [[10,16],[2,8],[1,6],[7,12]]
```

---

### Step 1: Sort by end

```
Sorted points:
[1,6], [2,8], [7,12], [10,16]
```

---

### Step 2: Initialize

```
arrows = 1
current_arrow_pos = 6
```

---

### Step 3: Process Each Balloon

#### Balloon: [2,8]

```
start = 2 ≤ current_arrow_pos = 6
→ overlaps
→ use same arrow
```

State:

```
arrows = 1
current_arrow_pos = 6
```

---

#### Balloon: [7,12]

```
start = 7 > current_arrow_pos = 6
→ no overlap
→ need new arrow
```

Update:

```
arrows = 2
current_arrow_pos = 12
```

---

#### Balloon: [10,16]

```
start = 10 ≤ current_arrow_pos = 12
→ overlaps
→ same arrow
```

---

### Final Result

```
Minimum arrows required = 2
```

---

## 7. Greedy Pattern Recognition

This problem belongs to the **“Interval Greedy”** family:

| Problem | Greedy Rule |
| --- | --- |
| 452. Min Arrows | Sort by end |
| 435. Non-overlapping Intervals | Sort by end |
| 56. Merge Intervals | Sort by start |
| 253. Meeting Rooms II | Min heap on end |

---

## 8. Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | `O(n log n)` (sorting) |
| Space | `O(1)` (in-place) |

---

## 9. Interview Tip

If stuck:

> Ask yourself: *“Where should I place the arrow so that it bursts the most balloons?”*

The answer always points to:

> **the earliest finishing balloon**

---

If you want, I can also:

* Draw the **interval overlap visualization**
* Compare this with **meeting rooms**
* Show **counterexample where sorting by start fails**
* Provide a **formal greedy proof**

Just tell me.