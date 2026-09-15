---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1386: Cinema Seat Allocation"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1386: Cinema Seat Allocation

Below is a **complete, structured explanation** of **LeetCode 1386 – Cinema Seat Allocation**, written in an interview-ready, systems-thinking style.

---

## 1. Problem Statement

You are given:

* An integer `n` — the number of rows in a cinema.
* A list `reservedSeats`, where each element `[row, seat]` indicates that `seat` in `row` is already reserved.

Each row has **10 seats**, labeled from **1 to 10**.

A family of **4 people** needs **4 consecutive seats** in the **same row**.

### Seating Rules

A family can sit in one of the following seat blocks:

* **Left block:** seats `[2, 3, 4, 5]`
* **Middle block:** seats `[4, 5, 6, 7]`
* **Right block:** seats `[6, 7, 8, 9]`

### Goal

Return the **maximum number of families** that can be seated.

---

## 2. Key Observations

### Observation 1: Independent Rows

Each row is independent.  
We can compute how many families fit **per row** and sum the result.

---

### Observation 2: Maximum per Row = 2

If a row has **no reserved seats**, we can seat:

* One family in `[2–5]`
* One family in `[6–9]`

So:

```
Max families per empty row = 2
```

---

### Observation 3: Only Seats 2–9 Matter

Seats **1 and 10** do not affect any valid family block.

We only care about:

```
Seats {2,3,4,5,6,7,8,9}
```

---

### Observation 4: Greedy Placement Order

For each row:

1. Try to place **left block** `[2–5]`
2. Try to place **right block** `[6–9]`
3. If neither fits, try **middle block** `[4–7]`

This order ensures **maximum utilization**.

---

## 3. Greedy Strategy (Core Idea)

### Step 1: Group Reserved Seats by Row

Use a map:

```
row → set of reserved seats
```

---

### Step 2: Initialize Result

Assume all rows are empty:

```
result = n * 2
```

---

### Step 3: Process Only Rows with Reservations

For each such row:

1. Subtract the assumed `2`
2. Check which blocks are free
3. Add back the actual number of families that fit

---

## 4. Python 3 Solution (with Typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def maxNumberOfFamilies(self, n: int, reservedSeats: List[List[int]]) -> int:
        # Group reserved seats by row
        rows = defaultdict(set)
        for r, c in reservedSeats:
            rows[r].add(c)

        # Assume all rows are empty initially
        result = n * 2

        for seats in rows.values():
            # Remove the optimistic assumption
            result -= 2

            left_free = all(seat not in seats for seat in (2, 3, 4, 5))
            right_free = all(seat not in seats for seat in (6, 7, 8, 9))
            middle_free = all(seat not in seats for seat in (4, 5, 6, 7))

            if left_free and right_free:
                result += 2
            elif left_free or right_free or middle_free:
                result += 1
            # else: no family can be seated in this row

        return result
```

---

## 5. Complete Worked Example (Step-by-Step)

### Input

```
n = 3
reservedSeats = [[1,2], [1,3], [1,8], [2,6], [3,1], [3,10]]
```

---

### Step 1: Initial Assumption

```
3 rows × 2 families = 6
result = 6
```

---

### Step 2: Group Reserved Seats

| Row | Reserved Seats |
| --- | --- |
| 1 | {2,3,8} |
| 2 | {6} |
| 3 | {1,10} |

---

### Step 3: Process Each Reserved Row

---

### Row 1: `{2,3,8}`

Check blocks:

* `[2–5]` ❌ (2,3 blocked)
* `[6–9]` ❌ (8 blocked)
* `[4–7]` ❌ (overlaps 8)

Families possible: **0**

Adjustment:

```
result = 6 - 2 + 0 = 4
```

---

### Row 2: `{6}`

Check blocks:

* `[2–5]` ✅
* `[6–9]` ❌ (6 blocked)
* `[4–7]` ❌ (6 blocked)

Families possible: **1**

Adjustment:

```
result = 4 - 2 + 1 = 3
```

---

### Row 3: `{1,10}`

Seats 1 and 10 do not matter.

Check blocks:

* `[2–5]` ✅
* `[6–9]` ✅

Families possible: **2**

Adjustment:

```
result = 3 - 2 + 2 = 3
```

---

## 6. Final Answer

```
Maximum number of families = 3
```

---

## 7. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | `O(n + r)` where `r = len(reservedSeats)` |
| Space | `O(r)` |
| Approach | Greedy + Bitmask-like seat checks |

---

## 8. Interview Takeaways

* Reduce problem scope early (ignore seats 1 & 10)
* Greedy works because seat blocks do not overlap arbitrarily
* Pre-subtract optimistic maximum and adjust per constraint
* Excellent example of **constraint-based greedy optimization**

If you want, I can also:

* Rewrite this using **bitmask optimization**
* Provide a **visual seat-block diagram**
* Compare greedy vs brute force reasoning