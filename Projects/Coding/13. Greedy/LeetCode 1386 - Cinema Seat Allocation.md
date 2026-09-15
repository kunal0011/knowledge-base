---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1386: Cinema Seat Allocation"
tags:
  - leetcode
  - coding
  - greedy
  - bit-manipulation
  - hash-table
  - amazon
  - google
---

# LeetCode 1386: Cinema Seat Allocation

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Bit Manipulation / Hash Table  

---

### Problem Statement

A cinema has $n$ rows of seats, numbered from $1$ to $n$ and there are $10$ seats in each row, labelled from $1$ to $10$ as shown below:

```
[1] [2 3 4 5] [6 7 8 9] [10]
```

Given the array `reservedSeats` containing the numbers of seats already reserved, for each $i$ from $1$ to $\text{reservedSeats.length}$, `reservedSeats[i] = [row_i, col_i]` means that the seat in row `row_i` and column `col_i` is already reserved.

Return the **maximum number of 4-person families** you can seat on the cinema. Each 4-person family needs to be seated in **consecutive seats in the same row**.

There are only three valid configurations for a 4-person family:
1. Seats **2, 3, 4, 5** (Left block across aisle)
2. Seats **6, 7, 8, 9** (Right block across aisle)
3. Seats **4, 5, 6, 7** (Middle block crossing the center)

---

### Input & Output Formats & Constraints

- **Input:**
  - `n`: `int` — total number of rows ($1 \le n \le 10^9$).
  - `reservedSeats`: `List[List[int]]` / `vector<vector<int>>` / `int[][]` ($1 \le \text{reservedSeats.length} \le \min(10 \cdot n, 10^4)$).
- **Output:**
  - `int` — the total maximum number of four-person families that can be accommodated.
- **Constraints:**
  - $1 \le n \le 10^9$
  - $1 \le \text{reservedSeats.length} \le 10^4$
  - `reservedSeats[i].length == 2`
  - $1 \le \text{row}_i \le n$
  - $1 \le \text{col}_i \le 10$
  - All `[row_i, col_i]` pairs are distinct.

---

### Key Idea & Intuition

Notice that seats **1** and **10** are outer edge seats and can never be part of any 4-person family group. We only care about seats **2 through 9**.

For any row, there are at most three candidate family blocks:
- **Left block:** `[2, 3, 4, 5]`
- **Right block:** `[6, 7, 8, 9]`
- **Middle block:** `[4, 5, 6, 7]`

Notice the overlap:
- The Middle block overlaps with the Left block on seats `[4, 5]`.
- The Middle block overlaps with the Right block on seats `[6, 7]`.
- The Left and Right blocks **do not overlap** with each other.

Therefore, for each row:
- The maximum number of families in a single row is at most **2** (one on Left, one on Right).
- If a row has **no reservations** in seats 2..9, it can hold **2 families**.
- If a row has reservations:
  - Can we place a family on Left (`2, 3, 4, 5` all free)? If yes, count $+1$.
  - Can we place a family on Right (`6, 7, 8, 9` all free)? If yes, count $+1$.
  - If we could NOT place both (i.e. neither or only one), check if Middle (`4, 5, 6, 7` all free) is valid and whether Left/Right wasn't already placed:
    - Specifically, if both Left and Right are blocked, but Middle is free, we can place **1** family in the Middle!

#### The $N = 10^9$ Scale:
Because $n \le 10^9$, we cannot iterate through all $n$ rows.
Instead, we only store rows that appear in `reservedSeats` inside a hash map using bitmasks.
If there are $U$ unique rows with reservations:
- Each unreserved row accommodates $2$ families: $(n - U) \times 2$.
- We calculate the families accommodated in the $U$ reserved rows individually.

---

### Solution Approach (Step-by-Step)

1. Create a hash map `reserved_map` mapping `row -> bitmask` of occupied seats (using bits 2 through 9).
2. For each `[row, col]` in `reservedSeats`:
   - If $2 \le col \le 9$:
     - `reserved_map[row] |= (1 << col)`
3. Initialize `ans = (n - len(reserved_map)) * 2`.
4. Define bitmask patterns:
   - `LEFT_MASK = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)`
   - `RIGHT_MASK = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)`
   - `MID_MASK = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)`
5. For each `mask` in `reserved_map.values()`:
   - `can_left = (mask & LEFT_MASK) == 0`
   - `can_right = (mask & RIGHT_MASK) == 0`
   - `can_mid = (mask & MID_MASK) == 0`
   - If `can_left and can_right`:
     - `ans += 2`
   - Else if `can_left or can_right or can_mid`:
     - `ans += 1`
6. Return `ans`.

---

### Visual Algorithm Walkthrough

Row Seat Layout:
```
Col:  1    2   3   4   5    6   7   8   9    10
     [ ]  [  Left Block  ] [  Right Block  ] [ ]
              [  Middle Block  ]
```

Case Analysis for a Row:
```
Case 1: Left free & Right free
  [2,3,4,5] and [6,7,8,9] both open
  -> Accommodates 2 families.

Case 2: Left occupied, Right occupied, Middle open (e.g. seats 2 and 9 reserved)
  Seat 2 reserved: Left blocked.
  Seat 9 reserved: Right blocked.
  Seats 4, 5, 6, 7 are free!
  -> Accommodates 1 family in the Middle!

Case 3: Only Left open, or only Right open
  -> Accommodates 1 family.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `n = 3`, `reservedSeats = [[1,2],[1,3],[1,8],[2,6],[3,1],[3,10]]`
- **Tracing:**
  - Row 1: reserved at 2, 3, 8. Left and Right blocked. Middle (4,5,6,7) free! -> 1 family.
  - Row 2: reserved at 6. Left (2,3,4,5) free! Right and Mid blocked. -> 1 family.
  - Row 3: reserved at 1, 10 (irrelevant!). Both Left and Right free -> 2 families.
  - Total = $1 + 1 + 2 = 4$.
- **Output:** `4`

#### Example 2:
- **Input:** `n = 2`, `reservedSeats = [[2,1],[1,8],[2,6]]`
- **Output:** `2`

#### Example 3 ($n = 4$, no reservations):
- **Input:** `n = 4`, `reservedSeats = []`
- **Tracing:** $4 \times 2 = 8$.
- **Output:** `8`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import defaultdict
from typing import List

class Solution:
    def maxNumberOfFamilies(self, n: int, reservedSeats: List[List[int]]) -> int:
        reserved_map = defaultdict(int)
        
        # Bitmask for relevant seats 2 through 9
        for r, c in reservedSeats:
            if 2 <= c <= 9:
                reserved_map[r] |= (1 << c)
                
        # Unreserved rows accommodate 2 families each
        total_families = (n - len(reserved_map)) * 2
        
        LEFT_MASK = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        RIGHT_MASK = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
        MID_MASK = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        
        for mask in reserved_map.values():
            can_left = (mask & LEFT_MASK) == 0
            can_right = (mask & RIGHT_MASK) == 0
            can_mid = (mask & MID_MASK) == 0
            
            if can_left and can_right:
                total_families += 2
            elif can_left or can_right or can_mid:
                total_families += 1
                
        return total_families
```

#### C++17
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    int maxNumberOfFamilies(int n, const std::vector<std::vector<int>>& reservedSeats) {
        std::unordered_map<int, int> reserved_map;
        
        for (const auto& seat : reservedSeats) {
            int r = seat[0];
            int c = seat[1];
            if (c >= 2 && c <= 9) {
                reserved_map[r] |= (1 << c);
            }
        }
        
        int total_families = (n - static_cast<int>(reserved_map.size())) * 2;
        
        const int LEFT_MASK = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);
        const int RIGHT_MASK = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9);
        const int MID_MASK = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7);
        
        for (const auto& [r, mask] : reserved_map) {
            bool can_left = (mask & LEFT_MASK) == 0;
            bool can_right = (mask & RIGHT_MASK) == 0;
            bool can_mid = (mask & MID_MASK) == 0;
            
            if (can_left && can_right) {
                total_families += 2;
            } else if (can_left || can_right || can_mid) {
                total_families += 1;
            }
        }
        
        return total_families;
    }
};
```

#### Java 17
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int maxNumberOfFamilies(int n, int[][] reservedSeats) {
        Map<Integer, Integer> reservedMap = new HashMap<>();
        
        for (int[] seat : reservedSeats) {
            int r = seat[0];
            int c = seat[1];
            if (c >= 2 && c <= 9) {
                reservedMap.put(r, reservedMap.getOrDefault(r, 0) | (1 << c));
            }
        }
        
        int totalFamilies = (n - reservedMap.size()) * 2;
        
        int leftMask = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);
        int rightMask = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9);
        int midMask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7);
        
        for (int mask : reservedMap.values()) {
            boolean canLeft = (mask & leftMask) == 0;
            boolean canRight = (mask & rightMask) == 0;
            boolean canMid = (mask & midMask) == 0;
            
            if (canLeft && canRight) {
                totalFamilies += 2;
            } else if (canLeft || canRight || canMid) {
                totalFamilies += 1;
            }
        }
        
        return totalFamilies;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M)$ where $M = \text{reservedSeats.length} \le 10^4$
  - Constructing bitmasks takes $\mathcal{O}(M)$.
  - Iterating over the unique reserved rows (at most $M$) and performing bitwise AND operations takes $\mathcal{O}(M)$.
  - Independent of $n$ (which can be up to $10^9$).
- **Space Complexity:** $\mathcal{O}(M)$ auxiliary space
  - The hash map stores at most $M$ unique row entries.

---

### Takeaway Pattern & Interview Traps

- **Sparse Row Compression:** When $N$ is astronomical ($10^9$) but the number of active points $M$ is small ($10^4$), never allocate an array of size $N$. Only record active rows in a hash map, and mathematically compute default values for untouched rows ($(N - U) \times 2$).
- **Bitmask Matching:** Using bitwise masks `mask & MASK == 0` evaluates 4 seat statuses simultaneously in a single CPU cycle.