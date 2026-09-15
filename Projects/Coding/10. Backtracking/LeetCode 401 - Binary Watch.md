---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 401: Binary Watch"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 401: Binary Watch

## Problem Statement (LeetCode 401)

A **binary watch** has:

* **4 LEDs** for hours (0–11)
* **6 LEDs** for minutes (0–59)

Each LED represents a binary value.

You are given an integer `turnedOn`, representing the **number of LEDs that are currently ON**.

**Task**  
Return **all possible times** the watch could represent.

### Output format

* Time is in `"H:MM"` format
* Hour has **no leading zero**
* Minute has **two digits** (leading zero allowed)

---

### Example

```text
Input: turnedOn = 1
Output: ["0:01","0:02","0:04","0:08","0:16","0:32","1:00","2:00","4:00","8:00"]
```

---

## 2. Key Observations (Very Important)

1. **Hour LEDs = 4 → values:** `[8, 4, 2, 1]`
2. **Minute LEDs = 6 → values:** `[32, 16, 8, 4, 2, 1]`
3. Total LEDs = `10`
4. We must pick exactly `turnedOn` LEDs **from these 10**
5. Valid constraints:

   * `hour < 12`
   * `minute < 60`
6. Order does **not** matter — combinations, not permutations.

---

## 3. All Possible Solution Approaches

### Approach 1: Brute Force (Bit Counting) ✅ Simplest

#### Idea

* Iterate over **all possible times**
* Count number of `1`s in binary representation of hour + minute

#### Complexity

* Time: `12 × 60 = 720` → constant
* Space: `O(1)` (output excluded)

#### Python 3 Code

```python
from typing import List

class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        result: List[str] = []

        for hour in range(12):
            for minute in range(60):
                if bin(hour).count("1") + bin(minute).count("1") == turnedOn:
                    result.append(f"{hour}:{minute:02d}")

        return result
```

#### Example Walkthrough

`turnedOn = 1`

* `hour = 1 (0001)` → 1 LED
* `minute = 0` → 0 LED  
  → `"1:00"` valid

---

### Approach 2: Precompute Combinations (Split LEDs)

#### Idea

* Choose `i` LEDs for hour
* Choose `turnedOn - i` LEDs for minute
* Combine valid values

#### Complexity

* Time: `C(4, i) × C(6, turnedOn - i)`
* Efficient due to small limits

#### Python Code

```python
from typing import List
from itertools import combinations

class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        hours = [8, 4, 2, 1]
        minutes = [32, 16, 8, 4, 2, 1]

        result: List[str] = []

        for h_count in range(min(4, turnedOn) + 1):
            m_count = turnedOn - h_count
            if m_count > 6:
                continue

            for h_combo in combinations(hours, h_count):
                hour = sum(h_combo)
                if hour >= 12:
                    continue

                for m_combo in combinations(minutes, m_count):
                    minute = sum(m_combo)
                    if minute < 60:
                        result.append(f"{hour}:{minute:02d}")

        return result
```

---

### Approach 3: Backtracking (DFS) ⭐ Interview Favorite

This is what you explicitly asked to visualize.

---

## 4. Backtracking Approach (With Tree Explanation)

### Core Idea

* We have **10 LEDs**
* Each LED can be either:

  * **ON (chosen)**
  * **OFF (skipped)**
* We explore all combinations using DFS

---

### LED Mapping

```
Index: 0  1  2  3   4   5   6  7  8  9
Value: 8  4  2  1  32  16  8  4  2  1
        ↑ Hour LEDs ↑    ↑ Minute LEDs ↑
```

---

### Backtracking Tree (Example: turnedOn = 2)

```
Start (count=0, hour=0, minute=0)
|
├── Choose LED[0]=8 (hour=8, count=1)
|   |
|   ├── Choose LED[1]=4 (hour=12 ❌ invalid)
|   |
|   ├── Choose LED[4]=32 (hour=8, minute=32 ✅)
|   |
|   └── Skip...
|
├── Choose LED[4]=32 (minute=32, count=1)
|   |
|   ├── Choose LED[5]=16 (minute=48 ✅)
|   |
|   └── Skip...
|
└── Skip LED[0]
    |
    └── Continue DFS
```

We **prune early** when:

* `count > turnedOn`
* `hour >= 12`
* `minute >= 60`

---

### Python 3 Backtracking Code (Typed)

```python
from typing import List

class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        leds: List[int] = [8, 4, 2, 1, 32, 16, 8, 4, 2, 1]
        result: List[str] = []

        def backtrack(index: int, count: int, hour: int, minute: int) -> None:
            # Pruning
            if hour >= 12 or minute >= 60:
                return

            if count == turnedOn:
                result.append(f"{hour}:{minute:02d}")
                return

            if index == len(leds):
                return

            # Choose current LED
            if index < 4:
                backtrack(index + 1, count + 1, hour + leds[index], minute)
            else:
                backtrack(index + 1, count + 1, hour, minute + leds[index])

            # Skip current LED
            backtrack(index + 1, count, hour, minute)

        backtrack(0, 0, 0, 0)
        return result
```

---

## 5. Example Walkthrough (turnedOn = 1)

### DFS Path

```
Start → choose LED[0] → hour=8 → "8:00"
Start → choose LED[1] → hour=4 → "4:00"
Start → choose LED[4] → minute=32 → "0:32"
...
```

Final output:

```
["8:00", "4:00", "2:00", "1:00", "0:32", "0:16", "0:08", "0:04", "0:02", "0:01"]
```

---

## 6. Which Approach Should You Use?

| Scenario | Best Approach |
| --- | --- |
| Coding round | Bit Counting |
| DSA interview | Backtracking |
| Teaching / clarity | Combination split |

---

## 7. Key Interview Takeaways

* Recognize **small bounded search space**
* Apply **combinatorics + pruning**
* Backtracking demonstrates **decision tree reasoning**

If you want, I can:

* Convert backtracking tree into a **diagram**
* Show **step-by-step DFS stack**
* Compare recursion vs bitmask approach

Just tell me.