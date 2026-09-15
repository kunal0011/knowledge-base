---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 401: Binary Watch"
tags:
  - leetcode
  - coding
  - backtracking
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 401: Binary Watch

**Target Companies:** Google, Amazon, Apple  
**Difficulty:** Easy  
**Topic:** Backtracking / Bit Manipulation / Exhaustive Search  

---

### Problem Statement

A binary watch has 4 LEDs on the top to represent the hours (`0-11`), and 6 LEDs on the bottom to represent the minutes (`0-59`). Each LED represents a zero or one, with the least significant bit on the right.

- The 4 hour LEDs represent values: `8, 4, 2, 1`.
- The 6 minute LEDs represent values: `32, 16, 8, 4, 2, 1`.

Given an integer `turnedOn` which represents the number of LEDs that are currently on, return *all possible times the watch could represent*. You may return the answer in **any order**.

**Format Rules:**
- The hour must not contain a leading zero (e.g., `"01:00"` is invalid, `"1:00"` is valid).
- The minute must be consistently two digits and may contain a leading zero (e.g., `"10:2"` is invalid, `"10:02"` is valid).

---

### Input & Output Formats & Constraints

- **Input:** `turnedOn: int`
- **Output:** `List[str]`
- **Constraints:**
  - $0 \le \text{turnedOn} \le 10$

---

### Key Idea & Intuition

- **Constraint Analysis & Universe Size:**
  - The watch display can only show valid clock times:
    - Hours: $0 \le h \le 11$ (12 possibilities)
    - Minutes: $0 \le m \le 59$ (60 possibilities)
  - The total number of valid times on the watch is strictly:
    $$12 \times 60 = 720 \text{ possible times}$$
- **Approach 1: Bit Count Enumeration ($\mathcal{O}(1)$):**
  - Iterate over all $h \in [0, 11]$ and $m \in [0, 59]$.
  - Count set bits: $\text{popcount}(h) + \text{popcount}(m)$.
  - If the sum equals `turnedOn`, format as `f"{h}:{m:02d}"` and record.
- **Approach 2: Backtracking over the 10 LEDs:**
  - We have an array of 10 LED weights:
    `[1, 2, 4, 8]` for hours, and `[1, 2, 4, 8, 16, 32]` for minutes.
  - Pick exactly `turnedOn` LEDs out of 10.
  - Prune immediately if $h \ge 12$ or $m \ge 60$.

---

### Solution Approach (Step-by-Step)

#### Direct Iteration (Clean & Interview Preferred):
1. If `turnedOn >= 9`, return `[]` (at most 3 bits for hours since $\max h = 11$, and at most 5 bits for minutes since $\max m = 59 \implies 3 + 5 = 8$ max on LEDs).
2. Initialize `results = []`.
3. Loop `h` from $0$ to $11$:
   - Loop `m` from $0$ to $59$:
     - If `bin(h).count('1') + bin(m).count('1') == turnedOn`:
       - Append formatted string `f"{h}:{m:02d}"` to `results`.
4. Return `results`.

---

### Visual Algorithm Walkthrough

Binary Watch Layout:
```
Hours LEDs (4 bits):    [8]  [4]  [2]  [1]   (Range: 0 - 11)
Minutes LEDs (6 bits):  [32] [16] [8]  [4]  [2]  [1] (Range: 0 - 59)
```

Example for `turnedOn = 1`:
- Exactly 1 bit is ON across the entire watch:
  - 1 bit in hours ($h \in \{1, 2, 4, 8\}$), 0 bits in minutes ($m = 0$):
    $\implies$ `"1:00", "2:00", "4:00", "8:00"`
  - 0 bits in hours ($h = 0$), 1 bit in minutes ($m \in \{1, 2, 4, 8, 16, 32\}$):
    $\implies$ `"0:01", "0:02", "0:04", "0:08", "0:16", "0:32"`
- Total valid outputs = $4 + 6 = 10$.

---

### Solved Examples with Multiple Inputs

| Test Case | `turnedOn` | Total Combinations | Example Valid Times |
| :--- | :--- | :--- | :--- |
| **Example 1** | `1` | `10` | `"0:01"`, `"0:02"`, `"1:00"`, `"8:00"` |
| **Zero LEDs** | `0` | `1` | `"0:00"` |
| **Too Many LEDs** | `9` | `0` | `[]` (Max possible is $3 + 5 = 8$) |
| **Example 2** | `2` | `44` | `"0:03"`, `"1:01"`, `"1:02"`, `"3:00"` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        """
        Enumerates all valid times matching the total turnedOn LEDs.
        O(1) time complexity over fixed 720 clock configurations.
        """
        if turnedOn > 8:
            return []

        results = []
        for h in range(12):
            for m in range(60):
                if bin(h).count('1') + bin(m).count('1') == turnedOn:
                    results.append(f"{h}:{m:02d}")
        return results
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <bitset>
#include <iomanip>
#include <sstream>

class Solution {
public:
    std::vector<std::string> readBinaryWatch(int turnedOn) {
        if (turnedOn > 8) return {};

        std::vector<std::string> results;
        for (int h = 0; h < 12; ++h) {
            for (int m = 0; m < 60; ++m) {
                if (__builtin_popcount(h) + __builtin_popcount(m) == turnedOn) {
                    std::string time = std::to_string(h) + ":" + (m < 10 ? "0" : "") + std::to_string(m);
                    results.push_back(time);
                }
            }
        }
        return results;
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<String> readBinaryWatch(int turnedOn) {
        List<String> results = new ArrayList<>();
        if (turnedOn > 8) {
            return results;
        }

        for (int h = 0; h < 12; h++) {
            for (int m = 0; m < 60; m++) {
                if (Integer.bitCount(h) + Integer.bitCount(m) == turnedOn) {
                    results.add(String.format("%d:%02d", h, m));
                }
            }
        }

        return results;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(1)$.
  - The nested loops execute exactly $12 \times 60 = 720$ iterations, which is a small constant.
  - Bit counting and string formatting execute in $\mathcal{O}(1)$ time per pair.
  - Runs in $< 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space beyond the output list.

---

### Takeaway Pattern & Interview Traps

- **Recognizing Bounded Universe Size:** When search boundaries are strictly fixed by physical limits ($12$ hours and $60$ minutes), brute-force iteration over the output domain is often vastly cleaner and less error-prone than complex recursive state-space backtracking.
- **Maximum Possible LEDs On:** Notice that the maximum possible bits for $h \in [0, 11]$ is 3 (for $h = 7$ or $11$), and for $m \in [0, 59]$ is 5 (for $m = 31, 47, 55, 59$). Hence, for $\text{turnedOn} \ge 9$, no valid time can ever exist; return `[]` immediately.