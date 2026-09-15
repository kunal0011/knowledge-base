---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 93: Restore IP Addresses"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 93: Restore IP Addresses

## LeetCode 93 — Restore IP Addresses

---

### Problem Statement

Given a string `s` containing only digits, return **all possible valid IP addresses** that can be formed by inserting dots (`.`) into `s`.

A **valid IP address**:

* Consists of **exactly 4 parts**
* Each part is between `0` and `255`
* No leading zeros unless the part is exactly `"0"`

**Constraints**

* `1 ≤ len(s) ≤ 20`

**Example**

```text
Input: s = "25525511135"
Output:
["255.255.11.135", "255.255.111.35"]
```

---

## Key Observations (Critical)

1. An IP address has **exactly 4 segments** → recursion depth is fixed.
2. Each segment can have **1 to 3 digits**.
3. **Pruning rules** are essential:

   * Segment value must be ≤ 255
   * No leading zeros (e.g., `"01"` is invalid)
4. If remaining characters are too many or too few to fill remaining segments, the branch can be cut early.
5. This is a **string partitioning backtracking** problem.

---

## Core Validity Rules

For a segment `part`:

* Length: `1 ≤ len(part) ≤ 3`
* If `len(part) > 1`, it must **not start with `'0'`**
* `int(part) ≤ 255`

---

## Approach (Backtracking)

### State

* `index`: current position in string
* `segments`: list of chosen IP segments

### Base Case

* If `len(segments) == 4`:

  * Valid only if `index == len(s)`

### Pruning

* Remaining characters must satisfy:

```
remaining_segments ≤ remaining_chars ≤ 3 × remaining_segments
```

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        result: List[str] = []
        segments: List[str] = []

        def backtrack(index: int) -> None:
            # If 4 segments are formed
            if len(segments) == 4:
                if index == len(s):
                    result.append(".".join(segments))
                return

            remaining_segments = 4 - len(segments)
            remaining_chars = len(s) - index

            # Prune invalid lengths
            if remaining_chars < remaining_segments or remaining_chars > remaining_segments * 3:
                return

            # Try segments of length 1 to 3
            for length in range(1, 4):
                if index + length > len(s):
                    break

                part = s[index:index + length]

                # Leading zero check
                if part[0] == "0" and len(part) > 1:
                    continue

                # Value check
                if int(part) > 255:
                    continue

                segments.append(part)
                backtrack(index + length)
                segments.pop()

        backtrack(0)
        return result
```

---

## Example Walkthrough

### Input

```
s = "25525511135"
```

### Valid Outputs

```
255.255.11.135
255.255.111.35
```

### Partial Path Explanation

* `255 → 255 → 11 → 135` ✓
* `255 → 255 → 111 → 35` ✓
* `255 → 25 → ...` ❌ (leads to invalid segmentation)

---

## Backtracking Tree Structure (Navigation View)

![https://i.ytimg.com/vi/h2HmKNdmZ_c/hq720.jpg?rs=AOn4CLDA7jxUQRzrT2rS2bk4CvRkL7PNEg&sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&utm_source=chatgpt.com](https://i.ytimg.com/vi/h2HmKNdmZ_c/hq720.jpg?rs=AOn4CLDA7jxUQRzrT2rS2bk4CvRkL7PNEg&sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&utm_source=chatgpt.com)

![https://assets.algo.monster/liteProblems/ip_address.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/IkVCKnicu5mZUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw4qdsnJyI-oCnMuCNS1KM4wqNK1TMvWjYwPybQIrDLLiTT39wgqdDQ0iA80cfTy8ylWKwYAWUQlbg?utm_source=chatgpt.com)

![https://www.researchgate.net/publication/344552220/figure/fig3/AS%3A944565909458959%401602213470315/An-Example-of-IP-TREE.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/nUmK9nicu5mVUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw4xKXIvcDbKsSxLzklONMyNCDEJMnKPjPLyL3cPCrXUDTYwzK5wzi6vzHb3M_Y3L89KVSsGAHD3Je8?utm_source=chatgpt.com)

### Conceptual Tree (Simplified)

```
[]
                  ------------------------------------------------
                 |                |                |
               2                  25              255
              /                   /                \
           [2]                 [25]                [255]
           /  \                   |               /      \
        5      55                ...          [255,2]   [255,25]
        |       |                                 |          |
      [2,5]   [2,55]                          [255,255]   [255,255]
         |        |                              /     \       |
       ...      ...                         [11]     [111]    [11]
                                             |         |       |
                                       [11,135]  [111,35]     ❌
```

---

## Why Pruning Is Crucial

Without pruning:

* Total partitions explode exponentially.  
  With pruning:
* Impossible segment lengths are eliminated early.
* Invalid values (`>255`) are never explored.

---

## Complexity Analysis

* **Time Complexity:** `O(1)` (bounded search space; max 3⁴ paths)
* **Space Complexity:** `O(1)` recursion depth (max depth = 4)

---

## Pattern Recognition

LeetCode 93 represents the **“Fixed Segments String Partitioning”** pattern:

* Fixed number of partitions
* Each partition has strict validation rules
* Strong pruning based on remaining length

---

### One-Line Interview Explanation

> “We backtrack by placing 1–3 digit segments, validate each segment, and prune branches where remaining characters cannot form valid IP segments.”

If you want next, I can:

* Show **what breaks if pruning is removed**
* Compare with **Palindrome Partitioning (LC 131)**
* Convert this into a **generic string segmentation template**