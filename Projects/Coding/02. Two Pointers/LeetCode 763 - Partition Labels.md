---
date: "2025-12-23"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 763: Partition Labels"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 763: Partition Labels

Below is a complete, structured explanation of **LeetCode 763 – Partition Labels**, aligned with your usual learning format.

---

## Problem Statement

You are given a string `s` consisting of lowercase English letters.  
Partition the string into as many parts as possible such that **each letter appears in at most one part**.

Return a list of integers representing the **lengths of these parts**.

### Example

```text
Input:  s = "ababcbacadefegdehijhklij"
Output: [9, 7, 8]
```

---

## Key Observation

The critical insight is:

> A character **cannot appear outside the partition** in which it first appears.

Therefore:

* For every character, we must know its **last occurrence** in the string.
* A partition can only end when **all characters seen so far do not appear later**.

This converts the problem into **interval merging over character ranges**, executed greedily in one pass.

---

## Two Pointer / Greedy Technique

Although commonly described as greedy, this is effectively a **two-pointer window expansion** technique.

### Pointers

* `start` → beginning of the current partition
* `end` → farthest last occurrence of any character seen so far

### Algorithm Logic

1. Precompute the **last index** of each character.
2. Traverse the string with index `i`.
3. For each character `s[i]`, update:

   ```
   end = max(end, last_index[s[i]])
   ```
4. When `i == end`:

   * All characters in the current window are fully contained.
   * Close the partition.
   * Record its length: `end - start + 1`
   * Move `start` to `i + 1`

---

## Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # Step 1: Record last occurrence of each character
        last = {ch: i for i, ch in enumerate(s)}

        result: List[int] = []
        start = 0
        end = 0

        # Step 2: Traverse the string
        for i, ch in enumerate(s):
            end = max(end, last[ch])

            # Step 3: Close partition
            if i == end:
                result.append(end - start + 1)
                start = i + 1

        return result
```

---

## Worked-Out Example (Step-by-Step)

### Input

```
s = "ababcbacadefegdehijhklij"
```

### Last Occurrence Map (partial)

```
a → 8
b → 5
c → 7
d → 14
e → 15
f → 11
g → 13
h → 19
i → 22
j → 23
k → 20
l → 21
```

---

### Traversal Walkthrough

| i | s[i] | end update | start | action |
| --- | --- | --- | --- | --- |
| 0 | a | max(0, 8) = 8 | 0 | continue |
| 1 | b | max(8, 5) = 8 | 0 | continue |
| 2 | a | max(8, 8) = 8 | 0 | continue |
| 3 | b | max(8, 5) = 8 | 0 | continue |
| 4 | c | max(8, 7) = 8 | 0 | continue |
| 5 | b | max(8, 5) = 8 | 0 | continue |
| 6 | a | max(8, 8) = 8 | 0 | continue |
| 7 | c | max(8, 7) = 8 | 0 | continue |
| 8 | a | max(8, 8) = 8 | 0 | **cut → length = 9** |

Partition: `"ababcbaca"`

---

Next partitions follow the same logic:

* `"defegde"` → length **7**
* `"hijhklij"` → length **8**

---

## Final Output

```
[9, 7, 8]
```

---

## Complexity Analysis

* **Time Complexity:** `O(n)`  
  (single pass + constant-time map access)
* **Space Complexity:** `O(1)`  
  (at most 26 lowercase letters)

---

## Core Takeaway

This problem is a textbook example of:

* Greedy window expansion
* Two-pointer boundary control
* Preprocessing with last-occurrence indexing

If you want, I can also:

* Convert this into a **formal interval-merging interpretation**
* Show a **counterexample for incorrect greedy cuts**
* Map this problem to a **general greedy pattern category**