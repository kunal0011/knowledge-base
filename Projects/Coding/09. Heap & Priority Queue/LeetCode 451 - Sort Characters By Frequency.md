---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 451: Sort Characters By Frequency"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 451: Sort Characters By Frequency

Below is a complete, structured explanation of **LeetCode 451 – Sort Characters By Frequency**, aligned with your usual preparation style (problem → observation → technique → implementation → worked examples).

---

## LeetCode 451: Sort Characters By Frequency

---

### Problem Statement

Given a string `s`, sort it in **decreasing order based on the frequency of characters**.  
Return the resulting string.

* Characters with higher frequency should appear before characters with lower frequency.
* If multiple characters have the same frequency, their relative order does **not** matter.

**Example**

```text
Input:  s = "tree"
Output: "eert"  or "eetr"
```

---

## Key Observations

1. This is a **frequency-based ordering problem**, not lexicographical.
2. The task naturally breaks into two phases:

   * **Count** frequency of each character.
   * **Order** characters by decreasing frequency.
3. We must repeatedly extract the character with **maximum frequency** → this maps directly to a **Max Heap / Priority Queue**.
4. Python’s `heapq` is a **min-heap**, so we simulate a max-heap by:

   * Storing frequencies as **negative values**.

---

## Priority Queue (Heap) Technique Used

### Why Priority Queue?

* We want:

  * Fast access to the **highest-frequency character**
  * Repeated extraction in sorted order

### Heap Structure

Each heap entry:

```
(-frequency, character)
```

Why this works:

* Heap orders by first element (frequency)
* Negative frequency converts min-heap → max-heap
* Character acts as a tie-breaker but order is irrelevant for same frequency

---

## Algorithm (Step-by-Step)

1. Count character frequencies using `Counter`
2. Build a heap with `(-freq, char)`
3. Repeatedly pop from heap
4. Append `char * freq` to result
5. Join and return final string

---

## Python 3 Solution (with Typing)

```python
from typing import Dict
from collections import Counter
import heapq

class Solution:
    def frequencySort(self, s: str) -> str:
        # Step 1: Frequency count
        freq: Dict[str, int] = Counter(s)

        # Step 2: Build max heap using negative frequencies
        max_heap = [(-count, char) for char, count in freq.items()]
        heapq.heapify(max_heap)

        # Step 3: Build result
        result = []

        while max_heap:
            count, char = heapq.heappop(max_heap)
            result.append(char * (-count))

        return "".join(result)
```

---

## Worked Example 1

### Input

```
s = "tree"
```

### Step 1: Frequency Count

```
{
  't': 1,
  'r': 1,
  'e': 2
}
```

### Step 2: Heap Construction

```
[(-1, 't'), (-1, 'r'), (-2, 'e')]
```

After heapify (conceptual order):

```
[(-2, 'e'), (-1, 'r'), (-1, 't')]
```

### Step 3: Heap Pops

| Pop | (count, char) | Append |
| --- | --- | --- |
| 1 | (-2, 'e') | "ee" |
| 2 | (-1, 'r') | "r" |
| 3 | (-1, 't') | "t" |

### Final Output

```
"eert"
```

---

## Worked Example 2

### Input

```
s = "cccaaa"
```

### Frequency

```
{
  'c': 3,
  'a': 3
}
```

### Heap

```
[(-3, 'a'), (-3, 'c')]
```

### Output Possibilities

```
"aaaccc" or "cccaaa"
```

(Both are valid since equal frequencies)

---

## Time & Space Complexity

### Time Complexity

* Frequency count: **O(n)**
* Heap operations: **O(k log k)**  
  where `k` = number of unique characters
* Total: **O(n + k log k)**

### Space Complexity

* Frequency map + heap + result: **O(n)**

---

## Why This Is a Classic Heap Problem

* Pattern: **“Sort elements by frequency”**
* Core idea:

  * Count → Heap → Extract max repeatedly
* Same technique appears in:

  * Top K Frequent Elements
  * Reorganize String
  * Task Scheduler

---

If you want, I can next:

* Show a **bucket sort** alternative (O(n))
* Map this problem to a **DSA pattern**
* Compare heap vs bucket for interviews