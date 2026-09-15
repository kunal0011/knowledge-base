---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 767: Reorganize String"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 767: Reorganize String

Below is a **complete, structured explanation** of **LeetCode 767 – Reorganize String**, aligned with how this problem is typically reasoned about in interviews and competitive programming.

---

## 1. Problem Statement

**LeetCode 767 — Reorganize String**

You are given a string `s`.  
Rearrange the characters of `s` so that **no two adjacent characters are the same**.

* If it is **possible**, return any valid rearrangement.
* If it is **not possible**, return an empty string `""`.

### Example

```text
Input:  s = "aab"
Output: "aba"
```

```text
Input:  s = "aaab"
Output: ""
```

---

## 2. Key Observation (Feasibility Condition)

Let:

* `n` = length of string
* `maxFreq` = frequency of the most common character

### Critical Condition

A valid reorganization is possible **if and only if**:

```
maxFreq ≤ (n + 1) // 2
```

### Why?

* The most frequent character must be placed with **at least one different character between each occurrence**.
* The maximum number of “safe slots” available is `(n + 1) // 2`.
* If one character exceeds this count, it is impossible to separate its occurrences.

#### Example

```
s = "aaab" → freq(a) = 3, n = 4
(4 + 1) // 2 = 2 → 3 > 2 → impossible
```

---

## 3. Why Priority Queue (Max Heap)?

### Greedy Strategy

At every step:

* Pick the **two characters with highest remaining frequencies**
* Place them next to each other (in alternating order)
* Decrease their frequencies
* Push them back if they still have remaining count

### Why this works

* Always using the most frequent characters first prevents them from clustering together.
* Greedy choice is safe because future placements have fewer constraints.

### Data Structure Choice

Python’s `heapq` is a **min-heap**, so:

* Store frequencies as **negative values** to simulate a **max-heap**

---

## 4. Algorithm (Step-by-Step)

1. Count character frequencies.
2. Check feasibility using the `(n + 1) // 2` rule.
3. Build a max-heap `(−frequency, character)`.
4. While heap has **at least two elements**:

   * Pop top two characters
   * Append them to result
   * Decrement frequencies
   * Push back if still remaining
5. If one character remains, append it (safe by feasibility check).
6. Return the result string.

---

## 5. Python 3 Solution (With Typing)

```python
from typing import Dict
import heapq
from collections import Counter

class Solution:
    def reorganizeString(self, s: str) -> str:
        freq: Dict[str, int] = Counter(s)
        n: int = len(s)

        # Feasibility check
        if max(freq.values()) > (n + 1) // 2:
            return ""

        # Max heap using negative frequencies
        max_heap = [(-count, char) for char, count in freq.items()]
        heapq.heapify(max_heap)

        result = []

        while len(max_heap) >= 2:
            count1, char1 = heapq.heappop(max_heap)
            count2, char2 = heapq.heappop(max_heap)

            result.append(char1)
            result.append(char2)

            if count1 + 1 < 0:
                heapq.heappush(max_heap, (count1 + 1, char1))
            if count2 + 1 < 0:
                heapq.heappush(max_heap, (count2 + 1, char2))

        # If one character remains
        if max_heap:
            result.append(max_heap[0][1])

        return "".join(result)
```

---

## 6. Worked Examples

---

### Example 1: `s = "aab"`

#### Step 1: Frequency

```
a → 2
b → 1
```

#### Step 2: Feasibility

```
maxFreq = 2
(3 + 1) // 2 = 2 → OK
```

#### Step 3: Max Heap

```
[(-2, 'a'), (-1, 'b')]
```

#### Step 4: Processing

* Pop: `a(2)`, `b(1)`
* Result: `"ab"`
* Push back: `a(1)`

Heap:

```
[(-1, 'a')]
```

#### Step 5: Remaining

* Append `a`

#### Final Output

```
"aba"
```

---

### Example 2: `s = "aaab"`

#### Frequency

```
a → 3
b → 1
```

#### Feasibility Check

```
maxFreq = 3
(4 + 1) // 2 = 2 → 3 > 2
```

#### Result

```
""
```

---

### Example 3: `s = "vvvlo"`

#### Frequency

```
v → 3
l → 1
o → 1
```

#### Heap Initialization

```
[(-3, 'v'), (-1, 'l'), (-1, 'o')]
```

#### Processing Steps

1. Pop `v, l` → `"vl"` → push back `v(2)`
2. Pop `v, o` → `"vlvo"` → push back `v(1)`
3. Append remaining `v`

#### Final Output

```
"vlvov"
```

(Any valid arrangement is acceptable)

---

## 7. Complexity Analysis

* **Time Complexity:**

  ```
  O(n log k)
  ```

  where `k` = number of unique characters
* **Space Complexity:**

  ```
  O(k)
  ```

---

## 8. Key Takeaways

* This is a **greedy + priority queue** problem.
* The **feasibility condition** is the most important insight.
* Always placing the **two most frequent remaining characters** avoids adjacency violations.
* Multiple correct answers may exist.

If you want, I can also:

* Show a **dry-run table**
* Explain **why a sorting-only approach fails**
* Compare this with **bucket placement / index-filling technique**