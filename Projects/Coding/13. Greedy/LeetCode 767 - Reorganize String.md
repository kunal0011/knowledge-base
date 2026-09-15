---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 767: Reorganize String"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 767: Reorganize String

Below is a **complete, structured, interview-grade explanation** of **LeetCode 767 – Reorganize String**, aligned with how you have been approaching greedy and backtracking problems.

---

## 1. Problem Statement (LeetCode 767)

You are given a string `s`.  
Rearrange the characters of `s` so that **no two adjacent characters are the same**.

Return **any valid rearrangement**.  
If no such arrangement is possible, return an empty string `""`.

### Constraints

* `1 <= len(s) <= 500`
* `s` consists of lowercase English letters

---

## 2. Key Observation (Feasibility Condition)

Let:

* `n = len(s)`
* `max_freq = maximum frequency of any character`

### Necessary and Sufficient Condition

A valid rearrangement exists **if and only if**:

```
max_freq <= (n + 1) // 2
```

### Why?

* The most frequent character must be placed in gaps between other characters.
* There are at most `(n + 1) // 2` such positions.
* If one character appears more than that, it is **forced** to sit next to itself.

Example:

```
s = "aaab"
n = 4
max_freq = 3
(4 + 1) // 2 = 2 → impossible
```

---

## 3. Greedy Strategy (Core Idea)

This is a **classic greedy + max-heap** problem.

### Greedy Invariant

At every step:

* Pick the **two most frequent remaining characters**
* Place them **alternately**
* Reduce their counts
* Push them back if still available

This guarantees:

* No adjacent duplicates
* Highest-risk characters are handled first

---

## 4. Greedy Solution Trick (Why Heap of Two?)

If you always place **only the single most frequent character**, you risk:

```
a a _ a _
```

Instead:

* Place the top two most frequent characters **together**
* This ensures they separate each other

This pattern is common in:

* Task scheduling
* String reorganization
* Priority-based greedy problems

---

## 5. Algorithm (Step-by-Step)

1. Count frequency of characters.
2. Check feasibility using `(n + 1) // 2`.
3. Push `(−frequency, character)` into a max-heap.
4. While heap has at least two elements:

   * Pop two most frequent characters
   * Append both to result
   * Decrease their frequencies
   * Push back if still remaining
5. If one character remains:

   * Append it (safe because feasibility already checked)

---

## 6. Python 3 Solution (With Typing)

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

        # Max heap: (-count, char)
        max_heap = [(-count, ch) for ch, count in freq.items()]
        heapq.heapify(max_heap)

        result = []

        while len(max_heap) >= 2:
            count1, ch1 = heapq.heappop(max_heap)
            count2, ch2 = heapq.heappop(max_heap)

            result.append(ch1)
            result.append(ch2)

            if count1 + 1 < 0:
                heapq.heappush(max_heap, (count1 + 1, ch1))
            if count2 + 1 < 0:
                heapq.heappush(max_heap, (count2 + 1, ch2))

        if max_heap:
            result.append(max_heap[0][1])

        return "".join(result)
```

---

## 7. Complete Worked Example (All Processing Steps)

### Input

```
s = "aaabbc"
```

### Step 1: Frequency Count

```
a → 3
b → 2
c → 1
```

### Step 2: Feasibility Check

```
n = 6
max_freq = 3
(6 + 1) // 2 = 3 → valid
```

### Step 3: Max Heap Initialization

```
[(-3, 'a'), (-2, 'b'), (-1, 'c')]
```

---

### Iteration 1

Pop two:

```
('a', 3), ('b', 2)
```

Append:

```
result = ['a', 'b']
```

Update counts:

```
a → 2, b → 1
```

Push back:

```
[(-2, 'a'), (-1, 'b'), (-1, 'c')]
```

---

### Iteration 2

Pop two:

```
('a', 2), ('b', 1)
```

Append:

```
result = ['a', 'b', 'a', 'b']
```

Update counts:

```
a → 1, b → 0
```

Heap:

```
[(-1, 'a'), (-1, 'c')]
```

---

### Iteration 3

Pop two:

```
('a', 1), ('c', 1)
```

Append:

```
result = ['a', 'b', 'a', 'b', 'a', 'c']
```

Heap becomes empty.

---

### Final Output

```
"ababac"
```

✔ No adjacent duplicates  
✔ All characters used

---

## 8. Time and Space Complexity

### Time

```
O(n log k)
```

* `k` = number of unique characters (≤ 26)

### Space

```
O(k)
```

---

## 9. Pattern Recognition (Important for Interviews)

This problem belongs to the **Greedy + Priority Queue** pattern, commonly used when:

* You must avoid adjacent conflicts
* You must always pick the “most dangerous” option first

Related problems:

* Task Scheduler (LC 621)
* Rearrange String k Distance Apart
* Minimum Cost to Hire Workers

---

If you want, next we can:

* Rewrite this using **array placement (even/odd indices)**
* Compare greedy vs bucket approach
* Convert this into a **generic greedy template** reusable across problems