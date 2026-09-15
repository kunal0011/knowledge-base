---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 781: Rabbits in Forest"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 781: Rabbits in Forest

Below is a complete, structured explanation of **LeetCode 781 – Rabbits in Forest**, tailored for algorithmic clarity and interview readiness.

---

## 1. Problem Statement

You are given an integer array `answers`, where each element represents a rabbit’s answer to the question:

> “How many **other** rabbits have the same color as you?”

Each rabbit tells the truth.

Return the **minimum number of rabbits** that could be in the forest.

---

## 2. Key Observation (Core Insight)

If a rabbit answers **`x`**, then:

* There must be **`x + 1` rabbits** of that same color **in total** (including itself).
* All rabbits that answer `x` can be grouped into color-groups of **size `x + 1`**.

Important implications:

* Multiple rabbits may give the same answer `x`.
* However, **only `x + 1` rabbits can belong to one color-group**.
* If more than `x + 1` rabbits answer `x`, they must form **multiple color-groups**.

---

## 3. Greedy Strategy (Why Greedy Works)

### Greedy Idea

For each distinct answer `x`:

1. Count how many rabbits answered `x` → `count`.
2. Each color-group can hold **`x + 1` rabbits**.
3. We greedily pack rabbits into as few groups as possible.

### Formula

Number of groups needed:

```
groups = ceil(count / (x + 1))
```

Total rabbits contributed by this answer:

```
groups * (x + 1)
```

This is greedy because:

* We always **maximize utilization** of each color-group.
* Any partially filled group still requires the **full size** (`x + 1`) since unseen rabbits of that color may exist.

---

## 4. Python 3 Solution (with Typing)

```python
from typing import List
from collections import Counter
import math

class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        freq = Counter(answers)
        total_rabbits = 0

        for x, count in freq.items():
            group_size = x + 1
            groups = math.ceil(count / group_size)
            total_rabbits += groups * group_size

        return total_rabbits
```

---

## 5. Complete Worked Example (Step-by-Step)

### Input

```
answers = [1, 1, 2]
```

---

### Step 1: Frequency Count

| Answer (`x`) | Count |
| --- | --- |
| 1 | 2 |
| 2 | 1 |

---

### Step 2: Process Each Answer

#### Case 1: `x = 1`

* Each group size = `1 + 1 = 2`
* Rabbits saying `1` = `2`
* Groups needed:

  ```
  ceil(2 / 2) = 1
  ```
* Rabbits contributed:

  ```
  1 × 2 = 2
  ```

Interpretation:

* These two rabbits can be the **same color**.

---

#### Case 2: `x = 2`

* Each group size = `2 + 1 = 3`
* Rabbits saying `2` = `1`
* Groups needed:

  ```
  ceil(1 / 3) = 1
  ```
* Rabbits contributed:

  ```
  1 × 3 = 3
  ```

Interpretation:

* This rabbit implies **two more rabbits of the same color**, even though they are unseen.

---

### Step 3: Final Answer

```
Total rabbits = 2 + 3 = 5
```

---

## 6. Why This Is the Minimum

* You **must** account for implied rabbits.
* You **cannot merge groups** of different answers.
* Any incomplete group still requires the full size because rabbits might exist but did not answer.

Thus, **5 is the minimum possible number of rabbits**.

---

## 7. Time & Space Complexity

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)` (frequency map)

---

## 8. Interview Takeaway

This problem is a classic example of:

* **Implicit constraints**
* **Ceiling-based grouping**
* **Greedy packing with forced group sizes**

If you recognize:

> “Each answer defines a fixed group size”

the solution becomes straightforward.

---

If you want, I can also:

* Derive this using a **visual grouping model**
* Explain common **wrong approaches**
* Compare with similar greedy problems (e.g., task batching, bucket packing)