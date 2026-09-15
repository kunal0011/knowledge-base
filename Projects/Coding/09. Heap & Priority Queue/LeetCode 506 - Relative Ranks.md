---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 506: Relative Ranks"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 506: Relative Ranks

Below is a structured, interview-grade explanation of **LeetCode 506 – Relative Ranks**, following the format you requested.

---

## Problem Statement (LeetCode 506 – Relative Ranks)

You are given an integer array `score` of size `n`, where `score[i]` is the score of the `i`-th athlete.

All scores are **unique**. The athletes are ranked based on their scores:

* The athlete with the **highest score** gets `"Gold Medal"`
* The athlete with the **second highest score** gets `"Silver Medal"`
* The athlete with the **third highest score** gets `"Bronze Medal"`
* All remaining athletes receive their **numeric rank** (starting from `"4"`)

Return an array `answer` of size `n`, where `answer[i]` is the rank of the `i`-th athlete.

---

## Key Observations

1. **Ranking is global, output is index-based**

   * Rankings depend on sorted scores.
   * Final output must preserve **original indices**.
2. **Scores are unique**

   * No tie handling is required.
   * This allows deterministic ranking.
3. **Top 3 ranks are special strings**

   * Position 1 → `"Gold Medal"`
   * Position 2 → `"Silver Medal"`
   * Position 3 → `"Bronze Medal"`
4. **This is a classic “top-k ranking” problem**

   * Naturally fits a **Priority Queue (Heap)** approach.

---

## Priority Queue Technique Used

### Why Priority Queue?

* We want to **repeatedly extract the maximum score**.
* Python provides a **min-heap**, so we simulate a max-heap by pushing **negative scores**.

### Heap Content Design

Each heap element stores:

```
(-score, original_index)
```

This ensures:

* Highest score is popped first
* We can assign the rank back to the correct athlete

---

## Algorithm (Heap-Based)

1. Initialize a result array `res` of size `n`
2. Push all `(−score[i], i)` into a heap
3. Pop elements one by one:

   * Assign medals for ranks 1–3
   * Assign numeric rank string for others
4. Return the result array

---

## Python 3 Solution (With Typing)

```python
from typing import List
import heapq

class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        n = len(score)
        res: List[str] = [""] * n

        # Max-heap using negative scores
        heap: List[tuple[int, int]] = []
        for i, s in enumerate(score):
            heapq.heappush(heap, (-s, i))

        rank = 1
        while heap:
            _, idx = heapq.heappop(heap)

            if rank == 1:
                res[idx] = "Gold Medal"
            elif rank == 2:
                res[idx] = "Silver Medal"
            elif rank == 3:
                res[idx] = "Bronze Medal"
            else:
                res[idx] = str(rank)

            rank += 1

        return res
```

---

## Worked Example 1

### Input

```
score = [5, 4, 3, 2, 1]
```

### Heap Construction

```
(-5, 0), (-4, 1), (-3, 2), (-2, 3), (-1, 4)
```

### Heap Pops & Rank Assignment

| Pop Order | Score | Index | Rank Assigned |
| --- | --- | --- | --- |
| 1 | 5 | 0 | Gold Medal |
| 2 | 4 | 1 | Silver Medal |
| 3 | 3 | 2 | Bronze Medal |
| 4 | 2 | 3 | "4" |
| 5 | 1 | 4 | "5" |

### Output

```
["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]
```

---

## Worked Example 2

### Input

```
score = [10, 3, 8, 9, 4]
```

### Sorted Scores (Descending)

```
10, 9, 8, 4, 3
```

### Rank Mapping

| Score | Index | Rank |
| --- | --- | --- |
| 10 | 0 | Gold Medal |
| 9 | 3 | Silver Medal |
| 8 | 2 | Bronze Medal |
| 4 | 4 | "4" |
| 3 | 1 | "5" |

### Output (Original Order)

```
["Gold Medal", "5", "Bronze Medal", "Silver Medal", "4"]
```

---

## Complexity Analysis

* **Time Complexity:**  
  `O(n log n)` due to heap insertions and removals
* **Space Complexity:**  
  `O(n)` for heap and result array

---

## Alternative Note (Without Heap)

This problem can also be solved by:

* Sorting indices by score (`O(n log n)`)
* Assigning ranks directly

However, the **priority queue solution generalizes well** to streaming or partial ranking problems and is commonly expected in interviews.

---

If you want, I can also:

* Compare heap vs sorting approach formally
* Show a one-pass map-based optimization
* Provide a dry-run visualization similar to backtracking trees