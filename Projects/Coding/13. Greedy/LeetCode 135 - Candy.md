---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 135: Candy"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 135: Candy

**LeetCode 135 – Candy**, aligned with your usual preference for conceptual clarity and step-by-step reasoning.

---

## LeetCode 135 — Candy

### Problem Statement

You are given an integer array `ratings` where `ratings[i]` represents the rating of the *i-th* child standing in a line.

You must distribute candies to these children subject to the following rules:

1. Each child must have **at least one candy**.
2. Children with a **higher rating than their immediate neighbor** must get **more candies** than that neighbor.

Return the **minimum total number of candies** required to satisfy these conditions.

---

### Key Observations

1. **Local constraint, global minimum**

   * Each child’s candy count depends only on **adjacent neighbors**.
   * However, naive local decisions can violate constraints later.
2. **One-directional greediness is insufficient**

   * Left-to-right ensures correctness w.r.t. left neighbor.
   * Right-to-left ensures correctness w.r.t. right neighbor.
   * **Both directions are required**.
3. **Final candy count for each child**

   * Must satisfy **both** constraints:

     ```
     candies[i] ≥ candies[i-1] + 1  (if rating[i] > rating[i-1])
     candies[i] ≥ candies[i+1] + 1  (if rating[i] > rating[i+1])
     ```
   * Hence:

     ```
     candies[i] = max(left_pass[i], right_pass[i])
     ```

---

## Greedy Strategy (Core Trick)

### Two-Pass Greedy Algorithm

1. **Initialize**

   * Give each child **1 candy** (minimum requirement).
2. **Left → Right pass**

   * If `ratings[i] > ratings[i-1]`

     ```
     candies[i] = candies[i-1] + 1
     ```
3. **Right → Left pass**

   * If `ratings[i] > ratings[i+1]`

     ```
     candies[i] = max(candies[i], candies[i+1] + 1)
     ```
4. **Sum all candies**

---

## Why This Works

* First pass enforces **left neighbor dominance**
* Second pass enforces **right neighbor dominance**
* `max()` ensures **both constraints are satisfied**
* Greedy works because:

  * We only increase candies when *forced by constraints*
  * We never increase unnecessarily

---

## Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        if n == 0:
            return 0

        # Step 1: each child gets at least one candy
        candies = [1] * n

        # Step 2: left to right
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1

        # Step 3: right to left
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)

        # Step 4: total candies
        return sum(candies)
```

---

## Complete Worked Example (Step-by-Step)

### Input

```
ratings = [1, 0, 2]
```

---

### Step 1: Initialization

```
candies = [1, 1, 1]
```

---

### Step 2: Left → Right Pass

| i | ratings | Condition | candies update |
| --- | --- | --- | --- |
| 1 | 0 vs 1 | 0 > 1 ❌ | no change |
| 2 | 2 vs 0 | 2 > 0 ✅ | candies[2] = 2 |

```
candies = [1, 1, 2]
```

---

### Step 3: Right → Left Pass

| i | ratings | Condition | candies update |
| --- | --- | --- | --- |
| 1 | 0 vs 2 | 0 > 2 ❌ | no change |
| 0 | 1 vs 0 | 1 > 0 ✅ | candies[0] = max(1, 1+1) = 2 |

```
candies = [2, 1, 2]
```

---

### Step 4: Final Sum

```
2 + 1 + 2 = 5
```

---

### Output

```
5
```

---

## Another Example (Monotonic Decrease)

### Input

```
ratings = [5, 4, 3, 2, 1]
```

### Final candies after both passes

```
[5, 4, 3, 2, 1]
```

### Total

```
15
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(n)** |
| Greedy Validity | Guaranteed minimal |

---

## Interview Notes

* This problem is a **classic greedy + two-pass** pattern
* Frequently asked follow-up:

  > Can we do it in O(1) space?

  * Yes, using slope counting (advanced variant), but **this solution is preferred and accepted**

---

If you want, I can next:

* Derive the **O(1) space slope-based solution**
* Explain **why a single pass fails**
* Convert this into a **pattern template** usable across greedy problems

Just tell me how deep you want to go.