---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1090: Largest Values From Labels"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1090: Largest Values From Labels

Below is a complete, structured explanation for **LeetCode 1090 – Largest Values From Labels**, aligned with your preferred problem-solving depth.

---

## 1. Problem Statement

You are given:

* `values[i]`: the value of the `i`-th item
* `labels[i]`: the label of the `i`-th item
* An integer `numWanted`: maximum number of items you can pick
* An integer `useLimit`: maximum number of items allowed **per label**

### Objective

Select **at most `numWanted` items** such that:

* No label is used more than `useLimit` times
* The **sum of selected values is maximized**

---

## 2. Key Observations

1. **No dependency between items**

   * Each item is independent except for:

     * global limit (`numWanted`)
     * per-label limit (`useLimit`)
2. **We always want higher values first**

   * There is no benefit in skipping a higher value in favor of a lower one unless constrained by label limits.
3. **Greedy is optimal**

   * Sorting items by value (descending) and picking valid ones yields the optimal result.
   * This works because:

     * Constraints are monotonic
     * There is no future tradeoff (no DP required)

---

## 3. Greedy Strategy (Core Trick)

### Greedy Rule

> Always pick the highest-value available item **if**:

* Total picked items `< numWanted`
* The item's label count `< useLimit`

### Why This Works

* Sorting ensures we consider the best possible candidate first
* Label constraints only block further picks of the same label
* Skipping a valid high-value item can never increase the final sum

---

## 4. Algorithm Steps

1. Combine `values` and `labels` into pairs
2. Sort pairs by value in descending order
3. Maintain:

   * `label_count[label]` → how many times a label is used
   * `picked` → number of selected items
   * `total_sum`
4. Iterate through sorted items:

   * If constraints allow → pick the item
   * Stop when `picked == numWanted`

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def largestValsFromLabels(
        self,
        values: List[int],
        labels: List[int],
        numWanted: int,
        useLimit: int
    ) -> int:
        # Step 1: Pair values with labels
        items = list(zip(values, labels))
        
        # Step 2: Sort by value descending
        items.sort(reverse=True, key=lambda x: x[0])
        
        label_count = defaultdict(int)
        total_sum = 0
        picked = 0
        
        # Step 3: Greedy selection
        for value, label in items:
            if picked == numWanted:
                break
            
            if label_count[label] < useLimit:
                total_sum += value
                label_count[label] += 1
                picked += 1
        
        return total_sum
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
values    = [5, 4, 3, 2, 1]
labels    = [1, 1, 2, 2, 3]
numWanted = 3
useLimit  = 1
```

---

### Step 1: Pair and Sort

```
Paired items (value, label):
[(5,1), (4,1), (3,2), (2,2), (1,3)]

Sorted descending:
[(5,1), (4,1), (3,2), (2,2), (1,3)]
```

---

### Step 2: Iteration Trace

| Step | Item | Label Count Before | Pick? | Reason | Total Sum |
| --- | --- | --- | --- | --- | --- |
| 1 | (5,1) | label 1 → 0 | YES | under limit | 5 |
| 2 | (4,1) | label 1 → 1 | NO | exceeds `useLimit` | 5 |
| 3 | (3,2) | label 2 → 0 | YES | under limit | 8 |
| 4 | (2,2) | label 2 → 1 | NO | exceeds `useLimit` | 8 |
| 5 | (1,3) | label 3 → 0 | YES | under limit | 9 |

---

### Step 3: Termination

* Picked items = 3 (`numWanted`)
* Stop processing

---

### Final Output

```
9
```

---

## 7. Complexity Analysis

* **Sorting:** `O(n log n)`
* **Iteration:** `O(n)`
* **Space:** `O(n)` for sorting + `O(k)` for label counts

---

## 8. When This Greedy Pattern Appears Again

You will see this exact pattern in:

* “Pick top K with constraints”
* “Maximize sum with category limits”
* “Priority + quota” problems

Common signals:

* Independent items
* Hard caps (count limits)
* Objective is monotonic (sum / max)

---

If you want, I can:

* Convert this into a **decision-tree / greedy pick tree**
* Compare with a **DP approach and show why it’s inferior**
* Generalize this into a **reusable greedy template**