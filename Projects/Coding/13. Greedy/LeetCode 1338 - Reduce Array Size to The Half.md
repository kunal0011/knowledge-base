---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1338: Reduce Array Size to The Half"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1338: Reduce Array Size to The Half

Below is a structured, end-to-end explanation of **LeetCode 1338 – Reduce Array Size to The Half**, aligned with interview expectations and greedy-algorithm reasoning.

---

## 1. Problem Statement

You are given an integer array `arr`.  
In one operation, you can choose **any integer value** and remove **all occurrences** of that value from the array.

Your task is to return the **minimum number of such operations** required to reduce the size of the array to **at most half**of its original size.

### Constraints

* `1 ≤ arr.length ≤ 10^5`
* `1 ≤ arr[i] ≤ 10^5`

---

## 2. Key Observation

* Removing a value with **higher frequency** eliminates **more elements per operation**.
* Since each operation removes all occurrences of one value, the **order of removal does not matter**, only the **frequencies** matter.
* To minimize the number of operations, we should:

  * Remove values with the **largest frequencies first**.

This immediately points to a **greedy strategy**.

---

## 3. Greedy Strategy (Why It Works)

### Greedy Choice

Always remove the number that appears **most frequently**.

### Justification

* Each operation has equal cost (1 operation).
* To reach the target (`remaining ≤ n/2`) in the fewest operations, we want to reduce the array size as fast as possible.
* Removing smaller frequencies first would require more operations to reach the same reduction.

This satisfies the **greedy-choice property** and **optimal substructure**.

---

## 4. Algorithm Steps

1. Count frequency of each number.
2. Sort frequencies in **descending order**.
3. Iteratively remove the largest frequency:

   * Keep subtracting from the array size.
   * Count operations.
4. Stop once removed elements ≥ `n/2`.

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List
from collections import Counter

class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        n: int = len(arr)
        target: int = n // 2

        # Step 1: Count frequencies
        freq: Counter[int] = Counter(arr)

        # Step 2: Sort frequencies descending
        counts: List[int] = sorted(freq.values(), reverse=True)

        removed: int = 0
        operations: int = 0

        # Step 3: Greedily remove largest frequencies
        for c in counts:
            removed += c
            operations += 1
            if removed >= target:
                break

        return operations
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
arr = [3,3,3,3,5,5,5,2,2,7]
```

### Step 1: Initial Size

```
n = 10
target = n // 2 = 5
```

---

### Step 2: Frequency Count

| Value | Frequency |
| --- | --- |
| 3 | 4 |
| 5 | 3 |
| 2 | 2 |
| 7 | 1 |

---

### Step 3: Sort Frequencies (Descending)

```
counts = [4, 3, 2, 1]
```

---

### Step 4: Greedy Removal Process

| Operation | Frequency Removed | Total Removed | Remaining | Condition |
| --- | --- | --- | --- | --- |
| 1 | 4 | 4 | 6 | Not enough |
| 2 | 3 | 7 | 3 | ≤ n/2 ✔ |

---

### Final Result

```
Minimum operations = 2
```

---

## 7. Why Not Dynamic Programming or Backtracking?

* DP would be overkill and inefficient given constraints.
* Backtracking is exponential and unnecessary.
* Greedy works optimally due to uniform operation cost and monotonic reduction.

---

## 8. Time and Space Complexity

### Time Complexity

* Frequency count: `O(n)`
* Sorting frequencies: `O(k log k)` where `k` = number of unique elements
* Overall: **O(n log n)** in worst case

### Space Complexity

* Frequency map: `O(k)`
* Sorted list: `O(k)`
* Overall: **O(n)**

---

## 9. Interview Takeaway

* This is a **classic frequency-based greedy problem**.
* Key trigger:

  > “Remove all occurrences” + “minimum operations” → **sort by frequency**.
* Very similar reasoning applies to problems like:

  * Top-K Frequent Elements
  * Remove Stones to Minimize the Total

If you want, I can also:

* Rewrite this using a **heap**
* Compare greedy vs incorrect approaches
* Map this problem to a broader **Greedy Pattern taxonomy**