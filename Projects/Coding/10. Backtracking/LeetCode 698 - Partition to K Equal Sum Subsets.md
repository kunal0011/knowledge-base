---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 698: Partition to K Equal Sum Subsets"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 698: Partition to K Equal Sum Subsets

## LeetCode 698 — Partition to K Equal Sum Subsets

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return `True` if it is possible to divide the array into `k` **non-empty subsets**whose sums are all equal. Otherwise, return `False`.

**Rules**

* Each number in `nums` must be used **exactly once**
* Subsets do **not need to be contiguous**

**Constraints**

* `1 ≤ k ≤ len(nums) ≤ 16`
* `1 ≤ nums[i] ≤ 10⁴`

---

## Key Observations

1. Let `total = sum(nums)`. If `total % k != 0`, partitioning is **impossible**.
2. Each subset must sum to `target = total / k`.
3. This is a **combinatorial partitioning** problem → exponential by nature.
4. Sorting `nums` in **descending order** dramatically improves pruning.
5. Once a subset reaches `target`, we **start filling the next subset**.
6. This is equivalent to placing numbers into `k` buckets of equal capacity.

---

## Core Backtracking Idea

* Maintain an array `buckets[k]`, where each bucket tracks its current sum.
* Try to place each number into one of the buckets.
* Prune when:

  * A number does not fit in a bucket
  * A symmetric bucket state is revisited

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        total = sum(nums)
        if total % k != 0:
            return False

        target = total // k
        nums.sort(reverse=True)

        if nums[0] > target:
            return False

        buckets = [0] * k

        def backtrack(index: int) -> bool:
            if index == len(nums):
                return True

            for i in range(k):
                if buckets[i] + nums[index] > target:
                    continue

                buckets[i] += nums[index]
                if backtrack(index + 1):
                    return True
                buckets[i] -= nums[index]

                # 🔴 Prune symmetric states
                if buckets[i] == 0:
                    break

            return False

        return backtrack(0)
```

---

## Example Explanation

### Input

```text
nums = [4,3,2,3,5,2,1]
k = 4
```

### Computation

```
total = 20
target = 5
```

Goal:

```
Subsets = [5], [4+1], [3+2], [3+2]
```

---

## Conceptual Backtracking Tree (COMPLETE STRUCTURE)

> This is a **bucket-placement tree**, not a permutation tree.

### Numbers (sorted descending)

```
[5,4,3,3,2,2,1]
```

### Tree Structure (Conceptual)

```
Start
                                        |
                                   Place 5
                 -------------------------------------------------
                 |                 |                 |           |
              B0=5 ✓           B1=5 ✓           B2=5 ✓       B3=5 ✓
                 |
                             (all symmetric → only one explored)

                                  Place 4
                           ------------------------
                           |                      |
                        B1=4                  B0 full ✗
                           |
                                Place 3
                     -----------------------------
                     |              |            |
                  B2=3          B3=3          B1 overflow ✗
                     |
                           Place 3
                 -----------------------------
                 |              |            |
              B3=3          B2 overflow ✗   B1 overflow ✗
                 |
                         Place 2
             -----------------------------
             |              |            |
          B2=5 ✓        B3 overflow ✗   B1 overflow ✗
             |
                       Place 2
         -----------------------------
         |              |            |
      B3=5 ✓        B2 full ✗     B1 overflow ✗
         |
                     Place 1
              -------------------------
              |          |            |
          B1=5 ✓     others full ✗   done
```

---

## How to Read This Tree

* Each **level** places one number
* Each **branch** represents choosing a bucket
* `✓` indicates bucket reaches `target`
* `✗` indicates pruning (overflow or symmetry)
* Only **one symmetric empty bucket branch** is explored

---

## Why Symmetry Pruning Works

If we try to place a number in an **empty bucket** and it fails later:

* Trying it in another empty bucket leads to the **same state**
* Hence, we break when `buckets[i] == 0`

---

## Complexity Analysis

* **Time Complexity:** `O(k^n)` worst case (n ≤ 16)
* **Space Complexity:** `O(k)` (bucket array + recursion)

---

## Pattern Recognition

This problem exemplifies the **Bucket Filling Backtracking Pattern**:

* Equal capacity constraints
* Symmetry pruning
* Descending sort for early failure

---

### One-Line Interview Explanation

> “We sort the numbers in descending order and try to place each number into one of k buckets whose sum does not exceed the target, pruning symmetric and overflow states.”