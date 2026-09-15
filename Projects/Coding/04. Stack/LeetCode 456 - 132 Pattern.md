---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 456: 132 Pattern"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 456: 132 Pattern

Below is a complete, interview-ready explanation of **LeetCode 456 – 132 Pattern**, structured the way you requested.

---

## LeetCode 456 — 132 Pattern

### Problem Statement

Given an array of integers `nums`, determine whether there exists a **132 pattern** in the array.

A **132 pattern** is a subsequence of three integers  
`nums[i]`, `nums[j]`, `nums[k]` such that:

* `i < j < k`
* `nums[i] < nums[k] < nums[j]`

Return `True` if such a pattern exists; otherwise, return `False`.

---

### Key Observation

A brute-force approach would try all triples `(i, j, k)` → **O(n³)**, which is infeasible.

Rewriting the condition:

* `nums[j]` is the **largest** element (the `3`)
* `nums[k]` is the **middle** element (the `2`)
* `nums[i]` is the **smallest** element (the `1`)

So the problem reduces to:

> Can we find a value `nums[k]` that lies **between** a smaller value on its left and a larger value on its left?

---

### Stack Key Insight (Core Idea)

We solve this in **O(n)** using a **monotonic decreasing stack**, scanning from **right to left**.

#### Why right to left?

Because when we scan from the end:

* We can treat the current element as a potential `nums[i]` (the `1`)
* The stack holds candidates for `nums[j]` (the `3`)
* A variable tracks the best candidate for `nums[k]` (the `2`)

---

### Stack Invariant

* Stack maintains values in **strictly decreasing order**
* A variable `third` stores the **largest valid `nums[k]`** seen so far

---

### Algorithm Steps

1. Initialize an empty stack
2. Initialize `third = -∞`
3. Traverse `nums` from right to left:

   * If `nums[i] < third`, we found `nums[i] < nums[k] < nums[j]` → return `True`
   * While stack is not empty and `nums[i] > stack[-1]`:

     * Pop stack and update `third`
   * Push `nums[i]` onto the stack
4. If traversal finishes, return `False`

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        stack: List[int] = []
        third = float('-inf')  # candidate for nums[k]

        # Traverse from right to left
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] < third:
                return True

            while stack and nums[i] > stack[-1]:
                third = stack.pop()

            stack.append(nums[i])

        return False
```

---

### Worked-Out Example

#### Input

```text
nums = [3, 1, 4, 2]
```

#### Expected Output

```
True
```

---

### Step-by-Step Execution

| Index | Value | Stack | third | Explanation |
| --- | --- | --- | --- | --- |
| 3 | 2 | [] → [2] | -∞ | Push |
| 2 | 4 | [2] → [] → [4] | 2 | Pop 2 → update `third = 2` |
| 1 | 1 | [4] | 2 | `1 < 2` → **132 found** |
|  |  |  |  | Pattern: `1 (i), 4 (j), 2 (k)` |

Condition satisfied:

```
1 < 2 < 4 and i < j < k
```

---

### Why This Works

* Stack guarantees we only keep valid `nums[j]` candidates
* `third` tracks the best possible `nums[k]`
* Single pass, no nested loops

---

### Complexity Analysis

* **Time:** O(n)
* **Space:** O(n) (stack)

---

### Mental Model (Interview Tip)

Think of it as:

* Stack → possible **peaks (`3`)**
* `third` → best **valley-middle (`2`)**
* Current element → checking for **lowest (`1`)**

If you want, I can also:

* Draw the stack evolution as a tree/diagram
* Show why left-to-right fails
* Compare with brute force and prefix-min approaches