---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 330: Patching Array"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 330: Patching Array

**LeetCode 330 – Patching Array**, exactly in the format you requested.

---

## LeetCode 330 — Patching Array

### Problem Statement

You are given a **sorted** integer array `nums` of **positive integers** and an integer `n`.

You can **add (patch)** any positive integer to `nums`.

Your goal is to **add the minimum number of patches** such that **every number in the range `[1, n]`** can be formed as the **sum of some subset** of the array elements.

Return the **minimum number of patches required**.

---

### Key Observation (Core Insight)

This problem is **not** about subset DP.

It is about **range coverage**.

#### Fundamental Greedy Invariant

If you can currently form **all numbers in the range**:

```
[1, miss - 1]
```

Then:

• If the next number in `nums` is `<= miss`,  
you can **extend coverage** to:

```
[1, miss + nums[i] - 1]
```

• If the next number is `> miss`,  
then **miss itself cannot be formed**, so you **must patch** `miss`.

Adding `miss` is optimal because it **maximally expands coverage**.

---

### Why Patching with `miss` Is Optimal

If current coverage is:

```
[1, miss - 1]
```

Adding `miss` gives:

```
New coverage = [1, 2*miss - 1]
```

This is the **largest possible expansion** using one number.

Any smaller number adds less coverage.  
Any larger number still leaves `miss` uncovered.

Hence:

> **Always patch with `miss`**

---

### Greedy Strategy Summary

Maintain:

* `miss`: the **smallest number we cannot form**
* `i`: index in `nums`
* `patches`: count of added numbers

Algorithm:

1. Start with `miss = 1`
2. While `miss <= n`:

   * If `nums[i] <= miss`: use it → `miss += nums[i]`
   * Else: patch `miss` → `miss *= 2`, increment patches

---

### Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        miss = 1        # smallest number we cannot form
        patches = 0
        i = 0

        while miss <= n:
            if i < len(nums) and nums[i] <= miss:
                miss += nums[i]
                i += 1
            else:
                # patch with miss
                miss += miss
                patches += 1

        return patches
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```text
nums = [1, 3]
n = 6
```

---

### Initial State

| Variable | Value |
| --- | --- |
| miss | 1 |
| coverage | [ ] |
| i | 0 |
| patches | 0 |

---

### Step 1

`nums[i] = 1 <= miss (1)` → use it

```
miss = miss + 1 = 2
coverage = [1]
```

| miss | coverage |
| --- | --- |
| 2 | [1] |

---

### Step 2

`nums[i] = 3 > miss (2)` → cannot form `2`

#### Patch with `miss = 2`

```
miss = 2 + 2 = 4
patches = 1
coverage = [1, 2, 3]
```

---

### Step 3

`nums[i] = 3 <= miss (4)` → use it

```
miss = 4 + 3 = 7
coverage = [1..6]
```

---

### Termination

`miss = 7 > n (6)` → done

---

### Final Answer

```
Minimum patches required = 1
```

---

### Coverage Growth Visualization

| Action | Added | Coverage After |
| --- | --- | --- |
| use 1 | 1 | [1] |
| patch | 2 | [1,2,3] |
| use 3 | 3 | [1..6] |

---

### Why This Is Greedy (and Correct)

• Always fixes the **smallest missing sum**  
• Maximizes range growth per operation  
• Guarantees minimal patches  
• Runs in **O(len(nums)) time**  
• Constant space

---

### Key Takeaways (Interview Ready)

* Think in terms of **coverage**, not subsets
* `miss` tracks the **first gap**
* Patching with `miss` **doubles coverage**
* Greedy works because **local optimal = global optimal**

---

If you want, next I can:

* Draw a **conceptual coverage expansion diagram**
* Compare this with **DP and explain why DP fails**
* Provide **edge case walkthroughs** (empty array, large `n`)