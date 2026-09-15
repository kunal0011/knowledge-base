---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 633: Sum of Square Numbers"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 633: Sum of Square Numbers

Below is a complete, structured explanation of **LeetCode 633 – Sum of Square Numbers**, aligned with your requested format.

---

## 📘 Problem Statement (LeetCode 633)

Given a **non-negative integer `c`**, determine whether there exist **two integers `a` and `b`** such that:

[  
a^2 + b^2 = c  
]

Return `True` if such integers exist, otherwise return `False`.

---

## 🔍 Key Observation

1. The equation involves **squares**, which are always **non-negative**.
2. We only need to consider values of `a` and `b` in the range:  
   [  
   0 \le a, b \le \lfloor \sqrt{c} \rfloor  
   ]
3. Brute force checking all `(a, b)` pairs is **O(n²)** and inefficient.
4. Since squares grow monotonically, we can leverage a **two-pointer technique** on the number line `[0 … √c]`.

---

## 👉 Two Pointer Technique (Core Idea)

### Pointer Setup

* `left = 0`
* `right = floor(sqrt(c))`

### Invariant

* We maintain:  
  [  
  current = left^2 + right^2  
  ]

### Pointer Movement Logic

* If `current == c` → ✅ Solution found
* If `current < c` → Increase `left` (to increase sum)
* If `current > c` → Decrease `right` (to decrease sum)

This works because:

* Increasing `left` strictly increases `left²`
* Decreasing `right` strictly decreases `right²`

### Complexity

* **Time:** `O(√c)`
* **Space:** `O(1)`

---

## 🧠 Algorithm Steps

1. Initialize `left = 0`
2. Initialize `right = int(sqrt(c))`
3. While `left <= right`:

   * Compute `left² + right²`
   * Adjust pointers based on comparison with `c`
4. If loop ends without a match, return `False`

---

## 🧪 Worked-Out Example

### Input

```
c = 5
```

### Iterations

| left | right | left² | right² | sum | Action |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | 0 | 4 | 4 | sum < 5 → left++ |
| 1 | 2 | 1 | 4 | 5 | sum == 5 → return True |

### Result

```
True (1² + 2² = 5)
```

---

## 🧩 Python 3 Solution (with Typing)

```python
from typing import *

class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        left: int = 0
        right: int = int(c ** 0.5)

        while left <= right:
            current_sum: int = left * left + right * right

            if current_sum == c:
                return True
            elif current_sum < c:
                left += 1
            else:
                right -= 1

        return False
```

---

## ✅ Why This Approach Is Optimal

* Avoids nested loops
* Uses mathematical monotonicity
* Runs efficiently even for large values of `c` (up to `2³¹ - 1`)

---

If you want, I can also:

* Derive this from a **binary search** perspective
* Compare with **hash set**–based solutions
* Explain why negative values are irrelevant mathematically

Just let me know.