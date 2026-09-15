---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 179: Largest Number"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 179: Largest Number

**LeetCode 179 – Largest Number**, structured exactly as requested.

---

## LeetCode 179 — Largest Number

---

## 1. Problem Statement

You are given an array of **non-negative integers** `nums`.

Your task is to arrange them such that they form the **largest possible number** and return it as a **string**.

### Constraints

* `1 <= nums.length <= 100`
* `0 <= nums[i] <= 10^9`

### Important

The result may be very large, so return it as a **string**, not an integer.

---

### Example

```text
Input:  nums = [3, 30, 34, 5, 9]
Output: "9534330"
```

---

## 2. Key Observation (Core Insight)

This is **not** a standard numeric sorting problem.

### Why normal sorting fails

If we sort numerically (descending):

```
[34, 30, 9, 5, 3]   ❌
```

Concatenation:

```
3430953   (not maximum)
```

But correct order:

```
[9, 5, 34, 3, 30]  ✅
```

Concatenation:

```
9534330
```

---

### Critical Insight

For **any two numbers `a` and `b`**, we must decide:

```
Should "a" come before "b" OR "b" before "a"?
```

The decision is based on **string concatenation**, not numeric value.

---

## 3. Greedy Decision Rule

For two numbers `a` and `b`:

* Convert to strings
* Compare:

  ```
  a + b  vs  b + a
  ```

### Rule

* If `a + b > b + a`, place `a` **before** `b`
* Else, place `b` **before** `a`

This greedy choice ensures the **locally optimal ordering** contributes to a **globally optimal result**.

---

### Example of Pairwise Decision

```
a = "3", b = "30"

a + b = "330"
b + a = "303"

"330" > "303"
⇒ "3" should come before "30"
```

---

## 4. Why Greedy Works Here

* The problem has **optimal substructure**
* Each local ordering decision maximizes the prefix
* String comparison ensures lexicographically largest number
* Sorting with this custom comparator yields the global maximum

This is a classic **custom sorting + greedy** problem.

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List
from functools import cmp_to_key

class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # Convert numbers to strings
        strs = list(map(str, nums))

        # Custom comparator
        def compare(a: str, b: str) -> int:
            if a + b > b + a:
                return -1   # a should come before b
            elif a + b < b + a:
                return 1    # b should come before a
            else:
                return 0

        # Sort using custom comparator
        strs.sort(key=cmp_to_key(compare))

        # Edge case: all zeros
        if strs[0] == "0":
            return "0"

        return "".join(strs)
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```text
nums = [3, 30, 34, 5, 9]
```

---

### Step 1: Convert to Strings

```
["3", "30", "34", "5", "9"]
```

---

### Step 2: Pairwise Comparisons (Key Decisions)

| a | b | a+b | b+a | Winner |
| --- | --- | --- | --- | --- |
| 3 | 30 | 330 | 303 | 3 |
| 34 | 3 | 343 | 334 | 34 |
| 5 | 34 | 534 | 345 | 5 |
| 9 | 5 | 95 | 59 | 9 |
| 9 | 34 | 934 | 349 | 9 |

---

### Step 3: Sorted Order After Applying Comparator

```
["9", "5", "34", "3", "30"]
```

---

### Step 4: Concatenate

```
"9" + "5" + "34" + "3" + "30"
= "9534330"
```

---

### Final Output

```
"9534330"
```

---

## 7. Edge Case Explanation (All Zeros)

### Input

```
[0, 0, 0]
```

After sorting:

```
["0", "0", "0"]
```

Naive join:

```
"000"
```

Correct output:

```
"0"
```

Handled via:

```
if strs[0] == "0":
    return "0"
```

---

## 8. Time and Space Complexity

### Time Complexity

* Sorting with custom comparator:

  ```
  O(n log n * k)
  ```

  where `k` = max digits per number (≤ 10)

### Space Complexity

```
O(n)
```

(for string conversion)

---

## 9. Pattern Recognition

This problem belongs to:

* **Greedy + Custom Sorting**
* **String comparison trick**
* Common in **system design style ordering problems**

---

If you want, I can next:

* Prove correctness formally
* Show why numeric sort fails with counterexamples
* Compare this with similar problems (e.g., arranging strings lexicographically)
* Convert this into a reusable comparator pattern for other problems