---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 738: Monotone Increasing Digits"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 738: Monotone Increasing Digits

Below is a complete, structured treatment of **LeetCode 738 – Monotone Increasing Digits**, aligned with standard DSA interview expectations.

---

## 1. Problem Statement

**LeetCode 738 – Monotone Increasing Digits**

Given a non-negative integer `n`, return the **largest number less than or equal to `n`** such that its digits are **monotone increasing**.

A number is said to have *monotone increasing digits* if and only if each digit is **less than or equal to the digit to its right**.

### Examples

* `1234` → monotone increasing
* `112233` → monotone increasing
* `132` → **not** monotone increasing (3 > 2)

---

## 2. Key Observation

### Core Insight

If a number violates the monotone condition at some position `i`:

```
digit[i] > digit[i + 1]
```

then:

1. We must **decrease digit[i] by 1**
2. All digits **to the right** of `i` must be set to `9` to maximize the number while keeping it ≤ `n`

### Why setting to `9` works

Once a digit is decreased, the best way to keep the number as large as possible (while remaining monotone) is to make all following digits as large as allowed, i.e., `9`.

---

## 3. Greedy Strategy (Why Greedy Works)

This is a **digit greedy problem**.

### Greedy Choice

* Fix the **leftmost violation** by decreasing the offending digit.
* Propagate the correction **backward if needed**.
* Fill the suffix with `9`s.

### Why backward propagation is required

Decreasing a digit may create a new violation with its left neighbor.

Example:

```
332 → decrease 3 → 322 → still invalid → decrease again → 299
```

Hence, we scan **right to left**.

---

## 4. Greedy Algorithm (Step-by-Step)

1. Convert `n` into a list of digits.
2. Traverse from **right to left**.
3. If `digits[i] < digits[i - 1]`:

   * Decrease `digits[i - 1]` by `1`
   * Mark position `i` as the start of suffix replacement
4. After traversal, set all digits from the marked position to the end as `9`
5. Convert digits back to integer

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        digits: List[int] = list(map(int, str(n)))
        length: int = len(digits)
        
        # Marker where digits should be set to 9
        marker: int = length
        
        # Traverse from right to left
        for i in range(length - 1, 0, -1):
            if digits[i] < digits[i - 1]:
                digits[i - 1] -= 1
                marker = i
        
        # Set all digits after marker to 9
        for i in range(marker, length):
            digits[i] = 9
        
        return int("".join(map(str, digits)))
```

---

## 6. Complete Worked Example (All Processing Steps)

### Input

```
n = 332
```

### Step 1: Convert to digits

```
digits = [3, 3, 2]
```

---

### Step 2: Right-to-left traversal

#### i = 2

```
digits[2] < digits[1]
2 < 3 → violation
```

Action:

```
digits[1] -= 1 → [3, 2, 2]
marker = 2
```

---

#### i = 1

```
digits[1] < digits[0]
2 < 3 → violation
```

Action:

```
digits[0] -= 1 → [2, 2, 2]
marker = 1
```

---

### Step 3: Replace suffix with 9s

From `marker = 1` onward:

```
digits = [2, 9, 9]
```

---

### Step 4: Final result

```
299
```

✔ Largest monotone increasing number ≤ 332

---

## 7. Another Example (No Backward Cascade)

### Input

```
n = 1234
```

### Processing

* No violations found
* No digit changes

### Output

```
1234
```

---

## 8. Time and Space Complexity

### Time Complexity

* **O(d)** where `d` is the number of digits in `n`

### Space Complexity

* **O(d)** for digit list

---

## 9. Common Pitfalls

1. **Left-to-right scanning fails**

   * Cannot correctly handle cascading violations
2. **Forgetting backward propagation**

   * Leads to invalid monotone sequence
3. **Not maximizing suffix**

   * Failing to set suffix digits to `9` gives suboptimal results

---

## 10. Pattern Classification

* **Greedy**
* **Digit Manipulation**
* **Monotonic Property Enforcement**

This problem is a canonical example of **digit greedy with backward correction**, commonly tested in system design-adjacent coding interviews.

---

If you want, I can also provide:

* A **backtracking-tree-style visualization** (conceptual)
* A **comparison with brute-force**
* Variants and similar problems (e.g., LeetCode 670, 402)