---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 402: Remove K Digits"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 402: Remove K Digits

Below is a structured, interview-grade explanation of **LeetCode 402 – Remove K Digits**, aligned with how this problem is typically evaluated in greedy/stack categories.

---

## LeetCode 402: Remove K Digits

### Problem Statement

You are given a string `num` representing a **non-negative integer**, and an integer `k`.  
Remove exactly `k` digits from `num` so that the resulting number is the **smallest possible**.

**Constraints**

* `1 ≤ len(num) ≤ 10^5`
* `0 ≤ k ≤ len(num)`
* `num` contains only digits

**Output**

* Return the smallest possible number as a string.
* If all digits are removed, return `"0"`.

---

## Key Observation (Greedy Insight)

To minimize the number:

* **More significant digits matter more** than less significant ones.
* If a digit on the left is **greater than a digit on the right**, removing the left digit will always produce a smaller number.

This naturally leads to a **monotonic increasing stack** strategy.

---

## Stack Key Insight

Maintain a stack such that:

* Digits in the stack are in **non-decreasing order** (left → right).
* While:

  * the stack is not empty,
  * `k > 0`,
  * and the **top of the stack is greater than the current digit**,

  → pop from the stack (remove that digit).

This ensures:

* Larger digits to the left are removed first.
* The resulting number is lexicographically smallest.

### Post-processing

1. If `k > 0`, remove digits from the **end** (least significant).
2. Strip leading zeros.
3. If empty → return `"0"`.

---

## Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack: List[str] = []

        for digit in num:
            while stack and k > 0 and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)

        # If k remains, remove from the end
        while k > 0:
            stack.pop()
            k -= 1

        # Remove leading zeros
        result = ''.join(stack).lstrip('0')

        return result if result else "0"
```

---

## Worked-Out Example

### Example 1

**Input**

```
num = "1432219"
k = 3
```

### Step-by-Step Stack Processing

| Digit | Stack Before | Action | Stack After | k |
| --- | --- | --- | --- | --- |
| 1 | [] | push | [1] | 3 |
| 4 | [1] | push | [1,4] | 3 |
| 3 | [1,4] | pop 4 | [1] | 2 |
|  | [1] | push | [1,3] | 2 |
| 2 | [1,3] | pop 3 | [1] | 1 |
|  | [1] | push | [1,2] | 1 |
| 2 | [1,2] | push | [1,2,2] | 1 |
| 1 | [1,2,2] | pop 2 | [1,2] | 0 |
|  | [1,2] | push | [1,2,1] | 0 |
| 9 | [1,2,1] | push | [1,2,1,9] | 0 |

### Final Stack

```
[1, 2, 1, 9]
```

### Result

```
"1219"
```

---

## Edge Case Example

### Input

```
num = "10200"
k = 1
```

### Process

* Remove `'1'` (greater than `'0'`)
* Result becomes `"0200"`
* Strip leading zeros

### Output

```
"200"
```

---

## Complexity Analysis

* **Time Complexity:** `O(n)`  
  Each digit is pushed and popped at most once.
* **Space Complexity:** `O(n)`  
  Stack storage.

---

## Classification (LeetCode Pattern)

* Greedy
* Monotonic Stack
* Digit removal / lexicographical minimization

---

If you want, I can also:

* Explain **why removing from the end is optimal when k remains**
* Provide a **visual stack diagram**
* Compare this with **LC 316 / LC 1081 (remove duplicate letters)** patterns