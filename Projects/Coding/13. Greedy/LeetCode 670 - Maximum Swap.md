---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 670: Maximum Swap"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 670: Maximum Swap

Below is a structured, interview-grade explanation of **LeetCode 670 – Maximum Swap**, aligned with your preference for clarity, state reasoning, and step-by-step processing.

---

## LeetCode 670 — Maximum Swap

---

### Problem Statement

You are given a **non-negative integer** `num`.  
You are allowed to **swap at most two digits** in the number **at most once**.

Return the **maximum possible value** of the number after performing **at most one swap**.

**Constraints**

* `0 ≤ num ≤ 10⁸`

---

### Key Observation

1. The most significant digit (leftmost) contributes **more** to the value than any digit to its right.
2. To maximize the number using **one swap**:

   * We want to **increase the leftmost digit possible**.
   * That means swapping it with a **larger digit appearing later**.
3. Among multiple candidates:

   * Choose the **rightmost occurrence** of the **largest possible digit** to swap.
   * This preserves maximum gain for future digits.

---

### Greedy Insight (Core Trick)

> For each digit, check whether there exists a **larger digit to its right**.  
> If yes, swap it with the **rightmost occurrence of the maximum digit** on the right.

Why rightmost?

* Swapping with the rightmost keeps larger digits earlier intact, producing a higher number.

---

### Greedy Strategy (High-Level)

1. Convert the number into a list of digits.
2. Track the **last index** of every digit (`0–9`).
3. Traverse digits from **left to right**:

   * For current digit `d`, check digits `9 → d+1`
   * If a higher digit exists **to the right**, perform the swap and stop.

Only **one swap** is allowed.

---

### Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def maximumSwap(self, num: int) -> int:
        digits: List[str] = list(str(num))
        
        # Step 1: Record last occurrence of each digit
        last_index = {}
        for i, d in enumerate(digits):
            last_index[d] = i
        
        # Step 2: Try to make the number larger from left to right
        for i, d in enumerate(digits):
            # Check if a larger digit exists later
            for bigger in map(str, range(9, int(d), -1)):
                if bigger in last_index and last_index[bigger] > i:
                    # Perform swap
                    j = last_index[bigger]
                    digits[i], digits[j] = digits[j], digits[i]
                    return int("".join(digits))
        
        return num
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```
num = 2736
```

---

#### Step 1: Convert to Digits

```
digits = ['2', '7', '3', '6']
```

---

#### Step 2: Track Last Occurrence

```
digit : last index
'2' → 0
'7' → 1
'3' → 2
'6' → 3
```

---

#### Step 3: Greedy Traversal

##### Index 0 → digit '2'

Check digits greater than '2' (from '9' down):

| Candidate | Exists? | Index | Valid? |
| --- | --- | --- | --- |
| '9' | No | — | ❌ |
| '8' | No | — | ❌ |
| '7' | Yes | 1 | ✅ (1 > 0) |

➡ Swap `digits[0]` and `digits[1]`

---

#### Step 4: Swap Execution

```
Before: ['2', '7', '3', '6']
After : ['7', '2', '3', '6']
```

---

#### Step 5: Result

```
Output = 7236
```

---

### Why Only One Swap Is Enough?

Because:

* We greedily maximize the **most significant digit first**
* Any further swap would affect **less significant digits**, yielding a smaller improvement

---

### Edge Case Example

#### Input

```
num = 9973
```

Traversal finds **no larger digit to the right**, so:

```
Output = 9973
```

---

### Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n × 10)** → effectively **O(n)** |
| Space | **O(1)** (only digit mapping) |

---

### Interview Summary (One-Liner)

> This is a greedy problem where we maximize the leftmost digit using at most one swap by exchanging it with the rightmost occurrence of the largest possible digit to its right.

If you want, I can also:

* Draw a **greedy decision tree**
* Explain **why brute force is inferior**
* Compare this with **next permutation logic**
* Provide a **constant-space math-only version**

Tell me how deep you want to go.