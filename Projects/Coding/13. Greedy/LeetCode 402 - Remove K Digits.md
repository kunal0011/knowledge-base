---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 402: Remove K Digits"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 402: Remove K Digits

**LeetCode 402 – Remove K Digits**, structured exactly as requested.

---

## LeetCode 402 — Remove K Digits

### 1. Problem Statement

You are given a string `num` representing a **non-negative integer** and an integer `k`.

Remove exactly `k` digits from `num` so that the resulting number is the **smallest possible**.

Return the result as a string.  
If the resulting number has leading zeros, remove them.  
If all digits are removed, return `"0"`.

**Constraints**

* `1 ≤ len(num) ≤ 10^5`
* `0 ≤ k ≤ len(num)`
* `num` consists of digits only.

---

### 2. Key Observation (Core Insight)

To make a number **as small as possible**:

* **Higher digits on the left are more significant** than digits on the right.
* If a digit is **larger than a digit coming after it**, keeping it makes the number larger.

#### Crucial Greedy Insight

> If the current digit is **smaller than the previous digit**, we should remove the previous digit (if we still can).

This leads directly to a **monotonic increasing stack** strategy.

---

### 3. Greedy Strategy (Why It Works)

We want digits to be:

```
as small as possible
as far left as possible
```

So we:

1. Traverse digits from left to right.
2. Maintain a stack where digits are **in increasing order**.
3. If the current digit is smaller than the top of the stack:

   * Remove (pop) the stack top **if we still have removals left (`k > 0`)**.
4. Push the current digit.
5. If `k > 0` after traversal, remove digits from the **end** (least significant).

This guarantees:

* All removals are **locally optimal**
* Local optimality ⇒ global minimum (classic greedy correctness)

---

### 4. Greedy “Tricks” to Remember (Interview Notes)

* **Why stack?**  
  Because we need to compare with the immediate previous digit efficiently.
* **Why pop larger digits first?**  
  A larger digit on the left contributes more to value than any digit on the right.
* **Why remove from the end if `k` remains?**  
  The number is already increasing → last digits are the largest and least significant.
* **Leading zeros handling**  
  Strip leading zeros at the end; do not worry during processing.

---

### 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack: List[str] = []

        for digit in num:
            # Remove previous larger digits if possible
            while k > 0 and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)

        # If removals remain, remove from the end
        while k > 0:
            stack.pop()
            k -= 1

        # Build result and strip leading zeros
        result = ''.join(stack).lstrip('0')

        return result if result else "0"
```

**Time Complexity:** `O(n)`  
**Space Complexity:** `O(n)`

---

### 6. Complete Worked Example (Step-by-Step)

#### Input

```
num = "1432219"
k = 3
```

---

#### Step-by-Step Stack Processing

| Digit | Stack Before | Action | Stack After | k |
| --- | --- | --- | --- | --- |
| '1' | [] | push | ['1'] | 3 |
| '4' | ['1'] | push | ['1','4'] | 3 |
| '3' | ['1','4'] | pop '4' | ['1'] | 2 |
|  | ['1'] | push '3' | ['1','3'] | 2 |
| '2' | ['1','3'] | pop '3' | ['1'] | 1 |
|  | ['1'] | push '2' | ['1','2'] | 1 |
| '2' | ['1','2'] | push | ['1','2','2'] | 1 |
| '1' | ['1','2','2'] | pop '2' | ['1','2'] | 0 |
|  | ['1','2'] | push '1' | ['1','2','1'] | 0 |
| '9' | ['1','2','1'] | push | ['1','2','1','9'] | 0 |

---

#### Stack After Traversal

```
['1', '2', '1', '9']
```

#### k Remaining?

```
k = 0 → no further removals
```

---

#### Final Result

```
"1219"
```

---

### 7. Edge Case Walkthrough

#### Case 1

```
num = "10200", k = 1
```

Processing removes `'1'` → stack becomes `"0200"`  
After stripping leading zeros → `"200"`

#### Case 2

```
num = "10", k = 2
```

All digits removed → return `"0"`

---

### 8. Pattern Recognition (Very Important)

This problem belongs to the pattern:

> **Greedy + Monotonic Stack (Increasing)**

Closely related problems:

* Remove Duplicate Letters
* Make Lexicographically Smallest String
* Remove K Letters
* Next Smaller Element variants

---

### 9. Final Takeaway

* Always remove **larger digits before smaller future digits**
* Leftmost digit matters most
* Stack ensures `O(n)` optimal processing
* One of the **cleanest greedy problems** asked frequently in interviews

If you want, I can also:

* Prove correctness formally
* Show why DP is worse here
* Convert this into a reusable greedy template
* Compare with similar problems (402 vs 316 vs 1081)

Just say the word.