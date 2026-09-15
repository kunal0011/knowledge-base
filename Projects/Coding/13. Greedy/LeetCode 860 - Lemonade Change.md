---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 860: Lemonade Change"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 860: Lemonade Change

Below is a complete, structured explanation for **LeetCode 860 — Lemonade Change**, aligned with your usual learning format.

---

## LeetCode 860 — Lemonade Change

### Problem Statement

At a lemonade stand, each lemonade costs **$5**.

You are given an integer array `bills`, where `bills[i]` is the bill the *i-th* customer pays with.  
Customers are served **in order**, and each customer buys exactly **one** lemonade.

Accepted bill denominations are **$5, $10, and $20**.

You must provide **correct change** to each customer at the time of purchase.  
Initially, you have **no money**.

Return `True` if you can provide change to every customer, otherwise return `False`.

---

### Key Observations

1. **Sequential constraint**

   * You must process customers **left to right**.
   * You cannot reorder customers or delay change.
2. **Limited denominations**

   * Only `$5`, `$10`, `$20` are involved.
   * Change needed is deterministic:

     * `$5` → no change
     * `$10` → need `$5`
     * `$20` → need `$15`
3. **No need to track total money**

   * Only the **count of $5 and $10 bills** matters.
   * `$20` bills are never useful for future change.
4. **Greedy choice matters**

   * When giving `$15` change:

     * Prefer **$10 + $5** over **$5 + $5 + $5**
     * Reason: `$5` bills are more flexible and critical for future `$10` payments.

---

### Greedy Strategy (Core Trick)

Maintain:

* `five`: count of `$5` bills
* `ten`: count of `$10` bills

Process each bill:

| Customer Pays | Action |
| --- | --- |
| `$5` | Increment `five` |
| `$10` | Must have `five >= 1` → decrement `five`, increment `ten` |
| `$20` | Prefer `ten >= 1 and five >= 1` → decrement both |
|  | Else if `five >= 3` → decrement `five` by 3 |
|  | Else → return `False` |

This greedy ordering ensures future feasibility.

---

### Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five = 0
        ten = 0

        for bill in bills:
            if bill == 5:
                five += 1

            elif bill == 10:
                if five == 0:
                    return False
                five -= 1
                ten += 1

            else:  # bill == 20
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False

        return True
```

---

### Complete Worked Example (Step-by-Step)

#### Example Input

```
bills = [5, 5, 5, 10, 20]
```

#### Initial State

```
five = 0
ten = 0
```

---

#### Customer 1 pays `$5`

* No change needed
* Collect `$5`

```
five = 1
ten = 0
```

---

#### Customer 2 pays `$5`

* No change needed
* Collect `$5`

```
five = 2
ten = 0
```

---

#### Customer 3 pays `$5`

* No change needed
* Collect `$5`

```
five = 3
ten = 0
```

---

#### Customer 4 pays `$10`

* Needs `$5` change
* Use one `$5`

```
five = 2
ten = 1
```

---

#### Customer 5 pays `$20`

* Needs `$15` change
* Prefer `$10 + $5`

```
five = 1
ten = 0
```

---

#### All customers served successfully

```
Return True
```

---

### Failure Case Example

#### Input

```
bills = [5, 10, 20]
```

#### Processing

* `$5` → OK (`five = 1`)
* `$10` → give `$5` (`five = 0`, `ten = 1`)
* `$20` → need `$15`

  * No `$5` available
  * Cannot give change

```
Return False
```

---

### Why This Is a Greedy Problem

* Each decision impacts future feasibility.
* Choosing `$10 + $5` over `$5 + $5 + $5` preserves **critical low-value bills**.
* Local optimal choices lead to a global solution.

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)` (constant counters)

---

If you want, I can also provide:

* A **counterexample showing why wrong greedy choices fail**
* A **state-transition table**
* Or a **visual greedy-decision tree** for interviews