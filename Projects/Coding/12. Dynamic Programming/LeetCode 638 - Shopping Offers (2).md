---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 638: Shopping Offers"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 638: Shopping Offers

**LeetCode 638 – Shopping Offers**, with explicit **state definition, transition, DP table (memo) creation**, and a **worked example**.

---

## Problem Statement (Condensed)

You are given:

* `price[i]`: price of the *i-th* item
* `special[j]`: an offer where `special[j][i]` is quantity of item *i* and `special[j][-1]` is total offer price
* `needs[i]`: how many units of item *i* you need

You may use any offer any number of times **as long as it does not exceed needs**.

Return the **minimum total cost** to satisfy exactly `needs`.

---

## Key Observation

* The order of buying items **does not matter**
* The remaining `needs` uniquely define the subproblem
* This is a **multi-dimensional unbounded knapsack**, best solved with **DP + memoization**

---

## DP State Definition

Let:

```
dp(needs) = minimum cost to satisfy exactly these remaining needs
```

Where:

* `needs` is a **tuple** of remaining quantities (hashable → memo key)

Example:

```
needs = (3, 2)
```

---

## Base Case

If all needs are zero:

```
dp((0, 0, ..., 0)) = 0
```

---

## Transition

From a given `needs`:

### Option 1: Buy remaining items individually

```
cost = sum(needs[i] * price[i])
```

### Option 2: Apply each valid special offer

For an offer `offer`:

* It is valid if for all `i`:

```
offer[i] <= needs[i]
```

* New state:

```
new_needs[i] = needs[i] - offer[i]
```

* Transition:

```
dp(needs) = min(
    dp(needs),
    offer_price + dp(new_needs)
)
```

---

## DP Recurrence

```
dp(needs) =
    min(
        sum(needs[i] * price[i]),
        min over all valid offers:
            offer_price + dp(needs - offer)
    )
```

---

## Why Memoization Is Required

* `needs` can repeat through different paths
* Total states ≤ `(needs[0]+1) * (needs[1]+1) * ...`
* Memo avoids exponential recomputation

---

## Python 3 Solution (Top-Down DP)

```python
from typing import List, Tuple
from functools import lru_cache

class Solution:
    def shoppingOffers(
        self,
        price: List[int],
        special: List[List[int]],
        needs: List[int]
    ) -> int:

        n = len(price)

        @lru_cache(None)
        def dp(cur_needs: Tuple[int, ...]) -> int:
            # Cost without using any offer
            min_cost = sum(cur_needs[i] * price[i] for i in range(n))

            # Try each special offer
            for offer in special:
                new_needs = []
                for i in range(n):
                    if offer[i] > cur_needs[i]:
                        break
                    new_needs.append(cur_needs[i] - offer[i])
                else:
                    # Offer is valid
                    min_cost = min(
                        min_cost,
                        offer[-1] + dp(tuple(new_needs))
                    )

            return min_cost

        return dp(tuple(needs))
```

---

## Example Walkthrough

### Input

```
price   = [2, 5]
special = [[3, 0, 5], [1, 2, 10]]
needs   = [3, 2]
```

---

### Step 1: Initial State

```
dp((3, 2))
```

Individual purchase:

```
3*2 + 2*5 = 6 + 10 = 16
```

---

### Step 2: Try Offers

#### Offer [3, 0, 5]

Valid since:

```
3 ≤ 3, 0 ≤ 2
```

New state:

```
(3-3, 2-0) = (0, 2)
```

Cost:

```
5 + dp((0, 2))
```

Compute:

```
dp((0,2)) = 2 * 5 = 10
```

Total:

```
5 + 10 = 15
```

---

#### Offer [1, 2, 10]

Valid since:

```
1 ≤ 3, 2 ≤ 2
```

New state:

```
(2, 0)
```

Cost:

```
10 + dp((2, 0))
```

Compute:

```
dp((2,0)) = 2 * 2 = 4
```

Total:

```
10 + 4 = 14
```

---

### Step 3: Choose Minimum

```
dp((3,2)) = min(16, 15, 14) = 14
```

---

## DP Table (Memo) Created During Execution

| State (needs) | dp value |
| --- | --- |
| (0, 0) | 0 |
| (0, 2) | 10 |
| (2, 0) | 4 |
| (3, 2) | **14** |

---

## Time & Space Complexity

Let:

```
N = number of item types
M = max quantity per item
```

* **States:** O((M+1)^N)
* **Transitions per state:** O(#special \* N)
* **Time:** Acceptable due to small constraints (LeetCode limits)
* **Space:** O((M+1)^N) for memo

---

## Why This DP Is Correct

* Optimal substructure: optimal cost for `needs` depends on optimal cost of smaller `needs`
* Overlapping subproblems: same `needs` reachable via different paths
* Memoization guarantees efficiency

---

If you want, I can also provide:

* Bottom-up DP (less common for this problem)
* State pruning optimization
* Visual recursion tree for this example
* Comparison with knapsack formulation

Tell me how deep you want to go.