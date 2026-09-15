---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 638: Shopping Offers"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 638: Shopping Offers

## LeetCode 638 — Shopping Offers

---

### Problem Statement

You are given:

* `price[i]`: price of the `i`-th item
* `special[j]`: a special bundle offer where  
  `special[j][i]` = quantity of item `i` in the bundle, and  
  `special[j][-1]` = total price of the bundle
* `needs[i]`: number of units of item `i` you need

Return the **minimum cost** to satisfy all shopping needs.

**Rules**

* You may use any special offer **any number of times**
* You may also buy items individually
* You cannot buy more than what you need

---

### Example

```text
price   = [2, 5]
special = [[3, 0, 5], [1, 2, 10]]
needs   = [3, 2]

Output: 14
```

---

## Key Observations (Critical)

1. This is **not** a greedy problem:

   * A cheap-looking offer may block better future combinations.
2. The state is fully defined by the **remaining needs vector**.
3. Buying items individually is always a valid fallback.
4. This is a **DFS / backtracking with memoization** problem:

   * Backtracking explores combinations of offers
   * Memoization avoids recomputing the same `needs` state
5. The recursion tree branches on:

   * Applying each valid special offer
   * Or stopping and buying remaining items directly

---

## Core Idea

At each step:

* Compute cost of buying remaining items **individually**
* Try applying each special offer that does **not exceed needs**
* Recurse on the reduced needs
* Take the minimum

---

## Python 3 Solution (with Typing)

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
        def dfs(state: Tuple[int, ...]) -> int:
            # Cost without any special offers
            min_cost = sum(state[i] * price[i] for i in range(n))

            for offer in special:
                new_state = []
                for i in range(n):
                    if offer[i] > state[i]:
                        break
                    new_state.append(state[i] - offer[i])
                else:
                    # Offer is valid
                    min_cost = min(
                        min_cost,
                        offer[-1] + dfs(tuple(new_state))
                    )

            return min_cost

        return dfs(tuple(needs))
```

---

## Example Explanation (Step-by-Step)

### Input

```
price   = [2, 5]
special = [[3,0,5], [1,2,10]]
needs   = [3,2]
```

### Options at Root

1. Buy everything individually:

```
3×2 + 2×5 = 16
```

2. Apply offer `[3,0,5]`:

```
Remaining needs = [0,2]
Cost = 5 + (0×2 + 2×5) = 15
```

3. Apply offer `[1,2,10]`:

```
Remaining needs = [2,0]
Cost = 10 + (2×2) = 14  ← minimum
```

---

## Backtracking Tree Structure

(**Complete Conceptual Tree — State Navigation View**)

![https://i.ytimg.com/vi/9R3aXaM8McA/maxresdefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/9R3aXaM8McA/maxresdefault.jpg?utm_source=chatgpt.com)

![https://i.sstatic.net/cNmhS.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/EMKFvXicu5mVUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw4MdwwPisi1rDKOKstNLzUyzjdJ9LD080or9LEI0o2KqPJODnLz8dHNL49wKTAu0rU0UysGAHW2JaM?utm_source=chatgpt.com)

![https://inventwithpython.com/recursion/images/000026.webp?utm_source=chatgpt.com](https://inventwithpython.com/recursion/images/000026.webp?utm_source=chatgpt.com)

### Conceptual Tree (State-Based)

```
needs = (3,2)
                    /                 |                  \
           no offer (16)        offer A [3,0,5]      offer B [1,2,10]
                                   |                     |
                             needs = (0,2)          needs = (2,0)
                               |                       |
                      no offer (10)              no offer (4)
                               |                       |
                         total = 15                total = 14  ✅
```

---

## How to Read This Tree

* **Each node** = remaining `needs`
* **Edges** = applying one special offer
* **Leaf cost** = buying remaining items individually
* DFS explores all valid branches
* Memoization ensures each `(needs)` state is computed once

---

## Why Memoization Is Essential

Without caching:

* Same `needs` state can be reached via different paths
* Exponential recomputation

With memoization:

* Time complexity becomes manageable
* State space is bounded by product of `needs[i]`

---

## Complexity Analysis

* **State Count:** `∏ (needs[i] + 1)`
* **Time Complexity:** proportional to number of valid states × offers
* **Space Complexity:** memo table size + recursion depth

---

## Pattern Recognition

LeetCode 638 is a **multi-dimensional knapsack with DFS + memo**:

* State compression via tuple
* Try-or-skip decisions
* Cost minimization

---

### One-Line Interview Explanation

> “We perform DFS over remaining needs, trying each valid offer and comparing it with direct purchase cost, while memoizing states to avoid recomputation.”

If you want next, I can:

* Show **what breaks without memoization**
* Compare **LC 638 vs LC 39 / 40**
* Convert this into a **generic state-DP template**