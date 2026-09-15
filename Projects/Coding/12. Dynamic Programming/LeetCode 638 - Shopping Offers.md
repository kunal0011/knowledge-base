---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 638: Shopping Offers"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - memoization
  - knapsack
  - google
  - amazon
---

# LeetCode 638: Shopping Offers

**Target Companies:** Google, Amazon, Microsoft  
**Difficulty:** Medium  
**Topic:** Multidimensional Unbounded Knapsack / Top-Down DP with Memoization  

---

### Problem Statement

In LeetCode Store, there are `n` items to sell. Each item has a price. However, there are some special offers, and a special offer consists of one or more different kinds of items with a sale price.

You are given an integer array `price` where `price[i]` is the price of the `i-th` item, and an integer array `needs` where `needs[i]` is the number of pieces of the `i-th` item you want to buy.

You are also given an array `special` where `special[i]` is of size `n + 1` where `special[i][j]` is the number of pieces of the `j-th` item in the `i-th` offer and `special[i][n]` (the last integer in the array) is the price of the `i-th` offer.

Return the **lowest price** you have to pay for exactly certain items as given, where you could make optimal use of the special offers. You are not allowed to buy more items than you want, even if that would lower the overall price. You could use any of the special offers as many times as you want.

---

### Input & Output Formats & Constraints

- **Input:**
  - `price: List[int]` — Price of each item.
  - `special: List[List[int]]` — List of offers; each offer contains item quantities and the package price at the end.
  - `needs: List[int]` — Target quantity of each item needed.
- **Output:**
  - `int` — Minimum cost to fulfill the exact needs.
- **Constraints:**
  - $n == \text{price.length} == \text{needs.length}$
  - $1 \le n \le 6$
  - $0 \le \text{price}[i] \le 10$
  - $0 \le \text{needs}[i] \le 10$
  - $1 \le \text{special.length} \le 100$
  - $\text{special}[i].\text{length} == n + 1$
  - $0 \le \text{special}[i][j] \le 50$

---

### Key Idea & Intuition

1. **Multidimensional State Space:**
   - There are $n \le 6$ items, and each item requirement is $\le 10$.
   - The state is completely captured by the remaining needs tuple:
     $$(k_0, k_1, \dots, k_{n-1}) \quad \text{where } 0 \le k_i \le \text{needs}[i]$$
   - The total number of distinct subproblem states is bounded by:
     $$\prod_{i=0}^{n-1} (\text{needs}[i] + 1) \le 11^6 \approx 1.77 \times 10^6$$
   - In practice, only a tiny fraction of these states are reachable via combinations of the given special offers. This makes **Top-Down Memoized DP** exceptionally fast and memory-efficient.

2. **Base Cost & Transitions:**
   - **Baseline (Buy Individually):** For any remaining needs state, we can always purchase all needed items at standard individual prices:
     $$\text{cost} = \sum_{i=0}^{n-1} \text{cur\_needs}[i] \times \text{price}[i]$$
   - **Offer Transitions:** For each special offer that does not exceed any of the current needs ($\forall i, \text{offer}[i] \le \text{cur\_needs}[i]$):
     $$\text{cost} = \min(\text{cost}, \ \text{offer\_price} + \text{dfs}(\text{cur\_needs} - \text{offer}))$$

3. **Offer Filtering (Pruning):**
   - Filter out invalid offers that provide zero items or cost more than purchasing their components individually:
     $$\sum_{i=0}^{n-1} \text{offer}[i] \times \text{price}[i] \le \text{offer\_price} \implies \text{Discard offer}$$

---

### Solution Approach (Step-by-Step)

1. **Filter Special Offers:**
   - Keep only offers that are strictly cheaper than buying the individual components and provide at least one item.
2. **Define Memoized DFS Function `dfs(needs_tuple)`:**
   - If `needs_tuple` is in memo, return cached result.
   - Compute baseline cost: $\sum_{i} \text{needs}[i] \times \text{price}[i]$.
   - For each valid offer:
     - Check if $\text{offer}[i] \le \text{needs}[i]$ for all $i$.
     - If valid:
       - Construct `new_needs = tuple(needs[i] - offer[i] for i in range(n))`.
       - Update `cost = min(cost, offer[-1] + dfs(new_needs))`.
   - Store and return `memo[needs_tuple] = cost`.
3. **Return:**
   - Return `dfs(tuple(needs))`.

---

### Visual Algorithm Walkthrough

`price = [2, 5]`, `special = [[3, 0, 5], [1, 2, 10]]`, `needs = [3, 2]`

```
State (3, 2):
  Baseline Cost (no offers): 3*2 + 2*5 = 6 + 10 = 16

  Branch 1: Try Offer [3, 0, 5] (Cost: 5):
    Can we take it? 3 <= 3 and 0 <= 2 (Yes!)
    New needs = (3-3, 2-0) = (0, 2)
    Evaluate (0, 2):
      Baseline cost = 0*2 + 2*5 = 10
      Offers [3, 0, 5] and [1, 2, 10] exceed needs (0 < 3, 0 < 1)
      Returns 10
    Total Branch 1 Cost = 5 + 10 = 15

  Branch 2: Try Offer [1, 2, 10] (Cost: 10):
    Can we take it? 1 <= 3 and 2 <= 2 (Yes!)
    New needs = (3-1, 2-2) = (2, 0)
    Evaluate (2, 0):
      Baseline cost = 2*2 + 0*5 = 4
      No offers applicable
      Returns 4
    Total Branch 2 Cost = 10 + 4 = 14

Optimal Minimum Cost = min(16, 15, 14) = 14!
```

---

### Solved Examples with Multiple Inputs

| Case | `price` | `special` | `needs` | Result | Explanation |
|---|---|---|---|---|---|
| **Standard** | `[2, 5]` | `[[3,0,5], [1,2,10]]` | `[3, 2]` | `14` | Use offer 2 once + buy 2 of item 0 individually |
| **No Beneficial Offers** | `[2, 3, 4]` | `[[1,1,0,4], [2,2,1,9]]` | `[1, 2, 1]` | `11` | Regular prices $2 + 2(3) + 4 = 12$; offer 1 gives $4 + 3 + 4 = 11$ |
| **Needs All Zero** | `[5, 5]` | `[[1,1,5]]` | `[0, 0]` | `0` | Nothing needed |
| **Exceeding Offer Disallowed** | `[2, 2]` | `[[2,0,1]]` | `[1, 1]` | `4` | Offer `[2,0,1]` gives 2 of item 0, but we only need 1; cannot use it |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Tuple, Dict

class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        n = len(price)
        memo: Dict[Tuple[int, ...], int] = {}
        
        # Filter out bogus offers that are more expensive than individual items
        valid_specials = []
        for sp in special:
            if sum(sp[i] * price[i] for i in range(n)) > sp[-1]:
                valid_specials.append(sp)

        def dfs(cur_needs: Tuple[int, ...]) -> int:
            if cur_needs in memo:
                return memo[cur_needs]
                
            # Baseline cost: buy all remaining items individually
            res = sum(cur_needs[i] * price[i] for i in range(n))
            
            # Try all valid special offers
            for offer in valid_specials:
                new_needs = []
                for i in range(n):
                    if offer[i] > cur_needs[i]:
                        break
                    new_needs.append(cur_needs[i] - offer[i])
                else:
                    res = min(res, offer[-1] + dfs(tuple(new_needs)))
                    
            memo[cur_needs] = res
            return res

        return dfs(tuple(needs))
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int shoppingOffers(std::vector<int>& price, std::vector<std::vector<int>>& special, std::vector<int>& needs) {
        int n = price.size();
        std::map<std::vector<int>, int> memo;

        // Filter out offers that do not provide a discount
        std::vector<std::vector<int>> valid_specials;
        for (const auto& sp : special) {
            int original_cost = 0;
            for (int i = 0; i < n; ++i) {
                original_cost += sp[i] * price[i];
            }
            if (original_cost > sp[n]) {
                valid_specials.push_back(sp);
            }
        }

        return dfs(price, valid_specials, needs, memo);
    }

private:
    int dfs(const std::vector<int>& price, const std::vector<std::vector<int>>& special,
            std::vector<int>& cur_needs, std::map<std::vector<int>, int>& memo) {
        if (memo.find(cur_needs) != memo.end()) {
            return memo[cur_needs];
        }

        int n = price.size();
        // Baseline cost: buy individually
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res += cur_needs[i] * price[i];
        }

        for (const auto& offer : special) {
            bool valid = true;
            std::vector<int> next_needs(n);
            for (int i = 0; i < n; ++i) {
                if (offer[i] > cur_needs[i]) {
                    valid = false;
                    break;
                }
                next_needs[i] = cur_needs[i] - offer[i];
            }

            if (valid) {
                res = std::min(res, offer[n] + dfs(price, special, next_needs, memo));
            }
        }

        return memo[cur_needs] = res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class Solution {
    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        int n = price.size();
        Map<List<Integer>, Integer> memo = new HashMap<>();

        // Filter out offers that do not provide a discount
        List<List<Integer>> validSpecials = new ArrayList<>();
        for (List<Integer> sp : special) {
            int originalCost = 0;
            for (int i = 0; i < n; i++) {
                originalCost += sp.get(i) * price.get(i);
            }
            if (originalCost > sp.get(n)) {
                validSpecials.add(sp);
            }
        }

        return dfs(price, validSpecials, needs, memo);
    }

    private int dfs(List<Integer> price, List<List<Integer>> special,
                    List<Integer> curNeeds, Map<List<Integer>, Integer> memo) {
        if (memo.containsKey(curNeeds)) {
            return memo.get(curNeeds);
        }

        int n = price.size();
        int res = 0;
        for (int i = 0; i < n; i++) {
            res += curNeeds.get(i) * price.get(i);
        }

        for (List<Integer> offer : special) {
            boolean valid = true;
            List<Integer> nextNeeds = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                if (offer.get(i) > curNeeds.get(i)) {
                    valid = false;
                    break;
                }
                nextNeeds.add(curNeeds.get(i) - offer.get(i));
            }

            if (valid) {
                res = Math.min(res, offer.get(n) + dfs(price, special, nextNeeds, memo));
            }
        }

        memo.put(curNeeds, res);
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\text{States} \times |\text{special}| \times n) \le \mathcal{O}(11^n \times |\text{special}| \times n)$  
  Where $n \le 6$ and $\text{needs}[i] \le 10$. In practice, only a few hundred states are ever reached during DFS due to offer step sizes, finishing in $< 10$ ms.
- **Space Complexity:** $\mathcal{O}(\text{States} \times n)$  
  The memoization map holds entries for all reached needs configurations, bounded well within memory limits.

---

### Takeaway Pattern & Interview Traps

1. **Exact Fulfillment Constraint:**
   - The problem explicitly states: *"You are not allowed to buy more items than you want, even if that would lower the overall price."*
   - Therefore, any offer where $\text{offer}[i] > \text{needs}[i]$ for any item $i$ must be strictly rejected.
2. **Offer Pruning:**
   - Always discard offers that cost more than purchasing their constituent components at standard unit price. This significantly trims search tree branching.