---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 638: Shopping Offers"
tags:
  - leetcode
  - coding
  - backtracking
  - dynamic-programming
  - memoization
  - array
  - amazon
  - google
---

# LeetCode 638: Shopping Offers

**Target Companies:** Google, Amazon, Uber  
**Difficulty:** Medium  
**Topic:** Backtracking / Multi-Dimensional Unbounded Knapsack / Memoization  

---

### Problem Statement

In LeetCode Store, there are `n` items to sell. Each item has a price. However, there are some special offers, and a special offer consists of one or more different items with a bundle price.

You are given:
- An integer array `price` where `price[i]` is the price of the $i$-th item.
- An integer array `special` where `special[i]` is of the form `[n_1, n_2, ..., n_n, price]` which means that using this special offer, you can buy $n_1, n_2, \dots, n_n$ items of type $1, 2, \dots, n$ respectively for the specified price.
- An integer array `needs` where `needs[i]` is the number of pieces of the $i$-th item you want to buy.

Return *the lowest price you have to pay for exactly certain items as given, where you could make optimal use of the special offers*. You are not allowed to buy more items than you want, even if that would lower the overall price. You could use any of the special offers as many times as you want.

---

### Input & Output Formats & Constraints

- **Input:** `price: List[int]`, `special: List[List[int]]`, `needs: List[int]`
- **Output:** `int` (minimum cost to satisfy needs)
- **Constraints:**
  - $n == \text{price.length} == \text{needs.length}$
  - $1 \le n \le 6$
  - $0 \le \text{price}[i] \le 10$
  - $0 \le \text{needs}[i] \le 10$
  - $1 \le \text{special.length} \le 100$
  - $\text{special}[i]\text{.length} == n + 1$
  - $0 \le \text{special}[i][j] \le 50$

---

### Key Idea & Intuition

- **Multi-Dimensional Knapsack:**
  - The problem is an unbounded multi-dimensional knapsack where the capacities are the remaining requirements `needs` for each of the $n \le 6$ items.
  - The state is completely defined by the tuple of remaining items needed: `tuple(current_needs)`.
- **Upper Bound (Direct Purchase):**
  - At any state `curr_needs`, we can always buy all remaining items individually at regular prices:
    $$\text{base\_cost} = \sum_{i=0}^{n-1} \text{curr\_needs}[i] \times \text{price}[i]$$
  - This provides a natural upper bound for our search.
- **Validating Special Offers:**
  - An offer is applicable to `curr_needs` if and only if for every item $i$, $\text{offer}[i] \le \text{curr\_needs}[i]$ (we cannot overshoot needs).
  - Also, any special offer that costs more than buying its individual items at retail price can be filtered out upfront as useless.
- **Top-Down DFS with Memoization:**
  - Cache `dfs(tuple(curr_needs))` in a hash map.
  - For each applicable offer, recursively compute the cost and take the minimum:
    $$\text{cost} = \min\left(\text{base\_cost}, \min_{\text{valid offers}} (\text{offer\_price} + \text{dfs}(\text{curr\_needs} - \text{offer}))\right)$$

---

### Solution Approach (Step-by-Step)

1. **Filter Inefficient Offers:**
   - Keep only offers where $\text{offer\_price} < \sum \text{offer}[i] \times \text{price}[i]$.
2. **Define Memoized DFS `dfs(curr_needs)`:**
   - If `tuple(curr_needs)` in `memo`: return `memo[tuple(curr_needs)]`.
   - Calculate baseline cost: buying all `curr_needs` at individual prices.
   - For each filtered offer in `special`:
     - Check if `offer[i] <= curr_needs[i]` for all $i \in [0, n - 1]$.
     - If yes, form `next_needs` by subtracting `offer[i]` from `curr_needs[i]`.
     - `candidate_cost = offer[-1] + dfs(next_needs)`.
     - Update `min_cost = min(min_cost, candidate_cost)`.
   - Store `memo[tuple(curr_needs)] = min_cost` and return it.
3. Return `dfs(needs)`.

---

### Visual Algorithm Walkthrough

Let `price = [2, 5]`, `special = [[3, 0, 5], [1, 2, 10]]`, `needs = [3, 2]`.

```
dfs([3, 2])
├── Base cost (retail): 3*2 + 2*5 = 16
│
├── Try Offer 1: [3, 0] for $5
│   - Valid: 3 <= 3 and 0 <= 2.
│   - next_needs = [3-3, 2-0] = [0, 2]
│   └── dfs([0, 2])
│       ├── Base cost: 0*2 + 2*5 = 10
│       ├── Try Offer 1: [3, 0] -> Invalid (3 > 0)
│       ├── Try Offer 2: [1, 2] -> Invalid (1 > 0)
│       └── Result for dfs([0, 2]) = 10
│   Total with Offer 1 = 5 + 10 = 15
│
└── Try Offer 2: [1, 2] for $10
    - Valid: 1 <= 3 and 2 <= 2.
    - next_needs = [3-1, 2-2] = [2, 0]
    └── dfs([2, 0])
        ├── Base cost: 2*2 + 0*5 = 4
        ├── Try Offer 1: [3, 0] -> Invalid (3 > 2)
        ├── Try Offer 2: [1, 2] -> Invalid (2 > 0)
        └── Result for dfs([2, 0]) = 4
    Total with Offer 2 = 10 + 4 = 14

Min cost = min(16, 15, 14) = 14!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `price` | `special` | `needs` | Min Cost | Optimal Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[2, 5]` | `[[3,0,5],[1,2,10]]` | `[3, 2]` | `14` | Offer `[1,2]` for 10 + 2 items of type 1 for $4$ |
| **Example 2** | `[2, 3, 4]` | `[[1,1,0,4],[2,2,1,9]]` | `[1, 2, 1]` | `11` | Offer `[1,1,0,4]` + 1 of item 2 (3) + 1 of item 3 (4) |
| **No Offers Better** | `[2, 5]` | `[[1,1,8]]` | `[2, 2]` | `14` | Retail: $2 \times 2 + 2 \times 5 = 14$ vs Offer: $8 \times 2 = 16$ |
| **Zero Needs** | `[2, 5]` | `[[1,1,1]]` | `[0, 0]` | `0` | Need nothing |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Tuple, Dict

class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        """
        Finds the minimum cost to satisfy exact shopping needs.
        Uses top-down memoized backtracking with offer filtering.
        """
        n = len(price)
        # Filter out offers that do not save money compared to individual retail
        filtered_special = []
        for offer in special:
            individual_cost = sum(offer[i] * price[i] for i in range(n))
            if offer[-1] < individual_cost:
                filtered_special.append(offer)

        memo: Dict[Tuple[int, ...], int] = {}

        def dfs(curr_needs: Tuple[int, ...]) -> int:
            if curr_needs in memo:
                return memo[curr_needs]

            # Upper bound: buy all remaining needs individually
            min_cost = sum(curr_needs[i] * price[i] for i in range(n))

            # Try applying each special offer
            for offer in filtered_special:
                # Check feasibility
                if all(curr_needs[i] >= offer[i] for i in range(n)):
                    next_needs = tuple(curr_needs[i] - offer[i] for i in range(n))
                    min_cost = min(min_cost, offer[-1] + dfs(next_needs))

            memo[curr_needs] = min_cost
            return min_cost

        return dfs(tuple(needs))
```

#### C++17
```cpp
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int shoppingOffers(std::vector<int>& price, std::vector<std::vector<int>>& special, std::vector<int>& needs) {
        int n = static_cast<int>(price.size());
        // Filter out useless offers
        std::vector<std::vector<int>> filtered_special;
        for (const auto& offer : special) {
            int retail_sum = 0;
            for (int i = 0; i < n; ++i) {
                retail_sum += offer[i] * price[i];
            }
            if (offer.back() < retail_sum) {
                filtered_special.push_back(offer);
            }
        }

        std::map<std::vector<int>, int> memo;
        return dfs(needs, price, filtered_special, memo);
    }

private:
    int dfs(const std::vector<int>& curr_needs, const std::vector<int>& price,
            const std::vector<std::vector<int>>& special,
            std::map<std::vector<int>, int>& memo) {
        if (memo.count(curr_needs)) {
            return memo[curr_needs];
        }

        int n = static_cast<int>(price.size());
        int min_cost = 0;
        for (int i = 0; i < n; ++i) {
            min_cost += curr_needs[i] * price[i];
        }

        for (const auto& offer : special) {
            bool valid = true;
            std::vector<int> next_needs(n);
            for (int i = 0; i < n; ++i) {
                if (curr_needs[i] < offer[i]) {
                    valid = false;
                    break;
                }
                next_needs[i] = curr_needs[i] - offer[i];
            }

            if (valid) {
                min_cost = std::min(min_cost, offer.back() + dfs(next_needs, price, special, memo));
            }
        }

        memo[curr_needs] = min_cost;
        return min_cost;
    }
};
```

#### Java
```java
import java.util.*;

class Solution {
    public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
        int n = price.size();
        List<List<Integer>> filteredSpecial = new ArrayList<>();

        for (List<Integer> offer : special) {
            int retailSum = 0;
            for (int i = 0; i < n; i++) {
                retailSum += offer.get(i) * price.get(i);
            }
            if (offer.get(n) < retailSum) {
                filteredSpecial.add(offer);
            }
        }

        Map<List<Integer>, Integer> memo = new HashMap<>();
        return dfs(needs, price, filteredSpecial, memo);
    }

    private int dfs(List<Integer> currNeeds, List<Integer> price,
                   List<List<Integer>> special, Map<List<Integer>, Integer> memo) {
        if (memo.containsKey(currNeeds)) {
            return memo.get(currNeeds);
        }

        int n = price.size();
        int minCost = 0;
        for (int i = 0; i < n; i++) {
            minCost += currNeeds.get(i) * price.get(i);
        }

        for (List<Integer> offer : special) {
            List<Integer> nextNeeds = new ArrayList<>();
            boolean valid = true;
            for (int i = 0; i < n; i++) {
                if (currNeeds.get(i) < offer.get(i)) {
                    valid = false;
                    break;
                }
                nextNeeds.add(currNeeds.get(i) - offer.get(i));
            }

            if (valid) {
                minCost = Math.min(minCost, offer.get(n) + dfs(nextNeeds, price, special, memo));
            }
        }

        memo.put(currNeeds, minCost);
        return minCost;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(S \cdot \prod (N_i + 1))$, where $S$ is the number of valid special offers ($S \le 100$) and $N_i$ is `needs[i]` ($N_i \le 10$).
  - For $n \le 6$ items with needs up to 10, the total state space reachable is typically $\ll 10^4$ states.
  - With memoization and upfront filtering of useless offers, the search runs in $< 20 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(\prod (N_i + 1))$ for the memoization map and recursion stack of depth $\le \sum \text{needs}[i] \le 60$.

---

### Takeaway Pattern & Interview Traps

- **Over-Buying Restriction:** You cannot buy more items than specified in `needs`, even if the bundle is cheaper than what you need. Any offer where `offer[i] > curr_needs[i]` must be strictly ignored.
- **Upfront Offer Filtering:** Some special offers cost *more* than buying individual items. Pruning these upfront avoids unnecessary branch expansions.