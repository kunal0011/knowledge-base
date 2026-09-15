---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 904: Fruit Into Baskets"
tags:
  - leetcode
  - coding
  - sliding-window
  - hash-table
  - two-pointers
  - array
  - amazon
  - google
---

# LeetCode 904: Fruit Into Baskets

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Sliding Window / Hash Table / Two Pointers  

---

### Problem Statement

You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array `fruits` where `fruits[i]` is the **type** of fruit the $i$-th tree produces.

You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:
- You only have **two baskets**, and each basket can only hold a **single type** of fruit. There is no limit on the amount of fruit each basket can hold.
- Starting from any tree of your choice, you must pick **exactly one fruit** from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
- Once you reach a tree with fruit that cannot fit in your baskets, you must stop.

Given the integer array `fruits`, return the **maximum number of fruits** you can pick.

---

### Input & Output Formats & Constraints

- **Input:**
  - `fruits`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{fruits.length} \le 10^5$).
- **Output:**
  - `int` — maximum length of a contiguous subarray containing at most 2 distinct integers.
- **Constraints:**
  - $1 \le \text{fruits.length} \le 10^5$
  - $0 \le \text{fruits}[i] < \text{fruits.length}$

---

### Key Idea & Intuition

Despite the story setting, the problem translates directly to a fundamental sliding window pattern:
> **Find the length of the longest contiguous subarray that contains at most 2 distinct integers.**

#### Dynamic Sliding Window:
1. Maintain a window $[l, r]$ and a hash map (or frequency array) `basket` storing the count of each fruit type currently inside the window.
2. Expand the window by advancing `r`:
   - Increment `basket[fruits[r]] += 1`.
3. If the number of distinct fruit types in `basket` exceeds 2 (`len(basket) > 2`):
   - Shrink the window from the left by decrementing `basket[fruits[l]]`.
   - If a fruit's count drops to 0, completely remove it from `basket`.
   - Increment `l += 1`.
   - Repeat until at most 2 fruit types remain.
4. Update `max_fruits = max(max_fruits, r - l + 1)`.

#### Alternative Non-Shrinking Window ($\mathcal{O}(1)$ conditional):
Instead of a `while` loop that shrinks the window, use a single `if len(basket) > 2:`. The window expands when valid and simply shifts right without shrinking when invalid, preserving the historical maximum window size. At the end, $r - l$ equals the maximum size.

---

### Solution Approach (Step-by-Step)

1. Initialize `basket = {}`, `left = 0`, and `max_fruits = 0`.
2. Iterate `right` from $0$ to $n - 1$:
   - Let `fruit = fruits[right]`.
   - Add `fruit` to `basket`: `basket[fruit] = basket.get(fruit, 0) + 1`.
   - While `len(basket) > 2`:
     - Decrement count of `fruits[left]`.
     - If count becomes $0$, delete key `fruits[left]` from `basket`.
     - Increment `left += 1`.
   - Update `max_fruits = max(max_fruits, right - left + 1)`.
3. Return `max_fruits`.

---

### Visual Algorithm Walkthrough

For `fruits = [1, 2, 3, 2, 2]`:

```
Indices:    0    1    2    3    4
Fruits:    [1,   2,   3,   2,   2]

r = 0 (fruit 1): basket = {1: 1}. Distinct = 1 <= 2.
  Window [0..0] -> [1], len = 1. max_fruits = 1.

r = 1 (fruit 2): basket = {1: 1, 2: 1}. Distinct = 2 <= 2.
  Window [0..1] -> [1, 2], len = 2. max_fruits = 2.

r = 2 (fruit 3): basket = {1: 1, 2: 1, 3: 1}. Distinct = 3 > 2! (INVALID)
  Shrink left:
    Decrement fruit 1 (count becomes 0 -> deleted).
    left moves to 1.
  basket now = {2: 1, 3: 1}. Distinct = 2 <= 2.
  Window [1..2] -> [2, 3], len = 2. max_fruits = 2.

r = 3 (fruit 2): basket = {2: 2, 3: 1}. Distinct = 2 <= 2.
  Window [1..3] -> [2, 3, 2], len = 3. max_fruits = 3.

r = 4 (fruit 2): basket = {2: 3, 3: 1}. Distinct = 2 <= 2.
  Window [1..4] -> [2, 3, 2, 2], len = 4. max_fruits = 4.

End of array.
Max Fruits Picked = 4 (trees [2, 3, 2, 2]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `fruits = [1, 2, 1]`
- **Output:** `3` (All fruits can be picked: types 1 and 2)

#### Example 2:
- **Input:** `fruits = [0, 1, 2, 2]`
- **Tracing:**
  - `[0, 1]` (len 2)
  - `[1, 2, 2]` (len 3, types 1 and 2)
- **Output:** `3`

#### Example 3:
- **Input:** `fruits = [1, 2, 3, 2, 2]`
- **Output:** `4`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import defaultdict
from typing import List

class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        basket = defaultdict(int)
        left = 0
        max_fruits = 0
        
        for right, fruit in enumerate(fruits):
            basket[fruit] += 1
            
            # If more than 2 types in basket, shrink from left
            while len(basket) > 2:
                left_fruit = fruits[left]
                basket[left_fruit] -= 1
                if basket[left_fruit] == 0:
                    del basket[left_fruit]
                left += 1
                
            max_fruits = max(max_fruits, right - left + 1)
            
        return max_fruits
```

#### C++17
```cpp
#include <vector>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    int totalFruit(const std::vector<int>& fruits) {
        std::unordered_map<int, int> basket;
        int left = 0;
        int max_fruits = 0;
        int n = static_cast<int>(fruits.size());
        
        for (int right = 0; right < n; ++right) {
            basket[fruits[right]]++;
            
            while (basket.size() > 2) {
                int left_fruit = fruits[left];
                basket[left_fruit]--;
                if (basket[left_fruit] == 0) {
                    basket.erase(left_fruit);
                }
                ++left;
            }
            
            max_fruits = std::max(max_fruits, right - left + 1);
        }
        
        return max_fruits;
    }
};
```

#### Java 17
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int totalFruit(int[] fruits) {
        Map<Integer, Integer> basket = new HashMap<>();
        int left = 0;
        int maxFruits = 0;
        
        for (int right = 0; right < fruits.length; right++) {
            basket.put(fruits[right], basket.getOrDefault(fruits[right], 0) + 1);
            
            while (basket.size() > 2) {
                int leftFruit = fruits[left];
                basket.put(leftFruit, basket.get(leftFruit) - 1);
                if (basket.get(leftFruit) == 0) {
                    basket.remove(leftFruit);
                }
                left++;
            }
            
            maxFruits = Math.max(maxFruits, right - left + 1);
        }
        
        return maxFruits;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - The right pointer visits each tree once.
  - The left pointer advances at most $n$ times across the entire array.
  - Hash map operations take $\mathcal{O}(1)$ average time since the map contains at most 3 distinct keys at any point.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The hash map contains at most 3 entries at any time.

---

### Takeaway Pattern & Interview Traps

- **De-contextualization:** Word problems with long thematic stories (orchards, trees, baskets) frequently mask standard patterns. Identifying "contiguous subarray with at most $K$ distinct elements" instantly reduces the problem to canonical sliding window.
- **Generalization:** If baskets $= K$, the exact same code with `len(basket) > K` solves the general problem (LeetCode 340: Longest Substring with At Most K Distinct Characters).