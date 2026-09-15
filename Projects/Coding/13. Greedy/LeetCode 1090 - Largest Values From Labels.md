---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1090: Largest Values From Labels"
tags:
  - leetcode
  - coding
  - greedy
  - sorting
  - hash-table
  - amazon
  - google
---

# LeetCode 1090: Largest Values From Labels

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Medium  
**Topic:** Greedy / Sorting / Hash Table / Priority Queue  

---

### Problem Statement

You are given $n$ items, each with a value and a label. You are also given two integers `numWanted` and `useLimit`:
- `values[i]`: the value of the $i$-th item.
- `labels[i]`: the label of the $i$-th item.
- `numWanted`: the maximum number of items you can choose.
- `useLimit`: the maximum number of items with the **same label** you can choose.

Select a subset of items such that:
1. The number of items chosen is at most `numWanted`.
2. The number of items with the same label is at most `useLimit`.

Return the **maximum possible sum** of the chosen subset.

---

### Input & Output Formats & Constraints

- **Input:**
  - `values`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{values.length} \le 2 \times 10^4$).
  - `labels`: `List[int]` / `vector<int>` / `int[]` ($\text{labels.length} == \text{values.length}$).
  - `numWanted`: `int` ($1 \le numWanted \le \text{values.length}$).
  - `useLimit`: `int` ($1 \le useLimit \le \text{values.length}$).
- **Output:**
  - `int` — the maximum sum of values of the selected items.
- **Constraints:**
  - $n == \text{values.length} == \text{labels.length}$
  - $1 \le n \le 2 \times 10^4$
  - $0 \le \text{values}[i] \le 2 \times 10^4$
  - $0 \le \text{labels}[i] \le 2 \times 10^4$
  - $1 \le numWanted, useLimit \le n$

---

### Key Idea & Intuition

Every item is an independent pair `(value, label)` with two independent quotas:
1. **Global quota:** total count of chosen items $\le numWanted$.
2. **Per-label quota:** count of chosen items with label $L \le useLimit$.

Because our goal is to maximize the sum of values:
- We should always prioritize items with **larger values**.
- There is no trade-off or sub-problem dependency: picking a larger value item will never block a better choice in the future, as long as neither the global quota nor the item's label quota is exceeded.
- Therefore, a pure **Greedy choice** is provably optimal (Matroid greedy theorem).

#### Algorithm:
1. Pair each value with its label: `items = zip(values, labels)`.
2. Sort `items` in descending order of value.
3. Traverse the sorted pairs:
   - For pair `(val, lab)`:
     - If `label_count[lab] < useLimit`:
       - Add `val` to total sum.
       - Increment `label_count[lab] += 1`.
       - Increment `items_picked += 1`.
       - If `items_picked == numWanted`: break early.
4. Return the total sum.

---

### Solution Approach (Step-by-Step)

1. Pair and sort: `items = sorted(zip(values, labels), key=lambda x: x[0], reverse=True)`.
2. Initialize `total_score = 0`, `picked = 0`, and a hash map `label_usage = defaultdict(int)`.
3. For each `(val, lab)` in `items`:
   - If `label_usage[lab] < useLimit`:
     - `label_usage[lab] += 1`
     - `total_score += val`
     - `picked += 1`
     - If `picked == numWanted`:
       - break
4. Return `total_score`.

---

### Visual Algorithm Walkthrough

For `values = [5, 4, 3, 2, 1]`, `labels = [1, 1, 2, 2, 3]`, `numWanted = 3`, `useLimit = 1`:

```
Paired & Sorted by value descending:
  (5, 1), (4, 1), (3, 2), (2, 2), (1, 3)

Iterate:
1. (val=5, lab=1):
   usage[1] = 0 < 1 -> ACCEPT
   score += 5 (5)
   usage[1] = 1, picked = 1

2. (val=4, lab=1):
   usage[1] = 1 == useLimit(1) -> REJECT (Label quota full)

3. (val=3, lab=2):
   usage[2] = 0 < 1 -> ACCEPT
   score += 3 (8)
   usage[2] = 1, picked = 2

4. (val=2, lab=2):
   usage[2] = 1 == useLimit(1) -> REJECT

5. (val=1, lab=3):
   usage[3] = 0 < 1 -> ACCEPT
   score += 1 (9)
   usage[3] = 1, picked = 3 == numWanted(3) -> STOP!

Final Result = 9 (items: 5, 3, 1).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `values = [5, 4, 3, 2, 1]`, `labels = [1, 1, 2, 2, 3]`, `numWanted = 3`, `useLimit = 1`
- **Output:** `9`

#### Example 2:
- **Input:** `values = [5, 4, 3, 2, 1]`, `labels = [1, 3, 3, 3, 2]`, `numWanted = 3`, `useLimit = 2`
- **Tracing:**
  - Pick (5, 1): score = 5
  - Pick (4, 3): score = 9
  - Pick (3, 3): score = 12 (label 3 has 2 items, reached useLimit)
  - Picked 3 items = numWanted. Stop.
- **Output:** `12`

#### Example 3:
- **Input:** `values = [9, 8, 8, 7, 6]`, `labels = [0, 0, 0, 1, 1]`, `numWanted = 3`, `useLimit = 1`
- **Tracing:** Pick (9, 0), skip (8, 0), skip (8, 0), pick (7, 1), skip (6, 1). Total = $9 + 7 = 16$.
- **Output:** `16`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import defaultdict
from typing import List

class Solution:
    def largestValsFromLabels(
        self, values: List[int], labels: List[int], numWanted: int, useLimit: int
    ) -> int:
        # Pair items and sort by value descending
        items = sorted(zip(values, labels), key=lambda x: x[0], reverse=True)
        
        label_usage = defaultdict(int)
        total_sum = 0
        picked = 0
        
        for val, lab in items:
            if label_usage[lab] < useLimit:
                label_usage[lab] += 1
                total_sum += val
                picked += 1
                if picked == numWanted:
                    break
                    
        return total_sum
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <unordered_map>

class Solution {
public:
    int largestValsFromLabels(
        const std::vector<int>& values, 
        const std::vector<int>& labels, 
        int numWanted, 
        int useLimit
    ) {
        int n = static_cast<int>(values.size());
        std::vector<std::pair<int, int>> items(n);
        for (int i = 0; i < n; ++i) {
            items[i] = {values[i], labels[i]};
        }
        
        // Sort descending by value
        std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        
        std::unordered_map<int, int> label_usage;
        int total_sum = 0;
        int picked = 0;
        
        for (const auto& [val, lab] : items) {
            if (label_usage[lab] < useLimit) {
                label_usage[lab]++;
                total_sum += val;
                picked++;
                if (picked == numWanted) {
                    break;
                }
            }
        }
        
        return total_sum;
    }
};
```

#### Java 17
```java
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int largestValsFromLabels(int[] values, int[] labels, int numWanted, int useLimit) {
        int n = values.length;
        int[][] items = new int[n][2];
        for (int i = 0; i < n; i++) {
            items[i][0] = values[i];
            items[i][1] = labels[i];
        }
        
        // Sort descending by value
        Arrays.sort(items, (a, b) -> Integer.compare(b[0], a[0]));
        
        Map<Integer, Integer> labelUsage = new HashMap<>();
        int totalSum = 0;
        int picked = 0;
        
        for (int[] item : items) {
            int val = item[0];
            int lab = item[1];
            
            int currentUsage = labelUsage.getOrDefault(lab, 0);
            if (currentUsage < useLimit) {
                labelUsage.put(lab, currentUsage + 1);
                totalSum += val;
                picked++;
                if (picked == numWanted) {
                    break;
                }
            }
        }
        
        return totalSum;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$
  - Zipping and sorting $n$ items takes $\mathcal{O}(n \log n)$ time.
  - The linear pass performs $\mathcal{O}(1)$ average hash map lookups/insertions.
  - Overall time is $\mathcal{O}(n \log n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space
  - The paired items list and the label frequency hash map store up to $n$ entries.

---

### Takeaway Pattern & Interview Traps

- **Independence Implies Greedy:** When constraints are simple capacities (global and categorical quotas) and values are additive without interactions, greedy sorting is mathematically guaranteed to be optimal.
- **Early Break:** Checking `if picked == numWanted: break` avoids scanning the rest of the array once the global quota is satisfied.