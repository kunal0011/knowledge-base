---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 446: Arithmetic Slices II (Subsequence)"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 446: Arithmetic Slices II (Subsequence)

**LeetCode 446 – Arithmetic Slices II (Subsequence)** with **formal state definition, transition, DP table construction, and a worked example**.  
This problem is subtle and often misunderstood, so I will be precise.

---

## 1. Problem Statement (Restated Precisely)

Given an integer array `nums`, return the **number of arithmetic subsequences of length ≥ 3**.

* A **subsequence** is formed by deleting zero or more elements without changing order.
* An **arithmetic sequence** has a constant difference between consecutive elements.

---

## 2. Why This Is NOT the Same as LeetCode 413

| Problem | Key Difference |
| --- | --- |
| LC 413 | Contiguous subarrays |
| **LC 446** | **Subsequences (can skip elements)** |

Because subsequences allow skipping, brute force is exponential → **Dynamic Programming is required**.

---

## 3. Core Insight (Key Observation)

For every index `i`, we want to know:

> How many arithmetic subsequences **end at index `i`** with a given **common difference `d`**?

If we know that, then:

* Extending an existing subsequence of length ≥ 2 creates a valid arithmetic subsequence of length ≥ 3.

---

## 4. DP State Definition

### DP State

```
dp[i][d] = number of arithmetic subsequences
           ending at index i
           with common difference d
           and length ≥ 2
```

* `i` → ending index
* `d` → difference between consecutive elements
* Length ≥ 2 is intentional (length 3 is counted when extending)

### Data Structure

* `dp` is an array of hash maps:

```
dp: List[Dict[int, int]]
```

Because:

* Differences can be large (`nums[i] - nums[j]`)
* Using a fixed 2D array is not feasible

---

## 5. DP Transition (Most Important Part)

For every pair `(j, i)` where `j < i`:

```
d = nums[i] - nums[j]
```

Two things happen:

### 1️⃣ Extend existing sequences ending at `j`

If `dp[j][d] = k`, then:

* We can append `nums[i]`
* This creates `k` new arithmetic subsequences of length ≥ 3

```
count += dp[j][d]
```

### 2️⃣ Create a new length-2 sequence

The pair `(nums[j], nums[i])` itself forms a sequence of length 2:

```
dp[i][d] += dp[j][d] + 1
```

* `+1` → new pair
* `dp[j][d]` → extended sequences

---

## 6. Why Length-2 Sequences Are Stored

Because:

* They may later become length ≥ 3 when extended
* But they are **NOT counted in the final answer**

Final answer only includes extensions.

---

## 7. Algorithm (Step-by-Step)

```
Initialize dp as array of empty hashmaps
result = 0

for i from 0 to n-1:
    for j from 0 to i-1:
        d = nums[i] - nums[j]
        prev = dp[j].get(d, 0)

        result += prev
        dp[i][d] += prev + 1
```

---

## 8. Python 3 Implementation (With Typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [defaultdict(int) for _ in range(n)]
        result = 0

        for i in range(n):
            for j in range(i):
                diff = nums[i] - nums[j]
                prev_count = dp[j][diff]

                # All sequences ending at j with diff can be extended
                result += prev_count

                # Update dp for index i
                dp[i][diff] += prev_count + 1

        return result
```

---

## 9. DP Table Construction Example

### Input

```text
nums = [2, 4, 6, 8]
```

---

### Step-by-Step DP Table

#### i = 1 (4)

| j | diff | dp[1] |
| --- | --- | --- |
| 0 | 2 | {2: 1} |

Explanation: `(2,4)` → length 2

---

#### i = 2 (6)

| j | diff | dp[2] update | result |
| --- | --- | --- | --- |
| 0 | 4 | {4: 1} | 0 |
| 1 | 2 | {4:1, 2:2} | +1 |

Explanation:

* `(4,6)` → new pair
* Extend `(2,4)` → `(2,4,6)` → **valid**

---

#### i = 3 (8)

| j | diff | dp[3] update | result |
| --- | --- | --- | --- |
| 0 | 6 | {6:1} | 1 |
| 1 | 4 | {6:1,4:1} | 1 |
| 2 | 2 | {6:1,4:1,2:3} | +2 |

Explanation:

* Extend `(4,6)` → `(4,6,8)`
* Extend `(2,4,6)` → `(2,4,6,8)`

---

### Final Result

```
Total = 3
```

Valid arithmetic subsequences:

```
[2,4,6]
[4,6,8]
[2,4,6,8]
```

---

## 10. Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n²)** |
| Space | **O(n²)** (worst case unique diffs) |

---

## 11. Conceptual Summary

* Store **length ≥ 2** arithmetic sequences
* Count only when **extending** to length ≥ 3
* DP indexed by **(ending index, difference)**

---

If you want next:

* Visualization as a **DP growth tree**
* Why this avoids double counting
* Comparison with LIS-style DP
* Optimization discussion (overflow handling)

Tell me how deep you want to go.