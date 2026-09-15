---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 410: Split Array Largest Sum"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 410: Split Array Largest Sum

**LeetCode 410 – Split Array Largest Sum**, aligned with how this problem is typically evaluated in senior DSA interviews.

---

## 1. Problem Statement

You are given an integer array `nums` of length `n` and an integer `k`.

You must split `nums` into **k non-empty contiguous subarrays**.

Return the **minimum possible value of the largest subarray sum** among these `k` subarrays.

### Constraints (important for approach selection)

* `1 ≤ n ≤ 1000`
* `1 ≤ k ≤ n`
* `0 ≤ nums[i] ≤ 10^6`

---

## 2. Key Observation (Core Insight)

This is **not** a pure greedy problem and **not** a pure DP problem.

### Critical Observations

1. **Search Space is Monotonic**

   * If you can split the array into ≤ `k` subarrays with max sum = `X`,  
     then you can also do it for any `X' > X`.
   * This monotonicity immediately suggests **Binary Search on the Answer**.
2. **Lower Bound**

   * The largest subarray must be **at least** the maximum element:

     ```
     low = max(nums)
     ```
3. **Upper Bound**

   * If you put everything into one subarray:

     ```
     high = sum(nums)
     ```
4. **Decision Problem**

   * Given a candidate max sum `mid`,  
     **can we split the array into ≤ k subarrays such that no subarray sum exceeds `mid`?**
   * This decision can be checked **greedily in O(n)**.

---

## 3. Greedy Feasibility Check (Core Trick)

For a fixed `mid`:

* Traverse the array
* Keep adding elements to the current subarray
* If adding an element exceeds `mid`, **start a new subarray**
* Count how many subarrays are formed

### Greedy Justification

* To minimize the number of subarrays, we **pack as many elements as possible** into the current subarray
* This greedy packing ensures **minimum splits**

If number of subarrays ≤ `k` → `mid` is feasible.

---

## 4. Algorithm Overview

### Binary Search + Greedy Validation

1. Set:

   ```
   low = max(nums)
   high = sum(nums)
   answer = high
   ```
2. While `low ≤ high`:

   * `mid = (low + high) // 2`
   * If feasible(mid):

     * update `answer = mid`
     * try smaller max sum → `high = mid - 1`
   * Else:

     * need larger max sum → `low = mid + 1`
3. Return `answer`

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        def can_split(max_allowed_sum: int) -> bool:
            subarrays = 1
            current_sum = 0

            for num in nums:
                if current_sum + num <= max_allowed_sum:
                    current_sum += num
                else:
                    subarrays += 1
                    current_sum = num
                    if subarrays > k:
                        return False

            return True

        low = max(nums)
        high = sum(nums)
        answer = high

        while low <= high:
            mid = (low + high) // 2

            if can_split(mid):
                answer = mid
                high = mid - 1
            else:
                low = mid + 1

        return answer
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```text
nums = [7, 2, 5, 10, 8]
k = 2
```

### Step 1: Define Search Space

```
low  = max(nums) = 10
high = sum(nums) = 32
```

---

### Step 2: Binary Search Iterations

#### Iteration 1

```
mid = (10 + 32) // 2 = 21
```

**Greedy Split Check (max = 21)**

| Element | Current Sum | Action |
| --- | --- | --- |
| 7 | 7 | continue |
| 2 | 9 | continue |
| 5 | 14 | continue |
| 10 | 24 | exceeds → new subarray |
| 8 | 18 | continue |

Subarrays formed = 2 ≤ k → feasible

```
answer = 21
high = 20
```

---

#### Iteration 2

```
mid = (10 + 20) // 2 = 15
```

**Greedy Split Check (max = 15)**

| Element | Current Sum | Action |
| --- | --- | --- |
| 7 | 7 | continue |
| 2 | 9 | continue |
| 5 | 14 | continue |
| 10 | 24 | new subarray |
| 8 | 18 | new subarray |

Subarrays = 3 > k → not feasible

```
low = 16
```

---

#### Iteration 3

```
mid = (16 + 20) // 2 = 18
```

**Greedy Split Check (max = 18)**

| Element | Current Sum | Action |
| --- | --- | --- |
| 7 | 7 | continue |
| 2 | 9 | continue |
| 5 | 14 | continue |
| 10 | 24 | new subarray |
| 8 | 18 | continue |

Subarrays = 2 → feasible

```
answer = 18
high = 17
```

---

#### Iteration 4

```
mid = (16 + 17) // 2 = 16
```

**Greedy Split Check**

Subarrays = 3 → not feasible

```
low = 17
```

---

#### Iteration 5

```
mid = (17 + 17) // 2 = 17
```

Subarrays = 3 → not feasible

```
low = 18
```

Loop ends.

---

### Final Answer

```
18
```

---

## 7. Time & Space Complexity

### Time Complexity

* Binary search range: `log(sum(nums))`
* Each feasibility check: `O(n)`

```
Total: O(n log(sum(nums)))
```

### Space Complexity

```
O(1)
```

---

## 8. Interview-Level Takeaways

* This is a **Binary Search on Answer + Greedy Validation** pattern
* Key is identifying **monotonic feasibility**
* Greedy is used only for **decision**, not optimization
* Commonly asked follow-ups:

  * Why DP is slower (`O(n^2 * k)`)
  * Why greedy alone fails
  * How bounds are derived

If you want, I can also:

* Show **DP solution and why it TLEs**
* Draw a **decision monotonicity diagram**
* Convert this into a **generic template for “minimize maximum” problems**