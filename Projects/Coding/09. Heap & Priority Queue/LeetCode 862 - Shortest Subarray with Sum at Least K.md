---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 862: Shortest Subarray with Sum at Least K"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 862: Shortest Subarray with Sum at Least K

Below is a complete, structured treatment of **LeetCode 862 – Shortest Subarray with Sum at Least K**, aligned with your usual learning format.

---

## 1. Problem Statement

**LeetCode 862 — Shortest Subarray with Sum at Least K**

You are given an integer array `nums` and an integer `k`.

Return the **length of the shortest, non-empty subarray** of `nums` with sum **at least `k`**.  
If there is no such subarray, return `-1`.

### Constraints

* `1 ≤ nums.length ≤ 10^5`
* `-10^5 ≤ nums[i] ≤ 10^5`
* `1 ≤ k ≤ 10^9`

---

## 2. Why This Problem Is Hard

* The array may contain **negative numbers**, so:

  * Sliding window **does not work**
  * Two pointers **break**
* Brute force is `O(n^2)` → **TLE**
* We need:

  * Efficient range sum queries
  * Ability to shrink the window optimally

This leads to **Prefix Sum + Monotonic Data Structure**

---

## 3. Key Observation

### Prefix Sum Transformation

Define:

```
prefix[i] = nums[0] + nums[1] + ... + nums[i-1]
```

Then the sum of subarray `[j, i)` is:

```
prefix[i] - prefix[j]
```

We need:

```
prefix[i] - prefix[j] ≥ k
→ prefix[j] ≤ prefix[i] - k
```

So for each `i`, we want the **smallest index `j < i`** such that:

```
prefix[j] ≤ prefix[i] - k
```

---

## 4. Priority Queue (Min-Heap) Technique

### Idea

* Maintain a **min-heap** of `(prefix_sum, index)`
* For current `prefix[i]`:

  * While the smallest prefix in heap satisfies  
    `prefix[i] - min_prefix ≥ k`

    * Update answer
    * Pop it (we want the shortest length)

### Why It Works

* Heap always gives the **smallest prefix sum**
* If smallest prefix works, larger ones might too, but:

  * Popping gives **shorter subarray**
* Each prefix enters and leaves heap once → `O(n log n)`

---

## 5. Algorithm Steps

1. Initialize:

   * `prefix_sum = 0`
   * `min_heap = []`
   * `ans = ∞`
2. Iterate through array:

   * Update `prefix_sum`
   * While heap top satisfies sum ≥ `k`:

     * Update `ans`
     * Pop heap
   * Push `(prefix_sum, index)`
3. Return `ans` if found else `-1`

---

## 6. Python 3 Solution (with Typing)

```python
from typing import List
import heapq

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        min_heap: List[tuple[int, int]] = []
        prefix_sum = 0
        ans = float('inf')

        for i, num in enumerate(nums):
            prefix_sum += num

            # Case: subarray starting from index 0
            if prefix_sum >= k:
                ans = min(ans, i + 1)

            # Try to shrink using previous prefix sums
            while min_heap and prefix_sum - min_heap[0][0] >= k:
                prev_sum, prev_idx = heapq.heappop(min_heap)
                ans = min(ans, i - prev_idx)

            heapq.heappush(min_heap, (prefix_sum, i))

        return ans if ans != float('inf') else -1
```

---

## 7. Worked Example (Step-by-Step)

### Example 1

```text
nums = [2, -1, 2]
k = 3
```

#### Prefix sums

```
Index:        0   1   2
Prefix sum:   2   1   3
```

---

### Iteration Details

#### i = 0, num = 2

```
prefix_sum = 2
heap = []
2 < 3 → no answer
push (2, 0)
```

Heap:

```
[(2, 0)]
```

---

#### i = 1, num = -1

```
prefix_sum = 1
1 < 3 → no
1 - 2 < 3 → no pop
push (1, 1)
```

Heap:

```
[(1, 1), (2, 0)]
```

---

#### i = 2, num = 2

```
prefix_sum = 3
3 >= 3 → ans = 3
```

Check heap:

```
3 - 1 >= 3 ? NO
3 - 2 >= 3 ? NO
```

Final Answer:

```
3
```

---

## 8. Another Example (Important Case)

### Example 2

```text
nums = [1, 2, -1, 2]
k = 3
```

Best subarray: `[2, -1, 2]` → length = 3  
Algorithm correctly finds it by **popping old prefix sums**.

---

## 9. Complexity Analysis

| Aspect | Complexity |
| --- | --- |
| Time | `O(n log n)` |
| Space | `O(n)` |

---

## 10. Important Notes

* This **priority queue solution is valid**, but:

  * The **optimal solution** uses a **monotonic deque** (`O(n)`)
* Heap approach is:

  * Easier to reason about
  * Good stepping stone before deque optimization

If you want, I can next:

* Convert this to the **monotonic deque O(n) solution**
* Show **why heap is slower but simpler**
* Compare **862 vs 209 (positive-only case)**