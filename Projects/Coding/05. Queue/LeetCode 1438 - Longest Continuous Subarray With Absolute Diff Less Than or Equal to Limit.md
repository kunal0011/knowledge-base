---
date: "2025-12-19"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

Below is a **structured, interview-ready explanation** of **LeetCode 1438 – Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit**, aligned with your usual format.

---

## 1. Problem Statement

**LeetCode 1438 — Longest Continuous Subarray With Absolute Diff ≤ Limit**

You are given an integer array `nums` and an integer `limit`.

Return the **length of the longest continuous subarray** such that the **absolute difference between any two elements** in this subarray is **less than or equal to `limit`**.

### Constraints

* `1 ≤ nums.length ≤ 10^5`
* `1 ≤ nums[i] ≤ 10^9`
* `0 ≤ limit ≤ 10^9`

---

## 2. Key Observation

For any subarray:

```
max(subarray) - min(subarray) ≤ limit
```

So the problem reduces to:

* Maintain a **sliding window**
* Efficiently track:

  * **maximum element**
  * **minimum element**
* Ensure their difference never exceeds `limit`

A brute-force solution is impossible due to `O(n²)` complexity.

---

## 3. Queue (Deque) Insights — Core Trick

We use **two monotonic deques**:

### 1️⃣ Max Deque (`max_dq`)

* Maintains elements in **decreasing order**
* Front always holds the **maximum** of the current window

### 2️⃣ Min Deque (`min_dq`)

* Maintains elements in **increasing order**
* Front always holds the **minimum** of the current window

### Why Deques Work

* Each element is added **once**
* Each element is removed **once**
* Overall time complexity: **O(n)**

---

## 4. Sliding Window Strategy

* Expand the window using `right`
* Insert `nums[right]` into both deques:

  * Pop smaller elements from `max_dq`
  * Pop larger elements from `min_dq`
* While window is invalid:

  ```
  max_dq[0] - min_dq[0] > limit
  ```

  * Shrink from the left
  * Remove outgoing elements from deques if needed
* Track the maximum window length

---

## 5. Python 3 Solution (With Typing)

```python
from collections import deque
from typing import List

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        max_dq = deque()  # decreasing
        min_dq = deque()  # increasing
        
        left = 0
        result = 0
        
        for right in range(len(nums)):
            # Maintain max deque
            while max_dq and nums[right] > max_dq[-1]:
                max_dq.pop()
            max_dq.append(nums[right])
            
            # Maintain min deque
            while min_dq and nums[right] < min_dq[-1]:
                min_dq.pop()
            min_dq.append(nums[right])
            
            # Shrink window if invalid
            while max_dq[0] - min_dq[0] > limit:
                if nums[left] == max_dq[0]:
                    max_dq.popleft()
                if nums[left] == min_dq[0]:
                    min_dq.popleft()
                left += 1
            
            result = max(result, right - left + 1)
        
        return result
```

---

## 6. Worked Out Example

### Input

```text
nums = [8, 2, 4, 7]
limit = 4
```

---

### Step-by-Step Window Processing

| Right | Window | Max | Min | Valid? | Length |
| --- | --- | --- | --- | --- | --- |
| 0 | [8] | 8 | 8 | Yes | 1 |
| 1 | [8,2] | 8 | 2 | No | — |
|  | [2] | 2 | 2 | Yes | 1 |
| 2 | [2,4] | 4 | 2 | Yes | 2 |
| 3 | [2,4,7] | 7 | 2 | No | — |
|  | [4,7] | 7 | 4 | Yes | 2 |

---

### Final Answer

```text
Output: 2
```

---

## 7. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(n)** (for deques) |

---

## 8. Key Takeaways (Interview Gold)

* Sliding window + **monotonic queues**
* Avoid recomputing min/max
* Each element processed at most **twice**
* Classic pattern for **range-based constraints**

---

If you want next:

* 🔁 Dry-run with a **large example**
* 🧠 Comparison with **heap-based approach**
* 📌 Pattern mapping: *Sliding Window + Monotonic Queue*

Tell me how you want to proceed.