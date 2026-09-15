---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 347: Top K Frequent Elements"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 347: Top K Frequent Elements

Below is a complete, interview-oriented treatment of **LeetCode 347 – Top K Frequent Elements**, structured exactly as requested.

---

## 📌 Problem Statement — LeetCode 347

Given an integer array `nums` and an integer `k`, return the **k most frequent elements**.

* The answer can be returned in **any order**
* It is **guaranteed** that the answer is unique
* You must solve it in **better than O(n log n)** time

**Example**

```text
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

---

## 🔑 Key Observations

1. **Frequency counting is mandatory**

   * Raw values are irrelevant without knowing how often they appear
   * Use a hash map: `num → frequency`
2. **Sorting by frequency is expensive**

   * Sorting all elements costs `O(n log n)`
   * The problem explicitly hints at avoiding this
3. **We only need top `k`**

   * This is a classic **Top-K problem**
   * Priority Queue (Heap) is the correct abstraction
4. **Use a Min-Heap of size `k`**

   * Keep only the top `k` frequent elements at any time
   * Smallest frequency stays at the top

---

## 🧠 Priority Queue Technique (Core Idea)

### Why Min-Heap and not Max-Heap?

* Max-Heap would require pushing **all elements**
* Min-Heap keeps heap size bounded to `k`
* Time complexity improves

### Heap Structure

Each heap entry:

```
(frequency, element)
```

### Algorithm

1. Count frequencies using a dictionary
2. Iterate through `(element, frequency)` pairs
3. Push into a min-heap
4. If heap size exceeds `k`, pop the smallest frequency
5. Heap now contains top `k` frequent elements

---

## ⏱️ Time & Space Complexity

| Component | Complexity |
| --- | --- |
| Frequency Map | `O(n)` |
| Heap Operations | `O(n log k)` |
| Total Time | **O(n log k)** |
| Space | `O(n + k)` |

---

## 🧩 Python 3 Solution (with typing)

```python
from typing import List
import heapq
from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Step 1: Count frequency
        freq_map = Counter(nums)

        # Step 2: Min-heap
        min_heap: List[tuple[int, int]] = []

        for num, freq in freq_map.items():
            heapq.heappush(min_heap, (freq, num))
            
            # Step 3: Maintain heap size k
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        # Step 4: Extract elements
        return [num for freq, num in min_heap]
```

---

## 🧪 Worked Example (Step-by-Step)

### Input

```text
nums = [1,1,1,2,2,3]
k = 2
```

---

### Step 1: Frequency Map

```
{
  1: 3,
  2: 2,
  3: 1
}
```

---

### Step 2: Heap Processing

| Operation | Heap Content (freq, num) |
| --- | --- |
| push (3,1) | [(3,1)] |
| push (2,2) | [(2,2), (3,1)] |
| push (1,3) | [(1,3), (3,1), (2,2)] |
| pop smallest | [(2,2), (3,1)] |

(heap size maintained at `k = 2`)

---

### Step 3: Extract Result

```
[2, 1]
```

✔ Correct — elements with highest frequencies

---

## 🎯 Key Takeaways for Interviews

* Recognize **Top-K** → think **Heap**
* Prefer **Min-Heap with bounded size**
* Avoid full sorting when only partial order is needed
* This pattern generalizes to:

  * Top K frequent words
  * K closest points
  * K largest/smallest elements

---

If you want, I can also explain:

* **Bucket sort solution (O(n))**
* **Why heap is preferred in interviews**
* **Dry-run with negative numbers or ties**
* **How this maps to streaming data problems**

Just let me know.