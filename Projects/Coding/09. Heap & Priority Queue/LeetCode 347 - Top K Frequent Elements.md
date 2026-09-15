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
  - bucket-sort
  - hash-map
  - amazon
  - google
---

# LeetCode 347: Top K Frequent Elements

**Target Companies:** Amazon, Google, Meta, Apple, Bloomberg, Microsoft  
**Difficulty:** Medium  
**Topic:** Priority Queue (Min-Heap) / Bucket Sort (Linear Time)

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return the $k$ most frequent elements. You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le \text{nums.length} \le 10^5$.
  - `k`: `int`, where $1 \le k \le \text{number of unique elements in } nums$.
  - $-10^4 \le nums[i] \le 10^4$.
- **Output:**
  - `List[int]`: The $k$ most frequent elements.
- **Constraints:**
  - The answer is guaranteed to be unique.
  - The algorithm's time complexity must be strictly better than $\mathcal{O}(N \log N)$.

---

### Key Idea & Intuition

Counting frequencies of all numbers takes $\mathcal{O}(N)$ using a hash map. Sorting the frequency entries takes $\mathcal{O}(U \log U)$ where $U$ is the number of unique elements ($\le N$).

To beat $\mathcal{O}(N \log N)$, there are two primary methods:

#### Paradigm 1: Min-Heap of Size $k$ ($\mathcal{O}(N \log k)$ Time, $\mathcal{O}(N)$ Space)
- Construct frequency map `freq_map`.
- Iterate through each `(num, freq)` pair.
- Maintain a **min-heap** of size $k$ ordered by frequency: `(freq, num)`.
- If the heap exceeds size $k$, pop the top (the lowest frequency element among the candidates).
- After scanning all unique numbers, the heap retains the $k$ most frequent elements!

#### Paradigm 2: Bucket Sort ($\mathcal{O}(N)$ Strictly Linear Time - Optimal)
- Notice that the frequency of any element is an integer in the range $[1, N]$.
- We can create an array of lists `buckets` of length $N + 1$, where `buckets[f]` stores all numbers that appear exactly $f$ times.
- Populate `buckets` in $\mathcal{O}(N)$ time.
- Traverse `buckets` backward from index $N$ down to $1$, collecting numbers into the output list until exactly $k$ elements are gathered.
- Achieves optimal $\mathcal{O}(N)$ runtime without any heap overhead!

---

### Solution Approach (Step-by-Step)

#### Approach 1: Bucket Sort ($\mathcal{O}(N)$ Time)
1. Compute `count = Counter(nums)`.
2. Create an array `buckets` of size $len(nums) + 1$, where each entry is an empty list.
3. For each `num, freq` in `count.items()`:
   - `buckets[freq].append(num)`
4. Initialize `result = []`.
5. Loop $f$ backward from $N$ down to $1$:
   - For `num` in `buckets[f]`:
     - `result.append(num)`
     - If `len(result) == k`: return `result`.

#### Approach 2: Min-Heap of Size $k$ ($\mathcal{O}(N \log k)$ Time)
1. Compute `count = Counter(nums)`.
2. `min_heap = []`.
3. For each `num, freq` in `count.items()`:
   - `heappush(min_heap, (freq, num))`
   - If `len(min_heap) > k`: `heappop(min_heap)`.
4. Return `[num for freq, num in min_heap]`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 1, 1, 2, 2, 3]`, $k = 2$:

```
Step 1: Count frequencies
  1 -> 3 times
  2 -> 2 times
  3 -> 1 time

Step 2: Place into frequency buckets (index = frequency):
  Bucket 0: []
  Bucket 1: [3]
  Bucket 2: [2]
  Bucket 3: [1]
  Bucket 4: []
  Bucket 5: []
  Bucket 6: []

Step 3: Collect from highest frequency bucket down:
  Check Bucket 6: empty
  Check Bucket 5: empty
  Check Bucket 4: empty
  Check Bucket 3: [1] -> append 1 (count = 1)
  Check Bucket 2: [2] -> append 2 (count = 2 == k!)

Result: [1, 2]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Multiset

- **Input:** `nums = [1, 1, 1, 2, 2, 3]`, `k = 2`
- **Output:** `[1, 2]`

#### Example 2: Single Element

- **Input:** `nums = [1]`, `k = 1`
- **Output:** `[1]`

#### Example 3: Negative Numbers with Equal Higher Frequencies

- **Input:** `nums = [-1, -1, 2, 2, 3]`, `k = 2`
- **Output:** `[-1, 2]`

---

### Multi-Language Implementations

#### Python 3

##### Optimal $\mathcal{O}(N)$ Bucket Sort
```python
from collections import Counter
from typing import List

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        n = len(nums)
        # buckets[i] will store all numbers that appear exactly i times
        buckets: List[List[int]] = [[] for _ in range(n + 1)]

        for num, freq in count.items():
            buckets[freq].append(num)

        result = []
        for freq in range(n, 0, -1):
            for num in buckets[freq]:
                result.append(num)
                if len(result) == k:
                    return result

        return result
```

##### Priority Queue Min-Heap Approach ($\mathcal{O}(N \log k)$)
```python
import heapq
from collections import Counter
from typing import List

class SolutionHeap:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        min_heap = []

        for num, freq in count.items():
            heapq.heappush(min_heap, (freq, num))
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        return [num for freq, num in min_heap]
```

#### C++17

```cpp
#include <vector>
#include <unordered_map>
#include <queue>

class Solution {
public:
    // Optimal O(N) Bucket Sort
    std::vector<int> topKFrequent(const std::vector<int>& nums, int k) {
        std::unordered_map<int, int> count;
        for (int num : nums) {
            count[num]++;
        }

        int n = static_cast<int>(nums.size());
        std::vector<std::vector<int>> buckets(n + 1);

        for (const auto& [num, freq] : count) {
            buckets[freq].push_back(num);
        }

        std::vector<int> result;
        result.reserve(k);

        for (int f = n; f >= 1 && static_cast<int>(result.size()) < k; --f) {
            for (int num : buckets[f]) {
                result.push_back(num);
                if (static_cast<int>(result.size()) == k) {
                    return result;
                }
            }
        }

        return result;
    }
};
```

#### Java

```java
import java.util.*;

public class Solution {
    // Optimal O(N) Bucket Sort
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }

        int n = nums.length;
        List<Integer>[] buckets = new List[n + 1];
        for (int i = 0; i <= n; i++) {
            buckets[i] = new ArrayList<>();
        }

        for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
            buckets[entry.getValue()].add(entry.getKey());
        }

        int[] result = new int[k];
        int idx = 0;

        for (int f = n; f >= 1 && idx < k; f--) {
            for (int num : buckets[f]) {
                result[idx++] = num;
                if (idx == k) {
                    return result;
                }
            }
        }

        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Bucket Sort Approach:** $\mathcal{O}(N)$. Counting frequencies takes $\mathcal{O}(N)$. Populating buckets takes $\mathcal{O}(U) \le \mathcal{O}(N)$. Scanning buckets takes at most $N + 1$ iterations. Total: strictly $\mathcal{O}(N)$.
  - **Min-Heap Approach:** $\mathcal{O}(N \log k)$ where $U$ unique elements are pushed into a heap of size $k$.
- **Space Complexity:** $\mathcal{O}(N)$ to store frequencies and bucket arrays.

---

### Takeaway Pattern & Interview Traps

1. **Why Bucket Sort Works Here:**
   - Bucket sort is generally risky when keys have arbitrary floating-point values. But here, frequencies are strictly integers bounded by $[1, N]$. This makes bucket sort deterministic, collision-free, and $\mathcal{O}(N)$ optimal!
2. **Min-Heap vs Max-Heap Trade-Off:**
   - A max-heap of all unique elements takes $\mathcal{O}(U \log U)$ to build/extract. A min-heap bounded at size $k$ takes $\mathcal{O}(U \log k)$ and consumes only $\mathcal{O}(k)$ auxiliary memory.