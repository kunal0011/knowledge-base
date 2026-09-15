---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 347: Top K Frequent Elements"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - heap
  - bucket-sort
  - amazon
  - google
  - meta
---

# LeetCode 347: Top K Frequent Elements

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Linear Time Bucket Sort / Min-Heap / Quickselect

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return *the `k` most frequent elements*. You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `List[int]` of length `k`
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^4 \le \text{nums}[i] \le 10^4$
  - `k` is in the range $[1, \text{number of unique elements in nums}]$.
  - It is **guaranteed** that the answer is unique.
- **Follow up:** Your algorithm's time complexity must be better than $\mathcal{O}(n \log n)$, where $n$ is the array's size.

---

### Key Idea & Intuition

#### 1. Comparison of Paradigms:
1. **Sorting Unique Pairs:** Count frequencies and sort pairs $(freq, num)$. Time: $\mathcal{O}(U \log U)$ where $U \le N$ is unique count.
2. **Min-Heap of Size $k$:** Maintain a min-heap of size $k$ based on frequency. Push each unique element, popping when size $> k$. Time: $\mathcal{O}(N + U \log k)$.
3. **Bucket Sort (Optimal Linear Time):**
   - The frequency of any element is an integer strictly bounded between $1$ and $N = \text{len}(nums)$.
   - We can create an array of $N + 1$ buckets, where index $f$ stores a list of all elements that appear exactly $f$ times.
   - Populating the buckets takes $\mathcal{O}(N)$ time.
   - We then iterate backwards from index $N$ down to 1, collecting elements until we have accumulated $k$ elements.
   - This achieves guaranteed **$\mathcal{O}(N)$** time complexity!

---

### Solution Approach (Step-by-Step)

1. **Count Frequencies:**
   - Use a hash map / frequency counter to tally occurrences of each number in `nums`.
2. **Populate Buckets:**
   - Initialize a list of $N + 1$ empty lists: `buckets = [[] for _ in range(len(nums) + 1)]`.
   - For each `(num, freq)` pair in the frequency map, append `num` to `buckets[freq]`.
3. **Collect Top $k$ Elements:**
   - Initialize an empty result list `res = []`.
   - Loop `freq` from $N$ down to $1$:
     - For each `num` in `buckets[freq]`:
       - Append `num` to `res`.
       - If `len(res) == k`, return `res` immediately.
4. **Return `res`.**

---

### Visual Algorithm Walkthrough

#### Example: `nums = [1, 1, 1, 2, 2, 3]`, `k = 2` ($N = 6$)

```
Frequency Count:
  1 -> 3 times
  2 -> 2 times
  3 -> 1 time

Bucket Array of size N+1 = 7:
  Index (Freq)  Elements
  ------------------------
  0             []
  1             [3]
  2             [2]
  3             [1]
  4             []
  5             []
  6             []

Traverse from freq = 6 down to 1:
  freq = 6: empty
  freq = 5: empty
  freq = 4: empty
  freq = 3: add 1       -> res = [1]
  freq = 2: add 2       -> res = [1, 2] -> len(res) == k (2)! Early return!

Result: [1, 2]
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | `k` | Frequencies | Bucket Layout | Output |
| :--- | :--- | :--- | :--- | :--- |
| `[1,1,1,2,2,3]` | `2` | `1:3, 2:2, 3:1` | `b[3]=[1], b[2]=[2], b[1]=[3]` | `[1, 2]` |
| `[1]` | `1` | `1:1` | `b[1]=[1]` | `[1]` |
| `[4,1,-1,2,-1,2,3]` | `2` | `-1:2, 2:2, 4:1, 1:1, 3:1` | `b[2]=[-1, 2]` | `[-1, 2]` |
| `[5, 5, 5, 5]` | `1` | `5:4` | `b[4]=[5]` | `[5]` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        n = len(nums)
        buckets: List[List[int]] = [[] for _ in range(n + 1)]
        
        for num, freq in count.items():
            buckets[freq].append(num)
            
        res: List[int] = []
        for freq in range(n, 0, -1):
            for num in buckets[freq]:
                res.append(num)
                if len(res) == k:
                    return res
                    
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    std::vector<int> topKFrequent(std::vector<int>& nums, int k) {
        std::unordered_map<int, int> count;
        for (int num : nums) {
            count[num]++;
        }

        int n = nums.size();
        std::vector<std::vector<int>> buckets(n + 1);
        for (const auto& [num, freq] : count) {
            buckets[freq].push_back(num);
        }

        std::vector<int> res;
        res.reserve(k);
        for (int freq = n; freq >= 1 && res.size() < static_cast<size_t>(k); --freq) {
            for (int num : buckets[freq]) {
                res.push_back(num);
                if (res.size() == static_cast<size_t>(k)) {
                    return res;
                }
            }
        }

        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }

        int n = nums.length;
        @SuppressWarnings("unchecked")
        List<Integer>[] buckets = new List[n + 1];
        for (int key : count.keySet()) {
            int freq = count.get(key);
            if (buckets[freq] == null) {
                buckets[freq] = new ArrayList<>();
            }
            buckets[freq].add(key);
        }

        int[] res = new int[k];
        int idx = 0;
        for (int freq = n; freq >= 1 && idx < k; freq--) {
            if (buckets[freq] != null) {
                for (int num : buckets[freq]) {
                    res[idx++] = num;
                    if (idx == k) {
                        return res;
                    }
                }
            }
        }

        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$, where $n$ is the length of `nums`. Counting frequencies takes $\mathcal{O}(n)$. Placing elements in buckets takes $\mathcal{O}(U) \le \mathcal{O}(n)$. Scanning buckets backwards visits at most $n$ bucket entries and extracts $k$ items. Overall time is strictly linear $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the frequency map and bucket arrays of total size $n + 1$.

---

### Takeaway Pattern & Interview Traps

1. **Bucket Sort vs Heap:**
   - Mentioning Min-Heap ($\mathcal{O}(n \log k)$) demonstrates standard data structure proficiency. Explaining Bucket Sort ($\mathcal{O}(n)$) showcases first-principles optimization because frequencies are strictly discrete integers in $[1, n]$.
2. **Bucket Array Size:**
   - The array size must be $N + 1$ (index range $[0, N]$), because an element appearing in every position has frequency $N$. Sizing it to $N$ causes an `IndexOutOfBounds` error.
